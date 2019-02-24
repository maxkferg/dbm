import os
import sys
import gym
import time
import math
import time
import random
import pybullet
import numpy as np
from gym import spaces
from gym.utils import seeding
from pprint import pprint
from PIL import Image, ImageDraw
from .utils import *
from .config import URDF_ROOT
from .robots.robot_models import Turtlebot


sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import tools.Math2D as m2d


COUNT = 0
RENDER_WIDTH = 960
RENDER_HEIGHT = 720
RENDER_SIZE = (RENDER_HEIGHT, RENDER_WIDTH)
EPISODE_LEN = 100
ROBOT_DANGER_DISTANCE = 0.8
ROBOT_CRASH_DISTANCE = 0.4
TARGET_REWARD = 1
CHECKPOINT_REWARD = 0.1
CHECKPOINT_DISTANCE = 0.5
BATTERY_THRESHOLD = 0.5
BATTERY_WEIGHT = -0.005
ROTATION_COST = -0.002
CRASHED_PENALTY = -1
TARGET_DISTANCE_THRESHOLD = 0.6 # Max distance to the target
HOST, PORT = "localhost", 9999
COUNT = 0



class Seeker():

    def __init__(self, world, config={}):
        print("Initializing new SeekerEnv")
        print("SimSeekerEnv Config:",config)
        super().__init__()

        self.world = world
        self.physics = world.physics
        self.color = random_color()
        self.actionRepeat = config.get("actionRepeat", 2) # Choose an action every 0.2 seconds
        self.resetOnTarget = config.get("resetOnTarget", True)
        self.debug = config.get("debug", False)
        self.renders = config.get("renders",False)
        self.isDiscrete = config.get("isDiscrete",False)
        self.messaging = config.get("messaging",False)
        self.previous_state = None
        self.ckpt_count = 5

        self.urdf_root = URDF_ROOT
        self.targetUniqueId = -1
        self.robot = None               # The controlled robot
        self.checkpoints = []           # Each checkpoint is given an id. Closest checkpoints are near zero index
        self.dead_checkpoints = []      # List of checkpoints that are not active
        self.collision_objects = []     # Other objects that can be collided with
        self.buildingIds = []           # Each plane is given an id
        self.width = 320                # The resolution of the sensor image (320x240)
        self.height = 240
        self.cam_dist = 3.
        self.cam_pitch = 0.
        self.cam_yaw = 0.
        self.cam_roll = 0.

        self.envStepCounter = 0
        self.startedTime = time.time()
        self.seed()

        ray_count = 12                    # 12 rays of 30 degrees each
        observationDim = 4                # These are positional coordinates
        highs = [10, 10, 10, 10, math.pi, 1, 1, 1] # x, y, pos, pos, theta, vx, vy, vz
        highs.extend([10]*2*self.ckpt_count)
        highs.extend([5]*ray_count)

        observation_high = np.array(highs)

        if self.isDiscrete:
            self.action_space = spaces.Discrete(9)
        else:
            action_dim = 2
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

        self.observation_space = spaces.Box(-observation_high, observation_high, dtype=np.float32)
        self.viewer = None

        # Generate the sensor rays so we don't have to do it repeatedly
        ray_angle = 2. * np.pi / ray_count
        print("Lidar Ray Angle:", ray_angle)

        self.rays = []
        for i in range(ray_count):
            q = make_quaternion([0, 0, 1], i*ray_angle)
            self.rays.append(q)

        # Load the floor file so we don't have to repeatedly read it
        self.build()
        self.reset_checkpoints()
        print("Initialization Complete")


    def __del__(self):
        self.physics = 0


    def build(self):
        """
        Build the environment. Only needs to be done once
        """
        target_pos = gen_start_position(.25, self.world.floor) + [.25]
        car_pos = gen_start_position(.3, self.world.floor) + [.25]
        self.targetUniqueId = self.world.create_shape(pybullet.GEOM_BOX, target_pos, size=0.2, color=self.color)
        #self.targetUniqueId = self.physics.loadURDF(os.path.join(self.urdf_root, "target.urdf"), target_pos)
        
        config = {
            'power': 20,
            'resolution': 1,
            'is_discrete': False,
            'target_pos': target_pos,
            'initial_pos': car_pos
        }
        
        self.robot = Turtlebot(self.physics, config=config)
        self.robot.set_position(car_pos)
        #pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        for i in range(10):
            self.physics.stepSimulation()


    def reset(self):
        """Reset the environment. Move the target and the car"""
        steps = self.envStepCounter / self.actionRepeat
        duration = time.time() - self.startedTime
        if self.debug:
            print("Reset after %i steps in %.2f seconds"%(steps,duration))

        self.startedTime = time.time()
        self.envStepCounter = 0

        # Reset the target and robot position
        self.reset_robot_position()
        self.reset_target_position()
        self.reset_checkpoints()

        # Allow all the objects to reach equilibrium
        for i in range(10):
            self.physics.stepSimulation()
        state = self.get_state()
        return self.get_observation(state)


    def reset_robot_position(self):
        """Move the robot to a new position"""
        car_pos = gen_start_position(.3, self.world.floor) + [.25]
        self.robot.set_position(car_pos)


    def reset_target_position(self):
        """Move the target to a new position"""
        target_pos = gen_start_position(.25, self.world.floor) + [.25]
        _, target_orn = self.physics.getBasePositionAndOrientation(self.targetUniqueId)
        self.physics.resetBasePositionAndOrientation(self.targetUniqueId, np.array(target_pos), target_orn)


    def reset_checkpoints(self):
        """Create new checkpoints at [(vx,yy)...] locations"""
        path = os.path.join(self.urdf_root, "checkpoint.urdf")

        # Remove old checkpoints
        for ckpt in self.checkpoints:
            self.remove_checkpoint(ckpt)

        # Use AStar to find checkpoint locations
        base_pos, carorn = self.physics.getBasePositionAndOrientation(self.robot.racecarUniqueId)
        target_pos, target_orn = self.physics.getBasePositionAndOrientation(self.targetUniqueId)
        nodes = self.world.grid.get_path(base_pos, target_pos)

        # Create new checkpoints
        if nodes is None:
            print("AStar Failed")
        else:
            for i,node in enumerate(nodes):
                if i>0 and i%3 == 0:
                    position = (node.x, node.y, 0.25)
                    self.create_checkpoint(position)
                    

    def create_checkpoint(self, position):
        """
        Create a new checkpoint object
        May take the checkpoint from the dead checkpoints list
        """
        orientation = (0,0,0,1)
        if len(self.dead_checkpoints):
            ckpt = self.dead_checkpoints.pop()
            self.physics.resetBasePositionAndOrientation(ckpt, position, orientation)
        else:
            ckpt = self.world.create_shape(pybullet.GEOM_CYLINDER,
                position,
                color=self.color,
                radius=0.15,
                length=0.04,
                specular=[0.3,0.3,0.3,0.3]
            )
        self.checkpoints.append(ckpt)
        return ckpt


    def remove_checkpoint(self, ckpt):
        """
        Remove a checkpoint from the map, and self.checkpoints
        Also moves the ckpt from self.checkpoints to self.dead_checkpoints
        """
        orientation = (0,0,0,1)
        self.checkpoints.remove(ckpt)
        self.physics.resetBasePositionAndOrientation(ckpt, (10,10,10), orientation)
        self.dead_checkpoints.append(ckpt)


    def read_lidar_values(self, robot_pos, robot_orn):
        """Read values from the laser scanner"""
        # The LIDAR is assumed to be attached to the top (to avoid self-intersection)
        lidar_pos = add_vec(robot_pos, [0, 0, .25])

        # The total length of the ray emanating from the LIDAR
        ray_len = 5

        # Rotate the ray vector and determine intersection
        intersections = []
        for ray in self.rays:
            rot = mul_quat(robot_orn, ray)
            dir_vec = rotate_vector(rot, [1, 0, 0])
            start_pos = add_vec(lidar_pos, scale_vec(.1, dir_vec))
            end_pos = add_vec(lidar_pos, scale_vec(ray_len, dir_vec))
            intersection = self.physics.rayTest(start_pos, end_pos)
            if intersection[0][0] == self.targetUniqueId:
                intersections.append(-1)
            elif intersection[0][0] == self.world.wallId:
                intersections.append(intersection[0][2]*ray_len)
            elif intersection[0][0] in self.collision_objects:
                intersections.append(intersection[0][2]*ray_len)
            else:
                intersections.append(ray_len)
        return intersections


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def get_state(self):
        """
        Return a dict that describes the state of the car
        Calculating the state is computationally intensive and should be done sparingly
        """
        state = {}
        robot_pos, robot_orn = self.physics.getBasePositionAndOrientation(self.robot.racecarUniqueId)
        robot_euler = pybullet.getEulerFromQuaternion(robot_orn)
        robot_theta = robot_euler[2]

        #carmat = self.physics.getMatrixFromQuaternion(robot_orn)
        tarpos, tarorn = self.physics.getBasePositionAndOrientation(self.targetUniqueId)
        invCarPos, invCarOrn = self.physics.invertTransform(robot_pos, robot_orn)
        tarPosInCar, tarOrnInCar = self.physics.multiplyTransforms(invCarPos, invCarOrn, tarpos, tarorn)

        lidar = self.read_lidar_values(robot_pos, robot_orn)

        # Iterate through checkpoints appending them to the distance list
        # Delete any checkpoints close to the robot, and the subsequent checkpoints
        ckpt_positions = []
        is_at_checkpoint = False
        for ckpt in reversed(self.checkpoints):
            pos, _ = self.physics.getBasePositionAndOrientation(ckpt)
            rel_pos = np.array(pos) - np.array(robot_pos)
            rel_distance = np.linalg.norm(rel_pos)
            if rel_distance < CHECKPOINT_DISTANCE:
                is_at_checkpoint = True
            if is_at_checkpoint:
                self.remove_checkpoint(ckpt)
            else:
                ckpt_positions.append(tuple(rel_pos[0:2]))

        # Sort checkpoints. Pad with zeros until length n_ckpt
        ckpt_positions = list(reversed(ckpt_positions)) + [(0,0)]*self.ckpt_count
        ckpt_positions = ckpt_positions[:self.ckpt_count]

        state = {
            "robot_pos": robot_pos,
            "robot_orn": robot_orn,
            "robot_theta": robot_theta,
            "robot_vx": 0,
            "robot_vy": 0,
            "robot_vt": 0,
            "rel_ckpt_positions": ckpt_positions,
            "rel_target_orientation": math.atan2(tarPosInCar[1], tarPosInCar[0]),
            "rel_target_distance": math.sqrt(tarPosInCar[1]**2 + tarPosInCar[0]**2),
            "lidar": lidar,
            "is_at_checkpoint": is_at_checkpoint,
            "is_crashed": self.is_crashed(),
            "is_at_target": self.is_at_target(),
            "is_broken": False,
        }

        if self.previous_state is not None:
            state["robot_vx"] = robot_pos[0] - self.previous_state["robot_pos"][0]
            state["robot_vy"] = robot_pos[1] - self.previous_state["robot_pos"][1]
            state["robot_vt"] = rotation_change(robot_theta, self.previous_state["robot_theta"])

        # Check if the simulation is broken
        if robot_pos[2] < 0 or robot_pos[2] > 1:
            print("Something went wrong with the simulation")
            state["is_broken"] = True

        # Calculate the distance to other robots
        state["other_robots"] = []
        for i in self.collision_objects:
            other_position, _ = self.physics.getBasePositionAndOrientation(i)
            state["other_robots"].append(np.linalg.norm(np.array(robot_pos) - np.array(other_position)))

        if np.any(np.less(state["other_robots"],[ROBOT_CRASH_DISTANCE])):
            state["is_crashed"] = True

        if self.debug:
            print("Target orientation:", state["rel_target_orientation"])
            print("Target position:", state["rel_target_distance"])

        if self.debug>1:
            print("State:")
            pprint(state)

        return state


    def get_observation(self, state):
        """
        Return the observation that is passed to the learning algorithm
        """
        observation = [
            state["rel_target_orientation"],
            state["rel_target_distance"],
            state["robot_pos"][0],
            state["robot_pos"][1],
            state["robot_theta"],
            state["robot_vx"],
            state["robot_vy"],
            state["robot_vt"],
        ]
        observation.extend(flatten(state["rel_ckpt_positions"]))
        observation.extend(state["lidar"])
        return np.array(observation)



    def act(self, action):
        """
        Move the simulation one step forward
        @action is the robot action, in the form [rotation, velocity]
        """
        if self.renders:
            basePos, orn = self.physics.getBasePositionAndOrientation(self.robot.racecarUniqueId)
            # Comment out this line to prevent the camera moving with the car
            #self.physics.resetDebugVisualizerCamera(1, 30, -40, basePos)

        if self.isDiscrete:
            fwd = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
            steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
            forward = fwd[action]
            steer = steerings[action]
            realaction = [forward, steer]
        else:
            realaction = action
        self.action = action
        self.robot.applyAction(realaction)


    def observe(self):
        # Keep the simulation loop as lean as possible.
        action = self.action
        state = self.get_state()
        observation = self.get_observation(state)
        reward = self.reward(state, action)
        done = self.termination(state)

        self.envStepCounter += 1
        self.previous_state = state

        # Respawn the target and clear the isAtTarget flag
        if not self.resetOnTarget and state["is_at_target"]:
            self.reset_target_position()
            self.reset_checkpoints()

        return observation, reward, done, {}


    def is_crashed(self):
        contact = self.physics.getContactPoints(self.robot.racecarUniqueId, self.world.wallId)
        return len(contact)>0


    def is_at_target(self):
        basePos, _ = self.physics.getBasePositionAndOrientation(self.robot.racecarUniqueId)
        targetPos, _ = self.physics.getBasePositionAndOrientation(self.targetUniqueId)
        return np.linalg.norm(np.array(basePos) - np.array(targetPos)) < TARGET_DISTANCE_THRESHOLD


    def termination(self, state):
        """Return True if the episode should end"""
        return state["is_crashed"] or state["is_broken"] or (self.resetOnTarget and state["is_at_target"])


    def reward(self, state, action):
        """
        Return the reward:
            Target Reward: 1 if target reached, else 0
            Collision Reward: -1 if crashed, else 0
            Battery Reward: Penalty if rotation or velocity exceeds 0.5
            Rotation Reward: Small penalty for any rotation
        """
        # Add positive reward if we are near the target
        if state["is_at_target"]:
            target_reward = TARGET_REWARD
        else:
            target_reward = 0

        # End the simulation with negative reward
        if state["is_crashed"]:
            crashed_reward = CRASHED_PENALTY
        else:
            crashed_reward = 0

        # Reward for reaching a checkpoint
        if state["is_at_checkpoint"]:
            checkpoint_reward = CHECKPOINT_REWARD
        else:
            checkpoint_reward = 0

        # Penalty for closeness
        danger_reward = 0
        for other in state["other_robots"]:
            if other < ROBOT_DANGER_DISTANCE:
                danger_reward -= math.exp(20*(ROBOT_CRASH_DISTANCE-other))
        danger_reward = max(-1, danger_reward)

        # There is a cost to acceleration and turning
        # We use the squared cost to incentivise careful use of battery resources
        battery_reward = BATTERY_WEIGHT*np.sum(positive_component(np.abs(action) - BATTERY_THRESHOLD))

        # There is an additional cost due to rotation
        rotation_reward = ROTATION_COST * abs(state["robot_vt"])

        # Total reward is the sum of components
        reward = target_reward + crashed_reward + battery_reward + rotation_reward + checkpoint_reward + danger_reward

        if self.debug:
            print("---- Step %i Summary -----"%self.envStepCounter)
            print("Action: ", action)
            print("Target Reward:  %.3f"%target_reward)
            print("Checkpoint Reward:  %.3f"%checkpoint_reward)
            print("Crashed Reward: %.3f"%crashed_reward)
            print("Battery Reward: %.3f"%battery_reward)
            print("Rotation Reward: %.3f"%rotation_reward)
            print("Danger Reward: %.3f"%danger_reward)
            print("Total Reward:   %.3f\n"%reward)

        return reward


    def render(self, mode='rgb_array', close=False, width=640, height=480):
        """Render the simulation to a frame"""
        if mode != "rgb_array":
            return np.array([])

        # Move the camera with the base_pos
        base_pos, carorn = self.physics.getBasePositionAndOrientation(self.robot.racecarUniqueId)

        # Position the camera behind the car, slightly above
        dir_vec = np.array(rotate_vector(carorn, [2, 0, 0]))
        cam_eye = np.subtract(np.array(base_pos), np.add(dir_vec, np.array([0, 0, -1])))
        cam_up = normalize(self.world.world_up - np.multiply(np.dot(self.world.world_up, dir_vec), dir_vec))

        view_matrix = self.physics.computeViewMatrix(
            cameraEyePosition=cam_eye,
            cameraTargetPosition=base_pos,
            cameraUpVector=cam_up)
        proj_matrix = self.physics.computeProjectionMatrixFOV(
            fov=60, aspect=float(width) / height,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, seg) = self.physics.getCameraImage(
            width=width, height=height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = rgb_array.reshape((height, width, 4))

        h,w = 256,256
        rgb_array[0:h,0:w] = self.render_observation(w,h)

        return rgb_array


    def render_observation(self, width=128, height=128):
        # Move the camera with the base_pos
        base_pos, carorn = self.physics.getBasePositionAndOrientation(self.robot.racecarUniqueId)

        # Position the camera behind the car, slightly above
        dir_vec = np.array(rotate_vector(carorn, [2, 0, 0]))
        cam_eye = np.subtract(np.array(base_pos), np.array([0, 0, -5]))
        cam_up = normalize(self.world.world_up - np.multiply(np.dot(self.world.world_up, dir_vec), dir_vec))

        view_matrix = self.physics.computeViewMatrix(
            cameraEyePosition=cam_eye,
            cameraTargetPosition=base_pos,
            cameraUpVector=cam_up)
        proj_matrix = self.physics.computeProjectionMatrixFOV(
            fov=60, aspect=float(width) / height,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, seg) = self.physics.getCameraImage(
            width=width, height=height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        #rgb_array = np.array(px, dtype=np.uint8)
        #rgb_array = rgb_array.reshape((height, width, 4))

        rgb_array = 40*np.array(seg, dtype=np.uint8)
        rgb_array = rgb_array.reshape((height, width, 1))
        rgb_array = np.tile(rgb_array, (1,1,4))

        return rgb_array


    def render_map(self, mode, width=640, height=480):
        """
        Render the map to a pixel buffer
        """
        im = Image.new('RGB', (width,height))
        draw = ImageDraw.Draw(im)

        base_pos, carorn = self.physics.getBasePositionAndOrientation(self.robot.racecarUniqueId)
        target_pos, target_orn = self.physics.getBasePositionAndOrientation(self.targetUniqueId)

        base_pos_pixels = self.world.scale(base_pos, width, height)
        target_pos_pixels = self.world.scale(target_pos, width, height)
        robot = [base_pos_pixels[0]-20, base_pos_pixels[1]-20, base_pos_pixels[0]+20, base_pos_pixels[1]+20]
        target = [target_pos_pixels[0]-20, target_pos_pixels[1]-20, target_pos_pixels[0]+20, target_pos_pixels[1]+20]

        for v0, v1, v2 in self.world.get_quads():
            v0 = self.world.scale(v0, width, height)
            v1 = self.world.scale(v1, width, height)
            v2 = self.world.scale(v2, width, height)
            xmin = min(v0[0], v1[0], v2[0])
            xmax = max(v0[0], v1[0], v2[0])
            ymin = min(v0[1], v1[1], v2[1])
            ymax = max(v0[1], v1[1], v2[1])
            draw.rectangle([(xmin,ymin), (xmax,ymax)], fill="#ff0000")
        draw.ellipse(robot, fill = 'blue', outline ='blue')
        draw.ellipse(target, fill = 'green', outline ='green')

        # Create the fastest path
        nodes = self.world.grid.get_path(base_pos, target_pos)
        if nodes is not None:
            for node in nodes:
                point = self.world.scale((node.x, node.y, 0), width, height)
                draw.text(point, "x", fill=(255,255,255,128))

        del draw
        return np.array(im)