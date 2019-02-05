import os
import gym
import time
import math
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet
from . import bullet_client
from .config import URDF_ROOT
from pprint import pprint
import random
from .robots.robot_models import Turtlebot
from random import random, randint
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from tools.MPQueueClient import MPQueueClient
from tools.TileGrid import TileGrid, compute_centre, AABB_to_vertices
import tools.Math2D as m2d

COUNT = 0
RENDER_WIDTH = 960
RENDER_HEIGHT = 720
EPISODE_LEN = 100



def normalize(vec):
    return np.multiply(vec, np.linalg.norm(vec))


def rotate_vector(quat, vec):
    n1 = quat[0] * 2.
    n2 = quat[1] * 2.
    n3 = quat[2] * 2.
    n4 = quat[0] * n1
    n5 = quat[1] * n2
    n6 = quat[2] * n3
    n7 = quat[0] * n2
    n8 = quat[0] * n3
    n9 = quat[1] * n3
    n10 = quat[3] * n1
    n11 = quat[3] * n2
    n12 = quat[3] * n3
    result = [0, 0, 0]
    result[0] = (1. - (n5 + n6)) * vec[0] + (n7 - n12) * vec[1] + (n8 + n11) * vec[2]
    result[1] = (n7 + n12) * vec[0] + (1. - (n4 + n6)) * vec[1] + (n9 - n10) * vec[2]
    result[2] = (n8 - n11) * vec[0] + (n9 + n10) * vec[1] + (1. - (n4 + n5)) * vec[2]
    return result


def make_quaternion(axis, angle_in_radians):
    n = (axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2])
    rad = angle_in_radians * .5
    sin_theta = np.sin(rad)
    cos_theta = np.cos(rad)
    s = sin_theta / np.sqrt(n)
    return [s*axis[0], s*axis[1], s*axis[2], cos_theta]


def mul_quat(qA, qB):
    return [
        qA[3]*qB[0] + qA[0]*qB[3] + qA[1]*qB[2] - qA[2]*qB[2],
        qA[3]*qB[1] + qA[1]*qB[3] + qA[2]*qB[0] - qA[0]*qB[2],
        qA[3]*qB[2] + qA[2]*qB[3] + qA[0]*qB[1] - qA[1]*qB[0],
        qA[3]*qB[3] - qA[0]*qB[0] - qA[1]*qB[1] - qA[2]*qB[2],
    ]


def positive_component(array):
    """Replace postive values with zero"""
    return (np.abs(array) + array)/2


def scale_vec(scale, vec):
    return [scale*vec[0], scale*vec[1], scale*vec[2]]


def add_vec(vA, vB):
    return [vA[0]+vB[0], vA[1]+vB[1], vA[2]+vB[2]]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def rotation_change(theta1,theta2):
    """Compute the change in rotation, assuming small angle change"""
    dt = theta2 - theta1
    dt1 = dt - 2*math.pi
    dt2 = dt + 2*math.pi
    return find_nearest([dt, dt1, dt2], 0)


def load_floor_file(path):
    file = open(path)

    vertices = []
    indices = []
    scale = 1
    dims = []

    for line in file:
        if line[0:2] == 'v ':
            els = line.split(' ')
            vertices.append([scale*float(els[1]), scale*float(els[2]), scale*float(els[3].strip('\n'))])
        elif line[0:2] == 'f ':
            els = line.split(' ')
            indices.append(
                [int(els[1].split('/')[0]) - 1, int(els[2].split('/')[0]) - 1, int(els[3].split('/')[0]) - 1])
        elif line[0:7] == '#scale ':
            scale = float(line.split(' ')[1])
        elif line[0:5] == '#dims':
            els = line.split(' ')
            dims = [int(els[1]), int(els[2])]

    file.close()

    return [vertices, indices, scale, dims]


def gen_start_position(radius, floor):
    def lerp(A, B, t):
        return (1 - t)*A + t*B

    # Select a random quad and generate a position on the quad with sufficient distance from the walls
    quad_count = len(floor[1])/2

    done = False
    ret = []
    while not done:
        quad = randint(0, quad_count - 1)
        qidx = 2 * quad
        f0, f1, f2 = floor[1][qidx]
        v0, v1, v2 = floor[0][f0], floor[0][f1], floor[0][f2]

        for i in range(20):
            u, v = random(), random()
            x, y = lerp(v0[0], v1[0], u), lerp(v0[1], v2[1], v)

            lb = [v0[0], v0[1]]
            rt = [v1[0], v2[1]]

            if lb[0] < x-radius and x+radius < rt[0] and lb[1] < y-radius and y+radius < rt[1]:
                ret = [x, y]
                done = True
                break

    return ret


COLLISION_DISTANCE = 0.2
TARGET_REWARD = 1
BATTERY_THRESHOLD = 0.5
BATTERY_WEIGHT = -0.005
ROTATION_COST = -0.002
CRASHED_PENALTY = -1
TARGET_DISTANCE_THRESHOLD = 0.6 # Max distance to the target
HOST, PORT = "localhost", 9999
COUNT = 0

class SeekerSimEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, config={}):
        print("Initializing new SeekerSimEnv")
        print("SimSeekerEnv Config:",config)

        self.timeStep = config.get("timestep", 0.1) # Simulate every 0.1 seconds
        self.urdfRoot = config.get("urdfRoot", URDF_ROOT)
        self.actionRepeat = config.get("actionRepeat", 4) # Choose an action every 0.4 seconds
        self.isEnableSelfCollision = config.get("isEnableSelfCollision", False)
        self.debug = config.get("debug", False)
        self.renders = config.get("renders",False)
        self.isDiscrete = config.get("isDiscrete",False)
        self.messaging = config.get("messaging",False)
        self.previous_state = None

        self.targetUniqueId = -1
        self.robot = None               # The controlled robot
        self.buildingIds = []           # Each plane is given an id
        self.width = 320                # The resolution of the sensor image (320x240)
        self.height = 240
        self.cam_dist = 3.
        self.cam_pitch = 0.
        self.cam_yaw = 0.
        self.cam_roll = 0.

        self.envStepCounter = 0
        self.startedTime = time.time()
        self.tile_grid = TileGrid(self.urdfRoot + "/output_floors.obj")
        self.mpqueue = None

        if self.renders:
            print("Creating new BulletClient (GUI)")
            self.physics = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            print(self.urdfRoot + "/output_floors.obj")
        else:
            print("Creating new BulletClient")
            self.physics = bullet_client.BulletClient()
            self.mpqueue = None

        if self.messaging:
            self.mpqueue = MPQueueClient(HOST, PORT)
            self.mpqueue.start(self.urdfRoot + "/output_floors.obj", self.urdfRoot + "/output_walls.obj")

        self.seed()
        ray_count = 12                    # 12 rays of 30 degrees each
        observationDim = 4                # These are positional coordinates
        highs = [10, 10, 10, 10, math.pi, 1, 1, 1] # x, y, pos, pos, theta, vx, vy, vz
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
        self.floor = load_floor_file(self.urdfRoot + "/output_floors.obj")
        self.world_up = np.array([0, 0, 1])
        self.build()
        self.reset()
        print("Initialization Complete")


    def __del__(self):
        self.physics = 0


    def build(self):
        """
        Build the environment. Only needs to be done once
        """
        print("Building simulation environment")
        self.physics.resetSimulation()
        self.physics.setTimeStep(self.timeStep)
        self.physics.setPhysicsEngineParameter(numSubSteps=50) # Adjust until stable
        self.buildingIds = self.physics.loadSDF(os.path.join(self.urdfRoot, "output.sdf"))

        target_pos = gen_start_position(.25, self.floor) + [.25]
        car_pos = gen_start_position(.3, self.floor) + [.25]
        self.targetUniqueId = self.physics.loadURDF(os.path.join(self.urdfRoot, "target.urdf"), target_pos)
        config = {
            'power': 20,
            'resolution': 1,
            'is_discrete': False,
            'target_pos': target_pos,
            'initial_pos': car_pos
        }
        self.robot = Turtlebot(self.physics, config=config)
        self.robot.set_position(car_pos)

        self.physics.setGravity(0, 0, -10)

        for i in range(100):
            self.physics.stepSimulation()

        state = self.get_state()
        return self.get_observation(state)


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

        # Allow all the objects to reach equilibrium
        for i in range(10):
            self.physics.stepSimulation()

        state = self.get_state()
        return self.get_observation(state)


    def reset_robot_position(self):
        """Move the robot to a new position"""
        car_pos = gen_start_position(.3, self.floor) + [.25]
        self.robot.set_position(car_pos)


    def reset_target_position(self):
        """Move the target to a new position"""
        target_pos = gen_start_position(.25, self.floor) + [.25]
        _, target_orn = self.physics.getBasePositionAndOrientation(self.targetUniqueId)
        self.physics.resetBasePositionAndOrientation(self.targetUniqueId, np.array(target_pos), target_orn)


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
            #print("lidar start:",start_pos, "lidar end:",end_pos)
            #time.sleep(1)
            intersection = self.physics.rayTest(start_pos, end_pos)
            if intersection[0][0] == self.targetUniqueId:
                intersections.append(-1)
            elif intersection[0][0] == self.buildingIds[0]:
                #print(intersection[0])
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

        if self.mpqueue:
            scale = self.floor[2]
            dims = self.floor[3]
            centre = compute_centre(self.tile_grid.bound)
            pos = [int((carpos[0]/scale) * dims[0] + dims[0]),
                   int((-carpos[1]/scale) * dims[1] + dims[1])]
            self.mpqueue.command_move(pos)
            dir_vec = rotate_vector(carorn, [1, 0, 0])
            angle = m2d.compute_angle(m2d.cp_mul(m2d.vec3_to_vec2n(dir_vec), [1, -1]))
            self.mpqueue.command_turn(angle)

        robot_pos, robot_orn = self.physics.getBasePositionAndOrientation(self.robot.racecarUniqueId)
        robot_euler = pybullet.getEulerFromQuaternion(robot_orn)
        robot_theta = robot_euler[2]

        #carmat = self.physics.getMatrixFromQuaternion(robot_orn)
        tarpos, tarorn = self.physics.getBasePositionAndOrientation(self.targetUniqueId)
        invCarPos, invCarOrn = self.physics.invertTransform(robot_pos, robot_orn)
        tarPosInCar, tarOrnInCar = self.physics.multiplyTransforms(invCarPos, invCarOrn, tarpos, tarorn)

        lidar = self.read_lidar_values(robot_pos, robot_orn)

        state = {
            "robot_pos": robot_pos,
            "robot_orn": robot_orn,
            "robot_theta": robot_theta,
            "robot_vx": 0,
            "robot_vy": 0,
            "robot_vt": 0,
            "rel_target_orientation": math.atan2(tarPosInCar[1], tarPosInCar[0]),
            "rel_target_distance": math.sqrt(tarPosInCar[1]**2 + tarPosInCar[0]**2),
            "lidar": lidar,
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
            state["robot_vt"]
        ]
        observation.extend(state["lidar"])
        return np.array(observation)


    def step(self, action):
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

        self.robot.applyAction(realaction)

        # Keep the simulation loop as lean as possible.
        # Technically we should check for crash or target state in the loop, but that is slow
        for i in range(self.actionRepeat):
            self.physics.stepSimulation()
            if self.is_crashed() or self.is_at_target():
                break

        state = self.get_state()
        observation = self.get_observation(state)
        reward = self.reward(state, action)
        done = self.termination(state)

        self.envStepCounter += 1
        self.previous_state = state

        # Respawn the target and clear the isAtTarget flag
        if state["is_at_target"]:
            self.reset_target_position()

        return observation, reward, done, {}


    def is_crashed(self):
        walls = self.buildingIds[0];
        contact = self.physics.getContactPoints(self.robot.racecarUniqueId, walls)
        return len(contact)>0


    def is_at_target(self):
        contact = self.physics.getContactPoints(self.robot.racecarUniqueId, self.targetUniqueId)
        return len(contact)>0


    def termination(self, state):
        """Return True if the episode should end"""
        return state["is_crashed"] or state["is_broken"]


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

        # There is a cost to acceleration and turning
        # We use the squared cost to incentivise careful use of battery resources
        battery_reward = BATTERY_WEIGHT*np.sum(positive_component(np.abs(action) - BATTERY_THRESHOLD))

        # There is an additional cost due to rotation
        rotation_reward = ROTATION_COST * abs(state["robot_vt"])

        # Total reward is the sum of components
        reward = target_reward + crashed_reward + battery_reward + rotation_reward

        if self.debug:
            print("---- Step %i Summary -----"%self.envStepCounter)
            print("Action: ", action)
            print("Target Reward:  %.3f"%target_reward)
            print("Crashed Reward: %.3f"%crashed_reward)
            print("Battery Reward: %.3f"%battery_reward)
            print("Rotation Reward: %.3f"%rotation_reward)
            print("Total Reward:   %.3f\n"%reward)

        return reward


    def render(self, mode='human', close=False):
        """Render the simulation to a frame"""
        if mode != "rgb_array":
            return np.array([])

        # Move the camera with the base_pos
        base_pos, carorn = self.physics.getBasePositionAndOrientation(self.robot.racecarUniqueId)

        # Position the camera behind the car, slightly above
        dir_vec = np.array(rotate_vector(carorn, [2, 0, 0]))
        cam_eye = np.subtract(np.array(base_pos), np.add(dir_vec, np.array([0, 0, -1])))
        cam_up = normalize(self.world_up - np.multiply(np.dot(self.world_up, dir_vec), dir_vec))

        view_matrix = self.physics.computeViewMatrix(
            cameraEyePosition=cam_eye,
            cameraTargetPosition=base_pos,
            cameraUpVector=cam_up)
        proj_matrix = self.physics.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self.physics.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = rgb_array.reshape((RENDER_HEIGHT, RENDER_WIDTH, 4))
        return rgb_array

