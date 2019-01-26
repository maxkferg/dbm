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


HOST, PORT = "localhost", 9999
COUNT = 0

class SeekerSimEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, urdfRoot=URDF_ROOT, actionRepeat=50,
                 isEnableSelfCollision=True, isDiscrete=False, renders=False, debug=0):
        print("Initializing new SeekerSimEnv")
        self.timeStep = .002
        self.urdfRoot = urdfRoot
        self.actionRepeat = actionRepeat
        self.isEnableSelfCollision = isEnableSelfCollision
        self.observation = []
        self.targetUniqueId = -1
        self.robot = None               # The controlled robot
        self.buildingIds = []           # Each plane is given an id
        self.width = 320                # The resolution of the sensor image (320x240)
        self.height = 240
        self.cam_dist = 3.
        self.cam_pitch = 0.
        self.cam_yaw = 0.
        self.cam_roll = 0.
        self.debug = debug

        self.envStepCounter = 0
        self.renders = renders
        self.isDiscrete = isDiscrete
        self.startedTime = time.time()
        self.tile_grid = TileGrid(self.urdfRoot + "/output_floors.obj")
        #self.tile_grid.build_grid()
        #self.tile_grid.build_map()

        if self.renders:
            print("Creating new BulletClient (GUI)")
            self.physics = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            print(self.urdfRoot + "/output_floors.obj")
            self.mpqueue = MPQueueClient(HOST, PORT)
            self.mpqueue.start(self.urdfRoot + "/output_floors.obj", self.urdfRoot + "/output_walls.obj")

        else:
            print("Creating new BulletClient")
            self.physics = bullet_client.BulletClient()
            self.mpqueue = None

        self.seed()
        ray_count = 12                      # 12 rays of 30 degrees each
        observationDim = 4                  # These are positional coordinates
        highs = [10, 10, 1, 1]            # Distance, angle and sine/consine of angles
        highs.extend([5]*ray_count)

        observation_high = np.array(highs)      #snp.ones(observationDim) * 1000  # np.inf

        if isDiscrete:
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
        print("ray_angle:", ray_angle)

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


    def build(self):
        """
        Build the environment. Only needs to be done once
        """
        print("Building simulation environment")
        self.physics.resetSimulation()
        self.physics.setTimeStep(self.timeStep)
        self.buildingIds = self.physics.loadSDF(os.path.join(self.urdfRoot, "output.sdf"))
        self.isCrashed = False
        self.isAtTarget = False

        target_pos = gen_start_position(.25, self.floor) + [.25]
        car_pos = gen_start_position(.3, self.floor) + [.25]
        self.targetUniqueId = self.physics.loadURDF(os.path.join(self.urdfRoot, "target.urdf"), target_pos)
        config = {
            'power': 40,
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

        self.observation = self.getExtendedObservation()
        return np.array(self.observation)


    def reset(self):
        """Reset the environment. Move the target and the car"""
        steps = self.envStepCounter / self.actionRepeat
        duration = time.time() - self.startedTime
        if self.debug:
            print("Reset after %i steps in %.2f seconds"%(steps,duration))

        self.isCrashed = False
        self.isAtTarget = False
        self.startedTime = time.time()
        self.envStepCounter = 0

        target_pos = gen_start_position(.25, self.floor) + [.25]
        car_pos = gen_start_position(.3, self.floor) + [.25]
        #self.targetUniqueId = self.physics.loadURDF(os.path.join(self.urdfRoot, "target.urdf"), target_pos)
        _, target_orn = self.physics.getBasePositionAndOrientation(self.targetUniqueId)
        self.physics.resetBasePositionAndOrientation(self.targetUniqueId, np.array(target_pos), target_orn)

        #target set position
        self.robot.set_position(car_pos)

        for i in range(100):
            self.physics.stepSimulation()

        self.observation = self.getExtendedObservation()

        return np.array(self.observation)


    def __del__(self):
        self.physics = 0


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def getExtendedObservation(self):
        self.observation = []

        carpos, carorn = self.physics.getBasePositionAndOrientation(self.robot.racecarUniqueId)

        if self.mpqueue:
            scale = self.floor[2]
            dims = self.floor[3]
            centre = compute_centre(self.tile_grid.bound)
            #print("centre:", centre)
            pos = [int((carpos[0]/scale) * dims[0] + dims[0]),
                   int((-carpos[1]/scale) * dims[1] + dims[1])]

            #print("POS:", pos, carpos)
            self.mpqueue.command_move(pos)

            dir_vec = rotate_vector(carorn, [1, 0, 0])
            angle = m2d.compute_angle(m2d.cp_mul(m2d.vec3_to_vec2n(dir_vec), [1, -1]))
            #print("Car Forward:", angle)

            self.mpqueue.command_turn(angle)

        carmat = self.physics.getMatrixFromQuaternion(carorn)
        tarpos, tarorn = self.physics.getBasePositionAndOrientation(self.targetUniqueId)
        invCarPos, invCarOrn = self.physics.invertTransform(carpos, carorn)
        tarPosInCar, tarOrnInCar = self.physics.multiplyTransforms(invCarPos, invCarOrn, tarpos, tarorn)

        self.observation.extend([
            tarPosInCar[0],
            tarPosInCar[1],
            math.atan2(tarPosInCar[1], tarPosInCar[0]),
            0 # dummy
        ])

        if self.debug:
            print("Target position:", tarPosInCar)
            print("Target orientation:", math.atan2(tarPosInCar[0], tarPosInCar[1]))

        # The LIDAR is assumed to be attached to the top (to avoid self-intersection)
        lidar_pos = add_vec(carpos, [0, 0, .25])

        # The total length of the ray emanating from the LIDAR
        ray_len = 5
        # Rotate the ray vector and determine intersection
        intersections = []
        for ray in self.rays:
            rot = mul_quat(carorn, ray)
            dir_vec = rotate_vector(rot, [1, 0, 0])
            start_pos = add_vec(lidar_pos, scale_vec(.1, dir_vec))
            end_pos = add_vec(lidar_pos, scale_vec(ray_len, dir_vec))
            intersection = self.physics.rayTest(start_pos, end_pos)
            if intersection[0][0] == self.targetUniqueId:
                intersections.append(-1)
            elif intersection[0][0] == self.buildingIds[0]:
                #print(intersection[0])
                #print("--------------------")
                intersections.append(intersection[0][2])
            else:
                intersections.append(ray_len)

        if self.debug>1:
            print("Lidar intersections:", intersections)

        self.observation.extend(intersections)
        return self.observation


    def step(self, action):
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
        for i in range(self.actionRepeat):
            self.physics.stepSimulation()
            if self.renders:
                time.sleep(self.timeStep)
            self.observation = self.getExtendedObservation()

            if self.termination():
                break

            self.envStepCounter += 1
        reward = self.reward()
        done = self.termination()
        self.last_action = action

        return np.array(self.observation), reward, done, {}


    # This function is not being called in the test SeekerSimEnv
    def render(self, mode='human', close=False):
        if mode != "rgb_array":
            return np.array([])

        # Move the camera with the base_pos
        base_pos, carorn = self.physics.getBasePositionAndOrientation(self.robot.racecarUniqueId)

        # Position the camera behind the car, slightly above
        dir_vec = np.array(rotate_vector(carorn, [1, 0, 0]))
        cam_eye = np.subtract(np.array(base_pos), np.add(dir_vec, np.array([0, 0, -.5])))
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

    # Note: The termination condition is specified in steps.  The step size is .01 and therefore the counter should be
    # divided by 100 to compute the number of seconds
    def termination(self):
        # Want agent to make 100 actions. 50 physics steps per action. Total duration 1000
        total_sim_duration = 5000  # 200 seconds for small model, 1000 for big model?
        return self.envStepCounter > total_sim_duration or self.isCrashed or self.isAtTarget

    def reward(self):
        # Adapt the reward to:
        # 1 if target reached, else 0
        # -1 if wall collision
        TARGET_DISTANCE_THRESHOLD = 0.6 # Max distance to the target
        COLLISION_DISTANCE = 0.04
        TARGET_REWARD = 10
        BATTERY_THRESHOLD = 0.005
        BATTERY_WEIGHT = -50
        CRASHED_PENALTY = -10
        closestPoints = self.physics.getClosestPoints(self.robot.racecarUniqueId, self.targetUniqueId, 10000)

        # Default reward is zero
        numPt = len(closestPoints)

        # Add positive reward if we are near the target
        # Distance based reward
        #if numPt > 0:
        #    target_reward = TARGET_WEIGHT*(MAX_DISTANCE - closestPoints[0][8])       # (contactFlag, bodyUniqueIdA, bodyUniqueIdB, linkIndexA, linkIndexB, positionOnA, positionOnB, contactNormalOnB, contactDistance, normalForce)
        #else:
        #    target_reward = 0
        if abs(closestPoints[0][8]) < TARGET_DISTANCE_THRESHOLD:
            target_reward = TARGET_REWARD
            self.isAtTarget = True
        else:
            target_reward = 0

        # If the robot is too closs to the wall (not the target) then end the simulation with negative reward
        crashed_reward = 0
        for observation in self.observation[4:]:
            if -1 < observation and observation < COLLISION_DISTANCE:
                self.isCrashed = True
                crashed_reward = CRASHED_PENALTY

        # There is a cost to acceleration and turning
        # We use the squared cost to incentivise careful use of battery resources
        battery_reward = BATTERY_WEIGHT*np.sum(positive_component(np.abs(self.robot.last_action) - BATTERY_THRESHOLD))

        # Total reward is the sum of components
        reward = target_reward + crashed_reward + battery_reward

        if self.debug:
            print("---- Step %i Summary -----"%self.envStepCounter)
            print("Action: ",self.robot.last_action)
            print("Target Reward:  %.3f"%target_reward)
            print("Crashed Reward: %.3f"%crashed_reward)
            print("Battery Reward: %.3f"%battery_reward)
            print("Total Reward:   %.3f\n"%reward)

        return reward
