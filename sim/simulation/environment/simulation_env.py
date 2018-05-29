import os
import gym
import math
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet
from . import SimRobot
from . import bullet_client
import random

RENDER_WIDTH = 960
RENDER_HEIGHT = 720


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


class SeekerSimEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, urdfRoot="/Users/otgaard/Development/dbm/sim/assets", actionRepeat=50,
                 isEnableSelfCollision=True, isDiscrete=False, renders=False):
        self.timeStep = .01
        self.urdfRoot = urdfRoot
        self.actionRepeat = actionRepeat
        self.isEnableSelfCollision = isEnableSelfCollision
        self.observation = []
        self.targetUniqueId = -1
        self.robot = None               # The controlled robot
        self.buildingIds = []           # Each plane is given an id
        self.width = 320                # The resolution of the sensor image (320x240)
        self.height = 240

        self.envStepCounter = 0
        self.renders = renders
        self.isDiscrete = isDiscrete
        if self.renders:
            self.physics = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.physics = bullet_client.BulletClient()

        self.seed()
        observationDim = 2  # len(self.getExtendedObservation())
        # print("observationDim")
        # print(observationDim)
        # observation_high = np.array([np.finfo(np.float32).max] * observationDim)
        observation_high = np.ones(observationDim) * 1000  # np.inf
        if isDiscrete:
            self.action_space = spaces.Discrete(9)
        else:
            action_dim = 2
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.viewer = None

    def reset(self):
        self.physics.resetSimulation()
        self.physics.setTimeStep(self.timeStep)
        self.buildingIds = self.physics.loadSDF(os.path.join(self.urdfRoot, "output.sdf"))

        # TODO: Determine location to spawn ball!
        #dist = 5 + 2. * random.random()
        #ang = 2. * 3.1415925438 * random.random()

        ballx = 1       #dist * math.sin(ang)
        bally = 0       #dist * math.cos(ang)
        ballz = .5      #1

        self.targetUniqueId = self.physics.loadURDF(os.path.join(self.urdfRoot, "target.urdf"), [ballx, bally, ballz])
        self.physics.setGravity(0, 0, -10)
        self.robot = SimRobot.SimRobot(self.physics, urdfRootPath=self.urdfRoot, timeStep=self.timeStep)
        self.envStepCounter = 0
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

        ray_count = 12          # 12 rays of 30 degrees each
        ray_angle = 2. * np.pi / ray_count
        print("ray_angle:", ray_angle)
        carpos, carorn = self.physics.getBasePositionAndOrientation(self.robot.racecarUniqueId)
        tarpos, tarorn = self.physics.getBasePositionAndOrientation(self.targetUniqueId)
        invCarPos, invCarOrn = self.physics.invertTransform(carpos, carorn)
        tarPosInCar, tarOrnInCar = self.physics.multiplyTransforms(invCarPos, invCarOrn, tarpos, tarorn)

        # Get the car's direction in Euler angles
        dir_vec = rotate_vector(carorn, [1, 0, 0])
        print("Dir:", dir_vec)

        # Test rotation by quaternion
        for i in range(ray_count):
            q = make_quaternion([0, 0, 1], i*ray_angle)

            R = mul_quat(carorn, q)

            print("q:", i, q)
            print("R:", i, R)
            dir_vec = rotate_vector(q, [1, 0, 0])
            print("Dir:", i, dir_vec)
            dir_vec = rotate_vector(R, [1, 0, 0])
            print("Dir:", i, dir_vec)

        self.observation.extend([tarPosInCar[0], tarPosInCar[1]])

        return self.observation

    #def getExtendedObservation(self):
    #    # TODO:  Add 12 angle ray-collision test (verify details)
    #    self.observation = []  # self._racecar.getObservation()
    #    carpos, carorn = self.physics.getBasePositionAndOrientation(self.robot.racecarUniqueId)
    #    ballpos, ballorn = self.physics.getBasePositionAndOrientation(self.targetUniqueId)
    #    invCarPos, invCarOrn = self.physics.invertTransform(carpos, carorn)
    #    ballPosInCar, ballOrnInCar = self.physics.multiplyTransforms(invCarPos, invCarOrn, ballpos, ballorn)

    #    self.observation.extend([ballPosInCar[0], ballPosInCar[1]])
    #    return self.observation

    #def getExtendedObservation(self):
    #    carpos, carorn = self.physics.getBasePositionAndOrientation(self.robot.racecarUniqueId)
    #    carmat = self.physics.getMatrixFromQuaternion(carorn)
    #    ballpos, ballorn = self.physics.getBasePositionAndOrientation(self.ballUniqueId)
    #    invCarPos, invCarOrn = self.physics.invertTransform(carpos, carorn)
    #    ballPosInCar, ballOrnInCar = self.physics.multiplyTransforms(invCarPos, invCarOrn, ballpos, ballorn)
    #    dist0 = 0.3
    #    dist1 = 1.
    #    eyePos = [carpos[0] + dist0 * carmat[0], carpos[1] + dist0 * carmat[3], carpos[2] + dist0 * carmat[6] + 0.3]
    #    targetPos = [carpos[0] + dist1 * carmat[0], carpos[1] + dist1 * carmat[3],
    #                 carpos[2] + dist1 * carmat[6] + 0.3]
    #    up = [carmat[2], carmat[5], carmat[8]]
    #    viewMat = self.physics.computeViewMatrix(eyePos, targetPos, up)
    #    # viewMat = self._p.computeViewMatrixFromYawPitchRoll(carpos,1,0,0,0,2)
    #    # print("projectionMatrix:")
    #    # print(self._p.getDebugVisualizerCamera()[3])
    #    projMatrix = [0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0,
    #                  0.0, 0.0, -0.02000020071864128, 0.0]
    #    img_arr = self.physics.getCameraImage(width=self.width, height=self.height, viewMatrix=viewMat,
    #                                     projectionMatrix=projMatrix)
    #    rgb = img_arr[2]
    #    np_img_arr = np.reshape(rgb, (self.height, self.width, 4))
    #    self.observation = np_img_arr
    #    return self.observation

    def step(self, action):
        if self.renders:
            basePos, orn = self.physics.getBasePositionAndOrientation(self.robot.racecarUniqueId)
            self.physics.resetDebugVisualizerCamera(1, 30, -40, basePos)

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
        print("len=%r" % len(self.observation))

        return np.array(self.observation), reward, done, {}

    # This function is not being called in the test environment
    def render(self, mode='human', close=False):
        if mode != "rgb_array":
            return np.array([])

        # Move the camera with the base_pos
        base_pos, orn = self.physics.getBasePositionAndOrientation(self.robot.racecarUniqueId)

        view_matrix = self.physics.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self.cam_dist,
            yaw=self.cam_yaw,
            pitch=self.cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self.physics.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self.physics.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def termination(self):
        return self.envStepCounter > 1000

    def reward(self):
        # Adapt the reward to:
        # 1 if target reached, else 0
        # -1 if wall collision
        closestPoints = self.physics.getClosestPoints(self.robot.racecarUniqueId, self.targetUniqueId, 10000)

        #print("Closest Points:", closestPoints)

        numPt = len(closestPoints)
        reward = -1000
        #print(numPt)
        if (numPt > 0):
            reward = -closestPoints[0][8]       # (contactFlag, bodyUniqueIdA, bodyUniqueIdB, linkIndexA, linkIndexB, positionOnA, positionOnB, contactNormalOnB, contactDistance, normalForce)
            print("Reward:", reward)
        return reward

