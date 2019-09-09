import os
import sys
import gym
import time
import math
import time
import scipy
import skimage
import random
import pybullet
import numpy as np
import roslibpy
from collections import OrderedDict
from gym import spaces
from gym.utils import seeding
from pprint import pprint
from PIL import Image, ImageDraw
from ..BuildingEnv.config import URDF_ROOT
from ..BuildingEnv.mapper import Mapper

sys.path.insert(1, os.path.join(sys.path[0], '../..'))


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
COUNT = 0

# ROS BRIDGE
PORT = 8080
HOST = "robot.com"

TARGETS = [
    [0.2, 4, 0.3],
    [-0.2, 0, 0.3]
]

OFFSET_X = 0.2
OFFSET_Y = 1.2
REAL = True

# Fix targets
TARGETS = [[x+OFFSET_X, y+OFFSET_Y, z] for x,y,z in TARGETS]

class RealMapper(Mapper):

    def __init__(self, *args, **kwargs):
        self.ros = roslibpy.Ros(host=HOST, port=PORT)
        self.ros_odom = roslibpy.Topic(self.ros, '/odom', 'nav_msgs/Odometry')
        self.ros_vel = roslibpy.Topic(self.ros, '/cmd_vel_mux/input/safety_controller', 'geometry_msgs/Twist')
        self.step = 0
        self.prev_robot_action = np.array([0,0])
        super().__init__(*args, **kwargs)
        if REAL:
            self.start_ros()

    def start_ros(self):
        """Setup a connection with the robot to wait for ODOM messages"""

        def receive_message(message):
            pos, orn = self.physics.getBasePositionAndOrientation(self.robot.racecarUniqueId)
            pos, orn = list(pos), list(orn)
            pos[0] = message['pose']['pose']['position']['x'] + OFFSET_X
            pos[1] = message['pose']['pose']['position']['y'] + OFFSET_Y
            orn[2] = message['pose']['pose']['orientation']['z']
            orn[3] = message['pose']['pose']['orientation']['w']
            self.physics.resetBasePositionAndOrientation(self.robot.racecarUniqueId, pos, orn)

        def start_listening():
            print("ROS: listening for odom messages")
            self.ros_odom.subscribe(receive_message)

        self.ros.get_topics(print)
        self.ros.on_ready(start_listening)
        self.ros.run()


    def send_control(self, control):
        """Send a control action to the robot"""
        rotation = control[0]
        velocity = control[1]

        linear = {
            'x': velocity,
            'y': 0,
            'z': 0
        }

        angular = {
            'x': 0,
            'y': 0,
            'z': 4*rotation
        }

        message = roslibpy.Message({
            'linear': linear,
            'angular': angular,
        })

        def send():
            print('Sending rotation: ', rotation, ' and velocity: ',velocity)
            self.ros_vel.publish(message)
        self.ros.on_ready(send)


    def reset_robot_position(self):
        """Move the robot to a new position"""
        car_pos = [OFFSET_X, OFFSET_Y, .25]
        self.robot.set_position(car_pos)


    def reset_target_position(self):
        """Move the target to a new position"""
        target_pos = random.choice(TARGETS)
        _, target_orn = self.physics.getBasePositionAndOrientation(self.targetUniqueId)
        self.physics.resetBasePositionAndOrientation(self.targetUniqueId, np.array(target_pos), target_orn)
        self.remap_targets()


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
        if not REAL:
            return

        print("ACTION:", action)

        # Ramp up to realaction
        current = self.prev_robot_action
        realaction = np.array(realaction)
        for i in range(8):
            current = (0.2 * realaction + 0.8 * current)
            control = (0.2 * current).tolist()
            control[0] = 3*control[0]
            self.send_control(control)
            time.sleep(0.1)

        self.prev_robot_action = current

        self.step +=1

        if self.step % 5 ==0:
            next_action = control
            for i in range(4):
                self.prev_robot_action = self.prev_robot_action/1.4
                self.send_control(self.prev_robot_action.tolist())
                time.sleep(0.05)
            input("Next?")

