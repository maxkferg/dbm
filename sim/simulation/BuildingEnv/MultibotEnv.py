import os
import math
import numpy as np
from astar import AStar
from .BuildingEnv import BuildingEnv
from PIL import Image, ImageDraw, ImageColor


class MultibotEnv():

    def __init__(self, env_config):
        pass

    def step(self):
        for robot in self.robots:
            robot.step()

    def reset(self):
        for robot in self.robots:
            robot.reset()

