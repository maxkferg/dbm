import os
import gym
import math
import random
import numpy as np
from astar import AStar
from .seeker import Seeker
from ray.rllib.env import MultiAgentEnv
from PIL import Image, ImageDraw, ImageColor

DEFAULT_ACTION_REPEAT = 2


class SingleRobot(gym.Env):

    def __init__(self, env_config={}):
        if 'world' in env_config:
            world = env_config['world']
        else:
            world = Playground()
        self.action_repeat = env_config.get("action_repeat") or DEFAULT_ACTION_REPEAT
        self.env = Seeker(world, env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space


    def step(self, action):
        """
        Step the simulation forward one timestep
        """
        self.env.act(action)
        for i in range(self.action_repeat):
            self.env.physics.stepSimulation()
            if self.env.is_crashed() or self.env.is_at_target():
                break
        return self.env.observe()


    def reset(self):
        """ 
        Reset the base environment
        """
        return self.env.reset()


    def render(self, *arg, **kwargs):
        return self.env.render(*arg, **kwargs)


    def render_map(self, *arg, **kwargs):
        return self.env.render_map(*arg, **kwargs)



class MultiRobot(gym.Env, MultiAgentEnv):

    def __init__(self, env_config={}):
        if 'world' in env_config:
            self.world = env_config['world']
        else:
            raise ValueError("The world was not specified")

        if 'num_robots' in env_config:
            self.num_robots = env_config['num_robots']
        else:
            raise ValueError("The number of robots was not specified")

        self.dones = set()
        self.action_repeat = env_config.get("action_repeat") or DEFAULT_ACTION_REPEAT
        self.env = {i: Seeker(self.world, env_config) for i in range(self.num_robots)}
        self.default_env = self.env[random.choice(list(self.env.keys()))]
        self.action_space = self.default_env.action_space
        self.observation_space = self.default_env.observation_space


    def step(self, actions):
        """
        Step the simulation forward one timestep
        """
        for key, action in actions.items():
            self.env[key].act(action)

        for i in range(self.action_repeat):
            self.world.physics.stepSimulation()
            is_crashed = any(e.is_crashed() for e in self.env.values())
            is_target = any(e.is_at_target() for e in self.env.values())
            if is_crashed or is_target:
                break

        obs, rew, done, info = {}, {}, {}, {}
        for i in actions.keys():
            obs[i], rew[i], done[i], info[i] = self.env[i].observe()
            if done[i]:
                self.dones.add(i)

        # Rllib requires a dones[__all__] key
        done["__all__"] = len(self.dones) == len(self.env)

        return obs, rew, done, info


    def reset(self):
        """ 
        Reset the base environment
        """
        self.dones = set()
        return {key:env.reset() for key,env in self.env.items()}


    def render(self, *arg, **kwargs):
        from PIL import Image
        im = Image.fromarray(self.default_env.render(*arg, **kwargs)[:,:,:3])
        im.save("your_file.jpeg")

        return self.default_env.render(*arg, **kwargs)


    def render_map(self, *arg, **kwargs):
        return self.default_env.render_map(*arg, **kwargs)




