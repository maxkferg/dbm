"""Example of a custom gym environment. Run this for a demo."""
import io
import yaml
import numpy as np
import gym
from pprint import pprint
from gym.spaces import Discrete, Box
from gym.envs.registration import EnvSpec
from gym.envs.registration import registry

import ray
from ray.tune import run_experiments
from simulation.SeekerSimEnv import SeekerSimEnv

CONFIG = "configs/seeker-appo.yaml"

with open(CONFIG, 'r') as stream:
    experiments = yaml.load(stream)

experiments["seeker-appo"]["env"] = SeekerSimEnv

pprint(experiments)

if __name__ == "__main__":
    ray.init()
    run_experiments(experiments)