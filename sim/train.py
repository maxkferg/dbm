"""
Train an agent on SeekerSimEnv

# For a lightweight test
python train.py configs/seeker-test.yaml

# For a GPU driven large test
python train.py configs/seeker-gpu.yaml
"""
import io
import ray
import yaml
import numpy as np
import gym
import argparse
from pprint import pprint
from gym.spaces import Discrete, Box
from gym.envs.registration import EnvSpec
from gym.envs.registration import registry
from ray.tune import run_experiments
from ray.tune.config_parser import make_parser
from simulation.SeekerSimEnv import SeekerSimEnv

def create_parser():
    parser = argparse.ArgumentParser(
        description='Process some integers.')
    parser.add_argument(
        "config",
        default="configs/seeker-test.yaml",
        type=str,
        help="The configuration file to use for the RL agent.")
    parser.add_argument(
        "node",
        default=False,
        type=bool,
        help="Just start the node.")
    return parser


def run(args):
    with open(args.config, 'r') as stream:
        experiments = yaml.load(stream)

    for experiment, settings in experiments.items():
        settings["env"] = SeekerSimEnv 

    pprint(experiments)
    run_experiments(experiments)

if __name__ == "__main__":
    ray.init()
    parser = create_parser()
    args = parser.parse_args()
    if not args.node:
        run(args)