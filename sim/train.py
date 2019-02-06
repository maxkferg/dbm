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
from random import choice
from pprint import pprint
from gym.spaces import Discrete, Box
from gym.envs.registration import EnvSpec
from gym.envs.registration import registry
from ray import tune
from ray.tune import run_experiments
from ray.tune.config_parser import make_parser
from ray.tune import grid_search
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
        "--node",
        default=False,
        type=bool,
        help="Just start the node.")
    return parser


def run(args):
    with open(args.config, 'r') as stream:
        experiments = yaml.load(stream)

    for experiment, settings in experiments.items():
        settings["env"] = SeekerSimEnv

    experiment = experimental_config(experiments)
    pprint(experiments)
    run_experiments(experiments, queue_trials=True, resume="prompt")


def experimental_config(config):
    changes = {
        'stop': {
            'episode_reward_mean': 1,
            'time_total_s': 3600,
        },
        'config': {
            'buffer_size': tune.sample_from([500000, 5000000]), 
            'train_batch_size': tune.sample_from([64, 128, 512]),
            'lr': tune.sample_from([0.00001, 0.0005, 0.0001]),
            'tau': tune.sample_from([0.005, 0.001]),
            'n_step': tune.sample_from([1, 2, 4]),
            'noise_scale': tune.sample_from([0.1, 0.2, 0.4])
        },
    }
    for context in changes.keys():
        for k,v in changes[context].items():
            config['seeker-td3'][context][k] = v
    return config


if __name__ == "__main__":
    ray.init()
    parser = create_parser()
    args = parser.parse_args()
    if not args.node:
        run(args)
