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

    experiment = experimental_config(experiment)
    pprint(experiments)
    run_experiments(experiments, queue_trials=True, resume="prompt")


def experimental_config(config):
    modify = {
        'stop': {
            'episode_reward_mean': 1,
            'time_total_s': 3600,
        },
        'buffer_size': grid_search([500000, 5000000]), 
        'train_batch_size': grid_search([64, 128, 512]),
        'lr': grid_search([0.00001, 0.0005, 0.0001]),
        'tau': grid_search([0.005, 0.001]),
        'n_step': grid_search([1, 2, 4]),
        'noise_scale': grid_search([0.1, 0.2, 0.4])
    }
    for k,v in modify.items():
        config['seeker-td3'][k] = v
    return config


if __name__ == "__main__":
    ray.init()
    parser = create_parser()
    args = parser.parse_args()
    if not args.node:
        run(args)
