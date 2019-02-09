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
from ray.tune.schedulers import PopulationBasedTraining
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
        "--pbt",
        default=False,
        type=bool,
        help="Run population based training.")
    return parser


def run(args):
    with open(args.config, 'r') as stream:
        experiments = yaml.load(stream)

    for experiment, settings in experiments.items():
        settings["env"] = SeekerSimEnv

    pprint(experiments)
    run_experiments(experiments, queue_trials=True, resume="prompt")


def run_pbt(args):
    pbt_scheduler = PopulationBasedTraining(
        time_attr='time_total_s',
        reward_attr='episode_reward_mean',
        perturbation_interval=3600.0,
        hyperparam_mutations={
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "tau": [0.005, 0.001],
            "target_noise": [0.1, 0.2],
            "train_batch_size": [64, 128, 512],
            "l2_reg": [1e-5, 1e-6, 1e-7],
        })
    # Prepare the default settings
    with open(args.config, 'r') as stream:
        experiments = yaml.load(stream)

    for experiment, settings in experiments.items():
        settings["env"] = SeekerSimEnv

    run_experiments(experiments, scheduler=pbt_scheduler)




if __name__ == "__main__":
    ray.init()
    parser = create_parser()
    args = parser.parse_args()
    if args.pbt:
        run_pbt(args)
    else:
        run(args)

