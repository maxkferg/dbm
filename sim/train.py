"""
Train an agent on SeekerSimEnv

# For a lightweight test
python train.py configs/seeker-test.yaml --dev=True

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
from simulation.BuildingEnv import BuildingEnv

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
    parser.add_argument(
        "--dev",
        default=False,
        type=bool,
        help="Use development cluster with local redis server")
    return parser


def run(args):
    with open(args.config, 'r') as stream:
        experiments = yaml.load(stream)

    for experiment, settings in experiments.items():
        settings["env"] = BuildingEnv

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
            "noise_scale": [0.1, 0.2],
            "train_batch_size": [2048, 4096, 80192],
            "l2_reg": [1e-5, 1e-6, 1e-7],
        })
    # Prepare the default settings
    with open(args.config, 'r') as stream:
        experiments = yaml.load(stream)

    for experiment, settings in experiments.items():
        settings["env"] = BuildingEnv

    run_experiments(experiments, scheduler=pbt_scheduler)




if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    if args.dev:
        ray.init()
    else:
        ray.init("localhost:6379")
    if args.pbt:
        run_pbt(args)
    else:
        run(args)

