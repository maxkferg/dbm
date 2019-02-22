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
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import run_experiments
from ray.tune.config_parser import make_parser
from ray.tune import grid_search
from ray.tune.registry import register_env
from ray.rllib.agents.ddpg.ddpg_policy_graph import DDPGPolicyGraph
from simulation.Worlds.worlds import Y2E2, Building, Playground, Maze
from simulation.BuildingEnv import MultiRobot


ENVIRONMENT = "MultiRobot-v0"

def robot_env_creator(env_config):
    return MultiRobot({
        "debug": 0,
        "num_robots": 2,
        "world": Playground()
    })



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
    register_env(ENVIRONMENT, lambda _: robot_env_creator({}))

    with open(args.config, 'r') as stream:
        experiments = yaml.load(stream)

    """
    def make_policy_graphs(policy_config):
        # Setup DDPG with an ensemble of `num_policies` different policy graphs
        policy_config = policy_config.copy() # Avoid recursion
        single_env = robot_env_creator({})
        obs_space = single_env.observation_space
        act_space = single_env.action_space
        del single_env
        return {"policy_1": (DDPGPolicyGraph, obs_space, act_space, policy_config)}

    def policy_mapping_fn(i):
        return "policy_1"
    """

    for experiment, settings in experiments.items():
        settings["env"] = ENVIRONMENT
        #settings["config"]["multiagent"] = {
        #    "policy_graphs": make_policy_graphs(settings["config"]),
        #    "policy_mapping_fn": tune.function(policy_mapping_fn)
        #}
    pprint(experiments)
    run_experiments(experiments, queue_trials=True, resume="prompt")



def run_pbt(args):
    pbt_scheduler = PopulationBasedTraining(
        time_attr='time_total_s',
        reward_attr='episode_reward_mean',
        perturbation_interval=4*3600.0,
        hyperparam_mutations={
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "tau": [0.005, 0.001],
            "target_noise": [0.01, 0.1, 0.2],
            "noise_scale": [0.01, 0.1, 0.2],
            "train_batch_size": [2048, 4096, 8192],
            "l2_reg": [1e-5, 1e-6, 1e-7],
        })

    # Prepare the default settings
    with open(args.config, 'r') as stream:
        experiments = yaml.load(stream)

    for experiment, settings in experiments.items():
        settings["env"] = ENVIRONMENT

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

