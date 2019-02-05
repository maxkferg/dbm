"""
Example of a custom gym environment. Run this for a demo.

Copying checkpoint files from the server
gcloud compute --project "stanford-projects" scp --zone "us-west1-b" --recurse "ray-trainer:~/ray_results/*" ~/ray_results


rllib rollout --checkpoint ~/ray_results/demo/experiment_state-2019-01-26_18-46-51.json
rllib rollout --checkpoint ~/ray_results/demo/experiment_state-2019-01-26_18-46-51.json


python rollout.py
"""
import io
import os
import json
import numpy as np
import gym
import ray
from pprint import pprint
from gym.spaces import Discrete, Box
from gym.envs.registration import EnvSpec
from gym.envs.registration import registry
from ray.rllib.rollout import rollout
from ray.rllib.agents.registry import get_agent_class
from simulation.SeekerSimEnv import SeekerSimEnv
from ray.tune.registry import register_env

CHECKPOINT = "~/ray_results/seeker-ppo-gae/PPO_SeekerSimEnv_0_2019-02-04_08-34-32qf6patqm/checkpoint_780/checkpoint-780"
CHECKPOINT = "~/ray_results/seeker-appo/APPO_SeekerSimEnv_0_2019-01-27_23-46-38eufch4md/checkpoint_860/checkpoint-860"
CHECKPOINT = os.path.expanduser(CHECKPOINT)
ENVIRONMENT = "SimSeekerEnv-v0"

def get_params(checkpoint):
    config_dir = os.path.dirname(checkpoint)
    config_path = os.path.join(config_dir, "params.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.json")
    if not os.path.exists(config_path):
        raise ValueError(
            "Could not find params.json in either the checkpoint dir or "
            "its parent directory.")
    with open(config_path, 'r') as stream:
        return json.load(stream)


config = get_params(CHECKPOINT)
config["num_workers"] = 1
config["monitor"] = True
config["env"] = ENVIRONMENT


AGENT = "APPO"
RENDER = True
OUT = "./stats/output.json"


def seeker_env_creator(env_config):
    env_config["renders"] = True
    return SeekerSimEnv(env_config)


register_env(ENVIRONMENT, seeker_env_creator)


ray.init()
cls = get_agent_class(AGENT)
agent = cls(env=ENVIRONMENT, config=config)
agent.restore(CHECKPOINT)
num_steps = int(10000)




if __name__ == "__main__":
    rollout(agent, ENVIRONMENT, num_steps, OUT, RENDER)
