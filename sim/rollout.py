"""
Example of a custom gym environment. Run this for a demo.

Copying checkpoint files from the server
gcloud compute --project "stanford-projects" scp --zone "us-west1-b" --recurse "ray-trainer:~/ray_results/*" ~/ray_results


rllib rollout --checkpoint ~/ray_results/demo/experiment_state-2019-01-26_18-46-51.json
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
from gym.envs.registration import registry

def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)

ENV_NAME = 'SeekerSimEnv-v0'
register(id=ENV_NAME,
     entry_point='simulation.SeekerSimEnv:SeekerSimEnv',
     reward_threshold=.5)


CHECKPOINT = "/Users/maxkferg/ray_results/seeker-appo/APPO_SeekerSimEnv_0_2019-01-27_23-46-38eufch4md/checkpoint_610/checkpoint-610"

conf = os.path.join(os.path.dirname(os.path.dirname(CHECKPOINT)),"params.json")
with open(conf, 'r') as stream:
    config = json.load(stream)
config["num_workers"] = 1
config["monitor"] = True


AGENT = "APPO"
RENDER = True
OUT = "./stats"





ray.init()
cls = get_agent_class(AGENT)
agent = cls(env=ENV_NAME, config=config)
agent.restore(CHECKPOINT)
num_steps = int(10000)




if __name__ == "__main__":
    rollout(agent, ENV_NAME, num_steps, OUT, RENDER)
