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
import argparse
from pprint import pprint
from gym.spaces import Discrete, Box
from gym.envs.registration import EnvSpec
from gym.envs.registration import registry
from ray.rllib.rollout import rollout
from ray.rllib.agents.registry import get_agent_class
from simulation.SeekerSimEnv import SeekerSimEnv
from ray.tune.registry import register_env

EXAMPLE_USAGE = """
Example Usage via RLlib CLI:
    python rollout.py --run DDPG_APEX --env SimSeekerEnv-v0 --steps 100 --no-render
"""

#CHECKPOINT = "~/ray_results/seeker-ppo-gae/PPO_SeekerSimEnv_0_2019-02-04_08-34-32qf6patqm/checkpoint_780/checkpoint-780"
#CHECKPOINT = "~/ray_results/seeker-appo/APPO_SeekerSimEnv_0_2019-01-27_23-46-38eufch4md/checkpoint_860/checkpoint-860"
#CHECKPOINT = "~/ray_results/seeker-apex-td3/APEX_DDPG_SeekerSimEnv_0_2019-02-05_09-48-37s38jziex/checkpoint_700/checkpoint-700"
#CHECKPOINT = "~/ray_results/humanoid-ppo-gae/PPO_SeekerSimEnv_0_2019-02-04_08-34-32qf6patqm/checkpoint_780/checkpoint-780"
#CHECKPOINT = "~/Google Drive/seeker-apex-td3/ray_results/APEX_DDPG_SeekerSimEnv_0_2019-02-09_03-26-01ze5dseda/checkpoint_100/checkpoint-100"
CHECKPOINT = "~/ray_results/seeker-apex-td3/APEX_DDPG_SeekerSimEnv_0_2019-02-10_01-47-50j4pfi3d3/checkpoint_300/checkpoint-300"


CHECKPOINT = os.path.expanduser(CHECKPOINT)
ENVIRONMENT = "SimSeekerEnv-v0"


def seeker_env_creator(env_config):
    return SeekerSimEnv(env_config)

register_env(ENVIRONMENT, seeker_env_creator)


def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
        "given a checkpoint.",
        epilog=EXAMPLE_USAGE)

    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=False,
        default=CHECKPOINT,
        help="Checkpoint from which to roll out.")

    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--run",
        type=str,
        required=True,
        help="The algorithm or model to train. This may refer to the name "
        "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
        "user-defined trainable function or class registered in the "
        "tune registry.")
    required_named.add_argument(
        "--env", type=str, help="The gym environment to use.")
    parser.add_argument(
        "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="Surpress rendering of the environment.")
    parser.add_argument(
        "--steps", default=10000, help="Number of steps to roll out.")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
        "Surpresses loading of configuration from checkpoint.")
    return parser



def run(args, parser):
    config = args.config
    if not config:
        # Load configuration from file
        checkpoint = os.path.expanduser(args.checkpoint)
        config_dir = os.path.dirname(checkpoint)
        config_path = os.path.join(config_dir, "params.json")
        if not os.path.exists(config_path):
            print("Could not find checkpoint in ", config_path)
            config_path = os.path.join(config_dir, "../params.json")
        if not os.path.exists(config_path):
            print("Could not find checkpoint in ", config_path)
            raise ValueError(
                "Could not find params.json in either the checkpoint dir or "
                "its parent directory.")
        with open(config_path) as f:
            config = json.load(f)
        if "num_workers" in config:
            config["num_workers"] = min(2, config["num_workers"])

    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    if not args.env:
        raise("No environment")

    ray.init()

    if not args.no_render:
        config["monitor"] = True

    config["exploration_final_eps"] = 0.5

    cls = get_agent_class(args.run)
    agent = cls(env=args.env, config=config)
    agent.restore(args.checkpoint)
    num_steps = int(args.steps)
    pprint(config)
    rollout(agent, args.env, num_steps, args.out, args.no_render)





if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
