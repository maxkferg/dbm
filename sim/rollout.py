"""
Example of a custom gym environment. Run this for a demo.

Copying checkpoint files from the server
gcloud compute --project "stanford-projects" scp --zone "us-west1-b" --recurse "ray-trainer:~/ray_results/*" ~/ray_results

ray rsync-down ray_results ~/ray_results  


python rollout.py --run APEX_DDPG --env BuildingEnv-v0 --steps 10000

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
from ray.rllib.agents.registry import get_agent_class
from simulation.BuildingEnv import BuildingEnv
from ray.tune.registry import register_env

EXAMPLE_USAGE = """
Example Usage via RLlib CLI:
    python rollout.py --run DDPG_APEX --env BuildingEnv-v0 --steps 100 --no-render
"""

#CHECKPOINT = "~/ray_results/seeker-ppo-gae/PPO_SeekerSimEnv_0_2019-02-04_08-34-32qf6patqm/checkpoint_780/checkpoint-780"
#CHECKPOINT = "~/ray_results/seeker-appo/APPO_SeekerSimEnv_0_2019-01-27_23-46-38eufch4md/checkpoint_860/checkpoint-860"
#CHECKPOINT = "~/ray_results/seeker-apex-td3/APEX_DDPG_SeekerSimEnv_0_2019-02-05_09-48-37s38jziex/checkpoint_700/checkpoint-700"
#CHECKPOINT = "~/ray_results/humanoid-ppo-gae/PPO_SeekerSimEnv_0_2019-02-04_08-34-32qf6patqm/checkpoint_780/checkpoint-780"
#CHECKPOINT = "~/Google Drive/seeker-apex-td3/ray_results/APEX_DDPG_SeekerSimEnv_0_2019-02-09_03-26-01ze5dseda/checkpoint_100/checkpoint-100"
CHECKPOINT = "~/ray_results/ray_results/seeker-apex-td3/APEX_DDPG_BuildingEnv_0_2019-02-17_10-28-35dht3zbpt/checkpoint_450/checkpoint-450"


CHECKPOINT = os.path.expanduser(CHECKPOINT)
ENVIRONMENT = "BuildingEnv-v0"

RESET_ON_TARGET = False
DEFAULT_TIMESTEP = 0.1
FRAME_MULTIPLIER = 1
EVAL_TIMESTEP = DEFAULT_TIMESTEP/FRAME_MULTIPLIER


def building_env_creator(env_config):
    env_config['timestep'] = EVAL_TIMESTEP
    env_config['resetOnTarget'] = RESET_ON_TARGET
    return BuildingEnv(env_config)

register_env(ENVIRONMENT, building_env_creator)


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


def rollout(agent, env_name, num_steps, out=None, no_render=True):
    if hasattr(agent, "local_evaluator"):
        env = agent.local_evaluator.env
    else:
        env = gym.make(env_name)

    if hasattr(agent, "local_evaluator"):
        state_init = agent.local_evaluator.policy_map[
            "default"].get_initial_state()
    else:
        state_init = []
    if state_init:
        use_lstm = True
    else:
        use_lstm = False

    if out is not None:
        rollouts = []
    steps = 0
    while steps < (num_steps or steps + 1):
        if out is not None:
            rollout = []
        state = env.reset()
        done = False
        reward_total = 0.0
        while not done and steps < (num_steps or steps + 1):
            if use_lstm:
                action, state_init, logits = agent.compute_action(
                    state, state=state_init)
            else:
                action = agent.compute_action(state)
            # Repeat this action n times, rendering each time
            for i in range(FRAME_MULTIPLIER):
                next_state, reward, done, _ = env.step(action)
                reward_total += reward
                if not no_render:
                    env.render()
                if done:
                    break
            if out is not None:
                rollout.append([state, action, next_state, reward, done])
            steps += 1
            state = next_state
        if out is not None:
            rollouts.append(rollout)
        print("Episode reward", reward_total)
    if out is not None:
        pickle.dump(rollouts, open(out, "wb"))


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
        if "horizon" in config:
            print("DEL")
            del config["horizon"]

    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    if not args.env:
        raise("No environment")

    ray.init()

    if not args.no_render:
        config["monitor"] = True

    config["exploration_final_eps"] = 0

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
