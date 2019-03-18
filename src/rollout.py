"""
Run and record a RL reollout

Copying checkpoint files from the server
gcloud compute --project "stanford-projects" scp --zone "us-west1-b" --recurse "ray-trainer:~/ray_results/*" ~/ray_results

ray rsync-down cluster.yaml ray_results ~/


python rollout.py --run APEX_DDPG --env BuildingEnv-v0 --steps 10000 --no-render

"""
import io
import os
import cv2
import json
import time
import datetime
import numpy as np
import gym
import ray
import argparse
import learning.model
import colored_traceback
from pprint import pprint
from gym.spaces import Discrete, Box
from gym.envs.registration import EnvSpec
from gym.envs.registration import registry
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models.preprocessors import get_preprocessor
from simulation.BuildingEnv import MultiRobot
from simulation.Worlds.worlds import Y2E2, Building, Playground, Maze, House
from learning.custom_policy_graph import CustomDDPGPolicyGraph
from ray.tune.registry import register_env
colored_traceback.add_hook()


EXAMPLE_USAGE = """
Example Usage via RLlib CLI:
    python rollout.py --run APEX_DDPG --env MultiRobot-v0 --steps 1000
"""

CHECKPOINT = "~/ray_results/seeker-apex-td3/APEX_DDPG_MultiRobot-v0_0_2019-02-23_01-54-076wuy8g2a/checkpoint_2200/checkpoint-2200"
CHECKPOINT = "~/ray_results/seeker-apex-td3/APEX_DDPG_MultiRobot-v0_0_2019-02-27_11-39-53d4m4cbl2/checkpoint_1350/checkpoint-1350"
CHECKPOINT = "~/ray_results/saved/mapper/checkpoint_1350/checkpoint-1350"

CHECKPOINT = os.path.expanduser(CHECKPOINT)
ENVIRONMENT = "MultiRobot-v0"

RESET_ON_TARGET = True
DEFAULT_TIMESTEP = 0.1
FRAME_MULTIPLIER = 5
EVAL_TIMESTEP = DEFAULT_TIMESTEP/FRAME_MULTIPLIER

RENDER_WIDTH = 1280
RENDER_HEIGHT = 720


timestamp = datetime.datetime.now().strftime("%I-%M-%S %p")
filename = 'videos/video %s.mp4'%timestamp

video_FourCC = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(filename, video_FourCC, fps=20, frameSize=(RENDER_WIDTH,RENDER_HEIGHT))



def building_env_creator(env_config):
    return MultiRobot({
        "monitor": True,
        "debug": 0,
        "renders": True,
        "num_robots": 2,
        "reset_on_target": False,
        "world": House(timestep=EVAL_TIMESTEP)
    })

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


def render_q(env, agent):
    action = np.array([0,0])
    prep = get_preprocessor(env.observation_space)(env.observation_space)

    start = time.time()
    observation, nx, ny = env.default_env.get_observation_array()
    print("Got %i observations in %.3f seconds"%(len(observation),time.time()-start))

    # Reshape action and observation so that the first dimension is the batch
    #nx, ny, ns = observation.shape
    #observation = np.reshape(observation, (-1, ns))
    #action = np.tile(action, (nx*ny,1))
    obs_t = []
    act_t = []
    for i in range(len(observation)):
        act_t.append(np.expand_dims(action, axis=0))
        obs_t.append(np.expand_dims(prep.transform(observation[i]), axis=0))
    print("Prep took %.3f seconds"%(time.time()-start))

    q, qt = agent.get_policy().compute_q(obs_t, act_t)
    q_img = np.reshape(q, (nx,ny,1))
    print("Policy took %.3f seconds"%(time.time()-start))

    import cv2
    import scipy.misc
    q_img = 127*(np.clip(q_img, -1, 1)+1)
    q_img = q_img.astype(np.uint8)
    q_img = np.tile(q_img, (1,1,3))
    q_img = cv2.applyColorMap(q_img, cv2.COLORMAP_JET)
    q_img = q_img[:,:,::-1] # Flip colormap to RGB
    return q_img


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
    print("Rolling out the policy:")
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
                action = {}
                for key,value in state.items():
                    action[key] = agent.compute_action(value)
                    # Compute the value of this default actor

            # Repeat this action n times, rendering each time
            for i in range(FRAME_MULTIPLIER):
                next_state, reward, dones, _ = env.step(action)
                reward_total += np.sum(list(reward.values()))
                done = dones['__all__']
                if not no_render:
                    q = render_q(env, agent)
                    frame = env.render(width=RENDER_WIDTH, height=RENDER_HEIGHT)
                    frame[0:q.shape[0], (frame.shape[1]-q.shape[1]):frame.shape[1], :3] = q
                    video.write(frame)
                if done:
                    break
            if out is not None:
                rollout.append([state, action, next_state, reward, done])
            steps += 1
            state = next_state
            print(".", end="", flush=True)
        if out is not None:
            rollouts.append(rollout)
        print("\nEpisode reward", reward_total)
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
            config["num_workers"] = min(1, config["num_workers"])
        if "horizon" in config:
            del config["horizon"]

    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    # Stop all the actor noise
    config['noise_scale'] = 0
    config['per_worker_exploration'] = False
    config['schedule_max_timesteps'] = 0

    if not args.env:
        raise("No environment")

    ray.init()

    if not args.no_render:
        config["monitor"] = True

    config["exploration_final_eps"] = 0

    cls = get_agent_class(args.run)
    cls._policy_graph = CustomDDPGPolicyGraph
    agent = cls(env=args.env, config=config)
    agent.restore(args.checkpoint)
    num_steps = int(args.steps)
    pprint(config)
    rollout(agent, args.env, num_steps, args.out, args.no_render)
    cv2.destroyAllWindows()
    video.release()




if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    try:
        run(args, parser)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        video.release()

