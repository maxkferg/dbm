#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym
from baselines.common import tf_util as U
from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv
from baselines.ppo1 import pposgd_simple, mlp_policy
import datetime




def train(num_timesteps):
    """Train the car to reach a target"""
    U.make_session(num_cpu=1).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space, hid_size=64, num_hid_layers=2)

    env = RacecarGymEnv(renders=False, isDiscrete=True)

    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=1024,
            clip_param=0.2,
            entcoeff=0.0,
            optim_epochs=10,
            optim_stepsize=3e-4,
            optim_batchsize=64,
            gamma=0.99,
            lam=0.95,
            schedule='linear',
        )
    env.close()



def main():
    #args = mujoco_arg_parser().parse_args()
    #logger.configure()
    num_timesteps = 50000
    train(num_timesteps)


if __name__ == '__main__':
    main()
