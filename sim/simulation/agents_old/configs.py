# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example configurations using the PPO algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from . import ppo
from . import networks
import tensorflow as tf


def default():
    """Default configuration for PPO."""
    # General
    algorithm = ppo.PPOAlgorithm
    num_agents = 8
    eval_episodes = 30
    use_gpu = False
    # Network
    network = networks.feed_forward_gaussian
    weight_summaries = dict(
        all=r'.*',
        policy=r'.*/policy/.*',
        value=r'.*/value/.*')
    policy_layers = 200, 100
    value_layers = 200, 100
    init_mean_factor = 0.1
    init_logstd = -1
    # Optimization
    update_every = 30
    update_epochs = 25
    optimizer = tf.train.AdamOptimizer
    update_epochs_policy = 64
    update_epochs_value = 64
    learning_rate = 1e-4
    # Losses
    discount = 0.99
    kl_target = 1e-2
    kl_cutoff_factor = 2
    kl_cutoff_coef = 1000
    kl_init_penalty = 1
    return locals()


def pybullet_seekersim():
    """Configuration for the SeekerSim agent"""
    locals().update(default())
    # The name of the environment must conform to ~/anaconda2/envs/sim/lib/python3.6/site-packages/gym/envs/registration.py
    env = 'SeekerSimEnv-v0'
    max_length = 1000 # Verify. This appears to be the maximum number of steps for the memory buffer
    update_every = 60
    update_epochs = 50
    steps = 1e7
    return locals()


def pybullet_racecar():
    """Configuration for Bullet MIT Racecar task."""
    locals().update(default())
    # Environment
    env = 'RacecarBulletEnv-v0' #functools.partial(racecarGymEnv.RacecarGymEnv, isDiscrete=False, renders=True)
    max_length = 10
    steps = 1e7  # 10M
    return locals()
