from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym.spaces import Box
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import ray
import ray.experimental.tf_utils
from ray.rllib.agents.dqn.dqn_policy_graph import (
    _minimize_and_clip, _scope_vars, _postprocess_dqn)
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.evaluation.policy_graph import PolicyGraph
from ray.rllib.evaluation.tf_policy_graph import TFPolicyGraph
from ray.rllib.agents.ddpg.ddpg_policy_graph import PNetwork, QNetwork, ActionNetwork, ActorCriticLoss

A_SCOPE = "a_func"
P_SCOPE = "p_func"
P_TARGET_SCOPE = "target_p_func"
Q_SCOPE = "q_func"
Q_TARGET_SCOPE = "target_q_func"
TWIN_Q_SCOPE = "twin_q_func"
TWIN_Q_TARGET_SCOPE = "twin_target_q_func"


class CustomDDPGPolicyGraph(TFPolicyGraph):
    def __init__(self, observation_space, action_space, config):
        config = dict(ray.rllib.agents.ddpg.ddpg.DEFAULT_CONFIG, **config)
        if not isinstance(action_space, Box):
            raise UnsupportedSpaceException(
                "Action space {} is not supported for DDPG.".format(
                    action_space))

        self.config = config
        self.cur_epsilon = 1.0
        self.dim_actions = action_space.shape[0]
        self.low_action = action_space.low
        self.high_action = action_space.high

        # create global step for counting the number of update operations
        self.global_step = tf.train.get_or_create_global_step()

        # Action inputs
        self.stochastic = tf.placeholder(tf.bool, (), name="stochastic")
        self.eps = tf.placeholder(tf.float32, (), name="eps")
        self.cur_observations = tf.placeholder(
            tf.float32,
            shape=(None, ) + observation_space.shape,
            name="cur_obs")

        # Actor: P (policy) network
        with tf.variable_scope(P_SCOPE) as scope:
            p_values, self.p_model = self._build_p_network(
                self.cur_observations, observation_space)
            self.p_func_vars = _scope_vars(scope.name)

        # Noise vars for P network except for layer normalization vars
        if self.config["parameter_noise"]:
            self._build_parameter_noise([
                var for var in self.p_func_vars if "LayerNorm" not in var.name
            ])

        # Action outputs
        with tf.variable_scope(A_SCOPE):
            self.output_actions = self._build_action_network(
                p_values, self.stochastic, self.eps)

        if self.config["smooth_target_policy"]:
            self.reset_noise_op = tf.no_op()
        else:
            with tf.variable_scope(A_SCOPE, reuse=True):
                exploration_sample = tf.get_variable(name="ornstein_uhlenbeck")
                self.reset_noise_op = tf.assign(exploration_sample,
                                                self.dim_actions * [.0])

        # Replay inputs
        self.obs_t = tf.placeholder(
            tf.float32,
            shape=(None, ) + observation_space.shape,
            name="observation")
        self.act_t = tf.placeholder(
            tf.float32, shape=(None, ) + action_space.shape, name="action")
        self.rew_t = tf.placeholder(tf.float32, [None], name="reward")
        self.obs_tp1 = tf.placeholder(
            tf.float32, shape=(None, ) + observation_space.shape)
        self.done_mask = tf.placeholder(tf.float32, [None], name="done")
        self.importance_weights = tf.placeholder(
            tf.float32, [None], name="weight")

        # p network evaluation
        with tf.variable_scope(P_SCOPE, reuse=True) as scope:
            prev_update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            self.p_t, _ = self._build_p_network(self.obs_t, observation_space)
            p_batchnorm_update_ops = list(
                set(tf.get_collection(tf.GraphKeys.UPDATE_OPS)) -
                prev_update_ops)

        # target p network evaluation
        with tf.variable_scope(P_TARGET_SCOPE) as scope:
            p_tp1, _ = self._build_p_network(self.obs_tp1, observation_space)
            target_p_func_vars = _scope_vars(scope.name)

        # Action outputs
        with tf.variable_scope(A_SCOPE, reuse=True):
            output_actions = self._build_action_network(
                self.p_t,
                stochastic=tf.constant(value=False, dtype=tf.bool),
                eps=.0)
            output_actions_estimated = self._build_action_network(
                p_tp1,
                stochastic=tf.constant(
                    value=self.config["smooth_target_policy"], dtype=tf.bool),
                eps=.0,
                is_target=True)

        # q network evaluation
        prev_update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        with tf.variable_scope(Q_SCOPE) as scope:
            q_t, self.q_model = self._build_q_network(
                self.obs_t, observation_space, self.act_t)
            self.q_func_vars = _scope_vars(scope.name)
            self.q_t = q_t
        self.stats = {
            "mean_q": tf.reduce_mean(q_t),
            "max_q": tf.reduce_max(q_t),
            "min_q": tf.reduce_min(q_t),
        }
        with tf.variable_scope(Q_SCOPE, reuse=True):
            q_tp0, _ = self._build_q_network(self.obs_t, observation_space,
                                             output_actions)
        if self.config["twin_q"]:
            with tf.variable_scope(TWIN_Q_SCOPE) as scope:
                twin_q_t, self.twin_q_model = self._build_q_network(
                    self.obs_t, observation_space, self.act_t)
                self.twin_q_func_vars = _scope_vars(scope.name)
                self.twin_q_t = twin_q_t
                self.stats.update({
                    "mean_twin_q_t": tf.reduce_mean(twin_q_t),
                    "max_twin_q_t": tf.reduce_max(twin_q_t),
                    "min_twin_q_t": tf.reduce_min(twin_q_t),
                })
        q_batchnorm_update_ops = list(
            set(tf.get_collection(tf.GraphKeys.UPDATE_OPS)) - prev_update_ops)

        # target q network evalution
        with tf.variable_scope(Q_TARGET_SCOPE) as scope:
            q_tp1, _ = self._build_q_network(self.obs_tp1, observation_space,
                                             output_actions_estimated)
            target_q_func_vars = _scope_vars(scope.name)
        if self.config["twin_q"]:
            with tf.variable_scope(TWIN_Q_TARGET_SCOPE) as scope:
                twin_q_tp1, _ = self._build_q_network(
                    self.obs_tp1, observation_space, output_actions_estimated)
                twin_target_q_func_vars = _scope_vars(scope.name)

        if self.config["twin_q"]:
            self.loss = self._build_actor_critic_loss(
                q_t, q_tp1, q_tp0, twin_q_t=twin_q_t, twin_q_tp1=twin_q_tp1)
        else:
            self.loss = self._build_actor_critic_loss(q_t, q_tp1, q_tp0)

        if config["l2_reg"] is not None:
            for var in self.p_func_vars:
                if "bias" not in var.name:
                    self.loss.actor_loss += (
                        config["l2_reg"] * 0.5 * tf.nn.l2_loss(var))
            for var in self.q_func_vars:
                if "bias" not in var.name:
                    self.loss.critic_loss += (
                        config["l2_reg"] * 0.5 * tf.nn.l2_loss(var))
            if self.config["twin_q"]:
                for var in self.twin_q_func_vars:
                    if "bias" not in var.name:
                        self.loss.critic_loss += (
                            config["l2_reg"] * 0.5 * tf.nn.l2_loss(var))

        # update_target_fn will be called periodically to copy Q network to
        # target Q network
        self.tau_value = config.get("tau")
        self.tau = tf.placeholder(tf.float32, (), name="tau")
        update_target_expr = []
        for var, var_target in zip(
                sorted(self.q_func_vars, key=lambda v: v.name),
                sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(
                var_target.assign(self.tau * var +
                                  (1.0 - self.tau) * var_target))
        if self.config["twin_q"]:
            for var, var_target in zip(
                    sorted(self.twin_q_func_vars, key=lambda v: v.name),
                    sorted(twin_target_q_func_vars, key=lambda v: v.name)):
                update_target_expr.append(
                    var_target.assign(self.tau * var +
                                      (1.0 - self.tau) * var_target))
        for var, var_target in zip(
                sorted(self.p_func_vars, key=lambda v: v.name),
                sorted(target_p_func_vars, key=lambda v: v.name)):
            update_target_expr.append(
                var_target.assign(self.tau * var +
                                  (1.0 - self.tau) * var_target))
        self.update_target_expr = tf.group(*update_target_expr)

        self.sess = tf.get_default_session()
        self.loss_inputs = [
            ("obs", self.obs_t),
            ("actions", self.act_t),
            ("rewards", self.rew_t),
            ("new_obs", self.obs_tp1),
            ("dones", self.done_mask),
            ("weights", self.importance_weights),
        ]
        input_dict = dict(self.loss_inputs)

        # Model self-supervised losses
        self.loss.actor_loss = self.p_model.custom_loss(
            self.loss.actor_loss, input_dict)
        self.loss.critic_loss = self.q_model.custom_loss(
            self.loss.critic_loss, input_dict)
        if self.config["twin_q"]:
            self.loss.critic_loss = self.twin_q_model.custom_loss(
                self.loss.critic_loss, input_dict)

        TFPolicyGraph.__init__(
            self,
            observation_space,
            action_space,
            self.sess,
            obs_input=self.cur_observations,
            action_sampler=self.output_actions,
            loss=self.loss.actor_loss + self.loss.critic_loss,
            loss_inputs=self.loss_inputs,
            update_ops=q_batchnorm_update_ops + p_batchnorm_update_ops)
        self.sess.run(tf.global_variables_initializer())

        # Note that this encompasses both the policy and Q-value networks and
        # their corresponding target networks
        self.variables = ray.experimental.tf_utils.TensorFlowVariables(
            tf.group(q_tp0, q_tp1), self.sess)

        # Hard initial update
        self.update_target(tau=1.0)

    @override(TFPolicyGraph)
    def optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.config["lr"])

    @override(TFPolicyGraph)
    def gradients(self, optimizer):
        if self.config["grad_norm_clipping"] is not None:
            actor_grads_and_vars = _minimize_and_clip(
                optimizer,
                self.loss.actor_loss,
                var_list=self.p_func_vars,
                clip_val=self.config["grad_norm_clipping"])
            critic_grads_and_vars = _minimize_and_clip(
                optimizer,
                self.loss.critic_loss,
                var_list=self.q_func_vars + self.twin_q_func_vars
                if self.config["twin_q"] else self.q_func_vars,
                clip_val=self.config["grad_norm_clipping"])
        else:
            actor_grads_and_vars = optimizer.compute_gradients(
                self.loss.actor_loss, var_list=self.p_func_vars)
            critic_grads_and_vars = optimizer.compute_gradients(
                self.loss.critic_loss,
                var_list=self.q_func_vars + self.twin_q_func_vars
                if self.config["twin_q"] else self.q_func_vars)
        actor_grads_and_vars = [(g, v) for (g, v) in actor_grads_and_vars
                                if g is not None]
        critic_grads_and_vars = [(g, v) for (g, v) in critic_grads_and_vars
                                 if g is not None]
        grads_and_vars = actor_grads_and_vars + critic_grads_and_vars
        return grads_and_vars


    def compute_q(self, observation, action):
        q_values, q_twin_values = [],[]
        for i in range(len(observation)):
            feed_dict = {self.obs_t: observation[i], self.act_t: action[i] }
            feed_dict.update(self.extra_compute_action_feed_dict())
            q_value, q_twin_value = self.sess.run([self.q_t, self.twin_q_t], feed_dict=feed_dict)
            q_values.append(q_value)
            q_twin_values.append(q_twin_value)
        return q_values, q_twin_values


    @override(TFPolicyGraph)
    def extra_compute_action_feed_dict(self):
        return {
            self.stochastic: True,
            self.eps: self.cur_epsilon,
        }

    @override(TFPolicyGraph)
    def extra_compute_grad_fetches(self):
        return {
            "td_error": self.loss.td_error,
            "stats": self.stats,
        }

    @override(PolicyGraph)
    def postprocess_trajectory(self,
                               sample_batch,
                               other_agent_batches=None,
                               episode=None):
        if self.config["parameter_noise"]:
            # adjust the sigma of parameter space noise
            states, noisy_actions = [
                list(x) for x in sample_batch.columns(["obs", "actions"])
            ]
            self.sess.run(self.remove_noise_op)
            clean_actions = self.sess.run(
                self.output_actions,
                feed_dict={
                    self.cur_observations: states,
                    self.stochastic: False,
                    self.eps: .0
                })
            distance_in_action_space = np.sqrt(
                np.mean(np.square(clean_actions - noisy_actions)))
            self.pi_distance = distance_in_action_space
            if distance_in_action_space < self.config["exploration_sigma"]:
                self.parameter_noise_sigma_val *= 1.01
            else:
                self.parameter_noise_sigma_val /= 1.01
            self.parameter_noise_sigma.load(
                self.parameter_noise_sigma_val, session=self.sess)

        return _postprocess_dqn(self, sample_batch)

    @override(TFPolicyGraph)
    def get_weights(self):
        return self.variables.get_weights()

    @override(TFPolicyGraph)
    def set_weights(self, weights):
        self.variables.set_weights(weights)

    @override(PolicyGraph)
    def get_state(self):
        return [TFPolicyGraph.get_state(self), self.cur_epsilon]

    @override(PolicyGraph)
    def set_state(self, state):
        TFPolicyGraph.set_state(self, state[0])
        self.set_epsilon(state[1])

    def _build_q_network(self, obs, obs_space, actions):
        q_net = QNetwork(
            ModelCatalog.get_model({
                "obs": obs,
                "is_training": self._get_is_training_placeholder(),
            }, obs_space, 1, self.config["model"]), actions,
            self.config["critic_hiddens"],
            self.config["critic_hidden_activation"])
        return q_net.value, q_net.model

    def _build_p_network(self, obs, obs_space):
        policy_net = PNetwork(
            ModelCatalog.get_model({
                "obs": obs,
                "is_training": self._get_is_training_placeholder(),
            }, obs_space, 1, self.config["model"]), self.dim_actions,
            self.config["actor_hiddens"],
            self.config["actor_hidden_activation"],
            self.config["parameter_noise"])
        return policy_net.action_scores, policy_net.model

    def _build_action_network(self, p_values, stochastic, eps,
                              is_target=False):
        return ActionNetwork(
            p_values, self.low_action, self.high_action, stochastic, eps,
            self.config["exploration_theta"], self.config["exploration_sigma"],
            self.config["smooth_target_policy"], self.config["act_noise"],
            is_target, self.config["target_noise"],
            self.config["noise_clip"]).actions

    def _build_actor_critic_loss(self,
                                 q_t,
                                 q_tp1,
                                 q_tp0,
                                 twin_q_t=None,
                                 twin_q_tp1=None):
        return ActorCriticLoss(
            q_t, q_tp1, q_tp0, self.importance_weights, self.rew_t,
            self.done_mask, twin_q_t, twin_q_tp1,
            self.config["actor_loss_coeff"], self.config["critic_loss_coeff"],
            self.config["gamma"], self.config["n_step"],
            self.config["use_huber"], self.config["huber_threshold"],
            self.config["twin_q"])

    def _build_parameter_noise(self, pnet_params):
        self.parameter_noise_sigma_val = self.config["exploration_sigma"]
        self.parameter_noise_sigma = tf.get_variable(
            initializer=tf.constant_initializer(
                self.parameter_noise_sigma_val),
            name="parameter_noise_sigma",
            shape=(),
            trainable=False,
            dtype=tf.float32)
        self.parameter_noise = list()
        # No need to add any noise on LayerNorm parameters
        for var in pnet_params:
            noise_var = tf.get_variable(
                name=var.name.split(':')[0] + "_noise",
                shape=var.shape,
                initializer=tf.constant_initializer(.0),
                trainable=False)
            self.parameter_noise.append(noise_var)
        remove_noise_ops = list()
        for var, var_noise in zip(pnet_params, self.parameter_noise):
            remove_noise_ops.append(tf.assign_add(var, -var_noise))
        self.remove_noise_op = tf.group(*tuple(remove_noise_ops))
        generate_noise_ops = list()
        for var_noise in self.parameter_noise:
            generate_noise_ops.append(
                tf.assign(
                    var_noise,
                    tf.random_normal(
                        shape=var_noise.shape,
                        stddev=self.parameter_noise_sigma)))
        with tf.control_dependencies(generate_noise_ops):
            add_noise_ops = list()
            for var, var_noise in zip(pnet_params, self.parameter_noise):
                add_noise_ops.append(tf.assign_add(var, var_noise))
            self.add_noise_op = tf.group(*tuple(add_noise_ops))
        self.pi_distance = None

    def compute_td_error(self, obs_t, act_t, rew_t, obs_tp1, done_mask,
                         importance_weights):
        td_err = self.sess.run(
            self.loss.td_error,
            feed_dict={
                self.obs_t: [np.array(ob) for ob in obs_t],
                self.act_t: act_t,
                self.rew_t: rew_t,
                self.obs_tp1: [np.array(ob) for ob in obs_tp1],
                self.done_mask: done_mask,
                self.importance_weights: importance_weights
            })
        return td_err

    def reset_noise(self, sess):
        sess.run(self.reset_noise_op)

    def add_parameter_noise(self):
        if self.config["parameter_noise"]:
            self.sess.run(self.add_noise_op)

    # support both hard and soft sync
    def update_target(self, tau=None):
        return self.sess.run(
            self.update_target_expr,
            feed_dict={self.tau: tau or self.tau_value})

    def set_epsilon(self, epsilon):
        self.cur_epsilon = epsilon