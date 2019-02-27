import tensorflow as tf
import tensorflow.contrib.slim as slim
from ray.rllib.models import ModelCatalog, Model
from ray.rllib.models.misc import get_activation_fn, flatten, normc_initializer

class Custom_CNN(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        """Define the layers of a custom model.

        Arguments:
            input_dict (dict): Dictionary of input tensors, including "obs",
                "prev_action", "prev_reward", "is_training".
            num_outputs (int): Output tensor must be of size
                [BATCH_SIZE, num_outputs].
            options (dict): Model options.

        Returns:
            (outputs, feature_layer): Tensors of size [BATCH_SIZE, num_outputs]
                and [BATCH_SIZE, desired_feature_size].

        When using dict or tuple observation spaces, you can access
        the nested sub-observation batches here as well:

        Examples:
            >>> print(input_dict)
            {'prev_actions': <tf.Tensor shape=(?,) dtype=int64>,
             'prev_rewards': <tf.Tensor shape=(?,) dtype=float32>,
             'is_training': <tf.Tensor shape=(), dtype=bool>,
             'obs': OrderedDict([
                ('sensors', OrderedDict([
                    ('front_cam', [
                        <tf.Tensor shape=(?, 10, 10, 3) dtype=float32>,
                        <tf.Tensor shape=(?, 10, 10, 3) dtype=float32>]),
                    ('position', <tf.Tensor shape=(?, 3) dtype=float32>),
                    ('velocity', <tf.Tensor shape=(?, 3) dtype=float32>)]))])}
        """
        obs = input_dict["obs"]

        # Dense connections processing sensor data
        with tf.name_scope("sensors"):
            dense = tf.concat([
                obs['robot_theta'], 
                obs['robot_velocity'],
                obs['target'],
                obs['ckpts'],
            ], axis=1, name='concat')

        # Convolutional layers processing maps
        with tf.name_scope("conv_net"):
            conv1 = slim.conv2d(
                obs['maps'],
                16,
                kernel_size=(3,3),
                stride=2,
                activation_fn=tf.nn.relu,
                padding="SAME",
                scope="conv1")
            conv2 = slim.conv2d(
                conv1,
                32,
                kernel_size=(3,3),
                stride=2,
                activation_fn=tf.nn.relu,
                scope="conv2")
            conv2 = slim.max_pool2d(
                conv2, [2, 2], scope='pool2'
            )
            conv3 = slim.conv2d(
                conv2,
                32,
                kernel_size=(3,3),
                stride=2,
                activation_fn=tf.nn.relu,
                scope="conv3")
            conv4 = slim.conv2d(
                conv3,
                32,
                kernel_size=(3,3),
                stride=1,
                activation_fn=tf.nn.relu,
                scope="conv4")
            print("CONV", conv4)
            conv_flat = flatten(conv4)

        # Combining with dense layers
        with tf.name_scope("combined"):
            combined = tf.concat([dense, conv_flat], axis=1)

            hidden = slim.fully_connected(
                combined, 
                128,
                weights_initializer=normc_initializer(1.0),
                activation_fn=tf.nn.relu)

            last_layer = slim.fully_connected(
                hidden, 
                64,
                weights_initializer=normc_initializer(1.0),
                activation_fn=tf.nn.relu)

            output = slim.fully_connected(
                last_layer, 
                num_outputs,
                weights_initializer=normc_initializer(0.01),
                activation_fn=None)

        return output, last_layer


ModelCatalog.register_custom_model("custom_cnn", Custom_CNN)
