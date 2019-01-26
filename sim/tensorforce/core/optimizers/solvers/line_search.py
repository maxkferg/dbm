# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf

from tensorforce import util, TensorForceError
from tensorforce.core.optimizers.solvers import Iterative


class LineSearch(Iterative):
    """
    Line search algorithm which iteratively optimizes the value $f(x)$ for $x$ on the line between  
    $x'$ and $x_0$ by optimistically taking the first acceptable $x$ starting from $x_0$ and  
    moving towards $x'$.
    """

    def __init__(self, max_iterations, accept_ratio, mode, parameter, unroll_loop=False):
        """
        Creates a new line search solver instance.

        Args:
            max_iterations: Maximum number of iterations before termination.
            accept_ratio: Lower limit of what improvement ratio over $x = x'$ is acceptable  
                (based either on a given estimated improvement or with respect to the value at  
                $x = x'$).
            mode: Mode of movement between $x_0$ and $x'$, either 'linear' or 'exponential'.
            parameter: Movement mode parameter, additive or multiplicative, respectively.
            unroll_loop: Unrolls the TensorFlow while loop if true.
        """
        assert accept_ratio >= 0.0
        self.accept_ratio = accept_ratio

        # TODO: Implement such sequences more generally, also useful for learning rate decay or so.
        if mode not in ('linear', 'exponential'):
            raise TensorForceError("Invalid line search mode: {}, please choose one of'linear' or 'exponential'".format(mode))
        self.mode = mode
        self.parameter = parameter

        super(LineSearch, self).__init__(max_iterations=max_iterations, unroll_loop=unroll_loop)

    def tf_solve(self, fn_x, x_init, base_value, target_value, estimated_improvement=None):
        """
        Iteratively optimizes $f(x)$ for $x$ on the line between $x'$ and $x_0$.

        Args:
            fn_x: A callable returning the value $f(x)$ at $x$.
            x_init: Initial solution guess $x_0$.
            base_value: Value $f(x')$ at $x = x'$.
            target_value: Value $f(x_0)$ at $x = x_0$.
            estimated_improvement: Estimated improvement for $x = x_0$, $f(x')$ if None.

        Returns:
            A solution $x$ to the problem as given by the solver.
        """
        return super(LineSearch, self).tf_solve(fn_x, x_init, base_value, target_value, estimated_improvement)

    def tf_initialize(self, x_init, base_value, target_value, estimated_improvement):
        """
        Initialization step preparing the arguments for the first iteration of the loop body.

        Args:
            x_init: Initial solution guess $x_0$.
            base_value: Value $f(x')$ at $x = x'$.
            target_value: Value $f(x_0)$ at $x = x_0$.
            estimated_improvement: Estimated value at $x = x_0$, $f(x')$ if None.

        Returns:
            Initial arguments for tf_step.
        """
        self.base_value = base_value

        if estimated_improvement is None:  # TODO: Is this a good alternative?
            estimated_improvement = tf.abs(x=base_value)

        first_step = super(LineSearch, self).tf_initialize(x_init)

        improvement = tf.divide(
            x=(target_value - self.base_value),
            y=tf.maximum(x=estimated_improvement, y=util.epsilon)
        )

        last_improvement = improvement - 1.0

        if self.mode == 'linear':
            deltas = [-t * self.parameter for t in x_init]
            self.estimated_incr = -estimated_improvement * self.parameter

        elif self.mode == 'exponential':
            deltas = [-t * self.parameter for t in x_init]

        return first_step + (deltas, improvement, last_improvement, estimated_improvement)

    def tf_step(self, x, iteration, deltas, improvement, last_improvement, estimated_improvement):
        """
        Iteration loop body of the line search algorithm.

        Args:
            x: Current solution estimate $x_t$.
            iteration: Current iteration counter $t$.
            deltas: Current difference $x_t - x'$.
            improvement: Current improvement $(f(x_t) - f(x')) / v'$.
            last_improvement: Last improvement $(f(x_{t-1}) - f(x')) / v'$.
            estimated_improvement: Current estimated value $v'$.

        Returns:
            Updated arguments for next iteration.
        """
        x, next_iteration, deltas, improvement, last_improvement, estimated_improvement = super(LineSearch, self).tf_step(
            x, iteration, deltas, improvement, last_improvement, estimated_improvement
        )

        next_x = [t + delta for t, delta in zip(x, deltas)]

        if self.mode == 'linear':
            next_deltas = deltas
            next_estimated_improvement = estimated_improvement + self.estimated_incr

        elif self.mode == 'exponential':
            next_deltas = [delta * self.parameter for delta in deltas]
            next_estimated_improvement = estimated_improvement * self.parameter

        target_value = self.fn_x(next_deltas)

        next_improvement = tf.divide(
            x=(target_value - self.base_value),
            y=tf.maximum(x=next_estimated_improvement, y=util.epsilon)
        )

        return next_x, next_iteration, next_deltas, next_improvement, improvement, next_estimated_improvement

    def tf_next_step(self, x, iteration, deltas, improvement, last_improvement, estimated_improvement):
        """
        Termination condition: max number of iterations, or no improvement for last step, or  
        improvement less than acceptable ratio, or estimated value not positive.

        Args:
            x: Current solution estimate $x_t$.
            iteration: Current iteration counter $t$.
            deltas: Current difference $x_t - x'$.
            improvement: Current improvement $(f(x_t) - f(x')) / v'$.
            last_improvement: Last improvement $(f(x_{t-1}) - f(x')) / v'$.
            estimated_improvement: Current estimated value $v'$.

        Returns:
            True if another iteration should be performed.
        """
        next_step = super(LineSearch, self).tf_next_step(
            x, iteration, deltas, improvement, last_improvement, estimated_improvement
        )

        def undo_deltas():
            value = self.fn_x([-delta for delta in deltas])
            with tf.control_dependencies(control_inputs=(value,)):
                # Trivial operation to enforce control dependency
                return tf.less(x=value, y=value)  # == False

        improved = tf.cond(
            pred=(improvement > last_improvement),
            true_fn=(lambda: True),
            false_fn=undo_deltas
        )

        next_step = tf.logical_and(x=next_step, y=improved)
        next_step = tf.logical_and(x=next_step, y=(improvement < self.accept_ratio))
        return tf.logical_and(x=next_step, y=(estimated_improvement > util.epsilon))
