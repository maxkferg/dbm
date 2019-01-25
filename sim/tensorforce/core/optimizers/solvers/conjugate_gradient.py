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

from tensorforce import util
from tensorforce.core.optimizers.solvers import Iterative


class ConjugateGradient(Iterative):
    """
    Conjugate gradient algorithm which iteratively finds a solution $x$ for a system of linear  
    equations of the form $A x = b$, where $A x$ could be, for instance, a locally linear  
    approximation of a high-dimensional function.

    See below pseudo-code taken from  
    [Wikipedia](https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm):

    ```text
    def conjgrad(A, b, x_0):
        r_0 := b - A * x_0
        c_0 := r_0
        r_0^2 := r^T * r

        for t in 0, ..., max_iterations - 1:
            Ac := A * c_t
            cAc := c_t^T * Ac
            \alpha := r_t^2 / cAc
            x_{t+1} := x_t + \alpha * c_t
            r_{t+1} := r_t - \alpha * Ac
            r_{t+1}^2 := r_{t+1}^T * r_{t+1}
            if r_{t+1} < \epsilon:
                break
            \beta = r_{t+1}^2 / r_t^2
            c_{t+1} := r_{t+1} + \beta * c_t

        return x_{t+1}
    ```

    """

    def __init__(self, max_iterations, damping, unroll_loop=False):
        """
        Creates a new conjugate gradient solver instance.

        Args:
            max_iterations: Maximum number of iterations before termination.
            damping: Damping factor.
            unroll_loop: Unrolls the TensorFlow while loop if true.
        """
        assert damping >= 0.0
        self.damping = damping

        super(ConjugateGradient, self).__init__(max_iterations=max_iterations, unroll_loop=unroll_loop)

    def tf_solve(self, fn_x, x_init, b):
        """
        Iteratively solves the system of linear equations $A x = b$.

        Args:
            fn_x: A callable returning the left-hand side $A x$ of the system of linear equations.
            x_init: Initial solution guess $x_0$, zero vector if None.
            b: The right-hand side $b$ of the system of linear equations.

        Returns:
            A solution $x$ to the problem as given by the solver.
        """
        return super(ConjugateGradient, self).tf_solve(fn_x, x_init, b)

    def tf_initialize(self, x_init, b):
        """
        Initialization step preparing the arguments for the first iteration of the loop body:  
        $x_0, 0, p_0, r_0, r_0^2$.

        Args:
            x_init: Initial solution guess $x_0$, zero vector if None.
            b: The right-hand side $b$ of the system of linear equations.

        Returns:
            Initial arguments for tf_step.
        """
        if x_init is None:
            # Initial guess is zero vector if not given.
            x_init = [tf.zeros(shape=util.shape(t)) for t in b]

        initial_args = super(ConjugateGradient, self).tf_initialize(x_init)

        # r_0 := b - A * x_0
        # c_0 := r_0
        conjugate = residual = [t - fx for t, fx in zip(b, self.fn_x(x_init))]

        # r_0^2 := r^T * r
        squared_residual = tf.add_n(inputs=[tf.reduce_sum(input_tensor=(res * res)) for res in residual])

        return initial_args + (conjugate, residual, squared_residual)

    def tf_step(self, x, iteration, conjugate, residual, squared_residual):
        """
        Iteration loop body of the conjugate gradient algorithm.

        Args:
            x: Current solution estimate $x_t$.
            iteration: Current iteration counter $t$.
            conjugate: Current conjugate $c_t$.
            residual: Current residual $r_t$.
            squared_residual: Current squared residual $r_t^2$.

        Returns:
            Updated arguments for next iteration.
        """
        x, next_iteration, conjugate, residual, squared_residual = super(ConjugateGradient, self).tf_step(
            x, iteration, conjugate, residual, squared_residual
        )

        # Ac := A * c_t
        A_conjugate = self.fn_x(conjugate)

        # TODO: reference?
        if self.damping > 0.0:
            A_conjugate = [A_conj + self.damping * conj for A_conj, conj in zip(A_conjugate, conjugate)]

        # cAc := c_t^T * Ac
        conjugate_A_conjugate = tf.add_n(
            inputs=[tf.reduce_sum(input_tensor=(conj * A_conj)) for conj, A_conj in zip(conjugate, A_conjugate)]
        )

        # \alpha := r_t^2 / cAc
        alpha = squared_residual / tf.maximum(x=conjugate_A_conjugate, y=util.epsilon)

        # x_{t+1} := x_t + \alpha * c_t
        next_x = [t + alpha * conj for t, conj in zip(x, conjugate)]

        # r_{t+1} := r_t - \alpha * Ac
        next_residual = [res - alpha * A_conj for res, A_conj in zip(residual, A_conjugate)]

        # r_{t+1}^2 := r_{t+1}^T * r_{t+1}
        next_squared_residual = tf.add_n(inputs=[tf.reduce_sum(input_tensor=(res * res)) for res in next_residual])

        # \beta = r_{t+1}^2 / r_t^2
        beta = next_squared_residual / tf.maximum(x=squared_residual, y=util.epsilon)

        # c_{t+1} := r_{t+1} + \beta * c_t
        next_conjugate = [res + beta * conj for res, conj in zip(next_residual, conjugate)]

        return next_x, next_iteration, next_conjugate, next_residual, next_squared_residual

    def tf_next_step(self, x, iteration, conjugate, residual, squared_residual):
        """
        Termination condition: max number of iterations, or residual sufficiently small.

        Args:
            x: Current solution estimate $x_t$.
            iteration: Current iteration counter $t$.
            conjugate: Current conjugate $c_t$.
            residual: Current residual $r_t$.
            squared_residual: Current squared residual $r_t^2$.

        Returns:
            True if another iteration should be performed.
        """
        next_step = super(ConjugateGradient, self).tf_next_step(x, iteration, conjugate, residual, squared_residual)
        return tf.logical_and(x=next_step, y=(squared_residual >= util.epsilon))
