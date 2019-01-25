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
import tensorforce.core.optimizers.solvers


class Solver(object):
    """
    Generic TensorFlow-based solver which solves a not yet further specified  
    equation/optimization problem.
    """

    def __init__(self):
        """
        Creates a new solver instance.
        """
        # TensorFlow function
        self.solve = tf.make_template(name_='solver', func_=self.tf_solve)

    def tf_solve(self, fn_x, *args):
        """
        Solves an equation/optimization for $x$ involving an expression $f(x)$.

        Args:
            fn_x: A callable returning an expression $f(x)$ given $x$.
            *args: Additional solver-specific arguments.

        Returns:
            A solution $x$ to the problem as given by the solver.
        """
        raise NotImplementedError

    @staticmethod
    def from_config(config, kwargs=None):
        """
        Creates a solver from a specification dict.
        """
        return util.get_object(
            obj=config,
            predefined=tensorforce.core.optimizers.solvers.solvers,
            kwargs=kwargs
        )
