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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorforce.core.preprocessors import Preprocessor


class ImageResize(Preprocessor):
    """
    Resize image to width x height.
    """

    def __init__(self, shape, width, height, scope='image_resize', summary_labels=()):
        self.size = (width, height)
        super(ImageResize, self).__init__(shape=shape, scope=scope, summary_labels=summary_labels)

    def tf_process(self, tensor):
        return tf.image.resize_images(images=tensor, size=self.size)

    def processed_shape(self, shape):
        return self.size + (shape[-1],)
