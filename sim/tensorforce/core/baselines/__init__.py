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

from tensorforce.core.baselines.baseline import Baseline
from tensorforce.core.baselines.aggregated_baseline import AggregatedBaseline
from tensorforce.core.baselines.network_baseline import NetworkBaseline
from tensorforce.core.baselines.mlp_baseline import MLPBaseline
from tensorforce.core.baselines.cnn_baseline import CNNBaseline


baselines = dict(
    aggregated=AggregatedBaseline,
    custom=NetworkBaseline,
    mlp=MLPBaseline,
    cnn=CNNBaseline
)


__all__ = [
    'baselines',
    'Baseline',
    'AggregatedBaseline',
    'NetworkBaseline',
    'MLPBaseline',
    'CNNBaseline'
]
