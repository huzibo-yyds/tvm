# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Custom Relay operators used by local experiments."""

from __future__ import absolute_import

from . import _custom_make


def layer_norm(data, gamma, beta, axis=-1, epsilon=1e-5, center=True, scale=True):
    """Custom layer normalization operator.

    This mirrors ``relay.nn.layer_norm`` at the Relay API level, but lowers
    through the separately registered ``custom.layer_norm`` op.
    """

    return _custom_make.layer_norm(data, gamma, beta, axis, epsilon, center, scale)


def numpy_layer_norm(data, gamma, beta, axis=-1, epsilon=1e-5, center=True, scale=True):
    """LayerNorm operator backed by a Python NumPy PackedFunc at runtime."""

    return _custom_make.numpy_layer_norm(data, gamma, beta, axis, epsilon, center, scale)


# hzb 将TVM注册的全局函数映射为Python函数
