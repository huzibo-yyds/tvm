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
"""Backend registrations for custom Relay operators."""

from __future__ import absolute_import

from tvm import te
from tvm.target import generic_func

from . import op as reg


@generic_func
def schedule_custom_layer_norm(attrs, outs, target):
    """朴素手写 schedule。

    FTVMCompute 已经在 C++ 里生成了 mean_sum/mean/var_sum/var/output
    这几个 TE tensor。这里先用 te.create_schedule 为最终输出创建
    schedule，TVM 会把依赖的中间 tensor 一起纳入 schedule。

    这个 schedule 的目标是先跑通 lowering/codegen，不做性能优化。
    后续如果要优化 CPU，可以在这里继续添加 split、parallel、
    vectorize、compute_at 等调度语句。
    """

    with target:
        return te.create_schedule([out.op for out in outs])


reg.register_schedule("custom.layer_norm", schedule_custom_layer_norm)
# hzb 注册
