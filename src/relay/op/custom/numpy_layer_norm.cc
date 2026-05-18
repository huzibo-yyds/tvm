/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/op/custom/numpy_layer_norm.cc
 * \brief NumPy runtime backed Relay layer_norm operator for correctness checks.
 *
 * 这个文件只注册 `custom.numpy_layer_norm` 的 Relay op、类型关系和
 * Python FFI Make 函数。它不注册 C++ FTVMCompute。
 *
 * 对应的 compute/schedule 在 python/tvm/relay/op/_custom.py 中注册：
 *
 *   custom.numpy_layer_norm
 *     -> te.extern
 *     -> tir.call_packed("custom.runtime.numpy_layer_norm", ...)
 *     -> Python NumPy PackedFunc
 *
 * 这样它和 `custom.layer_norm` 的 TE 手写 compute 实现完全解耦。
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>

#include "../op_common.h"

namespace tvm {
namespace relay {

// NumPy extern 版本复用 LayerNormAttrs，但拥有独立的 type relation。
// 规则和普通 LayerNorm 一致：
//   output shape/dtype == data shape/dtype
//   gamma/beta shape == data.shape[axis]
bool CustomNumpyLayerNormRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                             const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 4);
  ICHECK_EQ(num_inputs, 3);

  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const LayerNormAttrs* param = attrs.as<LayerNormAttrs>();
  ICHECK(param != nullptr);

  int axis = param->axis >= 0 ? param->axis : param->axis + data->shape.size();
  ICHECK(axis >= 0 && axis < static_cast<int>(data->shape.size()));

  reporter->Assign(types[1], TensorType({data->shape[axis]}, data->dtype));
  reporter->Assign(types[2], TensorType({data->shape[axis]}, data->dtype));
  reporter->Assign(types[3], TensorType(data->shape, data->dtype));
  return true;
}

// Python API relay.op.custom.numpy_layer_norm(...) 最终通过这个 Make 函数
// 构造 Relay Call(op="custom.numpy_layer_norm")。
Expr MakeCustomNumpyLayerNorm(Expr data, Expr gamma, Expr beta, int axis, double epsilon,
                              bool center, bool scale) {
  auto attrs = make_object<LayerNormAttrs>();
  attrs->axis = axis;
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;

  static const Op& op = Op::Get("custom.numpy_layer_norm");
  return Call(op, {data, gamma, beta}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.custom._make.numpy_layer_norm")
    .set_body_typed(MakeCustomNumpyLayerNorm);

RELAY_REGISTER_OP("custom.numpy_layer_norm")
    .describe(R"code(Custom layer normalization operator backed by Python NumPy at runtime.)code"
              TVM_ADD_FILELINE)
    .set_attrs_type<LayerNormAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "Input to which layer_norm will be applied.")
    .add_argument("gamma", "Tensor", "The gamma scale factor.")
    .add_argument("beta", "Tensor", "The beta offset factor.")
    .set_support_level(10)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .add_type_rel("CustomNumpyLayerNorm", CustomNumpyLayerNormRel);

}  // namespace relay
}  // namespace tvm

