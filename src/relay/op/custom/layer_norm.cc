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
 * \file src/relay/op/custom/layer_norm.cc
 * \brief 用于本地 pass 实验的自定义 Relay layer_norm 算子。
 *
 * 注册一个 Relay 算子通常需要下面几部分：
 *
 * 1. 属性类型
 *    描述 Relay Call 携带的参数。本例复用 Relay 已有的 LayerNormAttrs，
 *    没有重新定义新的 attrs node。
 *
 * 2. 类型关系
 *    告诉 Relay InferType 如何推导输入/输出 TensorType。
 *
 * 3. Make 函数 + TVM_REGISTER_GLOBAL
 *    通过 FFI 从 Python 构造 Relay Call。Python 侧调用
 *    relay.op.custom.layer_norm(...)，最终会进入下面的 MakeCustomLayerNorm。
 *
 * 4. RELAY_REGISTER_OP
 *    注册算子元信息，例如 op 名字、输入个数、参数说明、类型关系、
 *    fusion pattern，以及 lowering 相关属性。
 *
 * 5. compute/schedule 或 strategy
 *    告诉 relay.build 如何 lower 这个算子。本文件注册 FTVMCompute，
 *    python/tvm/relay/op/_custom.py 里注册 schedule。
 *
 * 本版本的 FTVMCompute 不再调用 TOPI layer_norm，而是直接使用 TE
 * 手写 LayerNorm 的计算：
 *
 *     mean = sum(data) / N
 *     var = sum((data - mean)^2) / N
 *     out = (data - mean) * rsqrt(var + epsilon) * gamma + beta
 *
 * 因此这里的 compute 是真正由本文件实现的；schedule 仍然沿用 Python
 * 侧注册的通用 schedule，后续如果要优化性能，可以再单独手写 schedule。
 */

#include <tvm/te/operation.h>
#include <tvm/tir/op.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>

#include "../op_common.h"

namespace tvm {
namespace relay {

// 1️⃣ 注册属性类型，复用 Relay 已有的 LayerNormAttrs。
// 这个 attrs 定义了 LayerNorm 的参数：include/tvm/relay/attrs/nn.h
 
// 2️⃣ 类型关系推导
// `types` 里包含每个输入的类型槽位，以及一个输出类型槽位：
//   types[0] = data
//   types[1] = gamma
//   types[2] = beta
//   types[3] = output
// 下面的关系表达了 LayerNorm 的 shape 规则：
//   gamma/beta 是 1-D tensor，长度等于 data.shape[axis]；
//   output 和 data 有相同 shape、相同 dtype。
bool CustomLayerNormRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                        const TypeReporter& reporter) {
  // 参数校验 三个输入类型，加一个输出类型。
  ICHECK_EQ(types.size(), 4);
  ICHECK_EQ(num_inputs, 3); // 这两行为防御性检查，确保传对了参数个数

  // 如果 data 类型暂时还不知道，返回 false，让 Relay type solver 在获得更多类型信息后再次尝试这个关系。
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false; // Relay动态处理形状关键机制

  // 复用内置 nn.layer_norm 的 attrs： axis, epsilon, center, scale.
  const LayerNormAttrs* param = attrs.as<LayerNormAttrs>(); // 获取属性
  ICHECK(param != nullptr);

  // 将负数 axis 转成规范化后的正数 axis。
  // 对 ViT-Tiny 的 LayerNorm 来说，这里通常是 axis=-1。
  int axis = param->axis >= 0 ? param->axis : param->axis + data->shape.size();
  ICHECK(axis >= 0 && axis < static_cast<int>(data->shape.size()));

  // 根据 data 和 axis 推导 gamma/beta 的类型。
  // 例如 data shape 是 (1, 197, 192)，axis=-1，则 gamma/beta shape 是 (192)。
  reporter->Assign(types[1], TensorType({data->shape[axis]}, data->dtype));
  reporter->Assign(types[2], TensorType({data->shape[axis]}, data->dtype));

  // LayerNorm 不改变 data 的 shape 和 dtype。
  reporter->Assign(types[3], TensorType(data->shape, data->dtype));

  return true;
}

// 3️⃣ Make｜Relay Call 节点的 C++ 构造函数。
// Python 不会直接实例化 C++ Call 节点，而是通过 Python helper
// 最终调用下面注册的 global PackedFunc：
//   relay.op.custom._make.layer_norm
// 这个函数把用户传入的参数打包进 LayerNormAttrs，然后返回：
//   Call(Op::Get("custom.layer_norm"), {data, gamma, beta}, attrs)
Expr MakeCustomLayerNorm(Expr data, Expr gamma, Expr beta, int axis, double epsilon, bool center,
                         bool scale) {
  auto attrs = make_object<LayerNormAttrs>();
  attrs->axis = axis;
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;

  // 这个字符串是 Relay op registry 里的名字，不是 C++ namespace。
  static const Op& op = Op::Get("custom.layer_norm");
  return Call(op, {data, gamma, beta}, Attrs(attrs), {});
}

// 将 C++ Make 函数暴露给 Python FFI（Foreign Function Interface）。
// python/tvm/relay/op/_custom_make.py 会用下面这句初始化这个命名空间：
//   tvm._ffi._init_api("relay.op.custom._make", __name__)
//
// 初始化之后，Python 就可以调用：
//   _custom_make.layer_norm(...)
TVM_REGISTER_GLOBAL("relay.op.custom._make.layer_norm").set_body_typed(MakeCustomLayerNorm);

// 5️⃣ 手写 TE compute。
//
// 这个函数是 custom.layer_norm 的真正数学实现。它没有调用 TOPI layer_norm，
// 而是直接构造五个 TE Tensor：
//
//   custom_layer_norm_mean_sum:
//     对 data 的 axis 维求和。
//
//   custom_layer_norm_mean:
//     mean_sum / N。
//
//   custom_layer_norm_var_sum:
//     对 (data - mean)^2 的 axis 维求和。
//
//   custom_layer_norm_var:
//     var_sum / N。
//
//   custom_layer_norm:
//     使用 mean/var/gamma/beta 计算最终输出。
//
// 当前版本为了和 ViT-Tiny pass 保持一致，只支持：
//   - float32
//   - center=True
//   - scale=True
//   - 单个归一化 axis
//
// schedule 仍在 python/tvm/relay/op/_custom.py 中注册。也就是说：
//   本文件负责“怎么算”；
//   Python schedule 负责“怎么把这个 TE compute lower 到 TIR”。
Array<te::Tensor> CustomLayerNormCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                         const Type& out_type) {
  ICHECK_EQ(inputs.size(), 3U);
  const LayerNormAttrs* param = attrs.as<LayerNormAttrs>();
  ICHECK(param != nullptr);

  ICHECK(param->center) << "custom.layer_norm currently expects center=True";
  ICHECK(param->scale) << "custom.layer_norm currently expects scale=True";

  te::Tensor data = inputs[0];
  te::Tensor gamma = inputs[1];
  te::Tensor beta = inputs[2];
  ICHECK_EQ(data->dtype, DataType::Float(32))
      << "manual custom.layer_norm currently supports float32 only";
  ICHECK_EQ(gamma->dtype, data->dtype);
  ICHECK_EQ(beta->dtype, data->dtype);

  const int ndim = static_cast<int>(data->shape.size());
  ICHECK_GT(ndim, 0) << "custom.layer_norm cannot reduce a 0-dim Tensor";
  int axis = param->axis >= 0 ? param->axis : param->axis + ndim;
  ICHECK_GE(axis, 0);
  ICHECK_LT(axis, ndim);

  Array<PrimExpr> reduced_shape;
  for (int i = 0; i < ndim; ++i) {
    if (i != axis) {
      reduced_shape.push_back(data->shape[i]);
    }
  }

  PrimExpr reduce_extent = tvm::cast(data->dtype, data->shape[axis]);

  auto make_data_indices = [axis, ndim](const Array<te::Var>& indices,
                                        const te::IterVar& reduce_axis) {
    Array<PrimExpr> data_indices;
    int non_reduce_index = 0;
    for (int i = 0; i < ndim; ++i) {
      if (i == axis) {
        data_indices.push_back(reduce_axis);
      } else {
        data_indices.push_back(indices[non_reduce_index]);
        ++non_reduce_index;
      }
    }
    return data_indices;
  };

  auto mean_reduce_axis = te::reduce_axis(Range(0, data->shape[axis]), "custom_ln_k_mean");
  te::Tensor mean_sum = te::compute(
      reduced_shape,
      [&](const Array<te::Var>& indices) {
        Array<PrimExpr> data_indices = make_data_indices(indices, mean_reduce_axis);
        return tvm::sum(data(data_indices), {mean_reduce_axis});
      },
      "custom_layer_norm_mean_sum", "comm_reduce");

  te::Tensor mean = te::compute(
      reduced_shape,
      [&](const Array<te::Var>& indices) { return mean_sum(indices) / reduce_extent; },
      "custom_layer_norm_mean", "injective");

  auto var_reduce_axis = te::reduce_axis(Range(0, data->shape[axis]), "custom_ln_k_var");
  te::Tensor var_sum = te::compute(
      reduced_shape,
      [&](const Array<te::Var>& indices) {
        Array<PrimExpr> data_indices = make_data_indices(indices, var_reduce_axis);
        PrimExpr diff = data(data_indices) - mean(indices);
        return tvm::sum(diff * diff, {var_reduce_axis});
      },
      "custom_layer_norm_var_sum", "comm_reduce");

  te::Tensor var = te::compute(
      reduced_shape,
      [&](const Array<te::Var>& indices) { return var_sum(indices) / reduce_extent; },
      "custom_layer_norm_var", "injective");

  te::Tensor output = te::compute(
      data->shape,
      [&](const Array<te::Var>& indices) {
        Array<PrimExpr> reduced_indices;
        for (int i = 0; i < ndim; ++i) {
          if (i != axis) {
            reduced_indices.push_back(indices[i]);
          }
        }

        Array<PrimExpr> param_indices;
        param_indices.push_back(indices[axis]);

        PrimExpr diff = data(indices) - mean(reduced_indices);
        PrimExpr inv_std =
            tvm::rsqrt(var(reduced_indices) + tir::make_const(data->dtype, param->epsilon));
        return diff * inv_std * gamma(param_indices) + beta(param_indices);
      },
      "custom_layer_norm", "injective");

  return {output};
}

// 4️⃣ 注册 Relay 算子本身。
//
// 这会告诉 Relay："custom.layer_norm" 是一个合法算子，它有几个输入、
// 拥有什么 attrs、如何推导类型，以及 relay.build 应该如何拿到它的 TE compute。
RELAY_REGISTER_OP("custom.layer_norm")
    .describe(R"code(Custom layer normalization operator used by the ViT-Tiny toy pass.)code"
              TVM_ADD_FILELINE) // 算子描述
    .set_attrs_type<LayerNormAttrs>() // 绑定属性类型
    .set_num_inputs(3) // 声明输入个数
    .add_argument("data", "Tensor", "Input to which layer_norm will be applied.")
    .add_argument("gamma", "Tensor", "The gamma scale factor.")
    .add_argument("beta", "Tensor", "The beta offset factor.") // 声明输入
    .set_support_level(10) //设置算子的支持级别，默认为3，10是实验性算子，分类标记，不影响编译
    // 对 Relay fusion 来说，把它当成 opaque op。
    // LayerNorm 内部包含 reduction，不应该把它声明成简单 elementwise/broadcast op。
    .set_attr<TOpPattern>("TOpPattern", kOpaque) // 算子属性，不参与算子融合

    // ⭐️ 为这个 op 注册 TE compute。 TE（Tensor Expression）
    //
    // 这里就是“算子实现”绑定的位置。relay.build 会调用这个 FTVMCompute，
    // 生成 TE tensors，然后应用 python/tvm/relay/op/_custom.py 里注册的 schedule，
    // lower 到 TIR，最后对目标后端 codegen。
    .set_attr<FTVMCompute>("FTVMCompute", CustomLayerNormCompute)
    .add_type_rel("CustomLayerNorm", CustomLayerNormRel); // 注册上面定义的类型关系函数

}  // namespace relay
}  // namespace tvm

// hzb
