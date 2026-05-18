"""将 ViT-Tiny 中的 LayerNorm 子图改写为 custom.layer_norm。

这个文件只负责 Relay IR 层面的图改写：

1. 如果图里已经有直接的 nn.layer_norm 调用，就直接替换成 custom.layer_norm。
2. 如果 ONNX frontend 已经把 LayerNorm 分解成 mean/subtract/power/sqrt/divide/multiply/add，
   就识别这个子图并重新合成为 custom.layer_norm。

真正的 custom.layer_norm 算子注册、类型关系、TE compute 和 schedule 不在这里，
而是在 TVM 源码和 Python op glue 中完成。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tvm
from tvm import relay
from tvm.relay import ExprMutator # 和之前pi0 pass的不同
from tvm.relay.expr import Call, Constant, Expr
from tvm.relay.op import custom


def _op_name(expr: Expr) -> Optional[str]:
    """返回 Relay Call 的 op 名字；非 Call 节点返回 None。"""
    if isinstance(expr, Call) and isinstance(expr.op, tvm.ir.Op):
        return expr.op.name
    return None


def _is_call(expr: Expr, name: str) -> bool:
    """判断 expr 是否是指定 op 名字的 Relay Call。"""
    return _op_name(expr) == name


def _int_tuple(value) -> tuple[int, ...]:
    """把 TVM attrs 中可能为空的 Array/列表转换成普通 Python tuple。"""
    if value is None:
        return ()
    return tuple(int(item) for item in value)


def _checked_rank(expr: Expr) -> Optional[int]:
    """从 checked_type 里读取张量 rank；类型尚未推导时返回 None。"""
    try:
        checked_type = getattr(expr, "checked_type", None)
    except ValueError:
        return None
    shape = getattr(checked_type, "shape", None)
    if shape is None:
        return None
    return len(shape)


def _checked_dtype(expr: Expr) -> Optional[str]:
    """从 checked_type 里读取 dtype；类型尚未推导时返回 None。"""
    try:
        checked_type = getattr(expr, "checked_type", None)
    except ValueError:
        return None
    dtype = getattr(checked_type, "dtype", None)
    return str(dtype) if dtype is not None else None


def _is_last_axis(axis: tuple[int, ...], rank: Optional[int]) -> bool:
    """判断归一化 axis 是否表示最后一维。"""
    if axis == (-1,):
        return True
    if rank is None:
        return False
    return axis == (rank - 1,)


def _scalar_const_value(expr: Expr) -> Optional[float]:
    """读取标量常量的 Python float 值；非标量常量返回 None。"""
    if not isinstance(expr, Constant):
        return None
    value = expr.data.numpy()
    if value.shape != ():
        return None
    return float(value)


def _is_one(expr: Expr) -> bool:
    """判断表达式是否是标量常量 1。"""
    value = _scalar_const_value(expr)
    return value is not None and np.isclose(value, 1.0)


def _structural_equal(lhs: Expr, rhs: Expr) -> bool:
    """使用 TVM 的结构等价判断，允许自由变量按名字映射。"""
    return bool(tvm.ir.structural_equal(lhs, rhs, map_free_vars=True))


@dataclass
class DecomposedLayerNorm:
    """保存从分解 LayerNorm 子图中提取出来的关键参数。"""

    data: Expr
    gamma: Expr
    beta: Expr
    axis: int
    epsilon: float


class LayerNormRewriter(ExprMutator): # 📌 使用ExprMutator的关键
    """遍历 Relay Function，把可识别的 LayerNorm 改写为 custom.layer_norm。"""

    def __init__(self) -> None:
        super().__init__()
        # replacements 用于端到端脚本检查：如果为 0，就说明自定义 pass 没真正生效。
        self.replacements = 0
        # skipped 记录保守跳过的原因，便于排查为什么某些 LayerNorm 没被替换。
        self.skipped: Counter[str] = Counter()

    def visit_call(self, call: Call) -> Expr:
        """后序遍历 Call 节点，并尝试 direct 和 decomposed 两种改写。"""
        # 先递归改写子表达式，保证当前节点拿到的是已经更新过的 args。
        new_call = super().visit_call(call)

        # 情况一：Relay 图里仍然保留着 nn.layer_norm。
        direct = self._rewrite_direct_layer_norm(new_call)
        if direct is not None:
            return direct

        # 情况二：ONNX frontend 已经把 LayerNorm 展开成一串基础算子。
        decomposed = self._match_decomposed_layer_norm(new_call)
        if decomposed is not None:
            self.replacements += 1
            return custom.layer_norm(
                decomposed.data,
                decomposed.gamma,
                decomposed.beta,
                axis=decomposed.axis,
                epsilon=decomposed.epsilon,
                center=True,
                scale=True,
            )

        return new_call

    def _rewrite_direct_layer_norm(self, call: Expr) -> Optional[Expr]:
        """匹配并替换直接的 nn.layer_norm 调用。"""
        if not _is_call(call, "nn.layer_norm"):
            return None

        attrs = call.attrs
        data, gamma, beta = call.args
        rank = _checked_rank(data)
        axis = int(attrs.axis)
        # 这个 demo 只处理 ViT-Tiny 中最常见的最后一维 LayerNorm。
        if not _is_last_axis((axis,), rank):
            self.skipped["direct_non_last_axis"] += 1
            return None
        # 先限定 float32，避免替换到未验证过的 dtype。
        dtype = _checked_dtype(data)
        if dtype is not None and dtype != "float32":
            self.skipped["direct_non_float32"] += 1
            return None
        # custom.layer_norm v1 要求 gamma/beta 都存在。
        if not bool(attrs.center) or not bool(attrs.scale):
            self.skipped["direct_without_center_or_scale"] += 1
            return None

        self.replacements += 1
        return custom.layer_norm(
            data,
            gamma,
            beta,
            axis=-1,
            epsilon=float(attrs.epsilon),
            center=True,
            scale=True,
        )

    def _match_decomposed_layer_norm(self, expr: Expr) -> Optional[DecomposedLayerNorm]:
        """匹配分解后的 LayerNorm 末端：add(mul(norm, gamma), beta)。"""
        if not _is_call(expr, "add"):
            return None

        lhs, rhs = expr.args
        # add 和 multiply 都是交换律算子，因此左右两边都尝试匹配。
        gamma_mul = self._match_gamma_mul(lhs)
        beta = rhs
        if gamma_mul is None:
            gamma_mul = self._match_gamma_mul(rhs)
            beta = lhs
        if gamma_mul is None:
            return None

        data, gamma, axis, epsilon = gamma_mul
        return DecomposedLayerNorm(data=data, gamma=gamma, beta=beta, axis=axis, epsilon=epsilon)

    def _match_gamma_mul(self, expr: Expr):
        """匹配 scale 部分：multiply(normalized, gamma)。"""
        if not _is_call(expr, "multiply"):
            return None

        lhs, rhs = expr.args
        # gamma 可能在 multiply 的任意一侧。
        norm = self._match_normalized(lhs)
        gamma = rhs
        if norm is None:
            norm = self._match_normalized(rhs)
            gamma = lhs
        if norm is None:
            return None

        data, axis, epsilon = norm
        return data, gamma, axis, epsilon

    def _match_normalized(self, expr: Expr):
        """匹配归一化主体：(data - mean) / std 或 (data - mean) * inv_std。"""
        # ViT-Tiny ONNX 当前常见形式是 divide(subtract, sqrt(...))。
        div_match = self._match_divide_normalized(expr)
        if div_match is not None:
            return div_match

        # 兼容另一类 frontend/优化后形式：subtract(data, mean) * (1 / sqrt(var + eps))。
        if not _is_call(expr, "multiply"):
            return None

        lhs, rhs = expr.args
        subtract = lhs if _is_call(lhs, "subtract") else rhs if _is_call(rhs, "subtract") else None
        inv_std = rhs if subtract is lhs else lhs
        if subtract is None:
            return None

        data, mean = subtract.args
        # mean 必须是对同一个 data 在最后一维求均值。
        mean_axis = self._match_mean(mean, data)
        if mean_axis is None:
            return None

        # inv_std 必须来自同一个 data/mean/axis 的方差。
        epsilon = self._match_inv_std(inv_std, data, mean, mean_axis)
        if epsilon is None:
            return None

        dtype = _checked_dtype(data)
        if dtype is not None and dtype != "float32":
            self.skipped["onnx_non_float32"] += 1
            return None

        return data, -1, epsilon

    def _match_divide_normalized(self, expr: Expr):
        """匹配 ONNX 常见形式：divide(subtract(data, mean), sqrt(var + eps))。"""
        if not _is_call(expr, "divide"):
            return None

        numerator, denominator = expr.args
        if not _is_call(numerator, "subtract") or not _is_call(denominator, "sqrt"):
            return None

        data, mean = numerator.args
        mean_axis = self._match_mean(mean, data)
        if mean_axis is None:
            return None

        # ONNX 导出的 variance 常见为 mean(power(data - mean, 2))。
        epsilon = self._match_sqrt_mean_power_variance(denominator, numerator, mean_axis)
        if epsilon is None:
            return None

        dtype = _checked_dtype(data)
        if dtype is not None and dtype != "float32":
            self.skipped["onnx_non_float32"] += 1
            return None

        return data, -1, epsilon

    def _match_mean(self, expr: Expr, data: Expr) -> Optional[tuple[int, ...]]:
        """匹配 mean(data, axis=last, keepdims=True)。"""
        if not _is_call(expr, "mean"):
            return None
        if not _structural_equal(expr.args[0], data):
            return None

        attrs = expr.attrs
        axis = _int_tuple(attrs.axis)
        if not bool(attrs.keepdims) or bool(attrs.exclude):
            return None
        if not _is_last_axis(axis, _checked_rank(data)):
            self.skipped["onnx_non_last_axis"] += 1
            return None
        return axis

    def _match_inv_std(
        self,
        expr: Expr,
        data: Expr,
        mean: Expr,
        axis: tuple[int, ...],
    ) -> Optional[float]:
        """匹配 inv_std = 1 / sqrt(variance(data, mean) + epsilon)。"""
        if not _is_call(expr, "divide"):
            return None

        numerator, denominator = expr.args
        if not _is_one(numerator) or not _is_call(denominator, "sqrt"):
            return None

        sqrt_arg = denominator.args[0]
        if not _is_call(sqrt_arg, "add"):
            return None

        lhs, rhs = sqrt_arg.args
        # epsilon 可能在 add 的任意一侧。
        variance = lhs if _is_call(lhs, "variance") else rhs if _is_call(rhs, "variance") else None
        epsilon_expr = rhs if variance is lhs else lhs
        if variance is None:
            return None

        if not _structural_equal(variance.args[0], data):
            return None
        if not _structural_equal(variance.args[1], mean):
            return None

        attrs = variance.attrs
        var_axis = _int_tuple(attrs.axis)
        if var_axis != axis or not bool(attrs.keepdims) or bool(attrs.exclude) or bool(attrs.unbiased):
            return None

        epsilon = _scalar_const_value(epsilon_expr)
        if epsilon is None:
            return None
        return epsilon

    def _match_sqrt_mean_power_variance(
        self,
        sqrt: Expr,
        subtract: Expr,
        axis: tuple[int, ...],
    ) -> Optional[float]:
        """匹配 sqrt(mean(power(data - mean, 2)) + epsilon) 形式的标准差。"""
        sqrt_arg = sqrt.args[0]
        if not _is_call(sqrt_arg, "add"):
            return None

        lhs, rhs = sqrt_arg.args
        # variance_mean 是 mean(power(...))，epsilon 是 add 的另一侧。
        variance_mean = lhs if _is_call(lhs, "mean") else rhs if _is_call(rhs, "mean") else None
        epsilon_expr = rhs if variance_mean is lhs else lhs
        if variance_mean is None:
            return None

        attrs = variance_mean.attrs
        var_axis = _int_tuple(attrs.axis)
        if var_axis != axis or not bool(attrs.keepdims) or bool(attrs.exclude):
            return None

        power = variance_mean.args[0]
        if not _is_call(power, "power"):
            return None
        base, exponent = power.args
        if not _structural_equal(base, subtract):
            return None

        exponent_value = _scalar_const_value(exponent)
        if exponent_value is None or not np.isclose(exponent_value, 2.0):
            return None

        epsilon = _scalar_const_value(epsilon_expr)
        if epsilon is None:
            return None
        return epsilon


@relay.transform.function_pass(opt_level=1)
class ReplaceLayerNormWithCustom:
    """Relay FunctionPass 包装类，供 Sequential 或脚本直接调用。"""

    def __init__(self) -> None:
        # 这两个字段会被 vit_tiny_layernorm_compare.py 读取并打印。
        self.replacement_count = 0
        self.skipped = Counter()

    def transform_function(self, func, mod, ctx):
        """TVM pass manager 对每个 Relay Function 调用的入口。"""
        rewriter = LayerNormRewriter()
        new_func = rewriter.visit(func)
        self.replacement_count += rewriter.replacements
        self.skipped.update(rewriter.skipped)
        return new_func

# hzb，pass识别替换逻辑