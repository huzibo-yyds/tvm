"""使用 DFPatternCallback 实现的 custom.layer_norm 替换 pass。

这个文件和 custom_layer_norm_pass.py 做同一件事：把 Relay 中的 LayerNorm
改写为 custom.layer_norm。区别在于这里使用 TVM 的 dataflow pattern API，
先声明要匹配的子图模板，再在 callback 中构造替换表达式。

适合把它和 ExprMutator 版本并排阅读：

- DFPatternCallback：pattern 写法更接近“画出子图”。
- ExprMutator：分支判断更自由，适合复杂语义检查。
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import DFPatternCallback, is_op, rewrite, wildcard
from tvm.relay.expr import Call, Constant, Expr
from tvm.relay.op import custom


def _int_tuple(value) -> tuple[int, ...]:
    """把 TVM attrs 中可能为空的 axis 转为 Python tuple。"""
    if value is None:
        return ()
    return tuple(int(item) for item in value)


def _checked_rank(expr: Expr) -> Optional[int]:
    """从 checked_type 读取 rank；如果类型尚不可用则返回 None。"""
    try:
        checked_type = getattr(expr, "checked_type", None)
    except ValueError:
        return None
    shape = getattr(checked_type, "shape", None)
    if shape is None:
        return None
    return len(shape)


def _checked_dtype(expr: Expr) -> Optional[str]:
    """从 checked_type 读取 dtype；如果类型尚不可用则返回 None。"""
    try:
        checked_type = getattr(expr, "checked_type", None)
    except ValueError:
        return None
    dtype = getattr(checked_type, "dtype", None)
    return str(dtype) if dtype is not None else None


def _is_last_axis(axis: tuple[int, ...], rank: Optional[int]) -> bool:
    """判断 axis 是否代表最后一维。"""
    if axis == (-1,):
        return True
    if rank is None:
        return False
    return axis == (rank - 1,)


def _scalar_const_value(expr: Expr) -> Optional[float]:
    """读取标量 Constant 的 float 值。"""
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
    """判断两个 Relay 表达式是否结构等价。"""
    return bool(tvm.ir.structural_equal(lhs, rhs, map_free_vars=True))


def _mean_axis_is_valid(mean: Expr, data: Expr) -> Optional[tuple[int, ...]]:
    """检查 mean(data, axis=last, keepdims=True) 并返回 axis。"""
    if not isinstance(mean, Call):
        return None
    if not _structural_equal(mean.args[0], data):
        return None

    attrs = mean.attrs
    axis = _int_tuple(attrs.axis)
    if not bool(attrs.keepdims) or bool(attrs.exclude):
        return None
    if not _is_last_axis(axis, _checked_rank(data)):
        return None
    return axis


def _dtype_is_supported(data: Expr) -> bool:
    """本 demo 只验证 float32；未知 dtype 交给后续 TVM 类型检查。"""
    dtype = _checked_dtype(data)
    return dtype is None or dtype == "float32"


class _DirectLayerNormCallback(DFPatternCallback):
    """匹配直接的 nn.layer_norm(data, gamma, beta)。"""

    def __init__(self, owner, require_type: bool = False) -> None:
        super().__init__(require_type)
        self.owner = owner
        self.data = wildcard()
        self.gamma = wildcard()
        self.beta = wildcard()
        self.pattern = is_op("nn.layer_norm")(self.data, self.gamma, self.beta)

    def callback(self, pre, post, node_map):
        if not isinstance(post, Call):
            return post

        data = node_map[self.data][0]
        gamma = node_map[self.gamma][0]
        beta = node_map[self.beta][0]
        attrs = post.attrs

        axis = int(attrs.axis)
        if not _is_last_axis((axis,), _checked_rank(data)):
            self.owner.skipped["direct_non_last_axis"] += 1
            return post
        if not _dtype_is_supported(data):
            self.owner.skipped["direct_non_float32"] += 1
            return post
        if not bool(attrs.center) or not bool(attrs.scale):
            self.owner.skipped["direct_without_center_or_scale"] += 1
            return post

        self.owner.replacement_count += 1
        return custom.layer_norm(
            data,
            gamma,
            beta,
            axis=-1,
            epsilon=float(attrs.epsilon),
            center=True,
            scale=True,
        )


class _DivideLayerNormCallback(DFPatternCallback):
    """匹配 ONNX 常见的 divide(subtract, sqrt(mean(power)+eps)) LayerNorm。"""

    def __init__(self, owner, require_type: bool = False) -> None:
        super().__init__(require_type)
        self.owner = owner

        self.data = wildcard()
        self.gamma = wildcard()
        self.beta = wildcard()
        self.epsilon = wildcard()
        self.exponent = wildcard()

        self.mean = is_op("mean")(self.data)
        self.subtract = is_op("subtract")(self.data, self.mean)
        self.power = is_op("power")(self.subtract, self.exponent)
        self.variance_mean = is_op("mean")(self.power)
        self.var_eps = is_op("add")(self.variance_mean, self.epsilon) | is_op("add")(
            self.epsilon, self.variance_mean
        )
        self.sqrt = is_op("sqrt")(self.var_eps)
        self.normalized = is_op("divide")(self.subtract, self.sqrt)
        self.scaled = is_op("multiply")(self.normalized, self.gamma) | is_op("multiply")(
            self.gamma, self.normalized
        )
        self.pattern = is_op("add")(self.scaled, self.beta) | is_op("add")(self.beta, self.scaled)

    def callback(self, pre, post, node_map):
        data = node_map[self.data][0]
        gamma = node_map[self.gamma][0]
        beta = node_map[self.beta][0]
        mean = node_map[self.mean][0]
        variance_mean = node_map[self.variance_mean][0]
        subtract = node_map[self.subtract][0]
        exponent = node_map[self.exponent][0]
        epsilon_expr = node_map[self.epsilon][0]

        axis = _mean_axis_is_valid(mean, data)
        if axis is None:
            self.owner.skipped["dfp_divide_invalid_mean"] += 1
            return post

        attrs = variance_mean.attrs
        var_axis = _int_tuple(attrs.axis)
        if var_axis != axis or not bool(attrs.keepdims) or bool(attrs.exclude):
            self.owner.skipped["dfp_divide_invalid_variance_mean"] += 1
            return post

        power_base = node_map[self.power][0].args[0]
        if not _structural_equal(power_base, subtract):
            self.owner.skipped["dfp_divide_invalid_power_base"] += 1
            return post

        exponent_value = _scalar_const_value(exponent)
        if exponent_value is None or not np.isclose(exponent_value, 2.0):
            self.owner.skipped["dfp_divide_invalid_power_exponent"] += 1
            return post

        epsilon = _scalar_const_value(epsilon_expr)
        if epsilon is None:
            self.owner.skipped["dfp_divide_non_scalar_epsilon"] += 1
            return post

        if not _dtype_is_supported(data):
            self.owner.skipped["onnx_non_float32"] += 1
            return post

        self.owner.replacement_count += 1
        return custom.layer_norm(data, gamma, beta, axis=-1, epsilon=epsilon, center=True, scale=True)


class _InvStdLayerNormCallback(DFPatternCallback):
    """匹配 subtract(data, mean) * (1 / sqrt(variance(data, mean) + eps)) 形式。"""

    def __init__(self, owner, require_type: bool = False) -> None:
        super().__init__(require_type)
        self.owner = owner

        self.data = wildcard()
        self.gamma = wildcard()
        self.beta = wildcard()
        self.epsilon = wildcard()
        self.one = wildcard()

        self.mean = is_op("mean")(self.data)
        self.subtract = is_op("subtract")(self.data, self.mean)
        self.variance = is_op("variance")(self.data, self.mean)
        self.var_eps = is_op("add")(self.variance, self.epsilon) | is_op("add")(
            self.epsilon, self.variance
        )
        self.sqrt = is_op("sqrt")(self.var_eps)
        self.inv_std = is_op("divide")(self.one, self.sqrt)
        self.normalized = is_op("multiply")(self.subtract, self.inv_std) | is_op("multiply")(
            self.inv_std, self.subtract
        )
        self.scaled = is_op("multiply")(self.normalized, self.gamma) | is_op("multiply")(
            self.gamma, self.normalized
        )
        self.pattern = is_op("add")(self.scaled, self.beta) | is_op("add")(self.beta, self.scaled)

    def callback(self, pre, post, node_map):
        data = node_map[self.data][0]
        gamma = node_map[self.gamma][0]
        beta = node_map[self.beta][0]
        mean = node_map[self.mean][0]
        variance = node_map[self.variance][0]
        one = node_map[self.one][0]
        epsilon_expr = node_map[self.epsilon][0]

        axis = _mean_axis_is_valid(mean, data)
        if axis is None:
            self.owner.skipped["dfp_inv_std_invalid_mean"] += 1
            return post

        attrs = variance.attrs
        var_axis = _int_tuple(attrs.axis)
        if var_axis != axis or not bool(attrs.keepdims) or bool(attrs.exclude) or bool(attrs.unbiased):
            self.owner.skipped["dfp_inv_std_invalid_variance"] += 1
            return post

        if not _is_one(one):
            self.owner.skipped["dfp_inv_std_numerator_not_one"] += 1
            return post

        epsilon = _scalar_const_value(epsilon_expr)
        if epsilon is None:
            self.owner.skipped["dfp_inv_std_non_scalar_epsilon"] += 1
            return post

        if not _dtype_is_supported(data):
            self.owner.skipped["onnx_non_float32"] += 1
            return post

        self.owner.replacement_count += 1
        return custom.layer_norm(data, gamma, beta, axis=-1, epsilon=epsilon, center=True, scale=True)


@relay.transform.function_pass(opt_level=1)
class ReplaceLayerNormWithCustomDFPattern:
    """使用 DFPatternCallback 的 Relay FunctionPass 包装类。"""

    def __init__(self) -> None:
        self.replacement_count = 0
        self.skipped = Counter()

    def transform_function(self, func, mod, ctx):
        """对一个 Relay Function 依次应用 direct、divide、inv_std 三类 pattern。"""
        func = rewrite(_DirectLayerNormCallback(self), func)
        func = rewrite(_DivideLayerNormCallback(self), func)
        func = rewrite(_InvStdLayerNormCallback(self), func)
        return func
