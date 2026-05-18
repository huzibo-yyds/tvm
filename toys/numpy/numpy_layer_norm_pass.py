"""Relay pass that rewrites ViT LayerNorm patterns to custom.numpy_layer_norm."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tvm
from tvm import relay
from tvm.relay import ExprMutator
from tvm.relay.expr import Call, Constant, Expr
from tvm.relay.op import custom


def _op_name(expr: Expr) -> Optional[str]:
    if isinstance(expr, Call) and isinstance(expr.op, tvm.ir.Op):
        return expr.op.name
    return None


def _is_call(expr: Expr, name: str) -> bool:
    return _op_name(expr) == name


def _int_tuple(value) -> tuple[int, ...]:
    if value is None:
        return ()
    return tuple(int(item) for item in value)


def _checked_rank(expr: Expr) -> Optional[int]:
    try:
        checked_type = getattr(expr, "checked_type", None)
    except ValueError:
        return None
    shape = getattr(checked_type, "shape", None)
    if shape is None:
        return None
    return len(shape)


def _checked_dtype(expr: Expr) -> Optional[str]:
    try:
        checked_type = getattr(expr, "checked_type", None)
    except ValueError:
        return None
    dtype = getattr(checked_type, "dtype", None)
    return str(dtype) if dtype is not None else None


def _is_last_axis(axis: tuple[int, ...], rank: Optional[int]) -> bool:
    if axis == (-1,):
        return True
    if rank is None:
        return False
    return axis == (rank - 1,)


def _scalar_const_value(expr: Expr) -> Optional[float]:
    if not isinstance(expr, Constant):
        return None
    value = expr.data.numpy()
    if value.shape != ():
        return None
    return float(value)


def _is_one(expr: Expr) -> bool:
    value = _scalar_const_value(expr)
    return value is not None and np.isclose(value, 1.0)


def _structural_equal(lhs: Expr, rhs: Expr) -> bool:
    return bool(tvm.ir.structural_equal(lhs, rhs, map_free_vars=True))


@dataclass
class LayerNormCandidate:
    data: Expr
    gamma: Expr
    beta: Expr
    axis: int
    epsilon: float


class NumpyLayerNormRewriter(ExprMutator):
    """Replace selected LayerNorm candidates with custom.numpy_layer_norm."""

    def __init__(self, replace_indices: Optional[set[int]] = None) -> None:
        super().__init__()
        self.replace_indices = replace_indices
        self.candidate_count = 0
        self.replacements = 0
        self.skipped: Counter[str] = Counter()
        self.replaced_indices: list[int] = []

    def visit_call(self, call: Call) -> Expr:
        new_call = super().visit_call(call)

        candidate = self._match_direct_layer_norm(new_call)
        if candidate is None:
            candidate = self._match_decomposed_layer_norm(new_call)
        if candidate is None:
            return new_call

        candidate_index = self.candidate_count
        self.candidate_count += 1
        if self.replace_indices is not None and candidate_index not in self.replace_indices:
            return new_call

        self.replacements += 1
        self.replaced_indices.append(candidate_index)
        return custom.numpy_layer_norm(
            candidate.data,
            candidate.gamma,
            candidate.beta,
            axis=candidate.axis,
            epsilon=candidate.epsilon,
            center=True,
            scale=True,
        )

    def _match_direct_layer_norm(self, call: Expr) -> Optional[LayerNormCandidate]:
        if not _is_call(call, "nn.layer_norm"):
            return None

        attrs = call.attrs
        data, gamma, beta = call.args
        rank = _checked_rank(data)
        axis = int(attrs.axis)
        if not _is_last_axis((axis,), rank):
            self.skipped["direct_non_last_axis"] += 1
            return None
        dtype = _checked_dtype(data)
        if dtype is not None and dtype != "float32":
            self.skipped["direct_non_float32"] += 1
            return None
        if not bool(attrs.center) or not bool(attrs.scale):
            self.skipped["direct_without_center_or_scale"] += 1
            return None

        return LayerNormCandidate(data, gamma, beta, -1, float(attrs.epsilon))

    def _match_decomposed_layer_norm(self, expr: Expr) -> Optional[LayerNormCandidate]:
        if not _is_call(expr, "add"):
            return None

        lhs, rhs = expr.args
        gamma_mul = self._match_gamma_mul(lhs)
        beta = rhs
        if gamma_mul is None:
            gamma_mul = self._match_gamma_mul(rhs)
            beta = lhs
        if gamma_mul is None:
            return None

        data, gamma, axis, epsilon = gamma_mul
        return LayerNormCandidate(data, gamma, beta, axis, epsilon)

    def _match_gamma_mul(self, expr: Expr):
        if not _is_call(expr, "multiply"):
            return None

        lhs, rhs = expr.args
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
        div_match = self._match_divide_normalized(expr)
        if div_match is not None:
            return div_match

        if not _is_call(expr, "multiply"):
            return None

        lhs, rhs = expr.args
        subtract = lhs if _is_call(lhs, "subtract") else rhs if _is_call(rhs, "subtract") else None
        inv_std = rhs if subtract is lhs else lhs
        if subtract is None:
            return None

        data, mean = subtract.args
        mean_axis = self._match_mean(mean, data)
        if mean_axis is None:
            return None

        epsilon = self._match_inv_std(inv_std, data, mean, mean_axis)
        if epsilon is None:
            return None

        dtype = _checked_dtype(data)
        if dtype is not None and dtype != "float32":
            self.skipped["onnx_non_float32"] += 1
            return None

        return data, -1, epsilon

    def _match_divide_normalized(self, expr: Expr):
        if not _is_call(expr, "divide"):
            return None

        numerator, denominator = expr.args
        if not _is_call(numerator, "subtract") or not _is_call(denominator, "sqrt"):
            return None

        data, mean = numerator.args
        mean_axis = self._match_mean(mean, data)
        if mean_axis is None:
            return None

        epsilon = self._match_sqrt_mean_power_variance(denominator, numerator, mean_axis)
        if epsilon is None:
            return None

        dtype = _checked_dtype(data)
        if dtype is not None and dtype != "float32":
            self.skipped["onnx_non_float32"] += 1
            return None

        return data, -1, epsilon

    def _match_mean(self, expr: Expr, data: Expr) -> Optional[tuple[int, ...]]:
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
        if not _is_call(expr, "divide"):
            return None

        numerator, denominator = expr.args
        if not _is_one(numerator) or not _is_call(denominator, "sqrt"):
            return None

        sqrt_arg = denominator.args[0]
        if not _is_call(sqrt_arg, "add"):
            return None

        lhs, rhs = sqrt_arg.args
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
        sqrt_arg = sqrt.args[0]
        if not _is_call(sqrt_arg, "add"):
            return None

        lhs, rhs = sqrt_arg.args
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

# 📌 Relay Pass
@relay.transform.function_pass(opt_level=1)
class ReplaceLayerNormWithNumpyExtern:
    """Relay FunctionPass wrapper for NumPy-backed LayerNorm replacement."""

    def __init__(self, replace_indices: Optional[set[int]] = None) -> None:
        self.replace_indices = replace_indices
        self.candidate_count = 0
        self.replacement_count = 0
        self.replaced_indices: list[int] = []
        self.skipped = Counter()

    def transform_function(self, func, mod, ctx):
        rewriter = NumpyLayerNormRewriter(self.replace_indices)
        new_func = rewriter.visit(func)
        self.candidate_count += rewriter.candidate_count
        self.replacement_count += rewriter.replacements
        self.replaced_indices.extend(rewriter.replaced_indices)
        self.skipped.update(rewriter.skipped)
        return new_func

