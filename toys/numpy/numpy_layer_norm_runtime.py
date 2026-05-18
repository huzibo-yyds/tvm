"""Python NumPy runtime for custom.numpy_layer_norm.

This module registers a TVM PackedFunc that can be called from TIR generated
by te.extern.  It is intentionally for correctness experiments only: every
call copies tensors through Python/NumPy and is not meant for performance.
"""

from __future__ import annotations

import numpy as np
import tvm


RUNTIME_FUNC_NAME = "custom.runtime.numpy_layer_norm"
CALL_COUNT = 0


def reset_call_count() -> None:
    """Reset the debug counter used by tests and scripts."""

    global CALL_COUNT
    CALL_COUNT = 0


def get_call_count() -> int:
    """Return how many times the runtime function has been invoked."""

    return CALL_COUNT


def _reshape_param(param: np.ndarray, data_ndim: int, axis: int) -> np.ndarray:
    """Reshape gamma/beta so NumPy broadcasts along the LayerNorm axis."""

    shape = [1] * data_ndim
    shape[axis] = param.shape[0]
    return param.reshape(shape)


def numpy_layer_norm_runtime(data, gamma, beta, out, axis, epsilon) -> None:
    """PackedFunc body called by generated TVM runtime code. Numpy实现"""

    global CALL_COUNT
    CALL_COUNT += 1

    x = data.numpy().astype("float32", copy=False)
    gamma_np = gamma.numpy().astype("float32", copy=False)
    beta_np = beta.numpy().astype("float32", copy=False)

    axis = int(axis)
    if axis < 0:
        axis += x.ndim
    epsilon = np.float32(float(epsilon))

    gamma_np = _reshape_param(gamma_np, x.ndim, axis)
    beta_np = _reshape_param(beta_np, x.ndim, axis)

    mean = np.mean(x, axis=axis, keepdims=True, dtype=np.float32)
    centered = x - mean
    var = np.mean(centered * centered, axis=axis, keepdims=True, dtype=np.float32)
    result = centered / np.sqrt(var + epsilon).astype("float32", copy=False)
    result = result * gamma_np + beta_np

    out.copyfrom(result.astype("float32", copy=False))


# 📌 注册 PackedFunc
def register_numpy_layer_norm_runtime() -> None:
    """Register the NumPy LayerNorm runtime PackedFunc in this Python process."""

    tvm.register_func(RUNTIME_FUNC_NAME, numpy_layer_norm_runtime, override=True)
