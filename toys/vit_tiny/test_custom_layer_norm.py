"""Small checks for custom.layer_norm and the replacement pass."""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
os.environ.setdefault("TOPHUB_LOCATION", "NONE")

import tvm  # noqa: E402
from tvm import relay  # noqa: E402
from tvm.contrib import graph_executor  # noqa: E402
from tvm.relay.op import custom  # noqa: E402

from custom_layer_norm_pass import ReplaceLayerNormWithCustom  # noqa: E402


def numpy_layer_norm(data, gamma, beta, epsilon):
    mean = np.mean(data, axis=-1, keepdims=True)
    var = np.var(data, axis=-1, keepdims=True)
    return (data - mean) / np.sqrt(var + epsilon) * gamma + beta


def run_relay(expr, params, inputs):
    func = relay.Function(relay.analysis.free_vars(expr), expr)
    mod = tvm.IRModule.from_expr(func)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target="llvm", params=params)

    dev = tvm.cpu(0)
    module = graph_executor.GraphModule(lib["default"](dev))
    for name, value in inputs.items():
        module.set_input(name, value)
    module.run()
    return module.get_output(0).numpy()


def check_custom_op():
    rng = np.random.default_rng(0)
    data_np = rng.normal(size=(2, 3, 4)).astype("float32")
    gamma_np = rng.normal(size=(4,)).astype("float32")
    beta_np = rng.normal(size=(4,)).astype("float32")
    epsilon = 1e-5

    data = relay.var("data", shape=data_np.shape, dtype="float32")
    gamma = relay.const(gamma_np)
    beta = relay.const(beta_np)
    expr = custom.layer_norm(data, gamma, beta, axis=-1, epsilon=epsilon)

    actual = run_relay(expr, params=None, inputs={"data": data_np})
    expected = numpy_layer_norm(data_np, gamma_np, beta_np, epsilon)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    print("custom.layer_norm single-op check: PASS")


def check_pass_direct_layer_norm():
    data = relay.var("data", shape=(2, 3, 4), dtype="float32")
    gamma = relay.var("gamma", shape=(4,), dtype="float32")
    beta = relay.var("beta", shape=(4,), dtype="float32")
    expr = relay.nn.layer_norm(data, gamma, beta, axis=-1, epsilon=1e-5)
    mod = tvm.IRModule.from_expr(relay.Function([data, gamma, beta], expr))

    replace_pass = ReplaceLayerNormWithCustom()
    seq = tvm.transform.Sequential(
        [relay.transform.InferType(), replace_pass, relay.transform.InferType()]
    )
    with tvm.transform.PassContext(opt_level=3):
        custom_mod = seq(mod)

    text = custom_mod.astext()
    assert "custom.layer_norm" in text, text
    assert replace_pass.replacement_count == 1
    print("direct nn.layer_norm pass check: PASS")


def main():
    check_custom_op()
    check_pass_direct_layer_norm()


if __name__ == "__main__":
    main()

'''
python test_custom_layer_norm.py 
custom.layer_norm single-op check: PASS
direct nn.layer_norm pass check: PASS
'''