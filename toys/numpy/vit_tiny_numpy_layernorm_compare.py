"""Compare baseline ViT-Tiny against NumPy-backed LayerNorm runtime calls."""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import tvm
from tvm import relay

from numpy_layer_norm_pass import ReplaceLayerNormWithNumpyExtern
from numpy_layer_norm_runtime import (
    get_call_count,
    register_numpy_layer_norm_runtime,
    reset_call_count,
)
from vit_tiny_numpy_common import (
    DEFAULT_IMAGE_URL,
    compile_and_run,
    count_op,
    load_vit_tiny_inputs,
    parse_replace_indices,
    print_topk,
    relay_from_onnx,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=pathlib.Path)
    parser.add_argument("--preprocessor-path", type=pathlib.Path)
    parser.add_argument("--image-path", type=pathlib.Path)
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--target", default="llvm")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--replace-indices")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--print-ir", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    register_numpy_layer_norm_runtime()
    reset_call_count()

    print("Loading ViT-Tiny ONNX")
    onnx_model, input_name, input_data, labels, image_path = load_vit_tiny_inputs(args)

    print("Converting ONNX to Relay")
    mod, params = relay_from_onnx(onnx_model, input_name, input_data)

    replace_pass = ReplaceLayerNormWithNumpyExtern(parse_replace_indices(args.replace_indices))
    seq = tvm.transform.Sequential(
        [relay.transform.InferType(), replace_pass, relay.transform.InferType()]
    )
    with tvm.transform.PassContext(opt_level=3):
        numpy_mod = seq(mod)

    numpy_count = count_op(numpy_mod["main"], "custom.numpy_layer_norm")
    if replace_pass.replacement_count == 0 or numpy_count == 0:
        print(f"candidates: {replace_pass.candidate_count}")
        print(f"skipped: {dict(replace_pass.skipped)}")
        raise RuntimeError("NumPy LayerNorm pass did not replace any LayerNorm expressions.")

    print(f"numpy_layer_norm candidates: {replace_pass.candidate_count}")
    print(f"numpy_layer_norm replacements: {replace_pass.replacement_count}")
    print(f"numpy_layer_norm replaced indices: {replace_pass.replaced_indices}")
    print(f"custom.numpy_layer_norm calls in IR: {numpy_count}")

    if args.print_ir:
        print("\n=== Baseline Relay IR ===")
        print(mod)
        print("\n=== NumPy Custom Relay IR ===")
        print(numpy_mod)

    print("\nCompiling/running baseline")
    baseline_output = compile_and_run(mod, params, input_name, input_data, args.target)

    print("Compiling/running NumPy custom")
    numpy_output = compile_and_run(numpy_mod, params, input_name, input_data, args.target)

    max_abs_diff = float(np.max(np.abs(baseline_output - numpy_output)))
    denom = np.maximum(np.abs(baseline_output), 1e-12)
    max_rel_diff = float(np.max(np.abs(baseline_output - numpy_output) / denom))
    allclose = bool(np.allclose(baseline_output, numpy_output, rtol=args.rtol, atol=args.atol))

    topk = min(args.topk, baseline_output.size)
    print_topk("baseline", baseline_output, labels, topk)
    print_topk("numpy custom", numpy_output, labels, topk)

    print("\nComparison:")
    print(f"image: {image_path}")
    print(f"input: {input_name} {input_data.shape}")
    print(f"numpy runtime calls: {get_call_count()}")
    print(f"max_abs_diff: {max_abs_diff:.8g}")
    print(f"max_rel_diff: {max_rel_diff:.8g}")
    print(f"allclose(rtol={args.rtol}, atol={args.atol}): {allclose}")

    if not allclose:
        raise RuntimeError("Baseline and NumPy custom outputs differ beyond the tolerance.")


if __name__ == "__main__":
    main()

