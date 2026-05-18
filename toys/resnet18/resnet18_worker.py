"""Offline ResNet18 image classification with TVM Relay.

All required files must be placed in this script's cache directory:

    cache/resnet18_Opset18.onnx
    cache/imagenet_classes.txt
    cache/kitten.jpg

Run from the apache-tvm-0.19.0 repository root:

    python toys/resnet18/resnet18_relay_classify.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import types

import numpy as np
from PIL import Image
import onnx


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CACHE_DIR = SCRIPT_DIR / "cache"
MODEL_FILE = CACHE_DIR / "resnet18_Opset18.onnx"
LABELS_FILE = CACHE_DIR / "imagenet_classes.txt"
DEFAULT_IMAGE_FILE = CACHE_DIR / "kitten.jpg"

IMAGE_NET_MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
IMAGE_NET_STD = np.array([0.229, 0.224, 0.225], dtype="float32")


# Force TVM Relay to stay offline. Without this, relay.build may try to fetch
# AutoTVM TopHub logs from the public internet.
os.environ.setdefault("TOPHUB_LOCATION", "NONE") #禁用TopHub下载，relay.build，tvm编译时会去下载调优好的算子调度文件
logging.getLogger("autotvm").setLevel(logging.ERROR)

# sys.path.insert(0, str(REPO_ROOT / "python"))

# TVM 0.19 expects onnx.mapping, while newer ONNX packages expose the same
# information through onnx._mapping.
if "onnx.mapping" not in sys.modules and hasattr(onnx, "_mapping"):
    mapping_module = types.ModuleType("onnx.mapping")
    mapping_module.TENSOR_TYPE_TO_NP_TYPE = {
        key: value.np_dtype for key, value in onnx._mapping.TENSOR_TYPE_MAP.items()
    }
    sys.modules["onnx.mapping"] = mapping_module

import tvm  # noqa: E402
from tvm import relay  # noqa: E402
from tvm.contrib import graph_executor  # noqa: E402


def require_file(path: pathlib.Path) -> pathlib.Path:
    if path.exists() and path.is_file() and path.stat().st_size > 0:
        return path
    raise FileNotFoundError(
        f"Required file is missing: {path}\n"
        "Put the required model, labels, and image files under this script's cache directory."
    )


def resize_shorter_side(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    if width < height:
        new_width = size
        new_height = round(height * size / width)
    else:
        new_height = size
        new_width = round(width * size / height)
    return image.resize((new_width, new_height), Image.Resampling.BILINEAR)


def center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = (width - size) // 2
    top = (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def preprocess_image(image_path: pathlib.Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = center_crop(resize_shorter_side(image, 256), 224)
    data = np.asarray(image).astype("float32") / 255.0
    data = (data - IMAGE_NET_MEAN) / IMAGE_NET_STD
    data = np.transpose(data, (2, 0, 1))
    return np.expand_dims(data, axis=0).astype("float32")


def load_labels(path: pathlib.Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    labels = json.loads(text) if text.startswith("[") else text.splitlines()
    return [label.strip() for label in labels]


def first_model_input_name(model: onnx.ModelProto) -> str:
    initializers = {initializer.name for initializer in model.graph.initializer}
    for model_input in model.graph.input:
        if model_input.name not in initializers:
            return model_input.name
    raise ValueError("Could not find an ONNX graph input.")


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.reshape(-1).astype("float64")
    logits = logits - np.max(logits)
    exps = np.exp(logits)
    return exps / np.sum(exps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-file",
        default=DEFAULT_IMAGE_FILE.name,
        help="Image filename under cache/. Default: kitten.jpg",
    )
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = require_file(MODEL_FILE)
    labels_path = require_file(LABELS_FILE)
    image_path = require_file(CACHE_DIR / args.image_file)

    labels = load_labels(labels_path)
    input_data = preprocess_image(image_path)

    print("Loading ONNX ResNet18")
    onnx_model = onnx.load(model_path)
    input_name = first_model_input_name(onnx_model)
    shape_dict = {input_name: input_data.shape}

    print("Converting ONNX to Relay graph")
    mod, params = relay.frontend.from_onnx(
        onnx_model,
        shape=shape_dict,
        freeze_params=True,
    )

    print("Compiling with relay.build for llvm")
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target="llvm", params=params)

    print("Running inference with graph executor")
    dev = tvm.cpu(0)
    module = graph_executor.GraphModule(lib["default"](dev))
    module.set_input(input_name, input_data)
    module.run()

    probs = softmax(module.get_output(0).numpy())
    top_indices = np.argsort(probs)[-min(args.topk, probs.size) :][::-1]

    print(f"image: {image_path}")
    print(f"input: {input_name} {input_data.shape}")
    print("top classifications:")
    for rank, index in enumerate(top_indices, start=1):
        label = labels[index] if index < len(labels) else f"class_{index}"
        print(f"{rank:>2}. {index:>4} {label:<30} {probs[index]:.4f}")


if __name__ == "__main__":
    main()
