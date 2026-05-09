"""Run ResNet18 image classification through TVM Relay.

Run from the apache-tvm-0.19.0 repository root:

    conda activate tvm-0.19
    python toys/resnet18_relay_classify.py

This example follows the classic Relay flow:

    ONNX model -> relay.frontend.from_onnx -> relay.build -> graph executor
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys
import types
import urllib.request

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
os.environ.setdefault("TOPHUB_LOCATION", "NONE")

from PIL import Image  # noqa: E402
import onnx  # noqa: E402

if "onnx.mapping" not in sys.modules and hasattr(onnx, "_mapping"):
    mapping_module = types.ModuleType("onnx.mapping")
    mapping_module.TENSOR_TYPE_TO_NP_TYPE = {
        key: value.np_dtype for key, value in onnx._mapping.TENSOR_TYPE_MAP.items()
    }
    sys.modules["onnx.mapping"] = mapping_module

import tvm  # noqa: E402
from tvm import relay  # noqa: E402
from tvm.contrib import graph_executor  # noqa: E402


CACHE_DIR = pathlib.Path(__file__).resolve().parent / "cache"
SIBLING_CACHE_DIR = REPO_ROOT.parent / "apache-tvm" / "toys" / "cache"

RESNET18_URL = (
    "https://github.com/onnx/models/raw/"
    "bec48b6a70e5e9042c0badbaafefe4454e072d08/"
    "Computer_Vision/resnet18_Opset18_timm/resnet18_Opset18.onnx"
)
DEFAULT_IMAGE_URL = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

IMAGE_NET_MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
IMAGE_NET_STD = np.array([0.229, 0.224, 0.225], dtype="float32")


def fetch_file(url: str, path: pathlib.Path, sibling_name: str | None = None) -> pathlib.Path:
    """Get a file from local cache, sibling cache, or the network."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path

    if sibling_name is None:
        sibling_name = path.name
    sibling_path = SIBLING_CACHE_DIR / sibling_name
    if sibling_path.exists() and sibling_path.stat().st_size > 0:
        shutil.copyfile(sibling_path, path)
        return path

    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:
        path.write_bytes(response.read())
    return path


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
    if text.startswith("["):
        labels = json.loads(text)
    else:
        labels = text.splitlines()
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
    parser.add_argument("--image-path", type=pathlib.Path)
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = fetch_file(
        RESNET18_URL,
        CACHE_DIR / "resnet18_Opset18.onnx",
        sibling_name="resnet18_Opset18.onnx",
    )
    labels_path = fetch_file(LABELS_URL, CACHE_DIR / "imagenet_classes.txt")
    image_path = args.image_path
    if image_path is None:
        image_path = fetch_file(
            args.image_url,
            CACHE_DIR / pathlib.Path(args.image_url).name,
            sibling_name="kitten.jpg",
        )

    labels = load_labels(labels_path)
    input_data = preprocess_image(image_path)

    '---------------------------------------------------'
    print("Loading ONNX ResNet18")
    onnx_model = onnx.load(model_path)
    input_name = first_model_input_name(onnx_model)
    shape_dict = {input_name: input_data.shape}

    '---------------------------------------------------'
    print("Converting ONNX to Relay graph")
    mod, params = relay.frontend.from_onnx(
        onnx_model,
        shape=shape_dict,
        freeze_params=True, # 将params作为relay常量嵌入计算图
    )

    '---------------------------------------------------'
    target = "llvm"
    print("Compiling with relay.build for llvm")
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params) # Relay 的 `relay.build` 返回的通常是一个 factory module
        # 编译+生成可执行runtime

    '---------------------------------------------------'
    print("Running inference with graph executor")
    dev = tvm.cpu(0)
    module = graph_executor.GraphModule(lib["default"](dev)) # relay编译后，常见运行方式 GraphModule
    module.set_input(input_name, input_data)
    module.run()
    output = module.get_output(0).numpy()
    probs = softmax(output)



    topk = min(args.topk, probs.size)
    top_indices = np.argsort(probs)[-topk:][::-1]

    print(f"image: {image_path}")
    print(f"input: {input_name} {input_data.shape}")
    print("top classifications:")
    for rank, index in enumerate(top_indices, start=1):
        label = labels[index] if index < len(labels) else f"class_{index}"
        print(f"{rank:>2}. {index:>4} {label:<30} {probs[index]:.4f}")


if __name__ == "__main__":
    main()
