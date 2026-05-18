"""Shared helpers for ViT-Tiny NumPy LayerNorm scripts."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import types
import urllib.request

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
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
VIT_TINY_CACHE_DIR = REPO_ROOT / "toys" / "vit_tiny" / "cache"
TOYS_CACHE_DIR = REPO_ROOT / "toys" / "cache"

MODEL_URL = (
    "https://huggingface.co/onnx-community/vit-tiny-patch16-224-ONNX/"
    "resolve/main/onnx/model.onnx"
)
PREPROCESSOR_URL = (
    "https://huggingface.co/onnx-community/vit-tiny-patch16-224-ONNX/"
    "resolve/main/preprocessor_config.json"
)
DEFAULT_IMAGE_URL = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"


def fetch_file(url: str, path: pathlib.Path, sibling_name: str | None = None) -> pathlib.Path:
    """Get a file from local cache, sibling cache, or the network."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path

    if sibling_name is None:
        sibling_name = path.name
    for sibling_dir in (VIT_TINY_CACHE_DIR, TOYS_CACHE_DIR):
        sibling_path = sibling_dir / sibling_name
        if sibling_path.exists() and sibling_path.stat().st_size > 0:
            shutil.copyfile(sibling_path, path)
            return path

    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:
        path.write_bytes(response.read())
    return path


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_labels(path: pathlib.Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    labels = json.loads(text) if text.startswith("[") else text.splitlines()
    return [label.strip() for label in labels]


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


def _size_value(value, default: int) -> int:
    if isinstance(value, dict):
        return int(value.get("shortest_edge") or value.get("height") or value.get("width") or default)
    if value is None:
        return default
    return int(value)


def preprocess_image(image_path: pathlib.Path, preprocessor: dict) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")

    resize_size = _size_value(preprocessor.get("size"), 224)
    crop_size = _size_value(preprocessor.get("crop_size"), resize_size)
    image = center_crop(resize_shorter_side(image, resize_size), crop_size)

    data = np.asarray(image).astype("float32")
    if preprocessor.get("do_rescale", True):
        data *= float(preprocessor.get("rescale_factor", 1.0 / 255.0))

    mean = np.array(preprocessor.get("image_mean", [0.5, 0.5, 0.5]), dtype="float32")
    std = np.array(preprocessor.get("image_std", [0.5, 0.5, 0.5]), dtype="float32")
    data = (data - mean) / std
    data = np.transpose(data, (2, 0, 1))
    return np.expand_dims(data, axis=0).astype("float32")


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


def print_topk(name: str, logits: np.ndarray, labels: list[str], topk: int) -> None:
    probs = softmax(logits)
    indices = np.argsort(probs)[-topk:][::-1]
    print(f"\n{name} top-{topk}:")
    for rank, index in enumerate(indices, start=1):
        label = labels[index] if index < len(labels) else f"class_{index}"
        print(f"{rank:>2}. {index:>4} {label:<30} {probs[index]:.6f}")


def count_op(expr: relay.Expr, op_name: str) -> int:
    class Counter(relay.ExprVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.count = 0

        def visit_call(self, call):
            if isinstance(call.op, tvm.ir.Op) and call.op.name == op_name:
                self.count += 1
            super().visit_call(call)

    visitor = Counter()
    visitor.visit(expr)
    return visitor.count


def parse_replace_indices(value: str | None) -> set[int] | None:
    if value is None or value.strip() == "":
        return None
    indices = {int(item.strip()) for item in value.split(",") if item.strip()}
    if any(index < 0 for index in indices):
        raise ValueError("--replace-indices must contain non-negative integers.")
    return indices


def load_vit_tiny_inputs(args):
    model_path = args.model_path or fetch_file(MODEL_URL, CACHE_DIR / "vit_tiny.onnx")
    preprocessor_path = args.preprocessor_path or fetch_file(
        PREPROCESSOR_URL, CACHE_DIR / "preprocessor_config.json"
    )
    labels_path = fetch_file(LABELS_URL, CACHE_DIR / "imagenet_classes.txt")
    image_path = args.image_path or fetch_file(
        args.image_url, CACHE_DIR / pathlib.Path(args.image_url).name, sibling_name="kitten.jpg"
    )

    onnx_model = onnx.load(model_path)
    input_data = preprocess_image(image_path, load_json(preprocessor_path))
    labels = load_labels(labels_path)
    input_name = first_model_input_name(onnx_model)
    return onnx_model, input_name, input_data, labels, image_path


def relay_from_onnx(onnx_model, input_name: str, input_data: np.ndarray):
    return relay.frontend.from_onnx(
        onnx_model,
        shape={input_name: input_data.shape},
        freeze_params=True,
    )


def compile_and_run(mod, params, input_name: str, input_data: np.ndarray, target: str):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    dev = tvm.cpu(0)
    module = graph_executor.GraphModule(lib["default"](dev))
    module.set_input(input_name, input_data)
    module.run()
    return module.get_output(0).numpy()

