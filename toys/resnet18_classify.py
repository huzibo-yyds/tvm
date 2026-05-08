"""Run ResNet18 image classification through TVM Relax.

This example mirrors the flow in the Tencent Cloud article, but uses an
ONNX ResNet18 model because MXNet is not available for this local Python
3.13 environment.

Run from the repository root:
    .venv/bin/python toys/resnet18_classify.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request

import numpy as np
from PIL import Image


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

import onnx  # noqa: E402
import tvm  # noqa: E402
from tvm import relax  # noqa: E402
from tvm.relax.frontend import onnx as tvm_onnx  # noqa: E402


CACHE_DIR = pathlib.Path(__file__).resolve().parent / "cache"

RESNET18_URL = (
    "https://github.com/onnx/models/raw/"
    "bec48b6a70e5e9042c0badbaafefe4454e072d08/"
    "Computer_Vision/resnet18_Opset18_timm/resnet18_Opset18.onnx"
)
DEFAULT_IMAGE_URL = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
LABELS_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
)

IMAGE_NET_MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
IMAGE_NET_STD = np.array([0.229, 0.224, 0.225], dtype="float32")


def download(url: str, path: pathlib.Path) -> pathlib.Path:
    """Download a file once and reuse it on later runs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
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


def to_numpy(output) -> np.ndarray:
    if isinstance(output, tvm.runtime.Tensor):
        return output.numpy()
    if isinstance(output, (list, tuple)):
        return to_numpy(output[0])
    if hasattr(output, "numpy"):
        return output.numpy()
    raise TypeError(f"Unsupported VM output type: {type(output)!r}")


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

    model_path = download(RESNET18_URL, CACHE_DIR / "resnet18_Opset18.onnx")
    labels_path = download(LABELS_URL, CACHE_DIR / "imagenet_classes.txt")
    image_path = args.image_path
    if image_path is None:
        image_path = download(args.image_url, CACHE_DIR / pathlib.Path(args.image_url).name)

    labels = load_labels(labels_path)
    input_data = preprocess_image(image_path)

    print("Loading ONNX ResNet18")# 1- 导入模型
    onnx_model = onnx.load(model_path)
    # print('onnx_model',onnx_model) #太大
    input_name = first_model_input_name(onnx_model)

    print("Converting ONNX to Relax IRModule")# 2- 将onnx模型转为IRmodel
    mod = tvm_onnx.from_onnx(
        onnx_model,
        shape_dict={input_name: input_data.shape},
        dtype_dict={input_name: "float32"},
    )
    print('Relax IRModule',mod)

    print("Compiling with TVM for llvm")
    target = tvm.target.Target("llvm")
    print('Relax IRModule before',mod)
    mod = relax.get_pipeline("zero")(mod) # 优化IR，编译优化
    print('Relax IRModule after',mod)
    executable = tvm.compile(mod, target=target) # 真正的编译，将优化后IR转为目标平台可执行代码
    # 真正编译，包含代码生成、backend lowering、生成可执行
    print('executable',executable)

    print("Running inference")
    dev = tvm.cpu()
    vm = relax.VirtualMachine(executable, dev) # tvm虚拟机用来运行编译出来的模型
    output = to_numpy(vm["main"](tvm.runtime.tensor(input_data, dev))) # 执行推理
    '''
    tvm.runtime.tensor(input_data, dev) # 将numpy转为tvm runtime能接受的tensor
    vm["main"]() # 调用模型的主函数
    结果转为numpy
    '''
    probs = softmax(output) # 归一化，转为概率

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
