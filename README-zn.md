TVM 快速上手（基于官方中文教程验真）
====================================

教程有效性
----------
- 参考来源：tvm.hyper.ai 中文镜像《从源码安装》（版本 0.21.0，2025-11-03 更新），内容与官方英文站一致，仅为中文镜像，可作为有效安装指引。
- 注意版本：若你使用更老/更新的 TVM，请以仓库当前 `cmake/config.cmake` 选项为准，必要时对照官网英文文档：https://tvm.apache.org/docs/get_started/install.html

目标
----
1) 构建 TVM（以 CPU/LLVM 为例，可按需开启 CUDA/ROCm/Metal/Vulkan 等）
2) 验证构建
3) 跑一个最小 TE 示例 + 一个 Relay 示例

环境准备（与官方要求对齐）
---------------------------
- 系统：Linux，建议 Ubuntu 20.04+/22.04+
- 依赖（C++17）：`git build-essential cmake ninja-build python3-dev python3-venv libtinfo-dev zlib1g-dev libedit-dev libxml2-dev`
- LLVM：>=15 推荐（可用 `llvm-config --version` 确认），构建时开启 `USE_LLVM=ON`
- Python：>=3.8；推荐虚拟环境或 Conda（官方示例用 `conda create -n tvm-build-venv`）
- GPU 可选：CUDA 11.8+ 或 ROCm 等，对应开关 `USE_CUDA/USE_ROCM/...`

步骤 1：获取源码与子模块
-------------------------
```bash
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
# 若忘记 --recursive，可补执行
git submodule update --init --recursive
```

步骤 2：配置与构建（CPU/LLVM 示例）
---------------------------------
```bash
export TVM_HOME=$(pwd)
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip wheel

mkdir -p build
cp cmake/config.cmake build/
# 按需编辑 build/config.cmake：
# set(USE_LLVM ON)            # 已安装 LLVM 时开启
# set(CMAKE_BUILD_TYPE RelWithDebInfo)  # 与官方推荐一致
# set(USE_CUDA OFF) 等，根据需求切换

cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build . --parallel "$(nproc)"

# 暴露 Python 与动态库
cd "$TVM_HOME"
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
export LD_LIBRARY_PATH=$TVM_HOME/build:$LD_LIBRARY_PATH
```

步骤 3：验证构建
-----------------
```bash
# 3.1 基础导入与库路径
python - <<'PY'
import tvm
print("TVM version:", tvm.__version__)
print("TVM library path:", tvm._ffi.base._LIB)
print("Build config sample:")
for k, v in list(tvm.support.libinfo().items())[:6]:
	print(f"{k}: {v}")
PY

# 3.2 最小 TE 编译 + 执行（需 USE_LLVM=ON）
python - <<'PY'
import numpy as np
import tvm
from tvm import te

n = 1024
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
sch = te.create_schedule(C.op)
f = tvm.build(sch, [A, B, C], target="llvm")

a = tvm.nd.array(np.ones(n, dtype="float32"))
b = tvm.nd.array(np.ones(n, dtype="float32"))
c = tvm.nd.empty((n,), dtype="float32")
f(a, b, c)
print("Add correct:", np.allclose(c.numpy(), a.numpy() + b.numpy()))
PY
```

步骤 4：Relay 简单示例（图执行器）
-------------------------------
```bash
python - <<'PY'
import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_executor

target = "llvm"  # GPU 时可改为 "cuda" 等
dtype = "float32"

# 1) 定义 Relay 计算图：y = relu(x @ w + b)
x = relay.var("x", shape=(1, 4), dtype=dtype)
w = relay.const(np.random.randn(4, 4).astype(dtype))
b = relay.const(np.random.randn(4).astype(dtype))
y = relay.nn.relu(relay.nn.dense(x, w) + b)
func = relay.Function([x], y)

# 2) 编译
with tvm.transform.PassContext(opt_level=3):
	lib = relay.build(func, target=target)

# 3) 运行
dev = tvm.device(target, 0)
module = graph_executor.GraphModule(lib["default"](dev))
inp = np.random.randn(1, 4).astype(dtype)
module.set_input("x", inp)
module.run()
out = module.get_output(0).asnumpy()
print("Output shape:", out.shape)
print("Sample output:", out)
PY
```

更多参考
--------
- 官方中文镜像教程（已验真）：https://tvm.hyper.ai/docs/getting-started/installing-tvm/install-from-source
- 官方英文文档（保持最新）：https://tvm.apache.org/docs
- 容器化/可复现环境：见仓库 `docker/` 与 `conda/` 目录。
