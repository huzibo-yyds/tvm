# TVM 项目说明与快速入门

本文档基于本机源码目录：

```bash
/Users/huzi/Documents/Code/tvm/apache-tvm
```

当前本地构建已经可用：

```text
TVM: 0.25.dev0
LLVM: enabled
Metal: enabled
CPU runtime: enabled
```

## 1. TVM 是什么

Apache TVM 是一个开源机器学习编译器框架。它的核心目标是把模型或张量程序编译成可以在不同设备上运行的高效部署模块。

你可以把 TVM 理解成一条编译流水线：

```text
模型 / 张量计算
  -> TVM IRModule
  -> 图级优化与算子级优化
  -> 面向目标硬件生成代码
  -> TVM runtime 执行
```

TVM 当前强调两个设计原则：

- Python-first：很多编译、变换、调度和优化逻辑都可以在 Python 里定制。
- Universal deployment：同一套编译基础设施可以面向 CPU、GPU、Metal、CUDA、Vulkan、移动端、边缘设备等运行环境。

在你的 Mac 上，目前已经构建的是 CPU/LLVM + Metal 版本，适合先学习 TVM 的核心概念、IR、编译流程和本地运行。

## 2. 本机环境怎么用

进入项目并激活虚拟环境：

```bash
cd /Users/huzi/Documents/Code/tvm/apache-tvm
source .venv/bin/activate
```

确认 TVM 可导入：

```bash
python -c "import tvm; print(tvm.__version__)"
```

确认后端能力：

```bash
python -c "import tvm; print('llvm', tvm.runtime.enabled('llvm')); print('metal', tvm.runtime.enabled('metal')); print('cpu', tvm.cpu(0).exist)"
```

重新编译：

```bash
cmake --build build --parallel 8
```

主要编译产物在：

```bash
build/lib/libtvm_compiler.dylib
build/lib/libtvm_runtime.dylib
build/lib/libtvm_ffi.dylib
```

本地构建配置在：

```bash
build/config.cmake
```

关键配置：

```cmake
set(USE_LLVM /opt/homebrew/opt/llvm/bin/llvm-config)
set(USE_METAL ON)
```

## 3. 一个最小可运行例子

保存为 `examples/quickstart_add_one.py`，或者直接在 Python 里运行：

```python
import numpy as np
import tvm
from tvm import te


A = te.placeholder((8,), name="A")
B = te.compute((8,), lambda i: A[i] + 1.0, name="B")

func = te.create_prim_func([B, A])
compiled = tvm.compile(func, target="llvm")

a_np = np.arange(8, dtype="float32")
out = tvm.runtime.tensor(np.zeros(8, dtype="float32"))

compiled(out, tvm.runtime.tensor(a_np))
print(out.numpy())
```

运行：

```bash
python examples/quickstart_add_one.py
```

期望输出：

```text
[1. 2. 3. 4. 5. 6. 7. 8.]
```

这个例子覆盖了 TVM 的最小闭环：

```text
TE 描述计算 -> 生成 PrimFunc/TIR -> tvm.compile 编译 -> runtime 执行
```

## 4. 快速入门需要掌握什么

### 4.1 先掌握整体流程

TVM 的典型使用流程是：

1. 构造或导入模型。
2. 得到 TVM 的 `IRModule`。
3. 对 `IRModule` 做优化变换。
4. 用 `tvm.compile` 面向某个 target 编译。
5. 在 TVM runtime 上执行。

官方快速入门里的神经网络流程大致是：

```python
import tvm
from tvm import relax

target = tvm.target.Target("llvm")
executable = tvm.compile(mod, target)
vm = relax.VirtualMachine(executable, tvm.cpu())
```

你初学时先盯住这几个词：

- `IRModule`：TVM 编译的核心容器。
- `Relax`：偏图级、模型级的高层 IR。
- `TensorIR` / `TIR`：偏底层张量程序和循环结构的 IR。
- `Target`：编译目标，例如 `llvm`、`metal`、`cuda`。
- `Runtime`：执行编译后模块的运行时。

### 4.2 理解 TVM 的几层抽象

从高到低可以这样看：

```text
模型前端
  PyTorch / ONNX / TensorFlow / Relax frontend

图级 IR
  Relax

张量程序 IR
  TensorIR / TIR

代码生成
  LLVM / Metal / CUDA / Vulkan / C source / 其他后端

运行时
  TVM runtime / VM / PackedFunc / Tensor
```

入门时不用一次吃完。建议顺序是：

1. 先跑通几个 `tvm.compile(..., target="llvm")` 的小例子。
2. 再看 `IRModule` 和 `PrimFunc` 长什么样。
3. 再学 Relax 模型导入和图级优化。
4. 最后学 TensorIR schedule、算子优化和后端代码生成。

### 4.3 必须会的 Python API

常用入口：

```python
import tvm
from tvm import te
from tvm import relax
```

常见对象：

- `tvm.IRModule`：编译单元。
- `te.placeholder`：声明输入张量。
- `te.compute`：声明张量计算。
- `te.create_prim_func`：从 TE 张量表达生成底层 PrimFunc。
- `tvm.compile`：把 IR 编译成可执行模块。
- `tvm.runtime.tensor`：创建 TVM runtime 张量。
- `tvm.cpu()` / `tvm.metal()`：选择运行设备。

### 4.4 你需要知道的新旧 API 差异

这个仓库是较新的主分支，版本是 `0.25.dev0`。一些旧教程里的 API 可能已经变化。

尤其注意：

```python
te.create_schedule(...)
```

在当前分支里不再是推荐入门路径。建议优先使用：

```python
func = te.create_prim_func([...])
compiled = tvm.compile(func, target="llvm")
```

如果你看老教程，遇到 `tvm.build`、`te.create_schedule`、`relay`，要意识到它们可能对应旧版 TVM 的风格。当前文档更强调 `Relax`、`TensorIR`、`tvm.compile`。

## 5. 推荐学习路线

### 第 1 天：跑通环境和最小例子

目标：

- 知道如何进入 `.venv`。
- 知道如何导入 `tvm`。
- 能跑通 `target="llvm"` 的最小计算。
- 能看懂 `tvm.runtime.tensor` 输入输出。

建议看：

```bash
docs/get_started/overview.rst
docs/get_started/tutorials/quick_start.py
```

### 第 2 天：理解 IRModule 和 TIR

目标：

- 知道 `IRModule` 是 TVM 编译的核心容器。
- 知道 `PrimFunc` 表示底层张量程序。
- 能用 `.show()` 或 `.script()` 看生成的 IR。

建议看：

```bash
docs/get_started/tutorials/ir_module.py
docs/deep_dive/tensor_ir/tutorials/tir_creation.py
```

### 第 3 天：理解 Relax 模型流程

目标：

- 知道 Relax 是图级 IR。
- 能理解模型导入、pipeline、VM 执行的大概流程。
- 能跑官方 quick start 里的 MLP 示例。

建议看：

```bash
docs/get_started/tutorials/quick_start.py
docs/deep_dive/relax/learning.rst
```

### 第 4 天以后：深入优化

目标：

- 学 schedule。
- 学 pass。
- 学 target 和 codegen。
- 学模型导入、导出、部署。

建议看：

```bash
docs/how_to/tutorials/import_model.py
docs/how_to/tutorials/export_and_load_executable.py
docs/how_to/tutorials/customize_opt.py
docs/arch/pass_infra.rst
docs/arch/codegen.rst
```

## 6. 当前本机构建的注意事项

这次为了快速在 Mac 上跑通，没有初始化 CUDA/CUTLASS/FlashAttention 这些可选子模块。它们主要面向 NVIDIA GPU 或特定高性能路径，不影响你现在学习 CPU/LLVM/Metal 的主流程。

当前适合做：

- CPU 后端学习。
- LLVM codegen 学习。
- Metal runtime/codegen 初步探索。
- Relax、TensorIR、IRModule、runtime 学习。

当前不适合直接做：

- CUDA 编译运行。
- CUTLASS 相关优化。
- FlashAttention 相关路径。
- NVIDIA GPU 专项性能调优。

如果以后要研究 CUDA，需要补齐相关子模块和 CUDA 工具链。

## 7. 常用排错命令

查看当前 TVM 从哪里导入：

```bash
python -c "import tvm; print(tvm.__file__)"
```

查看加载的是哪个动态库：

```bash
python -c "import tvm; print(tvm.base._LIB)"
```

查看构建选项：

```bash
python -c "import tvm; print('\\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))"
```

查看设备是否存在：

```bash
python -c "import tvm; print('cpu', tvm.cpu().exist); print('metal', tvm.metal().exist)"
```

重新配置 CMake：

```bash
cmake -S . -B build -G Ninja
```

重新编译：

```bash
cmake --build build --parallel 8
```

## 8. 这个仓库里值得先看的目录

```text
python/tvm
```

TVM Python API 主体。

```text
src
```

TVM C++ 编译器和 runtime 源码。

```text
include/tvm
```

C++ 头文件和核心 API 定义。

```text
docs/get_started
```

入门文档。

```text
docs/deep_dive
```

深入解释 Relax、TensorIR 等核心设计。

```text
tests/python
```

非常好的学习材料。很多当前 API 的真实用法都能在测试里找到。

## 9. 一句话学习建议

不要从“完整理解编译器”开始。先从三个问题入手：

1. 我如何用 TVM 表示一个计算？
2. TVM 把它变成了什么 IR？
3. TVM 如何把这个 IR 编译并在某个 target 上运行？

只要这三步清楚了，后面的图优化、schedule、codegen、runtime 都会自然接上。

## 10.git操作

以后同步官方 TVM 到你的 fork，用这个流程：

```
cd /Users/huzi/Documents/Code/tvm/apache-tvm 
git fetch upstream 
git checkout main 
git merge upstream/main 
git push origin main
```

