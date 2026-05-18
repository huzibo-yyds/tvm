# ViT-Tiny 自定义 LayerNorm Pass 实验报告

## 目标

本实验实现了一个真正参与 TVM 编译链路的 Relay 自定义算子 `custom.layer_norm`，并编写 Relay pass，将 ViT-Tiny ONNX 模型中的 LayerNorm 子图替换成该自定义算子。

实验同时保留 baseline 路径和 custom 路径：

- baseline：原始 ONNX 转 Relay 后直接 `relay.build` 并 runtime 执行。
- custom：ONNX 转 Relay 后先运行 `ReplaceLayerNormWithCustom`，再 `relay.build` 并 runtime 执行。

最后比较两边 graph executor 输出是否一致。

## 关键名词解释

### Relay

Relay 是 TVM 的高层神经网络 IR。ONNX、PyTorch、TensorFlow 等前端模型通常会先转换成 Relay。

在本实验中：

```text
ViT-Tiny ONNX
  -> relay.frontend.from_onnx
  -> Relay IR
```

自定义 pass 修改的就是 Relay IR。例如把：

```text
mean -> subtract -> power -> mean -> sqrt -> divide -> multiply -> add
```

替换为：

```text
custom.layer_norm
```

### Relay op

Relay op 是 Relay 图中的算子节点，例如：

```text
add
multiply
nn.layer_norm
custom.layer_norm
```

注册一个新的 Relay op，至少要告诉 TVM：

- 这个 op 的名字是什么。
- 它有哪些 attrs。
- 它的输入输出类型关系是什么。
- 它如何生成底层 compute。
- 它使用什么 schedule。

本实验新增的 Relay op 是：

```text
custom.layer_norm
```

### Type relation

Type relation 用于描述算子的输入输出类型约束。

例如 `custom.layer_norm(data, gamma, beta)` 中：

- 输出 shape/dtype 应该和 `data` 一样。
- `gamma` 和 `beta` 应该是一维张量。
- `gamma`/`beta` 的长度应该等于归一化轴的长度。

如果 type relation 不正确，Relay 的类型推导可能失败，或者后续 build 时出现 shape/dtype 错误。

### TE compute

TE 是 Tensor Expression，是 TVM 中描述张量计算公式的一层抽象。

TE compute 只描述“怎么算”，不描述“怎么调度”。例如 LayerNorm 的 compute 大致表达：

```text
mean = sum(data) / N
var = sum((data - mean)^2) / N
out = (data - mean) / sqrt(var + epsilon) * gamma + beta
```

在本实验当前版本中，`custom.layer_norm` 的 `FTVMCompute` 已经不再调用 TOPI
LayerNorm，而是在 `src/relay/op/custom/layer_norm.cc` 中直接用 TE 手写计算。

手写 compute 被拆成几步：

```text
custom_layer_norm_mean_sum = sum(data)
custom_layer_norm_mean     = mean_sum / N
custom_layer_norm_var_sum  = sum((data - mean)^2)
custom_layer_norm_var      = var_sum / N
custom_layer_norm          = (data - mean) * rsqrt(var + epsilon) * gamma + beta
```

这样做的意义是：`custom.layer_norm` 不只是一个新的 Relay op 名字，而是真的拥有一份本地实现的 TE compute。

### TOPI

TOPI 是 TVM Operator Inventory，可以理解为 TVM 内置的一批常用算子 compute 模板。

例如：

```text
topi::nn::layer_norm
topi::nn::conv2d
topi::add
topi::multiply
```

本实验早期版本曾经复用 `topi::nn::layer_norm(...)` 来快速打通链路。
当前版本已经改成手写 TE compute，因此 TOPI 在这里主要作为概念参照：

- TOPI 说明 TVM 里常用算子通常如何组织 compute 模板。
- 当前 `custom.layer_norm` 没有调用 TOPI LayerNorm。
- schedule 也已经在 Python 侧显式注册为一个朴素手写 schedule。

### Schedule

Schedule 描述“怎么执行”一个 TE compute。

同一个 compute 可以有不同 schedule，例如：

- 是否 inline 中间计算。
- reduction 如何展开。
- loop 如何 split/reorder/vectorize。
- 是否并行化。

在 Relay build 中，只有 compute 不够，还必须有 schedule。否则 TVM 知道这个算子怎么算，但不知道如何 lower 成可执行循环。

本实验当前版本中：

```python
@generic_func
def schedule_custom_layer_norm(attrs, outs, target):
    with target:
        return te.create_schedule([out.op for out in outs])

reg.register_schedule("custom.layer_norm", schedule_custom_layer_norm)
```

这个 schedule 只负责创建可 lower 的 TE schedule，没有做性能优化。
如果要继续优化 CPU 性能，可以在这里添加 `split`、`parallel`、`vectorize`、`compute_at` 等调度。

### TIR

TIR 是 Tensor IR，是 TVM 更底层的 IR，用来表达循环、buffer、load/store 等接近代码生成的结构。

Relay 到可执行代码的大致过程是：

```text
Relay op
  -> TE compute
  -> schedule
  -> TIR
  -> target code
```

所以 TIR 可以理解为“已经从高层神经网络图，降到循环程序”的阶段。

### Lowering

Lowering 指从高层表示逐步降到低层表示的过程。

在本实验里常说的 lowering 主要是：

```text
Relay
  -> TE compute/schedule
  -> TIR
```

如果 `custom.layer_norm` 只有 Relay op 名字，但没有 compute/schedule，那么 lowering 就无法继续。

### Codegen

Codegen 是代码生成阶段，把 TIR 继续生成目标平台代码。

本实验目标是本机 CPU：

```text
target="llvm"
```

因此 codegen 最终走 LLVM 路径，生成可以在 macOS CPU 上执行的代码。

### Graph executor

Graph executor 是 TVM 的一种 runtime。`relay.build` 会生成：

- graph JSON：描述运行时图结构。
- compiled library：编译后的算子代码。
- params：模型参数。

Python 中通过：

```python
graph_executor.GraphModule(lib["default"](dev))
```

创建 runtime module，然后设置输入、执行、读取输出。

本实验 baseline 和 custom 两条路径最终都通过 graph executor 执行，所以可以直接比较两边 logits。

## 修改内容

### TVM core 自定义算子

新增文件：

```text
src/relay/op/custom/layer_norm.cc
```

该文件注册了新的 Relay op：

```text
custom.layer_norm(data, gamma, beta, axis=-1, epsilon=1e-5, center=True, scale=True)
```

它完成了几件事：

- 注册 `custom.layer_norm` 这个 Relay op 名字。
- 复用 `LayerNormAttrs` 保存 `axis`、`epsilon`、`center`、`scale`。
- 注册 type relation，输出 shape/dtype 与输入 `data` 一致，`gamma`/`beta` shape 为归一化轴长度。
- 注册 `FTVMCompute`，在本文件中手写 TE compute，不再调用 TOPI LayerNorm。
- 设置 `TOpPattern = kOpaque`，避免普通 fusion 把它当作简单 elementwise 处理。

因为这是 C++ 层 op 注册，所以修改后必须重新编译 TVM，生成新的 `build/libtvm.dylib`。

与之前实现的pass（/Users/huzi/Documents/Code/calcc-analysis_log/tvm_pass/ref/tvm_nbu_ext/nn.cc）的区别：

- layer_norm.cc (line 37) 是一个能被 TVM 默认 relay.build(target="llvm") 编译执行的算子
- nn.cc，只是注册了一个Relay IR节点，通常还需要外部 backend/BYOC/lowering 才能真正跑。

### Python Relay API

新增文件：

```text
python/tvm/relay/op/custom.py
python/tvm/relay/op/_custom.py
python/tvm/relay/op/_custom_make.py
```

并修改：

```text
python/tvm/relay/op/__init__.py
```

作用：

- 暴露 Python API `relay.op.custom.layer_norm(...)`。
- 初始化 FFI constructor `relay.op.custom._make.layer_norm`。
- 给 `custom.layer_norm` 注册 schedule，目前是 `te.create_schedule` 的朴素手写版本，目标先验证 `llvm` CPU。

这三个新增 Python 文件的职责不同。

📌 `python/tvm/relay/op/custom.py` 是面向用户和 pass 的 Python API。它提供：

```python
relay.op.custom.layer_norm(data, gamma, beta, axis=-1, epsilon=1e-5)
```

该函数本身不实现 LayerNorm 数学逻辑，而是调用 `_custom_make.layer_norm(...)` 去构造 Relay Call。也就是说，它是一个易用的 Python wrapper。

调用关系是：

```text
relay.op.custom.layer_norm(...)
  -> _custom_make.layer_norm(...)
  -> C++ MakeCustomLayerNorm(...)
  -> Relay Call(op="custom.layer_norm")
```

📌 `python/tvm/relay/op/_custom_make.py` 是 FFI constructor 初始化文件。它执行：

```python
tvm._ffi._init_api("relay.op.custom._make", __name__)
```

这会把 C++ 侧注册的：

```cpp
TVM_REGISTER_GLOBAL("relay.op.custom._make.layer_norm")
```

动态绑定成 Python 可调用函数：

```python
_custom_make.layer_norm(...)
```

所以 `_custom_make.py` 的作用是把 C++ Make 函数接到 Python 侧。没有它，`custom.py` 里无法调用 C++ 的 `MakeCustomLayerNorm`。

📌 `python/tvm/relay/op/_custom.py` 是 backend registration 文件。它执行：

```python
@generic_func
def schedule_custom_layer_norm(attrs, outs, target):
    with target:
        return te.create_schedule([out.op for out in outs])

reg.register_schedule("custom.layer_norm", schedule_custom_layer_norm)
```

C++ 里已经通过 `FTVMCompute` 告诉 TVM：

```text
custom.layer_norm 如何生成 TE compute
```

但 `relay.build` 还需要知道这些 TE compute 如何 schedule。`_custom.py` 就负责注册一个朴素手写 schedule，让 build 能继续 lower 到 TIR 和 codegen。

完整关系可以理解为：

```text
custom.py
  负责“Python 里怎么调用”

_custom_make.py
  负责“Python 怎么连到 C++ Make 函数”

_custom.py
  负责“relay.build 怎么拿到 schedule”
```

同时，`python/tvm/relay/op/__init__.py` 里导入：

```python
from . import custom
from . import _custom
```

这样在 `import tvm.relay.op` 时：

- `relay.op.custom.layer_norm` API 会可见。
- `custom.layer_norm` 的 schedule 注册会被执行。

## 自定义 Pass

新增文件：

```text
toys/vit_tiny/custom_layer_norm_pass.py
```

核心 pass：

```python
@relay.transform.function_pass(opt_level=1)
class ReplaceLayerNormWithCustom:
    ...
```

内部使用 `ExprMutator` 遍历 Relay 表达式，支持两类替换。

第一类是直接 Relay op：

```text
nn.layer_norm(data, gamma, beta)
```

替换为：

```text
custom.layer_norm(data, gamma, beta)
```

第二类是 ONNX ViT-Tiny 实际转换出来的分解形态：

```text
mean(x, axis=-1, keepdims=True)
subtract(x, mean)
power(subtract, 2)
mean(power, axis=-1, keepdims=True)
add(var, epsilon)
sqrt(...)
divide(subtract, sqrt)
multiply(..., gamma)
add(..., beta)
```

替换为：

```text
custom.layer_norm(x, gamma, beta, axis=-1, epsilon=...)
```

本实验使用的 ViT-Tiny ONNX 中没有直接生成 `nn.layer_norm`，而是生成上述分解子图。因此 pass 需要识别真实 ONNX frontend 输出的 pattern。

### DFPatternCallback 与 ExprMutator 对比

在 TVM Relay 中，写自定义 pass 常见有两种方式：

```text
DFPatternCallback
  更像“声明一个固定子图模板，然后命中后替换”

ExprMutator
  更像“手写遍历器，逐个访问表达式，并在访问过程中做判断和改写”
```

#### DFPatternCallback 的使用方式

`DFPatternCallback` 来自：

```python
from tvm.relay.dataflow_pattern import *
```

它的典型写法是：

```python
class RwCallback(DFPatternCallback):
    def __init__(self):
        super().__init__()
        x = wildcard()
        mean = is_op("mean")(x)
        sub = is_op("subtract")(x, mean)
        self.pattern = sub

    def callback(self, pre, post, node_map):
        x = node_map[x][0]
        return new_expr

new_func = rewrite(RwCallback(), func)
```

这种方式适合匹配结构稳定、形态明确的子图。比如之前的：

```text
/Users/huzi/Documents/Code/calcc-analysis_log/tvm_pass/calcc/transforms/fuse_sinusoidal_pos_embedding.py
```

它识别的是一个比较固定的 sinusoidal position embedding 子图：

```text
timestep
  -> reshape
  -> cast
  -> multiply
  -> sin / cos
  -> concatenate
  -> cast
  -> reshape
  -> tile
  -> concatenate with data
  -> reshape
```

这个图的拓扑关系很明确，所以可以用：

```python
reshape1 = is_op("reshape")(self.timestep)
cast1 = is_op("cast")(reshape1)
mul1 = is_op("multiply")(cast1, self.scale_factor) | is_op("multiply")(self.scale_factor, cast1)
sin = is_op("sin")(mul1)
cos = is_op("cos")(mul1)
...
self.pattern = final_reshape
```

命中后，`callback()` 通过 `node_map` 取出 `data`、`timestep`、`scale_factor`，再构造后端融合算子。

它的优点是：

- 子图模式写出来很直观，像在画 Relay dataflow。
- 对固定 pattern 的融合很简洁。
- 不需要手写完整遍历逻辑，`rewrite()` 会负责遍历和替换。

它的缺点是：

- 当同一个语义有很多等价写法时，pattern 会变得很复杂。
- 对 `add`、`multiply` 这种可交换算子，需要显式写左右两种形式。
- 对 dtype、axis、epsilon、shape、checked_type 等语义约束，通常仍然要在 `callback()` 里补充检查。

#### ExprMutator 的使用方式

`ExprMutator` 来自：

```python
from tvm.relay import ExprMutator
```

它的典型写法是：

```python
class MyRewriter(ExprMutator):
    def visit_call(self, call):
        new_call = super().visit_call(call)
        if is_target_pattern(new_call):
            return replacement_expr
        return new_call
```

`ExprMutator` 会递归访问 Relay 表达式。重写 `visit_call()` 后，可以在每个 Call 节点上执行自定义逻辑：

```text
先递归改写子节点
再判断当前节点是不是目标 pattern 的末端
如果是，返回新的 Relay 表达式
如果不是，原样返回
```

本实验的：

```text
toys/vit_tiny/custom_layer_norm_pass.py
```

选择 `ExprMutator`，主要是因为 LayerNorm 在 Relay 中可能有多种形态。

第一种是直接 op：

```text
nn.layer_norm(data, gamma, beta)
```

第二种是 ONNX frontend 分解后的基础算子组合：

```text
add(
  multiply(
    divide(
      subtract(data, mean(data)),
      sqrt(mean(power(subtract(data, mean(data)), 2)) + epsilon)
    ),
    gamma
  ),
  beta
)
```

还可能出现一些等价变体：

```text
add(beta, multiply(norm, gamma))
multiply(gamma, norm)
multiply(subtract(data, mean), divide(1, sqrt(var + epsilon)))
```

如果用 `DFPatternCallback`，这些变体需要写多个 pattern，或者写一个很大的组合 pattern。即使 pattern 命中，仍然需要在 `callback()` 中检查：

- 是否是最后一维归一化。
- 输入 dtype 是否是 `float32`。
- `mean` 和 `variance` 是否来自同一个 `data`。
- `epsilon` 是否是标量常量。
- `center=True`、`scale=True` 是否满足。

因此这里使用 `ExprMutator`，把识别逻辑拆成多个小函数：

```text
_rewrite_direct_layer_norm
_match_decomposed_layer_norm
_match_gamma_mul
_match_normalized
_match_divide_normalized
_match_mean
_match_inv_std
_match_sqrt_mean_power_variance
```

这样写更长，但每个函数只验证一个局部结构，调试时也更容易知道是哪个条件没有匹配成功。

#### 两个 pass 风格差异的本质

之前的 `FuseSinusoidalPosEmbedding` 更像是：

```text
固定子图融合 pass
```

它关心的是：

- 找到一段固定的 sinusoidal position embedding dataflow。
- 把它替换成 NBU 后端融合算子。
- 在替换前后插入 `mat2hwptn`、`hwptn2mat` 这类后端布局转换。

本实验的 `ReplaceLayerNormWithCustom` 更像是：

```text
语义识别 + 算子重组 pass
```

它关心的是：

- 不管 LayerNorm 是直接 op 还是 ONNX 分解图，都识别出同一个数学语义。
- 提取 `data`、`gamma`、`beta`、`axis`、`epsilon`。
- 重组为 `custom.layer_norm`，让后续 `relay.build` 走自定义算子的 compute、schedule、lowering 和 codegen。

所以二者风格差异大，不是 Python 版本问题，也不是哪种写法一定更高级，而是任务目标不同：

```text
固定、规则、拓扑稳定的子图
  优先考虑 DFPatternCallback

有多种等价形态、需要大量语义检查和分支判断的改写
  优先考虑 ExprMutator
```

## Runtime 对比脚本

新增文件：

```text
toys/vit_tiny/vit_tiny_layernorm_compare.py
```

该脚本完成完整流程：

1. 下载/cache ViT-Tiny ONNX、preprocessor config、ImageNet labels 和测试图片。
2. 使用 `relay.frontend.from_onnx` 转换为 Relay。
3. baseline 路径直接 `relay.build`。
4. custom 路径先运行 `ReplaceLayerNormWithCustom`，再 `relay.build`。
5. 两边都用 `graph_executor.GraphModule` runtime 执行。
6. 比较 top-k、`max_abs_diff`、`max_rel_diff`、`np.allclose`。

模型来源：

```text
https://huggingface.co/onnx-community/vit-tiny-patch16-224-ONNX
```

缓存目录：

```text
toys/vit_tiny/cache/
```

缓存文件通过 `.gitignore` 忽略，不作为源码提交。

## 编译方式

因为新增了 C++ op 注册，必须重新编译 TVM：

```bash
cd /Users/huzi/Documents/Code/tvm/apache-tvm-0.19.0/build
/Users/huzi/miniconda3/envs/tvm-0.19/bin/cmake .. -G Ninja -DCMAKE_MAKE_PROGRAM=/Users/huzi/miniconda3/envs/tvm-0.19/bin/ninja
/Users/huzi/miniconda3/envs/tvm-0.19/bin/ninja -j8
```

### 为什么必须编译

这次新增的 `custom.layer_norm` 不是纯 Python pass，而是在 C++ 层通过 `RELAY_REGISTER_OP` 注册进 TVM 全局 op registry：

```text
src/relay/op/custom/layer_norm.cc
```

因此只有重新生成 `libtvm.dylib` 后，新的 C++ 注册逻辑才会在 Python `import tvm` 时生效。否则 Python 里调用：

```python
relay.op.custom.layer_norm(...)
```

会找不到底层 FFI constructor 或 build 时找不到 op 的 compute/strategy。

### 使用哪个环境

本机当前可用环境是：

```text
/Users/huzi/miniconda3/envs/tvm-0.19
```

建议始终显式使用这个环境里的工具，避免系统 PATH 中找不到 `cmake` 或误用其它版本：

```bash
/Users/huzi/miniconda3/envs/tvm-0.19/bin/python
/Users/huzi/miniconda3/envs/tvm-0.19/bin/cmake
/Users/huzi/miniconda3/envs/tvm-0.19/bin/ninja
```

本次实际编译时，裸 `cmake` 命令不可用，所以使用了 conda 环境里的 `cmake`。

### 首次配置或 CMake cache 异常

如果 `build/` 目录已经存在，但 CMake cache 里记录了失效的 ninja 路径，例如：

```text
.venv-0.19/bin/ninja: no such file or directory
```

可以重新指定当前可用的 ninja：

```bash
cd /Users/huzi/Documents/Code/tvm/apache-tvm-0.19.0/build
/Users/huzi/miniconda3/envs/tvm-0.19/bin/cmake .. \
  -G Ninja \
  -DCMAKE_MAKE_PROGRAM=/Users/huzi/miniconda3/envs/tvm-0.19/bin/ninja
```

这一步只重新生成 build files，不等于真正编译源码。

### 真正编译

配置完成后执行：

```bash
cd /Users/huzi/Documents/Code/tvm/apache-tvm-0.19.0/build
/Users/huzi/miniconda3/envs/tvm-0.19/bin/ninja -j8
```

如果机器核心数较多，也可以用：

```bash
/Users/huzi/miniconda3/envs/tvm-0.19/bin/ninja -j$(sysctl -n hw.ncpu)
```

编译成功后会重新链接：

```text
build/libtvm.dylib
build/libtvm_runtime.dylib
```

本实验中，最终看到类似输出：

```text
Linking CXX shared library libtvm_runtime.dylib
Linking CXX shared library libtvm.dylib
```

表示 C++ op 注册已经进入新的动态库。

### 修改后的增量编译

如果后续只修改：

```text
src/relay/op/custom/layer_norm.cc
```

通常不需要重新跑 CMake，直接运行：

```bash
cd /Users/huzi/Documents/Code/tvm/apache-tvm-0.19.0/build
/Users/huzi/miniconda3/envs/tvm-0.19/bin/ninja -j8
```

如果新增了新的 `.cc` 文件，因为 TVM 的 CMake 使用 glob 收集源码，保险做法是先跑一次 CMake，再跑 ninja：

```bash
cd /Users/huzi/Documents/Code/tvm/apache-tvm-0.19.0/build
/Users/huzi/miniconda3/envs/tvm-0.19/bin/cmake .. \
  -G Ninja \
  -DCMAKE_MAKE_PROGRAM=/Users/huzi/miniconda3/envs/tvm-0.19/bin/ninja
/Users/huzi/miniconda3/envs/tvm-0.19/bin/ninja -j8
```

如果只修改 Python 文件，例如：

```text
toys/vit_tiny/custom_layer_norm_pass.py
python/tvm/relay/op/custom.py
```

一般不需要重新编译 C++，重新启动 Python 进程即可。

### 确认 Python 加载的是源码树 TVM

不需要重新安装 Python 包。当前环境中的 TVM Python package 来自源码目录：

```text
/Users/huzi/Documents/Code/tvm/apache-tvm-0.19.0/python/tvm
```

动态库来自：

```text
/Users/huzi/Documents/Code/tvm/apache-tvm-0.19.0/build/libtvm.dylib
```

因此重编 `libtvm.dylib` 后，重新启动 Python 进程即可加载新的 op 注册。

可以用下面命令确认：

```bash
cd /Users/huzi/Documents/Code/tvm/apache-tvm-0.19.0
/Users/huzi/miniconda3/envs/tvm-0.19/bin/python - <<'PY'
import tvm
print("tvm file:", tvm.__file__)
print("lib path:", tvm._ffi.base._LIB._name)
PY
```

期望输出路径类似：

```text
tvm file: /Users/huzi/Documents/Code/tvm/apache-tvm-0.19.0/python/tvm/__init__.py
lib path: /Users/huzi/Documents/Code/tvm/apache-tvm-0.19.0/build/libtvm.dylib
```

### 编译后快速验证 op 是否生效

编译成功后先运行小测试：

```bash
cd /Users/huzi/Documents/Code/tvm/apache-tvm-0.19.0
/Users/huzi/miniconda3/envs/tvm-0.19/bin/python toys/vit_tiny/test_custom_layer_norm.py
```

如果看到：

```text
custom.layer_norm single-op check: PASS
direct nn.layer_norm pass check: PASS
```

说明：

- Python 能找到 `relay.op.custom.layer_norm`。
- C++ op/type relation/compute 注册已生效。
- `relay.build(..., target="llvm")` 可以把该 op lower 到可执行代码。

然后再运行完整 ViT-Tiny 对比：

```bash
cd /Users/huzi/Documents/Code/tvm/apache-tvm-0.19.0
/Users/huzi/miniconda3/envs/tvm-0.19/bin/python toys/vit_tiny/vit_tiny_layernorm_compare.py
```

## 运行方式

小型验证：

```bash
cd /Users/huzi/Documents/Code/tvm/apache-tvm-0.19.0
/Users/huzi/miniconda3/envs/tvm-0.19/bin/python toys/vit_tiny/test_custom_layer_norm.py
```

完整 ViT-Tiny 对比：

```bash
cd /Users/huzi/Documents/Code/tvm/apache-tvm-0.19.0
/Users/huzi/miniconda3/envs/tvm-0.19/bin/python toys/vit_tiny/vit_tiny_layernorm_compare.py
```

如果已经有本地 ONNX，也可以指定：

```bash
/Users/huzi/miniconda3/envs/tvm-0.19/bin/python toys/vit_tiny/vit_tiny_layernorm_compare.py \
  --model-path /path/to/model.onnx \
  --preprocessor-path /path/to/preprocessor_config.json
```

## 验证结果

小型验证结果：

```text
custom.layer_norm single-op check: PASS
direct nn.layer_norm pass check: PASS
```

完整 ViT-Tiny runtime 结果：

```text
custom.layer_norm replacements: 25
custom.layer_norm calls in custom IR: 25
```

baseline top-5：

```text
1.  282 tiger cat
2.  281 tabby
3.  285 Egyptian cat
4.  287 lynx
5.  761 remote control
```

custom top-5：

```text
1.  282 tiger cat
2.  281 tabby
3.  285 Egyptian cat
4.  287 lynx
5.  761 remote control
```

数值对比：

```text
max_abs_diff: 6.8247318e-06
max_rel_diff: 0.003125615
allclose(rtol=0.0001, atol=0.0001): True
```

这说明自定义 pass 替换后的 `custom.layer_norm` 路径能够完成 Relay build、TIR lowering、LLVM codegen 和 graph executor runtime 执行，并且输出与 baseline 保持一致。

## 当前限制

当前实现是实验版，目标是打通完整链路，而不是做高性能 LayerNorm kernel。

目前限制：

- 主要验证 `target="llvm"` CPU。
- pass 只替换 ViT-Tiny 常见的 float32、最后一维归一化、带 gamma/beta 的 LayerNorm。
- `custom.layer_norm` 的 TE compute 已经手写；schedule 也已手写为朴素版本，但没有做性能优化。
- ONNX pattern matcher 针对当前 ViT-Tiny ONNX frontend 结果编写，若模型导出图结构不同，可能需要扩展 matcher。

## 编译链路理解

本实验中的完整路径是：

```text
ONNX model
  -> relay.frontend.from_onnx
  -> Relay IR
  -> ReplaceLayerNormWithCustom pass
  -> Relay IR with custom.layer_norm
  -> relay.build
  -> FTVMCompute: 手写 TE layer_norm compute
  -> TE compute
  -> schedule
  -> TIR
  -> LLVM codegen
  -> graph executor runtime
```

其中 pass 只负责改 Relay IR；真正让 `custom.layer_norm` 可以被 build 和 runtime 执行的是 C++ op 注册、type relation、compute 和 schedule 绑定。
