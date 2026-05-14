# ResNet18 Relay 分类示例说明

这份文档对应同目录下的：

```text
resnet18_relay_classify.py
```

你之后主要使用 Relay，可以先把这个脚本当成一个最小完整模板：

```text
外部模型 -> Relay IRModule -> relay.build -> Graph Executor -> 推理结果
```

## 1. 运行方式

在 TVM 0.19.0 仓库根目录运行：

```bash
cd /Users/huzi/Documents/Code/tvm/apache-tvm-0.19.0
.venv-0.19/bin/python toys/resnet18_relay_classify.py
```

输出类似：

```text
Loading ONNX ResNet18
Converting ONNX to Relay graph
Compiling with relay.build for llvm
Running inference with graph executor
image: /Users/huzi/Documents/Code/tvm/apache-tvm-0.19.0/toys/cache/kitten.jpg
input: x (1, 3, 224, 224)
top classifications:
 1.  282 tiger cat                      0.7885
 2.  281 tabby                          0.1399
 3.  285 Egyptian cat                   0.0451
 4.  287 lynx                           0.0039
 5.  292 tiger                          0.0007
```

换自己的图片：

```bash
.venv-0.19/bin/python toys/resnet18_relay_classify.py --image-path /path/to/image.jpg
```

只看前 3 个分类：

```bash
.venv-0.19/bin/python toys/resnet18_relay_classify.py --topk 3
```

## 2. 脚本整体流程

脚本从 `main()` 开始，主流程是：

```python
model_path = fetch_file(...)
labels_path = fetch_file(...)
image_path = fetch_file(...)

labels = load_labels(labels_path)
input_data = preprocess_image(image_path)

onnx_model = onnx.load(model_path)
input_name = first_model_input_name(onnx_model)
shape_dict = {input_name: input_data.shape}

mod, params = relay.frontend.from_onnx(...)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target="llvm", params=params)

module = graph_executor.GraphModule(lib["default"](dev))
module.set_input(input_name, input_data)
module.run()
output = module.get_output(0).numpy()
```

压缩成一张流程图：

```text
ONNX ResNet18
  |
  v
relay.frontend.from_onnx()
  |
  v
Relay IRModule + params
  |
  v
relay.build()
  |
  v
GraphExecutorFactoryModule
  |
  v
graph_executor.GraphModule
  |
  v
set_input() + run() + get_output()
  |
  v
分类结果
```

## 3. 图片预处理在做什么

ResNet18 不能直接接收 JPG 文件。

模型需要的输入是：

```text
(1, 3, 224, 224)
```

含义是：

```text
1    batch size，一次输入 1 张图片
3    RGB 三个通道
224  高度
224  宽度
```

`preprocess_image()` 做了这些事：

1. 打开图片并转成 RGB
2. 短边缩放到 256
3. 中心裁剪成 224 x 224
4. 像素值从 `0 ~ 255` 变成 `0 ~ 1`
5. 使用 ImageNet 的 mean/std 做归一化
6. 从 HWC 格式转成 CHW 格式
7. 加上 batch 维度，变成 NCHW

关键代码：

```python
data = np.asarray(image).astype("float32") / 255.0
data = (data - IMAGE_NET_MEAN) / IMAGE_NET_STD
data = np.transpose(data, (2, 0, 1))
return np.expand_dims(data, axis=0).astype("float32")
```

这一步可以理解为：

```text
人能看的图片 -> 模型能吃的张量
```

## 4. Relay 导入模型

核心代码：

```python
mod, params = relay.frontend.from_onnx(
    onnx_model,
    shape=shape_dict,
    freeze_params=True,
)
```

这里做的是：

```text
ONNX 模型 -> Relay IRModule
```

`mod` 是 Relay 的模型计算图。

`params` 是模型参数，也就是卷积权重、BatchNorm 参数、全连接层权重等。

`shape_dict` 告诉 TVM 输入张量的名字和形状：

```python
shape_dict = {"x": (1, 3, 224, 224)}
```

如果 shape 写错，常见后果是：

1. 转换时报 shape 不匹配
2. 编译时报类型推导错误
3. 运行时报输入维度错误
4. 结果能跑但分类很差

## 5. Relay 编译模型

核心代码：

```python
target = "llvm"

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
```

这里做的是：

```text
Relay IRModule -> 可执行 runtime module
```

`target = "llvm"` 表示编译到本机 CPU。

`PassContext(opt_level=3)` 表示启用较常见的 Relay 优化级别。

`relay.build(...)` 是 Relay 编译的核心入口。

你同事截图里的写法：

```python
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)
```

和这个示例里的写法本质一样。



### 疑问解答❓

- 上下文（PassContext）
  - with tvm.transform.PassContext(opt_level=3) 设置 Relay 的优化等级，影响后续的优化/下发/代码生成流程（会改变 relay.build 执行的编译通道和传递的编译选项）。
- relay.build 做什么
  - relay -> TIR -> 机器码
  - 编译优化在build的时候完成
  - 编译+打包，返回结果为 factory module
  - factory 的创建和注册在 build 中完成，但实例化是在之后发生。GraphExecutorFactory 的 C++ 实现在



## 6. Graph Executor 执行模型

Relay 编译后，最常见的运行方式是 Graph Executor。

核心代码：

```python
dev = tvm.cpu(0)
module = graph_executor.GraphModule(lib["default"](dev))
module.set_input(input_name, input_data)
module.run()
output = module.get_output(0).numpy()
```

逐句解释：

```python
dev = tvm.cpu(0)
```

选择 CPU 设备。

```python
module = graph_executor.GraphModule(lib["default"](dev))
```

从编译结果里创建运行时模块。

```python
module.set_input(input_name, input_data)
```

设置模型输入。

```python
module.run()
```

执行推理。

```python
output = module.get_output(0).numpy()
```

取出第 0 个输出，并转成 NumPy。

## 7. Relay 和 Relax 的区别

你之前在新版 `apache-tvm` 目录里跑的是 Relax 版。

现在这个 `apache-tvm-0.19.0` 示例是 Relay 版。

两者都能表达神经网络计算图，但定位和 API 不一样。

## 7.1 一句话区别

Relay 是 TVM 里更传统、更成熟的一代深度学习图 IR。

Relax 是 TVM 新一代 IR，设计上更强调动态 shape、统一 TensorIR、现代深度学习模型和更灵活的编译流程。

你之后主要使用 TVM 0.19.0 和已有教程、同事代码时，Relay 会更直接。

## 7.2 流程对比

Relay 版(旧)：

```python
mod, params = relay.frontend.from_onnx(onnx_model, shape=shape_dict)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target="llvm", params=params)

module = graph_executor.GraphModule(lib["default"](dev))
module.set_input(input_name, input_data)
module.run()
output = module.get_output(0).numpy()
```

Relax 版（新）：

```python
mod = tvm_onnx.from_onnx(
    onnx_model,
    shape_dict={input_name: input_data.shape},
    dtype_dict={input_name: "float32"},
)

mod = relax.get_pipeline("zero")(mod)
executable = tvm.compile(mod, target="llvm")

vm = relax.VirtualMachine(executable, dev)
output = vm["main"](tvm.runtime.tensor(input_data, dev))
```

## 7.3 对比表

| 项目 | Relay | Relax |
|---|---|---|
| 所属阶段 | TVM 传统主力图 IR | TVM 新一代高层 IR |
| 常见版本 | TVM 0.19 及更早教程中很常见 | 新版 TVM 文档和开发中更常见 |
| 导入 ONNX | `relay.frontend.from_onnx` | `tvm.relax.frontend.onnx.from_onnx` |
| 编译入口 | `relay.build` | `tvm.compile` |
| 基础优化配置 | `PassContext(opt_level=3)` | `relax.get_pipeline(...)` |
| 常见运行器 | `graph_executor.GraphModule` | `relax.VirtualMachine` |
| 参数形式 | 常见为 `mod, params` 分开 | 常见为参数在 IRModule 中或作为 VM 参数 |
| 初学难度 | 更贴近老教程，资料多 | 概念更新，API 变化较多 |
| 适合你现在 | 是 | 暂时作为了解即可 |

## 7.4 编译产物也不同

Relay 的 `relay.build` 返回的通常是一个 factory module。

常见用法：

```python
lib = relay.build(mod, target="llvm", params=params)
module = graph_executor.GraphModule(lib["default"](dev))
```

Relax 的 `tvm.compile` 返回的是可交给 VM 使用的 executable。

常见用法：

```python
executable = tvm.compile(mod, target="llvm")
vm = relax.VirtualMachine(executable, dev)
```

所以不是简单地把 `relay.build` 替换成 `tvm.compile` 就完事。

前后的 runtime 也跟着变了。

## 8. Relay 常用模型导入方式

Relay 可以从很多框架导入模型。

常见入口包括：

```python
relay.frontend.from_onnx(...)
relay.frontend.from_mxnet(...)
relay.frontend.from_keras(...)
relay.frontend.from_tensorflow(...)
relay.frontend.from_tflite(...)
relay.frontend.from_pytorch(...)
relay.frontend.from_coreml(...)
```

大致模式都类似：

```python
mod, params = relay.frontend.from_xxx(model, shape_dict)
```

然后：

```python
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target="llvm", params=params)
```

再用 Graph Executor：

```python
module = graph_executor.GraphModule(lib["default"](dev))
module.set_input(input_name, input_data)
module.run()
output = module.get_output(0).numpy()
```

## 9. Relay 常用 target

`target` 决定编译到哪里运行。

本机 CPU：

```python
target = "llvm"
dev = tvm.cpu(0)
```

CUDA GPU：

```python
target = "cuda"
dev = tvm.cuda(0)
```

OpenCL：

```python
target = "opencl"
dev = tvm.opencl(0)
```

ARM CPU 交叉编译时常见：

```python
target = "llvm -mtriple=aarch64-linux-gnu"
```

你现在本机跑示例，用 `llvm` 就够了。

## 10. Relay 常用 PassContext

最常见写法：

```python
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
```

`opt_level` 可以粗略理解为优化强度：

```text
0  基本不优化，调试时可能有用
1  基础优化
2  更多图优化
3  常用默认选择，示例和教程里很常见
```

一般先用 `opt_level=3`。

如果遇到奇怪问题，可以降到 `opt_level=0` 或 `1` 做排查。

## 11. Relay 参数 params 是什么

很多 Relay frontend 会返回：

```python
mod, params = ...
```

`mod` 是计算图。

`params` 是模型权重。

比如 ResNet18 里的：

1. 卷积层权重
2. BatchNorm 的 gamma、beta、moving mean、moving variance
3. 全连接层权重和 bias

编译时传入：

```python
relay.build(mod, target=target, params=params)
```

这样 TVM 可以把权重绑定进编译结果。

如果忘记传 `params`，可能会导致：

1. 编译失败
2. 运行时还需要额外 `set_input` 很多权重
3. 输出结果不对

## 12. Graph Executor 常用方法

创建模块：

```python
module = graph_executor.GraphModule(lib["default"](dev))
```

设置一个输入：

```python
module.set_input("x", input_data)
```

设置多个输入：

```python
module.set_input("input_ids", input_ids)
module.set_input("attention_mask", attention_mask)
```

执行：

```python
module.run()
```

取第一个输出：

```python
out = module.get_output(0).numpy()
```

如果模型有多个输出：

```python
out0 = module.get_output(0).numpy()
out1 = module.get_output(1).numpy()
```

查看输出数量：

```python
num_outputs = module.get_num_outputs()
```

## 13. 查看 Relay IR

初学时很建议打印一下 Relay IR。

导入模型后可以加：

```python
print(mod)
```

也可以只看 main：

```python
print(mod["main"])
```

如果模型很大，ResNet18 会打印很多内容。

但你可以从里面看到类似：

```text
nn.conv2d
nn.batch_norm
nn.relu
nn.max_pool2d
nn.global_avg_pool2d
dense
```

这些就是 Relay 表示的神经网络算子。

## 14. 保存和加载 Relay 编译结果

Relay 编译后可以导出动态库：

```python
lib.export_library("resnet18_relay.so")
```

之后加载：

```python
loaded_lib = tvm.runtime.load_module("resnet18_relay.so")
module = graph_executor.GraphModule(loaded_lib["default"](dev))
```

这适合部署场景。

不过初学阶段先直接在 Python 里 `relay.build` 和运行，会更容易理解。

## 15. 调优提示是什么意思

运行时你可能看到：

```text
One or more operators have not been tuned. Please tune your model for better performance.
```

这是性能提示，不是错误。

它表示：

TVM 没有找到 AutoTVM/TopHub 的调优日志，所以会使用默认 schedule。

结果仍然是对的，只是性能不一定最好。

当前脚本里设置了：

```python
os.environ.setdefault("TOPHUB_LOCATION", "NONE")
```

这是为了避免 `relay.build` 自动尝试下载 TopHub 日志并写入 `~/.tvm/tophub`。

在这个本地沙箱环境里，写用户家目录会失败。

## 16. 这个脚本里两个兼容处理

### 16.1 复用另一个虚拟环境的 ONNX/Pillow

当前 `.venv-0.19` 里没有安装 `onnx` 和 `Pillow`。

脚本里有：

```python
SIBLING_SITE_PACKAGES = (
    REPO_ROOT.parent / "apache-tvm" / ".venv" / "lib" / "python3.13" / "site-packages"
)
if SIBLING_SITE_PACKAGES.exists():
    sys.path.append(str(SIBLING_SITE_PACKAGES))
```

这是为了复用新版 TVM 示例环境里已经安装好的依赖。

如果你之后在 `.venv-0.19` 里正式安装了依赖，这段也不会影响正常运行。

### 16.2 兼容新版 ONNX 的 `onnx.mapping`

TVM 0.19 的 Relay ONNX frontend 会导入：

```python
onnx.mapping
```

但新版 ONNX 已经没有这个模块了。

所以脚本里做了一个兼容映射：

```python
if "onnx.mapping" not in sys.modules and hasattr(onnx, "_mapping"):
    mapping_module = types.ModuleType("onnx.mapping")
    mapping_module.TENSOR_TYPE_TO_NP_TYPE = {
        key: value.np_dtype for key, value in onnx._mapping.TENSOR_TYPE_MAP.items()
    }
    sys.modules["onnx.mapping"] = mapping_module
```

这只是为了让旧版 TVM frontend 可以和新版 ONNX 包配合使用。

如果你以后使用旧版 ONNX，比如 ONNX 1.14 左右，可能不需要这段兼容代码。

## 17. 以后写 Relay 程序的常用模板

你可以把下面当成最小模板：

```python
import tvm
from tvm import relay
from tvm.contrib import graph_executor

target = "llvm"
dev = tvm.cpu(0)

# 1. 从外部框架导入模型
mod, params = relay.frontend.from_onnx(
    onnx_model,
    shape={"x": (1, 3, 224, 224)},
    freeze_params=True,
)

# 2. 编译
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# 3. 创建运行时
module = graph_executor.GraphModule(lib["default"](dev))

# 4. 设置输入并运行
module.set_input("x", input_data)
module.run()

# 5. 获取输出
output = module.get_output(0).numpy()
```

再简化成概念：

```text
from_xxx() 负责导入
relay.build() 负责编译
GraphModule 负责运行
```

## 18. 你现在应该优先记住的几个名字

```text
relay.frontend.from_onnx
relay.build
tvm.transform.PassContext
graph_executor.GraphModule
module.set_input
module.run
module.get_output
target = "llvm"
dev = tvm.cpu(0)
```

这些就是你之后用 Relay 跑大多数模型时最常碰到的 API。

