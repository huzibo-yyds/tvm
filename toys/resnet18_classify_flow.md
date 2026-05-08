# ResNet18 分类示例执行流程说明

这份说明对应同目录下的 `resnet18_classify.py`。

如果你刚接触 TVM，可以先抓住一句话：

> 这个脚本做的事情是：准备模型和图片，把 ONNX 格式的 ResNet18 模型转换成 TVM 能理解的 IRModule，然后让 TVM 编译它，最后用编译后的模型对图片做分类。

## 一、整体流程

脚本的主流程在 `main()` 函数里，大概分成 6 步：

1. 下载或读取 ResNet18 模型
2. 下载或读取 ImageNet 标签
3. 下载或读取待分类图片
4. 把图片预处理成模型需要的输入格式
5. 把 ONNX 模型导入 TVM，并编译到 CPU
6. 用 TVM runtime 执行推理，打印 top-k 分类结果

对应代码里的核心顺序是：

```python
model_path = download(...)
labels_path = download(...)
image_path = download(...)

labels = load_labels(labels_path)
input_data = preprocess_image(image_path)

onnx_model = onnx.load(model_path)
input_name = first_model_input_name(onnx_model)

mod = tvm_onnx.from_onnx(...)
mod = relax.get_pipeline("zero")(mod)
executable = tvm.compile(mod, target=target)

vm = relax.VirtualMachine(executable, dev)
output = to_numpy(vm["main"](...))
probs = softmax(output)
```

## 二、下载文件：模型、图片、标签

脚本会把下载的文件放到：

```text
toys/cache/
```

主要有三个文件：

```text
resnet18_Opset18.onnx
imagenet_classes.txt
kitten.jpg
```

`download()` 函数做了一件很朴素的事：

```python
if path.exists() and path.stat().st_size > 0:
    return path
```

意思是：

如果本地已经有这个文件，就直接复用；如果没有，才联网下载。

所以第一次运行会慢一点，后面再运行就不需要重新下载。

## 三、图片为什么要预处理

ResNet18 不能直接吃一张普通 JPG 图片。

模型希望输入是一个形状为：

```text
(1, 3, 224, 224)
```

的 `float32` 数组。

这个形状的含义是：

```text
1    batch size，一次输入 1 张图片
3    RGB 三个颜色通道
224  图片高度
224  图片宽度
```

普通图片通常是：

```text
(height, width, 3)
```

也就是 HWC 格式。

而 ResNet18 通常需要：

```text
(1, 3, height, width)
```

也就是 NCHW 格式。

`preprocess_image()` 主要做了这些事：

1. 打开图片，并转换成 RGB
2. 把图片较短边缩放到 256
3. 从中心裁剪出 224 x 224
4. 把像素值从 `0 ~ 255` 变成 `0 ~ 1`
5. 按 ImageNet 的 mean/std 做归一化
6. 把维度从 HWC 转成 CHW
7. 在最前面加一个 batch 维度，变成 NCHW

关键代码是：

```python
data = np.asarray(image).astype("float32") / 255.0
data = (data - IMAGE_NET_MEAN) / IMAGE_NET_STD
data = np.transpose(data, (2, 0, 1))
return np.expand_dims(data, axis=0).astype("float32")
```

可以把它理解成：

> 把“人能看的图片”整理成“模型能吃的张量”。

## 四、ONNX 是什么

脚本里用的是：

```python
onnx_model = onnx.load(model_path)
```

ONNX 可以简单理解为一种通用模型文件格式。

这个例子里，我们没有用 PyTorch 或 MXNet 直接跑模型，而是下载了一个已经导出的 ResNet18 ONNX 文件。

这样 TVM 可以通过 ONNX frontend 把模型读进来：

```python
mod = tvm_onnx.from_onnx(
    onnx_model,
    shape_dict={input_name: input_data.shape},
    dtype_dict={input_name: "float32"},
)
```

这里有两个重要参数：

```python
shape_dict={input_name: input_data.shape}
```

告诉 TVM：模型输入的形状是 `(1, 3, 224, 224)`。

```python
dtype_dict={input_name: "float32"}
```

告诉 TVM：模型输入的数据类型是 `float32`。

转换之后得到的 `mod` 是一个 TVM IRModule。

你可以把 IRModule 暂时理解为：

> TVM 内部表示的模型计算图。

## 五、TVM 编译在做什么

这几行是 TVM 的核心：

```python
target = tvm.target.Target("llvm")
mod = relax.get_pipeline("zero")(mod)
executable = tvm.compile(mod, target=target)
```

逐句解释：

```python
target = tvm.target.Target("llvm")
```

表示目标平台是 CPU。

这里的 `llvm` 可以先理解为：

> 让 TVM 生成能在本机 CPU 上运行的代码。

```python
mod = relax.get_pipeline("zero")(mod)
```

表示对模型做一轮基础处理。

这里用了 `"zero"` 管线，它比较轻量，不做耗时的自动调优，适合初学者快速跑通流程。

```python
executable = tvm.compile(mod, target=target)
```

表示真正开始编译。

编译完成后，`executable` 就是 TVM 生成的可执行模块。

可以把这一段理解成：

> 把“TVM 内部的模型计算图”变成“本机 CPU 可以执行的程序”。

## 六、VirtualMachine 是什么

编译完之后，脚本用 TVM 的 VM 执行模型：

```python
dev = tvm.cpu()
vm = relax.VirtualMachine(executable, dev)
```

这里：

```python
dev = tvm.cpu()
```

表示运行设备是 CPU。

```python
vm = relax.VirtualMachine(executable, dev)
```

表示创建一个 TVM 虚拟机，用来运行刚刚编译出来的模型。

执行推理的是这一句：

```python
output = to_numpy(vm["main"](tvm.runtime.tensor(input_data, dev)))
```

拆开看：

```python
tvm.runtime.tensor(input_data, dev)
```

把 NumPy 数组包装成 TVM runtime 能接受的 Tensor。

```python
vm["main"](...)
```

调用模型的主函数。

```python
to_numpy(...)
```

把 TVM 输出再转回 NumPy，方便后面处理和打印。

## 七、模型输出为什么还要 softmax

ResNet18 的原始输出通常是 1000 个数字。

这 1000 个数字对应 ImageNet 的 1000 个类别。

不过这些数字一开始不是概率，而是 logits。

脚本里用：

```python
probs = softmax(output)
```

把 logits 转成概率。

简单理解：

> softmax 会把模型输出变成总和为 1 的概率分布。

之后用：

```python
top_indices = np.argsort(probs)[-topk:][::-1]
```

找出概率最大的几个类别。

最后打印：

```python
for rank, index in enumerate(top_indices, start=1):
    label = labels[index]
    print(...)
```

也就是把类别编号转换成人类可读的标签。

## 八、一次完整运行的结果

运行：

```bash
.venv/bin/python toys/resnet18_classify.py
```

输出类似：

```text
Loading ONNX ResNet18
Converting ONNX to Relax IRModule
Compiling with TVM for llvm
Running inference
image: /Users/huzi/Documents/Code/tvm/apache-tvm/toys/cache/kitten.jpg
input: x (1, 3, 224, 224)
top classifications:
 1.  282 tiger cat                      0.7885
 2.  281 tabby                          0.1399
 3.  285 Egyptian cat                   0.0451
 4.  287 lynx                           0.0039
 5.  292 tiger                          0.0007
```

这里说明：

模型认为这张图片最像 `tiger cat`，概率约为 `0.7885`。

## 九、如果换自己的图片

可以这样运行：

```bash
.venv/bin/python toys/resnet18_classify.py --image-path /path/to/your/image.jpg
```

脚本会跳过默认的 `kitten.jpg`，改用你传入的图片。

## 十、把流程压缩成一张图

```text
JPG 图片
  |
  v
preprocess_image()
  |
  v
NumPy 输入张量 (1, 3, 224, 224)
  |
  v
ONNX ResNet18
  |
  v
tvm_onnx.from_onnx()
  |
  v
TVM IRModule
  |
  v
relax.get_pipeline("zero")
  |
  v
tvm.compile(target="llvm")
  |
  v
TVM executable
  |
  v
Relax VirtualMachine
  |
  v
模型输出 logits
  |
  v
softmax + top-k
  |
  v
分类结果
```

## 十一、初学者最容易困惑的几个点

### 1. TVM 不是直接拿图片分类

TVM 的作用不是“识别猫”本身。

真正学会识别猫的是 ResNet18 模型。

TVM 做的是：

> 把这个模型转换、优化、编译，并在目标硬件上执行。

### 2. ONNX 不是运行时

ONNX 文件更像是模型说明书。

它描述了模型有哪些层、每层怎么计算、参数是多少。

TVM 读取 ONNX 后，会把它转换成自己的 IRModule，再进行编译。

### 3. `compile` 之后才是真正可执行的东西

`from_onnx()` 得到的是 TVM 内部表示。

`tvm.compile()` 得到的才是可以交给 runtime 执行的模块。

### 4. 图片预处理很重要

如果图片没有按训练时的方式处理，模型输出会很差。

所以 resize、crop、归一化、NCHW 转换都不是随便写的，它们是 ResNet/ImageNet 分类任务的常规输入格式。

