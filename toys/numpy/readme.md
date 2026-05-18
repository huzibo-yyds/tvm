# NumPy LayerNorm Runtime 验证说明

📌 路线：**Relay custom op + te.extern + Python PackedFunc**

## 目标

这个目录实现了一条只用于数学验证的 TVM runtime 路径：

```text
Relay custom op
  -> te.extern
  -> tir.call_packed
  -> Python PackedFunc
  -> NumPy LayerNorm
```

它的目标不是性能优化，也不是部署，而是验证：

```text
如果把 ViT-Tiny 里的一个或多个 LayerNorm 换成 Python NumPy 实现，
整个模型最终输出是否仍然和原始 TVM baseline 一致。
```

为避免影响之前已经实现的 TE 自定义算子，本实验新增独立算子：

```text
custom.numpy_layer_norm
```

原有算子仍然保持独立：

```text
custom.layer_norm
```

## 整体流程

baseline 路径：

```text
ViT-Tiny ONNX
  -> relay.frontend.from_onnx
  -> Relay IR
  -> relay.build
  -> graph_executor
  -> baseline logits
```

NumPy custom 路径：

```text
ViT-Tiny ONNX
  -> relay.frontend.from_onnx
  -> Relay IR
  -> ReplaceLayerNormWithNumpyExtern pass
  -> Relay IR with custom.numpy_layer_norm
  -> relay.build
  -> graph_executor
  -> te.extern
  -> tir.call_packed("custom.runtime.numpy_layer_norm", ...)
  -> Python NumPy LayerNorm
  -> numpy custom logits
```

最后比较：

```text
baseline logits vs numpy custom logits
```

## 为什么不能直接用 NumPy 做 FTVMCompute

普通 `FTVMCompute` 需要返回：

```text
Array<te::Tensor>
```

也就是 TVM 的符号张量表达式，后续才能进入：

```text
TE compute
  -> schedule
  -> TIR
  -> LLVM codegen
  -> runtime
```

NumPy 是立即执行的真实数组计算：

```text
np.ndarray -> np.mean / np.sqrt / ... -> np.ndarray
```

它不是符号表达式，TVM 不能把 NumPy 自动 lower 成 TIR，也不能对 NumPy 代码做 schedule/codegen。

因此本实验使用折中方案：

```text
FTVMCompute 返回 te.extern
te.extern 生成外部 runtime 调用
runtime 调用 Python PackedFunc
Python PackedFunc 内部用 NumPy 算 LayerNorm
```

核心调用链是：

```text
custom.numpy_layer_norm
  -> te.extern(...)
  -> tir.call_packed("custom.runtime.numpy_layer_norm", data, gamma, beta, out, axis, epsilon)
  -> numpy_layer_norm_runtime(...)
```

## 文件职责

### C++ Relay op 注册

文件：

```text
src/relay/op/custom/numpy_layer_norm.cc
```

职责：

- 注册 Relay op 名字 `custom.numpy_layer_norm`。
- 复用 `LayerNormAttrs`，保存 `axis`、`epsilon`、`center`、`scale`。
- 注册 type relation：
  - 输出 shape/dtype 等于 `data`。
  - `gamma`/`beta` shape 等于归一化轴长度。
- 注册 FFI Make 函数：

```text
relay.op.custom._make.numpy_layer_norm
```

注意：这个文件不注册 C++ `FTVMCompute`。它只负责让 Relay 认识这个 op。

### Python Relay API

文件：

```text
python/tvm/relay/op/custom.py
```

新增 API：

```python
relay.op.custom.numpy_layer_norm(data, gamma, beta, axis=-1, epsilon=1e-5)
```

这个函数只负责构造 Relay Call，最终调用 C++ FFI：

```text
_custom_make.numpy_layer_norm(...)
```

### te.extern compute 和 schedule

文件：

```text
python/tvm/relay/op/_custom.py
```

这里给 `custom.numpy_layer_norm` 注册 Python 侧 `FTVMCompute`：

```python
@reg.register_compute("custom.numpy_layer_norm")
def compute_custom_numpy_layer_norm(attrs, inputs, out_type):
    return [
        te.extern(
            out_type.shape,
            [data, gamma, beta],
            lambda ins, outs: tvm.tir.call_packed(
                "custom.runtime.numpy_layer_norm",
                ins[0],
                ins[1],
                ins[2],
                outs[0],
                axis,
                epsilon,
            ),
            dtype=out_type.dtype,
        )
    ]
```

同时注册 schedule：

```python
reg.register_schedule("custom.numpy_layer_norm", schedule_custom_numpy_layer_norm)
```

这个 schedule 只创建 extern op 的 schedule，不做性能优化。

### Python NumPy runtime

文件：

```text
toys/numpy/numpy_layer_norm_runtime.py
```

注册 PackedFunc：

```python
tvm.register_func("custom.runtime.numpy_layer_norm", numpy_layer_norm_runtime, override=True)
```

NumPy 计算公式：

```python
mean = np.mean(x, axis=axis, keepdims=True, dtype=np.float32)
centered = x - mean
var = np.mean(centered * centered, axis=axis, keepdims=True, dtype=np.float32)
y = centered / np.sqrt(var + epsilon) * gamma + beta
```

结果通过：

```python
out.copyfrom(...)
```

写回 TVM runtime 分配的输出 buffer。

### Relay pass

文件：

```text
toys/numpy/numpy_layer_norm_pass.py
```

Pass 名字：

```python
ReplaceLayerNormWithNumpyExtern
```

作用：

- 识别直接的 `nn.layer_norm`。
- 识别 ONNX frontend 分解后的 LayerNorm 子图。
- 将匹配到的 LayerNorm 替换为：

```text
custom.numpy_layer_norm(data, gamma, beta, axis=-1, epsilon=...)
```

支持：

```text
--replace-indices 0,3,7
```

用于只替换指定序号的 LayerNorm。不传时默认替换所有匹配到的 LayerNorm。

### 脚本

公共工具：

```text
toys/numpy/vit_tiny_numpy_common.py
```

单算子测试：

```text
toys/numpy/test_numpy_layer_norm.py
```

只跑 NumPy custom 路径：

```text
toys/numpy/vit_tiny_numpy_layernorm_run.py
```

baseline vs NumPy custom 对比：

```text
toys/numpy/vit_tiny_numpy_layernorm_compare.py
```

## 和 custom.layer_norm 的区别

`custom.layer_norm`：

```text
src/relay/op/custom/layer_norm.cc
  -> C++ FTVMCompute
  -> 手写 TE compute
  -> schedule
  -> TIR
  -> LLVM codegen
  -> runtime
```

`custom.numpy_layer_norm`：

```text
src/relay/op/custom/numpy_layer_norm.cc
  -> Relay op 注册
python/tvm/relay/op/_custom.py
  -> te.extern
  -> tir.call_packed
toys/numpy/numpy_layer_norm_runtime.py
  -> Python PackedFunc
  -> NumPy compute
```

核心区别：

```text
custom.layer_norm 是真正走 TVM TE/TIR/codegen 的算子实现。
custom.numpy_layer_norm 是 runtime 外部调用 NumPy，用于数学验证。
```

因此 `custom.numpy_layer_norm`：

- 可以参与 `relay.build`。
- 可以在 graph executor 里运行。
- 但不能被 TVM 做正常 TE 优化。
- 依赖当前 Python 进程注册 PackedFunc。
- 不适合部署和性能测试。

## 编译

因为新增了 C++ Relay op 注册文件：

```text
src/relay/op/custom/numpy_layer_norm.cc
```

所以需要重新编译 TVM：

```bash
cd /Users/huzi/Documents/Code/tvm/apache-tvm-0.19.0/build
/Users/huzi/miniconda3/envs/tvm-0.19/bin/ninja -j8
```

如果是首次新增文件，CMake 会检测到 glob 变化并自动重新配置。

## 运行验证

进入 TVM 源码根目录：

```bash
cd /Users/huzi/Documents/Code/tvm/apache-tvm-0.19.0
```

### 1. 单算子测试

```bash
/Users/huzi/miniconda3/envs/tvm-0.19/bin/python toys/numpy/test_numpy_layer_norm.py
```

期望输出：

```text
custom.numpy_layer_norm extern check: PASS
direct nn.layer_norm numpy pass check: PASS
```

这个测试确认：

- `custom.numpy_layer_norm` 可以被 Relay 构造。
- `relay.build` 可以 lower 它。
- graph executor 运行时会调用 Python NumPy PackedFunc。
- 输出和 NumPy reference 一致。

### 2. 只跑 NumPy custom 路径

```bash
/Users/huzi/miniconda3/envs/tvm-0.19/bin/python toys/numpy/vit_tiny_numpy_layernorm_run.py
```

默认替换所有匹配到的 LayerNorm。

期望看到：

```text
numpy_layer_norm candidates: 25
numpy_layer_norm replacements: 25
custom.numpy_layer_norm calls in IR: 25
numpy runtime calls: 25
```

并输出 top-5 分类结果。

### 3. 只替换指定 LayerNorm

只替换第 0 个 LayerNorm：

```bash
/Users/huzi/miniconda3/envs/tvm-0.19/bin/python toys/numpy/vit_tiny_numpy_layernorm_run.py --replace-indices 0
```

期望看到：

```text
numpy_layer_norm candidates: 25
numpy_layer_norm replacements: 1
numpy_layer_norm replaced indices: [0]
custom.numpy_layer_norm calls in IR: 1
numpy runtime calls: 1
```

替换多个：

```bash
/Users/huzi/miniconda3/envs/tvm-0.19/bin/python toys/numpy/vit_tiny_numpy_layernorm_run.py --replace-indices 0,5,10
```

### 4. baseline vs NumPy custom 对比

```bash
/Users/huzi/miniconda3/envs/tvm-0.19/bin/python toys/numpy/vit_tiny_numpy_layernorm_compare.py
```

这个脚本会跑：

```text
baseline TVM 完整模型
NumPy custom 完整模型
```

然后比较最终 logits。

当前验证结果：

```text
numpy_layer_norm candidates: 25
numpy_layer_norm replacements: 25
custom.numpy_layer_norm calls in IR: 25
numpy runtime calls: 25
max_abs_diff: 1.001358e-05
max_rel_diff: 0.0039828327
allclose(rtol=0.0001, atol=0.0001): True
```

这说明替换所有 LayerNorm 后，最终模型输出仍然和 baseline 保持一致。

## 输出差异为什么不是 0

baseline 路径里 LayerNorm 是由 ONNX frontend 分解出的 Relay 基础算子计算：

```text
mean -> subtract -> power -> mean -> add -> sqrt -> divide -> multiply -> add
```

NumPy custom 路径里 LayerNorm 是 Python NumPy 计算。

两边数学语义一致，但执行实现不同：

- reduction 顺序可能不同。
- 中间结果舍入点不同。
- NumPy 和 TVM/LLVM 的 `sqrt`、`divide` 实现路径不同。
- float32 本身不保证 bitwise 一致。

所以应该使用：

```python
np.allclose(..., rtol=1e-4, atol=1e-4)
```

而不是要求完全相等。

## 注意事项

- `custom.runtime.numpy_layer_norm` 必须在运行脚本里先注册。
- 编译出的 TVM module 不能脱离 Python/NumPy 独立运行。
- 这个路径会频繁在 TVM NDArray 和 NumPy ndarray 之间转换，性能不是目标。
- `--replace-indices` 的序号来自 pass 遍历过程中发现 LayerNorm 的顺序。
- 当前只验证 ViT-Tiny 常见形态：
  - `float32`
  - 最后一维 LayerNorm
  - `center=True`
  - `scale=True`
  - gamma/beta 都存在

## 已验证命令

```bash
/Users/huzi/miniconda3/envs/tvm-0.19/bin/python toys/numpy/test_numpy_layer_norm.py
/Users/huzi/miniconda3/envs/tvm-0.19/bin/python toys/numpy/vit_tiny_numpy_layernorm_run.py --replace-indices 0
/Users/huzi/miniconda3/envs/tvm-0.19/bin/python toys/numpy/vit_tiny_numpy_layernorm_compare.py
/Users/huzi/miniconda3/envs/tvm-0.19/bin/python toys/vit_tiny/test_custom_layer_norm.py
```

这些命令均已通过。

