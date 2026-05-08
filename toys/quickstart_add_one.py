import numpy as np
import tvm
from tvm import te


# 1. 用 TE（Tensor Expression）描述一个最简单的逐元素计算：
#    输入张量 A 的每个元素都加 1，得到输出张量 B。
#    这里的 placeholder 只定义形状和名字，不保存具体数据。
A = te.placeholder((8,), name="A")  # 声明输入张量
# 2. compute 定义输出张量 B 的计算规则。
#    lambda i: A[i] + 1.0 表示 B[i] 依赖于 A[i]，并在元素级别执行加一操作。
B = te.compute((8,), lambda i: A[i] + 1.0, name="B") # 声明张量计算

# 3. 将 TE 计算转换成 PrimFunc。
#    TE 更偏向高层的算子描述，PrimFunc/TIR 则是 TVM 后端编译更直接消费的中间表示。
#    这里同时传入 [B, A]，表示函数签名中输出张量和输入张量的顺序。
func = te.create_prim_func([B, A]) # 生产底层PrimFunc。
# 4. 调用 tvm.compile 把 PrimFunc 编译成 LLVM 目标上的可执行函数。
#    target="llvm" 表示生成 CPU 上运行的代码。
compiled = tvm.compile(func, target="llvm") # 将IR编译成可执行模块

# 4.1 显式指定运行设备。
#     tvm.cpu(0) 表示使用第 0 个 CPU 设备，也就是本地主机 CPU。
dev = tvm.cpu(0)

# 5. 准备运行时输入数据。
#    np.arange(8) 生成 [0, 1, 2, ..., 7]，作为输入 A。
#    dtype="float32" 与 TE 中的浮点加法一致，避免类型不匹配。
a_np = np.arange(8, dtype="float32")
# 6. 创建输出缓冲区。
#    tvm.runtime.tensor 包装 NumPy 数组，使其可以作为 TVM runtime 的输入/输出张量。
#    这里输出先初始化为全零，执行后会被 compiled 写入结果。
out = tvm.runtime.tensor(np.zeros(8, dtype="float32"), device=dev)

# 7. 调用编译后的函数。
#    参数顺序必须与 create_prim_func([B, A]) 对应：先传输出 out，再传输入 A。
#    执行完成后，out 中会保存 A + 1 的结果。
compiled(out, tvm.runtime.tensor(a_np, device=dev))
# 8. 将 TVM runtime 张量转回 NumPy，便于查看结果。
print(out.numpy())

# 整体流程：TE 描述计算 -> 生成 PrimFunc/TIR -> tvm.compile 编译 -> runtime 执行