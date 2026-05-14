import numpy as np

import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay import ExprMutator


class Mul2ToAdd(ExprMutator): # 1️⃣ 定义改写器
    """把 Relay 表达式里的 `x * 2.0` 改写成 `x + x`。

    ExprMutator 是 Relay 提供的表达式改写基类：
    - 它会按照 Relay AST 的结构递归访问每一个表达式节点；
    - 默认行为是保持原节点语义不变；
    - 只需要重写自己关心的 visit_xxx 方法，就能在遍历过程中替换节点。

    这里我们只关心 Call 节点，因为 `relay.multiply(...)` 在 Relay IR 中
    会表示成一次函数/算子的调用，也就是 Call。
    """
    def visit_call(self, call):
        # 先调用父类逻辑递归访问当前 Call 的所有子表达式。
        #
        # 这样做有两个好处：
        # 1. 如果参数里还嵌套着其它表达式，嵌套表达式会先被改写；
        # 2. 后面的匹配逻辑基于改写后的 new_call，避免漏掉深层节点。
        new_call = super().visit_call(call)

        # Relay 的 Call.op 不一定总是 tvm.ir.Op，也可能是 Function、GlobalVar 等。
        # 所以这里先确认它是内置算子 Op，再通过名字判断是不是 multiply。
        if isinstance(new_call.op, tvm.ir.Op) and new_call.op.name == "multiply":
            lhs, rhs = new_call.args # 取出左右参数

            # 本例只处理形如 `lhs * 2.0` 的情况：
            # - rhs 必须是 Relay 常量；
            # - 常量必须是标量，也就是 shape == ()；
            # - 标量值必须等于 2.0。
            #
            # 注意：这里没有处理 `2.0 * lhs`，因为那种情况常量在左边；
            # 如果要支持交换律，可以再额外判断 lhs 是否为常量 2.0。
            if isinstance(rhs, relay.Constant):
                # relay.Constant 的真实数据保存在 NDArray 里，转成 numpy 后
                # 才方便检查 shape 和数值。
                value = rhs.data.numpy()
                if value.shape == () and float(value) == 2.0:
                    # 匹配成功后返回一个新的 Relay 表达式节点。
                    # 原来的 `lhs * 2.0` 被替换为 `lhs + lhs`。
                    #
                    # 这只是演示自定义 pass 的最小例子；真实优化里还需要考虑
                    # dtype、广播、数值精度、常量类型等更多边界情况。
                    return relay.add(lhs, lhs)

        # 没有匹配到目标模式时，保持父类递归改写后的 Call 不变。
        return new_call


@relay.transform.function_pass(opt_level=1) # 2️⃣ 修饰器将python class，注册为1个自定义pass
class MyOptimizePass:
    def transform_function(self, func, mod, ctx): # 必须复写 ！真正pass入口
        # function_pass 要实现 transform_function 这个入口。
        #
        # 参数含义：
        # - func: 当前正在处理的 Relay Function；
        # - mod: 整个 IRModule，可以用来查询其它全局函数或类型信息；
        # - ctx: 当前 PassContext，里面包含 opt_level、配置项等。
        #
        # 这里的优化只需要改写当前函数体，所以直接创建 Mul2ToAdd，
        # 对整个 func 做一次 ExprMutator 遍历并返回新函数。
        return Mul2ToAdd().visit(func)


# 3️⃣ 构造一个简单 Relay module: y = x * 2
# relay.var 创建一个 Relay 输入变量，相当于函数参数。
x = relay.var("x", shape=(4,), dtype="float32")
# relay.multiply 创建 multiply Call 节点；右侧 relay.const 是标量常量 2.0。
# 这个表达式正好满足 Mul2ToAdd 里的匹配条件。
y = relay.multiply(x, relay.const(2.0, "float32"))
# 把表达式包装成 Relay Function，再从 Function 构造 IRModule。
# Relay pass 通常以 IRModule 为输入输出单位运行。
func = relay.Function([x], y)
mod = tvm.IRModule.from_expr(func)

print("=== Before ===")
print(mod)

# 3️⃣ 构造一个 pass pipeline，只跑优化 pass，不进入 build/codegen。
#
# InferType 放在自定义 pass 前后各一次：
# - 前一次让输入 IR 尽量带上完整类型信息；
# - 后一次为新生成的 `add(lhs, lhs)` 补齐类型信息。
seq = tvm.transform.Sequential([
    relay.transform.InferType(), # 内建pass，类型推导 ！TVMIR是强类，shape、dtype
    MyOptimizePass(),
    relay.transform.InferType(),
])

# 4️⃣ PassContext 控制 pass 运行环境。这里 opt_level=3 表示允许执行
# opt_level <= 3 的 pass；MyOptimizePass 的 opt_level=1，因此会被执行。
with tvm.transform.PassContext(opt_level=3):
    mod_opt = seq(mod)

print("=== After ===")
print(mod_opt)


target = "llvm"

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod_opt, target=target) # 编译成可执行 runtime module

dev = tvm.cpu(0)
module = graph_executor.GraphModule(lib["default"](dev))

input_data = np.array([1, 2, 3, 4], dtype="float32")
module.set_input("x", input_data)
module.run()

out = module.get_output(0).numpy()
print(out)


'''
relay.build 内部流程：
Relay Function
    ↓
Relay 优化
    ↓
FuseOps
    ↓
TECompiler
    ↓
查 add 的 Op Strategy
    ↓
找到 add 在 llvm target 上的实现
    ↓
调用 TOPI compute
    ↓
调用 schedule
    ↓
Lower 到 TIR (TIR 是 TVM 更底层的循环 IR)
    ↓
TIR 优化
    ↓
LLVM codegen
    ↓
生成 CPU 可执行代码

'''