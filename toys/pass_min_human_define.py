import tvm
from tvm import relay
from tvm.relay import ExprMutator


class Mul2ToAdd(ExprMutator):
    def visit_call(self, call):
        # 先递归访问子节点
        new_call = super().visit_call(call)

        # 判断是不是 multiply
        if isinstance(new_call.op, tvm.ir.Op) and new_call.op.name == "multiply":
            lhs, rhs = new_call.args

            # 判断右边是不是常量 2.0
            if isinstance(rhs, relay.Constant):
                value = rhs.data.numpy()
                if value.shape == () and float(value) == 2.0:
                    return relay.add(lhs, lhs)

        return new_call


@relay.transform.function_pass(opt_level=1)
class MyOptimizePass:
    def transform_function(self, func, mod, ctx):
        return Mul2ToAdd().visit(func)


# 构造一个简单 Relay module: y = x * 2
x = relay.var("x", shape=(4,), dtype="float32")
y = relay.multiply(x, relay.const(2.0, "float32"))
func = relay.Function([x], y)
mod = tvm.IRModule.from_expr(func)

print("=== Before ===")
print(mod)

# 只跑 pass，不 build
seq = tvm.transform.Sequential([
    relay.transform.InferType(),
    MyOptimizePass(),
    relay.transform.InferType(),
])

with tvm.transform.PassContext(opt_level=3):
    mod_opt = seq(mod)

print("=== After ===")
print(mod_opt)