# 使用relay，加载-编译-运行，最小化模板
import onnx

import tvm
from tvm import relay
from tvm.contrib import graph_executor

target = "llvm"
dev = tvm.cpu(0)

onnx_model=onnx.load('xxx')

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