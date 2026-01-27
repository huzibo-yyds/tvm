"""Quick start demo: define a tiny MLP in Relax frontend, compile, and run on CPU."""

import numpy as np
import tvm
from tvm import relax
from tvm.relax.frontend import nn

# 使用 TVM 前端定义了一个两层 MLP网络
class MLPModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(784, 256)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(256, 10)

	def forward(self, x):
		x = self.fc1(x)
		x = self.relu1(x)
		x = self.fc2(x)
		return x


def main():
	# 1) 构建模型并导出为 IRModule
	mod, param_spec = MLPModel().export_tvm(
		spec={"forward": {"x": nn.spec.Tensor((1, 784), "float32")}}
	)
	print('---------')
	mod.show()
	print('---------\n')

	# 2) 执行零优化 pipeline（示例用，不做特定 target 调优）
	# 目标：模型优化、张量程序优化
	mod = relax.get_pipeline("zero")(mod)
	print('---------')
	mod.show()
	print('---------\n')

	# 3) 编译并创建虚拟机
	target = tvm.target.Target("llvm")
	ex = tvm.compile(mod, target=target)
	dev = tvm.cpu()
	vm = relax.VirtualMachine(ex, dev)

	# 4) 准备输入与参数
	data = np.random.rand(1, 784).astype("float32")
	tvm_data = tvm.runtime.tensor(data, device=dev)
	params = [np.random.rand(*p.shape).astype("float32") for _, p in param_spec]
	params = [tvm.runtime.tensor(p, device=dev) for p in params]

	# 5) 运行并打印输出
	out = vm["forward"](tvm_data, *params).numpy()
	print("Output shape:", out.shape)
	print("Sample output:", out)


if __name__ == "__main__":
	main()
