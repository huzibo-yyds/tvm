import tvm
import tvm_ffi
# tvm._ffi 在当前版本不存在，FFI 句柄在独立的 tvm_ffi 包里

## 1、定位 TVM python 包
print(tvm.__file__)
print("TVM version:", tvm.__version__)

## 2、确定所使用的 TVM 库
print("FFI library handle:", tvm_ffi.LIB)
print("FFI version:", getattr(tvm_ffi, "__version__", "unknown"))

## 3、检查TVM构建选项
# print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))


## 4、检查设备检查
print("cuda exist:", tvm.cuda().exist)
print("vulkan exist:", tvm.vulkan().exist)
print("opencl exist:", tvm.opencl().exist)