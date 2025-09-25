# ort_enable_debug.py
import sys
from importlib.abc import MetaPathFinder, Loader
from importlib.util import spec_from_loader
import types
import numpy as np
import os
import onnxruntime as original_ort  # 导入原始的 onnxruntime

def get_model_name(model_path):
    """从模型路径中提取模型名称"""
    if isinstance(model_path, str):
        return os.path.splitext(os.path.basename(model_path))[0]
    return "memory_model"

class DebugCounter:
    """用于记录每个模型的调用次数"""
    _counters = {}

    @classmethod
    def get_count(cls, model_name):
        if model_name not in cls._counters:
            cls._counters[model_name] = 0
        cls._counters[model_name] += 1
        return cls._counters[model_name]

class DebuggingInferenceSession(original_ort.InferenceSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_path = args[0] if args else kwargs.get('path')
        self._original_run = super().run  # 保存原始的 run 方法

    def run(self, output_names, input_feed, **kwargs):
        debug_level = int(os.environ.get('ONNX_DEBUG', '1'))
        if debug_level == 0:
            return self._original_run(output_names, input_feed, **kwargs)

        model_name = get_model_name(self._model_path)
        run_count = DebugCounter.get_count(model_name)
        print(f"\n\n<<<<<<<<<<<<<<")
        # 打印模型信息
        print(f"正在运行模型: {self._model_path}")
        print(f"输出名称: {output_names}")

        # 打印输入信息
        print("输入信息:")
        input_arrays = {}
        for name, value in input_feed.items():
            if isinstance(value, np.ndarray):
                print(f"- {name}: shape={value.shape}, dtype={value.dtype}")
                input_arrays[name] = value
            else:
                print(f"- {name}: type={type(value)}")

        # 保存输入数据
        if debug_level >= 2 and input_arrays:
            input_file = f"{model_name}_input_{run_count}.npz"
            np.savez(input_file, **input_arrays)
            print(f"输入数据已保存到: {input_file}")

        try:
            # 执行原始 run 方法
            outputs = self._original_run(output_names, input_feed, **kwargs)

            # 检查outputs
            if outputs is None:
                print("警告: 模型运行返回None")
                return outputs

            # 打印输出信息
            print("输出信息:")
            output_arrays = {}

            # 处理output_names为None的情况
            if output_names is None:
                # 如果outputs是tuple或list,直接遍历
                for i, value in enumerate(outputs):
                    if isinstance(value, np.ndarray):
                        print(f"- output_{i}: shape={value.shape}, dtype={value.dtype}")
                        output_arrays[f"output_{i}"] = value
                    else:
                        print(f"- output_{i}: type={type(value)}")
            else:
                # 原来的处理方式
                for name, value in zip(output_names, outputs):
                    if isinstance(value, np.ndarray):
                        print(f"- {name}: shape={value.shape}, dtype={value.dtype}")
                        output_arrays[name] = value
                    else:
                        print(f"- {name}: type={type(value)}")

            # 保存输出数据
            if debug_level >= 2 and output_arrays:
                output_file = f"{model_name}_output_{run_count}.npz"
                np.savez(output_file, **output_arrays)
                print(f"输出数据已保存到: {output_file}")

        except Exception as e:
            print(f"运行时发生错误: {str(e)}")
            raise

        print(">>>>>>>>>>>>>>>\n\n")

        return outputs

class ONNXRuntimeDebugLoader(Loader):
    def create_module(self, spec):
        return types.ModuleType(spec.name)

    def exec_module(self, module):
        # 复制原始 onnxruntime 的所有属性
        for attr in dir(original_ort):
            setattr(module, attr, getattr(original_ort, attr))
        # 将 InferenceSession 替换为调试版本
        module.InferenceSession = DebuggingInferenceSession

class ONNXRuntimeDebugFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'onnxruntime':
            return spec_from_loader(fullname, ONNXRuntimeDebugLoader())
        return None

# 清除已有的 onnxruntime 缓存
if 'onnxruntime' in sys.modules:
    del sys.modules['onnxruntime']

# 安装 import hook
sys.meta_path.insert(0, ONNXRuntimeDebugFinder())
