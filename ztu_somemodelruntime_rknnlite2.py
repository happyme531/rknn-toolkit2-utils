# 模块级常量和函数
from rknnlite.api import RKNNLite
import numpy as np
import os
import warnings
import logging
from typing import List, Dict, Union, Optional

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    warnings.warn("onnxruntime未安装,只能使用RKNN后端", ImportWarning)

# 配置日志
logger = logging.getLogger("somemodelruntime_rknnlite2")
logger.setLevel(logging.ERROR)  # 默认只输出错误信息
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# ONNX Runtime日志级别到Python logging级别的映射
_LOGGING_LEVEL_MAP = {
    0: logging.DEBUG,    # Verbose
    1: logging.INFO,     # Info
    2: logging.WARNING,  # Warning
    3: logging.ERROR,    # Error
    4: logging.CRITICAL  # Fatal
}

# 检查环境变量中的日志级别设置
try:
    env_log_level = os.getenv('ZTU_MODELRT_RKNNL2_LOG_LEVEL')
    if env_log_level is not None:
        log_level = int(env_log_level)
        if log_level in _LOGGING_LEVEL_MAP:
            logger.setLevel(_LOGGING_LEVEL_MAP[log_level])
            logger.info(f"从环境变量设置日志级别: {log_level}")
        else:
            logger.warning(f"环境变量ZTU_MODELRT_RKNNL2_LOG_LEVEL的值无效: {log_level}, 应该是0-4之间的整数")
except ValueError:
    logger.warning(f"环境变量ZTU_MODELRT_RKNNL2_LOG_LEVEL的值无效: {env_log_level}, 应该是0-4之间的整数")


def set_default_logger_severity(level: int) -> None:
    """
    Sets the default logging severity. 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal
    
    Args:
        level: 日志级别(0-4)
    """
    if level not in _LOGGING_LEVEL_MAP:
        raise ValueError(f"无效的日志级别: {level}, 应该是0-4之间的整数")
    logger.setLevel(_LOGGING_LEVEL_MAP[level])

def set_default_logger_verbosity(level: int) -> None:
    """
    Sets the default logging verbosity level. To activate the verbose log, 
    you need to set the default logging severity to 0:Verbose level.
    
    Args:
        level: 日志级别(0-4)
    """
    set_default_logger_severity(level)

# RKNN tensor type到numpy dtype的映射
RKNN_DTYPE_MAP = {
    0: np.float32,  # RKNN_TENSOR_FLOAT32
    1: np.float16,  # RKNN_TENSOR_FLOAT16
    2: np.int8,     # RKNN_TENSOR_INT8
    3: np.uint8,    # RKNN_TENSOR_UINT8
    4: np.int16,    # RKNN_TENSOR_INT16
    5: np.uint16,   # RKNN_TENSOR_UINT16
    6: np.int32,    # RKNN_TENSOR_INT32
    7: np.uint32,   # RKNN_TENSOR_UINT32
    8: np.int64,    # RKNN_TENSOR_INT64
    9: bool,        # RKNN_TENSOR_BOOL
    10: np.int8,    # RKNN_TENSOR_INT4 (用int8表示)
}

def get_available_providers() -> List[str]:
    """
    获取可用的设备提供者列表(为保持接口兼容性的占位函数)
    
    Returns:
        list: 可用的设备提供者列表,总是返回["CPUExecutionProvider", "somemodelruntime_rknnlite2_ExecutionProvider"]
    """
    return ["CPUExecutionProvider", "somemodelruntime_rknnlite2_ExecutionProvider"]


def get_device() -> str:
    """
    获取当前设备

    Returns:
        str: 当前设备
    """
    return "RKNN2"

def get_version_info() -> Dict[str, str]:
    """
    获取版本信息
    
    Returns:
        dict: 包含API和驱动版本信息的字典
    """
    runtime = RKNNLite()
    version = runtime.get_sdk_version()
    return {
        "api_version": version.split('\n')[2].split(': ')[1].split(' ')[0],
        "driver_version": version.split('\n')[3].split(': ')[1]
    }

class IOTensor:
    """输入/输出张量的信息封装类"""
    def __init__(self, name, shape, type=None):
        self.name = name.decode() if isinstance(name, bytes) else name
        self.shape = shape
        self.type = type

    def __str__(self):
        return f"IOTensor(name='{self.name}', shape={self.shape}, type={self.type})"

class SessionOptions:
    """会话选项类"""
    def __init__(self):
        self.enable_profiling = False  # 是否使用性能分析
        self.intra_op_num_threads = 1  # 设置RKNN的线程数, 对应rknn的core_mask
        self.log_severity_level = -1 # 另一个设置日志级别的参数
        self.log_verbosity_level = -1 # 另一个设置日志级别的参数


class InferenceSession:
    """
    RKNNLite运行时封装类,API风格类似ONNX Runtime
    """

    def __new__(cls, model_path: str, sess_options: Optional[SessionOptions] = None, **kwargs):
        processed_path = InferenceSession._process_model_path(model_path, sess_options)
        if isinstance(processed_path, str) and processed_path.lower().endswith('.onnx'):
            logger.info("使用ONNX Runtime加载模型")
            if not HAS_ORT:
                raise RuntimeError("未安装onnxruntime,无法加载ONNX模型")
            return ort.InferenceSession(processed_path, sess_options=sess_options, **kwargs)
        else:
            # 如果不是 ONNX 模型，则调用父类的 __new__ 创建 InferenceSession 实例
            instance = super().__new__(cls)
            # 保存处理后的路径
            instance._processed_path = processed_path
            return instance

    def __init__(self, model_path: str, sess_options: Optional[SessionOptions] = None, **kwargs):
        """
        初始化运行时并加载模型
        
        Args:
            model_path: 模型文件路径(.rknn或.onnx)
            sess_options: 会话选项
            **kwargs: 其他初始化参数
        """
        options = sess_options or SessionOptions()

        # 只在未设置环境变量时使用SessionOptions中的日志级别
        if os.getenv('ZTU_MODELRT_RKNNL2_LOG_LEVEL') is None:
            if options.log_severity_level != -1:
                set_default_logger_severity(options.log_severity_level)
            if options.log_verbosity_level != -1:
                set_default_logger_verbosity(options.log_verbosity_level)
            
        # 使用__new__中处理好的路径
        model_path = getattr(self, '_processed_path', model_path)
        if isinstance(model_path, str) and model_path.lower().endswith('.onnx'):
            # 避免重复加载 ONNX 模型
            return

        # ... 现有的 RKNN 模型加载和初始化代码 ...
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            logger.error(f"模型文件不存在: {self.model_path}")
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        self.runtime = RKNNLite(verbose=options.enable_profiling)

        logger.debug(f"正在加载模型: {self.model_path}")
        ret = self.runtime.load_rknn(self.model_path)
        if ret != 0:
            logger.error(f"加载RKNN模型失败: {self.model_path}")
            raise RuntimeError(f'加载RKNN模型失败: {self.model_path}')
        logger.debug("模型加载成功")


        if options.intra_op_num_threads == 1:
            core_mask = RKNNLite.NPU_CORE_AUTO
        elif options.intra_op_num_threads == 2:
            core_mask = RKNNLite.NPU_CORE_0_1
        elif options.intra_op_num_threads == 3:
            core_mask = RKNNLite.NPU_CORE_0_1_2
        else:
            raise ValueError(f"intra_op_num_threads的值无效: {options.intra_op_num_threads}, 只能是1,2或3")

        logger.debug("正在初始化运行时环境")
        ret = self.runtime.init_runtime(core_mask=core_mask)
        if ret != 0:
            logger.error("初始化运行时环境失败")
            raise RuntimeError('初始化运行时环境失败')

        logger.debug("运行时环境初始化成功")

        # 在 runtime 初始化后，按环境变量自动注册自定义算子插件库
        try:
            # 注册用户指定路径插件（逗号/分号分隔）
            env_custom = os.getenv('ZTU_MODELRT_RKNN2_REG_CUSTOM_OP_LIB', '').strip()
            if env_custom:
                paths = [seg.strip() for seg in re.split(r"[,;:]", env_custom) if seg.strip()]
                ok = 0
                for p in paths:
                    if self.register_custom_op_lib(p):
                        ok += 1
                if ok > 0:
                    logger.info(f"已注册 {ok}/{len(paths)} 个自定义算子插件")
                        # 注册系统目录下插件
            if os.getenv('ZTU_MODELRT_RKNN2_REG_SYSTEM_CUSTOM_OP_LIB', '1') == '1':
                cnt = self.register_system_custom_op_lib()
                if cnt > 0:
                    logger.info(f"已从系统目录注册 {cnt} 个自定义算子插件")
        except Exception as e:
            logger.warning(f"自动注册自定义算子插件失败: {e}")

        # 可选：按环境变量注册内置(基于Python)捆绑算子
        if os.getenv('ZTU_MODELRT_RKNN2_REG_BUNDLED_OPS', '0') == '1':
            logger.info("根据环境变量注册捆绑算子")
            self.register_bundled_ops()

        self._init_io_info()
        self.options = options

    def get_performance_info(self) -> Dict[str, float]:
        """
        获取性能信息
        
        Returns:
            dict: 包含性能信息的字典
        """
        if not self.options.perf_debug:
            raise RuntimeError("性能分析未启用,请在SessionOptions中设置perf_debug=True")
            
        perf = self.runtime.rknn_runtime.get_run_perf()
        return {
            "run_duration": perf.run_duration / 1000.0  # 转换为毫秒
        }

    def set_core_mask(self, core_mask: int) -> None:
        """
        设置NPU核心使用模式
        
        Args:
            core_mask: NPU核心掩码,使用NPU_CORE_*常量
        """
        ret = self.runtime.rknn_runtime.set_core_mask(core_mask)
        if ret != 0:
            raise RuntimeError("设置NPU核心模式失败")

    @staticmethod
    def _process_model_path(model_path, sess_options):
        """
        处理模型路径,支持.onnx和.rknn文件
        
        Args:
            model_path: 模型文件路径
        """
        # 如果是ONNX文件,检查是否需要自动加载RKNN
        if model_path.lower().endswith('.onnx'):
            logger.info("检测到ONNX模型文件")
            
            # 获取需要跳过自动加载的模型列表
            skip_models = os.getenv('ZTU_MODELRT_RKNNL2_SKIP', '').strip()
            if skip_models:
                skip_list = [m.strip() for m in skip_models.split(',')]
                # 获取模型文件名(不含路径)用于匹配
                model_name = os.path.basename(model_path)
                if model_name.lower() in [m.lower() for m in skip_list]:
                    logger.info(f"模型{model_name}在跳过列表中,将使用ONNX Runtime")
                    return model_path
            
            # 构造RKNN文件路径
            rknn_path = os.path.splitext(model_path)[0] + '.rknn'
            if os.path.exists(rknn_path):
                logger.info(f"找到对应的RKNN模型,将使用RKNN: {rknn_path}")
                return rknn_path
            else:
                logger.info("未找到对应的RKNN模型,将使用ONNX Runtime")
                return model_path
            
        return model_path
        
    def _convert_nhwc_to_nchw(self, shape):
        """将NHWC格式的shape转换为NCHW格式"""
        if len(shape) == 4:
            # NHWC -> NCHW
            n, h, w, c = shape
            return [n, c, h, w]
        return shape
        
    def _init_io_info(self):
        """初始化模型的输入输出信息"""
        runtime = self.runtime.rknn_runtime
        
        # 获取输入输出数量
        n_input, n_output = runtime.get_in_out_num()
        
        # 获取输入信息
        self.input_tensors = []
        for i in range(n_input):
            attr = runtime.get_tensor_attr(i)
            shape = [attr.dims[j] for j in range(attr.n_dims)]
            # 对四维输入进行NHWC到NCHW的转换
            shape = self._convert_nhwc_to_nchw(shape)
            # 获取dtype
            dtype = RKNN_DTYPE_MAP.get(attr.type, None)
            tensor = IOTensor(attr.name, shape, dtype)
            self.input_tensors.append(tensor)
            
        # 获取输出信息
        self.output_tensors = []
        for i in range(n_output):
            attr = runtime.get_tensor_attr(i, is_output=True)
            shape = runtime.get_output_shape(i)
            # 获取dtype
            dtype = RKNN_DTYPE_MAP.get(attr.type, None)
            tensor = IOTensor(attr.name, shape, dtype)
            self.output_tensors.append(tensor)
        
    def get_inputs(self):
        """
        获取模型输入信息
        
        Returns:
            list: 包含输入信息的列表
        """
        return self.input_tensors
        
    def get_outputs(self):
        """
        获取模型输出信息
        
        Returns:
            list: 包含输出信息的列表
        """
        return self.output_tensors
        
    def run(self, output_names=None, input_feed=None, data_format="nchw", **kwargs):
        """
        执行模型推理
        
        Args:
            output_names: 输出节点名称列表,指定需要返回哪些输出
            input_feed: 输入数据字典或列表
            data_format: 输入数据格式,"nchw"或"nhwc"
            **kwargs: 其他运行时参数
            
        Returns:
            list: 模型输出结果列表,如果指定了output_names则只返回指定的输出
        """
        if input_feed is None:
            logger.error("input_feed不能为None")
            raise ValueError("input_feed不能为None")
            
        # 准备输入数据
        if isinstance(input_feed, dict):
            # 如果是字典,按照模型输入顺序排列
            inputs = []
            input_map = {tensor.name: i for i, tensor in enumerate(self.input_tensors)}
            for tensor in self.input_tensors:
                if tensor.name not in input_feed:
                    raise ValueError(f"缺少输入: {tensor.name}")
                inputs.append(input_feed[tensor.name])
        elif isinstance(input_feed, (list, tuple)):
            # 如果是列表,确保长度匹配
            if len(input_feed) != len(self.input_tensors):
                raise ValueError(f"输入数量不匹配: 期望{len(self.input_tensors)}, 实际{len(input_feed)}")
            inputs = list(input_feed)
        else:
            logger.error("input_feed必须是字典或列表类型")
            raise ValueError("input_feed必须是字典或列表类型")
            
        # 执行推理
        try:
            logger.debug("开始执行推理")
            all_outputs = self.runtime.inference(inputs=inputs, data_format=data_format)
            
            # 如果没有指定output_names,返回所有输出
            if output_names is None:
                return all_outputs
                
            # 获取指定的输出
            output_map = {tensor.name: i for i, tensor in enumerate(self.output_tensors)}
            selected_outputs = []
            for name in output_names:
                if name not in output_map:
                    raise ValueError(f"未找到输出节点: {name}")
                selected_outputs.append(all_outputs[output_map[name]])
                    
            return selected_outputs
            
        except Exception as e:
            logger.error(f"推理执行失败: {str(e)}")
            raise RuntimeError(f"推理执行失败: {str(e)}")
        
    def close(self):
        """
        关闭会话,释放资源
        """
        if self.runtime is not None:
            logger.info("正在释放运行时资源")
            self.runtime.release()
            self.runtime = None
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def end_profiling(self) -> Optional[str]:
        """
        结束性能分析的存根方法
        
        Returns:
            Optional[str]: None
        """
        warnings.warn("end_profiling()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return None
        
    def get_profiling_start_time_ns(self) -> int:
        """
        获取性能分析开始时间的存根方法
        
        Returns:
            int: 0
        """
        warnings.warn("get_profiling_start_time_ns()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return 0
        
    def get_modelmeta(self) -> Dict[str, str]:
        """
        获取模型元数据的存根方法
        
        Returns:
            Dict[str, str]: 空字典
        """
        warnings.warn("get_modelmeta()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return {}
        
    def get_session_options(self) -> SessionOptions:
        """
        获取会话选项
        
        Returns:
            SessionOptions: 当前会话选项
        """
        return self.options
        
    def get_providers(self) -> List[str]:
        """
        获取当前使用的providers的存根方法
        
        Returns:
            List[str]: ["CPUExecutionProvider"]
        """
        warnings.warn("get_providers()是存根方法,始终返回CPUExecutionProvider", RuntimeWarning, stacklevel=2)
        return ["CPUExecutionProvider"]
        
    def get_provider_options(self) -> Dict[str, Dict[str, str]]:
        """
        获取provider选项的存根方法
        
        Returns:
            Dict[str, Dict[str, str]]: 空字典
        """
        warnings.warn("get_provider_options()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return {} 

    def get_session_config(self) -> Dict[str, str]:
        """
        获取会话配置的存根方法
        
        Returns:
            Dict[str, str]: 空字典
        """
        warnings.warn("get_session_config()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return {}

    def get_session_state(self) -> Dict[str, str]:
        """
        获取会话状态的存根方法
        
        Returns:
            Dict[str, str]: 空字典
        """
        warnings.warn("get_session_state()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return {}

    def set_session_config(self, config: Dict[str, str]) -> None:
        """
        设置会话配置的存根方法
        
        Args:
            config: 会话配置字典
        """
        warnings.warn("set_session_config()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)

    def get_memory_info(self) -> Dict[str, int]:
        """
        获取内存使用信息的存根方法
        
        Returns:
            Dict[str, int]: 空字典
        """
        warnings.warn("get_memory_info()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return {}

    def set_memory_pattern(self, enable: bool) -> None:
        """
        设置内存模式的存根方法
        
        Args:
            enable: 是否启用内存模式
        """
        warnings.warn("set_memory_pattern()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)

    def disable_memory_pattern(self) -> None:
        """
        禁用内存模式的存根方法
        """
        warnings.warn("disable_memory_pattern()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)

    def get_optimization_level(self) -> int:
        """
        获取优化级别的存根方法
        
        Returns:
            int: 0
        """
        warnings.warn("get_optimization_level()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return 0

    def set_optimization_level(self, level: int) -> None:
        """
        设置优化级别的存根方法
        
        Args:
            level: 优化级别
        """
        warnings.warn("set_optimization_level()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)

    def get_model_metadata(self) -> Dict[str, str]:
        """
        获取模型元数据的存根方法(与get_modelmeta不同的接口)
        
        Returns:
            Dict[str, str]: 空字典
        """
        warnings.warn("get_model_metadata()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return {}

    def get_model_path(self) -> str:
        """
        获取模型路径
        
        Returns:
            str: 模型文件路径
        """
        return self.model_path

    def get_input_type_info(self) -> List[Dict[str, str]]:
        """
        获取输入类型信息的存根方法
        
        Returns:
            List[Dict[str, str]]: 空列表
        """
        warnings.warn("get_input_type_info()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return []

    def get_output_type_info(self) -> List[Dict[str, str]]:
        """
        获取输出类型信息的存根方法
        
        Returns:
            List[Dict[str, str]]: 空列表
        """
        warnings.warn("get_output_type_info()是存根方法,不提供实际功能", RuntimeWarning, stacklevel=2)
        return [] 

    ################### 自定义算子 ###################

    def _init_custom_op_types(self):
        """初始化自定义算子的类型定义"""
        # 常量
        self._RKNN_TENSOR_FLOAT32 = 0
        self._RKNN_TENSOR_UINT8 = 3
        self._RKNN_TENSOR_INT64 = 8
        self._RKNN_TARGET_TYPE_CPU = 1

        # 结构体定义
        class RKNN_TensorAttr(ctypes.Structure):
            _fields_ = [
                ("index", ctypes.c_uint32),
                ("n_dims", ctypes.c_uint32),
                ("dims", ctypes.c_uint32 * RKNN_MAX_DIMS),
                ("name", ctypes.c_char * RKNN_MAX_NAME_LEN),
                ("n_elems", ctypes.c_uint32),
                ("size", ctypes.c_uint32),
                ("fmt", ctypes.c_int),
                ("type", ctypes.c_int),
                ("qnt_type", ctypes.c_int),
                ("fl", ctypes.c_int8),
                ("zp", ctypes.c_int32),
                ("scale", ctypes.c_float),
                ("w_stride", ctypes.c_uint32),
                ("size_with_stride", ctypes.c_uint32),
                ("pass_through", ctypes.c_uint8),
                ("h_stride", ctypes.c_uint32),
            ]

        class RKNN_TensorMem(ctypes.Structure):
            _fields_ = [
                ("virt_addr", ctypes.c_void_p),
                ("phys_addr", ctypes.c_uint64),
                ("fd", ctypes.c_int32),
                ("offset", ctypes.c_int32),
                ("size", ctypes.c_uint32),
                ("flags", ctypes.c_uint32),
                ("priv_data", ctypes.c_void_p),
            ]

        class RKNN_CustomOpTensor(ctypes.Structure):
            _fields_ = [
                ("attr", RKNN_TensorAttr),
                ("mem", RKNN_TensorMem),
            ]

        class RKNN_GPUOpContext(ctypes.Structure):
            _fields_ = [
                ("cl_context", ctypes.c_void_p),
                ("cl_command_queue", ctypes.c_void_p),
                ("cl_kernel", ctypes.c_void_p),
            ]

        InternalCtxType = (
            ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32
        )

        class RKNN_CustomOpContext(ctypes.Structure):
            _fields_ = [
                ("target", ctypes.c_int),
                ("internal_ctx", InternalCtxType),
                ("gpu_ctx", RKNN_GPUOpContext),
                ("priv_data", ctypes.c_void_p),
            ]

        class RKNN_CustomOpAttr(ctypes.Structure):
            _fields_ = [
                ("name", ctypes.c_char * RKNN_MAX_NAME_LEN),
                ("dtype", ctypes.c_int),
                ("n_elems", ctypes.c_uint32),
                ("data", ctypes.c_void_p),
            ]

        CB_SIG = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.POINTER(RKNN_CustomOpContext),
            ctypes.POINTER(RKNN_CustomOpTensor),
            ctypes.c_uint32,
            ctypes.POINTER(RKNN_CustomOpTensor),
            ctypes.c_uint32,
        )

        DESTROY_SIG = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.POINTER(RKNN_CustomOpContext)
        )

        class RKNN_CustomOp(ctypes.Structure):
            _fields_ = [
                ("version", ctypes.c_uint32),
                ("target", ctypes.c_int),
                ("op_type", ctypes.c_char * RKNN_MAX_NAME_LEN),
                ("cl_kernel_name", ctypes.c_char * RKNN_MAX_NAME_LEN),
                ("cl_kernel_source", ctypes.c_char_p),
                ("cl_source_size", ctypes.c_uint64),
                ("cl_build_options", ctypes.c_char * RKNN_MAX_NAME_LEN),
                ("init", CB_SIG),
                ("prepare", CB_SIG),
                ("compute", CB_SIG),
                ("compute_native", CB_SIG),
                ("destroy", DESTROY_SIG),
            ]

        # 保存类型定义
        self._RKNN_TensorAttr = RKNN_TensorAttr
        self._RKNN_TensorMem = RKNN_TensorMem
        self._RKNN_CustomOpTensor = RKNN_CustomOpTensor
        self._RKNN_CustomOpContext = RKNN_CustomOpContext
        self._RKNN_CustomOpAttr = RKNN_CustomOpAttr
        self._RKNN_CustomOp = RKNN_CustomOp
        self._CB_SIG = CB_SIG
        self._DESTROY_SIG = DESTROY_SIG

    def _create_attr_readers(self, get_op_attr):
        """创建属性读取函数"""
        def read_attr_int64(op_ctx_ptr, key: str, default: int = 0) -> int:
            attr = self._RKNN_CustomOpAttr()
            get_op_attr(op_ctx_ptr, key.encode("utf-8"), ctypes.byref(attr))
            if attr.n_elems == 1 and attr.dtype == self._RKNN_TENSOR_INT64 and attr.data:
                return ctypes.c_int64.from_address(attr.data).value
            return default

        def read_attr_float32(op_ctx_ptr, key: str, default: float = 0) -> float:
            attr = self._RKNN_CustomOpAttr()
            get_op_attr(op_ctx_ptr, key.encode("utf-8"), ctypes.byref(attr))
            if attr.n_elems == 1 and attr.dtype == self._RKNN_TENSOR_FLOAT32 and attr.data:
                return ctypes.c_float.from_address(attr.data).value
            return default

        def read_attr_str(op_ctx_ptr, key: str, default: str = "") -> str:
            attr = self._RKNN_CustomOpAttr()
            get_op_attr(op_ctx_ptr, key.encode("utf-8"), ctypes.byref(attr))
            if attr.n_elems > 0 and attr.dtype == self._RKNN_TENSOR_UINT8 and attr.data:
                buf = (ctypes.c_ubyte * attr.n_elems).from_address(attr.data)
                try:
                    return bytes(buf).decode("utf-8", errors="ignore").strip('"')
                except Exception:
                    return default
            return default
        
 
        return read_attr_int64, read_attr_str, read_attr_float32
  
    def _build_py_custom_op(self,
                            op_type: str,
                            n_inputs: int,
                            n_outputs: int,
                            on_init,
                            on_compute):
        """通用的Python自定义算子构造器

        Args:
            op_type: 算子类型名(字符串)
            n_inputs: 输入个数
            n_outputs: 输出个数
            on_init: 回调,签名 on_init(op_ctx_p, read_attr_int64, read_attr_str) -> state
            on_compute: 回调,签名 on_compute(op_ctx_p, inputs_p, outputs_p, state) -> int(0成功)
        Returns:
            (RKNN_CustomOp对象, 回调tuple)
        """
        @self._CB_SIG
        def _py_init(op_ctx_p, inputs_p, n_inputs_p, outputs_p, n_outputs_p):
            try:
                # 允许无需提前读取属性
                runtime = self.runtime.rknn_base.rknn_runtime
                read_attr_int64, read_attr_str, read_attr_float32 = self._create_attr_readers(runtime.lib.rknn_custom_op_get_op_attr)
                user_state = on_init(op_ctx_p, read_attr_int64, read_attr_str, read_attr_float32)
                # 为该实例分配唯一ID, 并写入priv_data
                if not hasattr(self, "_custom_op_states"):
                    self._custom_op_states = {}
                if not hasattr(self, "_next_custom_op_id"):
                    self._next_custom_op_id = 1
                inst_id = int(self._next_custom_op_id)
                self._next_custom_op_id += 1
                # 保存Python侧状态
                self._custom_op_states[inst_id] = user_state
                # 将实例ID写入priv_data
                try:
                    op_ctx_p.contents.priv_data = ctypes.c_void_p(inst_id)
                except Exception:
                    # 回退: 直接写入整数
                    op_ctx_p.contents.priv_data = inst_id
                return 0
            except Exception as e:
                logger.error(f"{op_type} init失败: {e}")
                return -1

        @self._CB_SIG
        def _py_prepare(op_ctx_p, inputs_p, n_inputs_p, outputs_p, n_outputs_p):
            return 0

        @self._CB_SIG
        def _py_compute(op_ctx_p, inputs_p, n_inputs_p, outputs_p, n_outputs_p):
            try:
                if n_inputs_p != n_inputs or n_outputs_p != n_outputs:
                    return -1
                # 通过priv_data取回该实例的状态
                try:
                    inst_id = int(op_ctx_p.contents.priv_data) if op_ctx_p.contents.priv_data else 0
                except Exception:
                    inst_id = 0
                user_state = None
                if hasattr(self, "_custom_op_states") and inst_id in self._custom_op_states:
                    user_state = self._custom_op_states.get(inst_id)
                else:
                    logger.error(f"{op_type} compute失败: 找不到实例状态, inst_id={inst_id}")
                    return -1
                return on_compute(op_ctx_p, inputs_p, outputs_p, user_state)
            except Exception as e:
                logger.error(f"{op_type} compute失败: {e}")
                import traceback
                logger.error(f"{op_type} compute失败: {traceback.format_exc()}")
                return -1

        @self._DESTROY_SIG
        def _py_destroy(op_ctx_p):
            try:
                # 清理该实例的状态
                try:
                    inst_id = int(op_ctx_p.contents.priv_data) if op_ctx_p.contents.priv_data else 0
                except Exception:
                    inst_id = 0
                if hasattr(self, "_custom_op_states") and inst_id in self._custom_op_states:
                    del self._custom_op_states[inst_id]
                # 将priv_data清空
                try:
                    op_ctx_p.contents.priv_data = ctypes.c_void_p(0)
                except Exception:
                    op_ctx_p.contents.priv_data = 0
                return 0
            except Exception:
                return -1

        op = self._RKNN_CustomOp()
        op.version = 1
        op.target = self._RKNN_TARGET_TYPE_CPU
        op.op_type = op_type.encode("utf-8")
        op.cl_kernel_name = b""
        op.cl_kernel_source = None
        op.cl_source_size = 0
        op.cl_build_options = b""
        op.init = _py_init
        op.prepare = _py_prepare
        op.compute = _py_compute
        op.compute_native = self._CB_SIG()  # NULL
        op.destroy = _py_destroy

        return op, (_py_init, _py_prepare, _py_compute, _py_destroy)


    def _tensor_to_numpy(self, rknn_tensor):
        """将 RKNN_CustomOpTensor 转换为 Numpy 数组视图"""
        # 确定Numpy数据类型
        # 您可以扩展这个映射
        dtype_map = {
            self._RKNN_TENSOR_FLOAT32: (ctypes.c_float, np.float32),
            self._RKNN_TENSOR_UINT8: (ctypes.c_uint8, np.uint8),
            self._RKNN_TENSOR_INT64: (ctypes.c_int64, np.int64),
        }
        c_type, np_dtype = dtype_map.get(rknn_tensor.attr.type, (None, None))
        if c_type is None:
            raise TypeError(f"不支持的RKNN张量类型: {rknn_tensor.attr.type}")

        # 获取内存地址和形状
        addr = (rknn_tensor.mem.virt_addr or 0) + int(rknn_tensor.mem.offset)
        ptr = ctypes.cast(addr, ctypes.POINTER(c_type))
        shape = tuple(rknn_tensor.attr.dims[i] for i in range(rknn_tensor.attr.n_dims))
        
        # 创建Numpy数组视图
        return np.ctypeslib.as_array(ptr, shape=shape)


    def _create_onnxscript_op_creator(self,
                                      op_type: str,
                                      # 现在接收一个"函数模板构造器"
                                      onnxscript_func_builder,
                                      n_inputs: int,
                                      n_outputs: int,
                                      attributes: dict = {},
                                      constants: dict = {}):
        """
        一个高阶工厂函数，用于创建基于ONNXScript的自定义算子构造器。
        它在 on_init 阶段动态生成最终的 onnxscript 计算函数。

        Args:
            op_type (str): 算子类型名。
            onnxscript_func_builder: 一个函数，它接收所有属性和常量作为关键字参数，
                                     并返回一个编译好的 onnxscript 函数。
                                     例如: def builder(mean, scale):
                                               @onnxscript.script()
                                               def compute(like):
                                                  return opset.RandomNormalLike(like, mean=mean, scale=scale)
                                               return compute
            attributes (dict): 从模型中读取的属性字典。
            constants (dict): 编译时常量字典。
            n_inputs (int): 输入个数。
            n_outputs (int): 输出个数。
        """

        def creator_func():
            def on_init(op_ctx_p, read_i64, read_s, read_f32):
                # 1. 读取所有动态属性
                attr_values = {}
                for name, (attr_type, default) in attributes.items():
                    if attr_type == 'int64':
                        attr_values[name] = read_i64(op_ctx_p, name, default)
                    elif attr_type == 'str':
                        attr_values[name] = read_s(op_ctx_p, name, default)
                    elif attr_type == 'float32':
                        attr_values[name] = read_f32(op_ctx_p, name, default)
                    else:
                        raise ValueError(f"不支持的属性类型: {attr_type}")

                # 2. 合并常量和属性
                final_kwargs = {**constants, **attr_values}

                # 3. 动态构建 onnxscript 函数！ <<<<< 核心修改
                #    这确保了所有属性值都作为常量被闭包捕获
                compute_func = onnxscript_func_builder(**final_kwargs)

                # 4. 将最终生成的、已编译的函数存入 state
                return {"compute_func": compute_func}

            def on_compute(op_ctx_p, inputs_p, outputs_p, state):
                compute_func = state["compute_func"]

                input_nps = [self._tensor_to_numpy(inputs_p[i]) for i in range(n_inputs)]
                output_nps = [self._tensor_to_numpy(outputs_p[i]) for i in range(n_outputs)]
                
                results = compute_func(*input_nps)

                if n_outputs == 1:
                    result_val = results[0] if isinstance(results, tuple) else results
                    output_nps[0][...] = result_val
                else:
                    for i in range(n_outputs):
                        output_nps[i][...] = results[i]
                
                return 0

            return self._build_py_custom_op(
                op_type=op_type,
                n_inputs=n_inputs,
                n_outputs=n_outputs,
                on_init=on_init,
                on_compute=on_compute
            )
        
        return creator_func

    def _create_gridsample_op(self):
        import onnxscript
        from onnxscript import opset17 as opset

        def grid_sample_builder(align_corners, mode, padding_mode):
            @onnxscript.script()
            def grid_sample_compute(X, G):
                return opset.GridSample(X, G, align_corners=align_corners, mode=mode, padding_mode=padding_mode)
            return grid_sample_compute

        grid_sample_creator = self._create_onnxscript_op_creator(
            op_type="GridSample",
            onnxscript_func_builder=grid_sample_builder, # << 传入 builder
            attributes={
                "align_corners": ("int64", 0),
                "mode": ("str", "bilinear"),
                "padding_mode": ("str", "zeros"),
            },
            n_inputs = 2,
            n_outputs = 1
        )
        return grid_sample_creator

    def _create_scatterelements_op(self):
        import onnxscript
        from onnxscript import opset17 as opset
        
        @onnxscript.script()
        def scatter_elements_compute(data, indices, updates):
            indices_i64 = opset.Cast(indices, to=onnxscript.INT64.dtype)
            return opset.ScatterElements(data, indices_i64, updates)
        
        scatter_elements_creator = self._create_onnxscript_op_creator(
            op_type="ScatterElements",
            onnxscript_func_builder=lambda: scatter_elements_compute,
            n_inputs = 3,
            n_outputs = 1
        )
        return scatter_elements_creator

    def _create_randomnormallike_op(self):
        import onnxscript
        from onnxscript import opset17 as opset

        def random_normal_like_builder(mean, scale):
            @onnxscript.script()
            def random_normal_like_compute(like):
                return opset.RandomNormalLike(like, mean=mean, scale=scale)
            
            return random_normal_like_compute
        
        # 3. 使用新的工厂函数
        random_normal_like_creator = self._create_onnxscript_op_creator(
            op_type="RandomNormalLike",
            onnxscript_func_builder=random_normal_like_builder, # << 传入 builder
            attributes={
                "mean": ("float32", 0.0),
                "scale": ("float32", 1.0),
            },
            n_inputs = 1,
            n_outputs = 1
        )
        return random_normal_like_creator

    def _create_einsum_op(self): 
        import onnxscript
        from onnxscript import opset17 as opset

        def einsum_builder(equation):

            @onnxscript.script()
            def einsum_compute(in1, in2):
                return opset.Einsum(in1, in2, equation=equation)
            
            return einsum_compute
        
        # 3. 使用新的工厂函数
        einsum_creator = self._create_onnxscript_op_creator(
            op_type="Einsum",
            onnxscript_func_builder=einsum_builder, # << 传入 builder
            attributes={
                "equation": ("str", ""),
            },
            n_inputs = 2,
            n_outputs = 1
        )
        return einsum_creator

    def register_bundled_ops(self) -> None:
        """注册自定义操作"""
        if getattr(self, "_custom_ops_registered", False):
            return

        runtime = self.runtime.rknn_base.rknn_runtime
        lib = runtime.lib
        ctx = runtime.context

        try:
            _ = lib.rknn_register_custom_ops
            _ = lib.rknn_custom_op_get_op_attr
        except AttributeError as e:
            logger.debug(f"SDK不支持自定义算子注册: {e}")
            return

        self._init_custom_op_types()

        # 注意：插件库注册已在模型加载后由环境变量控制，不在此处重复触发

        # 算子创建函数的列表现在更加清晰
        op_creator_factories = [
            self._create_gridsample_op,
            self._create_scatterelements_op,
            self._create_randomnormallike_op,
            self._create_einsum_op,
            # self._create_my_custom_add_op, # 添加新算子非常简单
        ]

        ops_to_register = []
        all_callbacks = []

        for factory in op_creator_factories:
            try:
                # 调用工厂获得真正的构造器
                creator_func = factory()
                # 调用构造器生成算子实例
                op, callbacks = creator_func()
                ops_to_register.append(op)
                all_callbacks.extend(callbacks)
                logger.debug(f"成功创建自定义算子: {op.op_type.decode()}")
            except Exception as e:
                logger.warning(f"创建自定义算子失败: {e}", exc_info=True)

        if not ops_to_register:
            logger.debug("没有可注册的自定义算子")
            return

        # 创建一个ctypes数组以包含所有要注册的算子, 然后一次性注册
        num_ops = len(ops_to_register)
        op_array = (self._RKNN_CustomOp * num_ops)(*ops_to_register)
        ret = lib.rknn_register_custom_ops(ctx, op_array, num_ops)
        if ret != 0:
            logger.error(f"注册自定义算子失败, ret={ret} (可能是误报, 继续执行...)")
            # raise RuntimeError(f"rknn_register_custom_ops 失败, ret={ret}")

        logger.info(f"成功注册 {len(ops_to_register)} 个自定义算子")

        self._custom_ops_registered = True
        self._registered_ops = ops_to_register
        self._op_callbacks = all_callbacks

    def _load_and_register_plugin_op(self, so_path: str) -> bool:
        """加载单个插件库并注册其中的自定义算子。

        要求插件实现 get_rknn_custom_op()，返回 rknn_custom_op*。
        我们将该 C 指针直接传递给 rknn_register_custom_ops，避免复制。
        """
        if not os.path.isfile(so_path):
            logger.warning(f"插件库不存在: {so_path}")
            return False

        runtime = self.runtime.rknn_base.rknn_runtime
        lib = runtime.lib
        ctx = runtime.context

        # 根据平台位宽设置 rknn_context 的 ctypes 类型
        ContextCType = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32
        # 设置 rknn_register_custom_ops(ctx, op_ptr, num) 签名。第二参数按 void* 传递，避免结构体布局不一致
        try:
            lib.rknn_register_custom_ops.argtypes = [ContextCType, ctypes.c_void_p, ctypes.c_uint32]
            lib.rknn_register_custom_ops.restype = ctypes.c_int
        except Exception:
            pass

        # 加载插件
        try:
            handle = ctypes.CDLL(so_path)
        except Exception as e:
            logger.error(f"dlopen 失败: {so_path}, err={e}")
            return False

        # 获取 get_rknn_custom_op 符号
        try:
            get_sym = getattr(handle, "get_rknn_custom_op")
        except AttributeError:
            logger.error(f"插件缺少符号 get_rknn_custom_op: {so_path}")
            return False

        # 返回类型直接使用 void*，避免 Python 解析第三方结构体
        try:
            get_sym.argtypes = []
        except Exception:
            pass
        get_sym.restype = ctypes.c_void_p

        op_void_ptr = get_sym()
        if not op_void_ptr:
            logger.error(f"get_rknn_custom_op 返回空指针: {so_path}")
            return False

        # 直接使用原生指针注册（零拷贝）
        ctx_val = ContextCType(runtime.context)
        ret = lib.rknn_register_custom_ops(ctx_val, ctypes.c_void_p(op_void_ptr), 1)
        if ret != 0:
            logger.error(f"rknn_register_custom_ops 失败, ret={ret}, so={so_path} (可能是误报, 继续执行...)")
            # return False

        # 保留句柄，避免被垃圾回收卸载
        if not hasattr(self, "_plugin_handles"):
            self._plugin_handles = []
        self._plugin_handles.append(handle)
        logger.info(f"成功注册插件自定义算子: {so_path}")
        return True

    def register_plugin_ops(self, plugin_paths: List[str]) -> int:
        """按给定路径列表注册插件库中的自定义算子。返回成功数量。"""
        if not plugin_paths:
            return 0
        success = 0
        for path in plugin_paths:
            try:
                if self._load_and_register_plugin_op(path):
                    success += 1
            except Exception as e:
                logger.error(f"注册插件失败: {path}, err={e}")
        return success

    # 对外API：注册单个自定义算子插件库
    def register_custom_op_lib(self, path: str) -> bool:
        return self._load_and_register_plugin_op(path)

    # 对外API：扫描并注册 Linux 系统目录下所有插件库（Android 不处理）
    def register_system_custom_op_lib(self) -> int:
        if os.name != 'posix':
            return 0
        # 仅 Linux：RKNN 官方默认目录
        system_dir = "/usr/lib/rknpu/op_plugins/"
        if not os.path.isdir(system_dir):
            return 0
        try:
            entries = os.listdir(system_dir)
        except Exception:
            return 0
        so_list = []
        for name in entries:
            # 官方要求文件名以 librkcst_ 开头
            if name.startswith("librkcst_") and name.endswith('.so'):
                so_list.append(os.path.join(system_dir, name))
        return self.register_plugin_ops(so_list)