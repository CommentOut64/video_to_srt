"""
核心硬件检测服务
"""
import os
import shutil
import tempfile
import logging
from typing import List, Dict, Optional, Tuple

from app.models.hardware_models import HardwareInfo, OptimizationConfig

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True  
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False


class CoreHardwareDetector:
    """核心硬件检测器，专注于影响转录性能的关键硬件信息"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect(self) -> HardwareInfo:
        """执行全面的硬件检测"""
        try:
            # 并行检测各个硬件组件
            gpu_info = self._detect_gpu()
            cpu_info = self._detect_cpu() 
            memory_info = self._detect_memory()
            storage_info = self._detect_storage()
            
            # 合并检测结果
            hardware = HardwareInfo(
                # GPU信息
                gpu_count=gpu_info.get("gpu_count", 0),
                gpu_memory_mb=gpu_info.get("gpu_memory_mb", []),
                cuda_available=gpu_info.get("cuda_available", False),
                gpu_name=gpu_info.get("gpu_name"),
                
                # CPU信息
                cpu_cores=cpu_info.get("cpu_cores", 1),
                cpu_threads=cpu_info.get("cpu_threads", 1),
                cpu_name=cpu_info.get("cpu_name"),
                cpu_max_frequency=cpu_info.get("cpu_max_frequency"),
                
                # 内存信息
                memory_total_mb=memory_info.get("memory_total_mb", 0),
                memory_available_mb=memory_info.get("memory_available_mb", 0),
                
                # 存储信息
                temp_space_available_gb=storage_info.get("temp_space_available_gb", 0)
            )
            
            self.logger.info(f"硬件检测完成: GPU={hardware.gpu_count}个, "
                           f"CPU={hardware.cpu_cores}核/{hardware.cpu_threads}线程, "
                           f"内存={hardware.memory_total_mb}MB, "
                           f"临时空间={hardware.temp_space_available_gb}GB")
            return hardware
            
        except Exception as e:
            self.logger.error(f"硬件检测失败: {e}")
            return self._get_fallback_hardware_info()
    
    def _detect_gpu(self) -> Dict:
        """检测GPU核心信息"""
        gpu_info = {
            "gpu_count": 0,
            "gpu_memory_mb": [],
            "cuda_available": False,
            "gpu_name": None  # 添加GPU名称字段
        }
        
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch未安装，跳过GPU检测")
            return gpu_info
        
        try:
            # 检测CUDA可用性
            cuda_available = torch.cuda.is_available()
            gpu_info["cuda_available"] = cuda_available
            
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                gpu_info["gpu_count"] = gpu_count
                
                # 检测每个GPU的显存和名称
                gpu_memory_list = []
                gpu_names = []
                for i in range(gpu_count):
                    try:
                        # 获取GPU显存信息（MB）
                        device_props = torch.cuda.get_device_properties(i)
                        memory_bytes = device_props.total_memory
                        memory_mb = memory_bytes // (1024 * 1024)
                        gpu_memory_list.append(memory_mb)
                        
                        # 获取GPU名称
                        gpu_name = device_props.name
                        gpu_names.append(gpu_name)
                        
                        self.logger.info(f"检测到GPU {i}: {gpu_name}, {memory_mb}MB显存")
                    except Exception as e:
                        self.logger.warning(f"检测GPU {i}显存失败: {e}")
                        gpu_memory_list.append(0)
                        gpu_names.append("Unknown GPU")
                
                gpu_info["gpu_memory_mb"] = gpu_memory_list
                # 使用第一个GPU的名称作为主要GPU名称
                gpu_info["gpu_name"] = gpu_names[0] if gpu_names else None
            else:
                self.logger.info("CUDA不可用，将使用CPU模式")
                
        except Exception as e:
            self.logger.error(f"GPU检测失败: {e}")
        
        return gpu_info
    
    def _detect_cpu(self) -> Dict:
        """检测CPU关键信息"""
        cpu_info = {
            "cpu_cores": 1,
            "cpu_threads": 1,
            "cpu_name": None,  # 添加CPU名称字段
            "cpu_max_frequency": None  # 添加最大频率字段
        }
        
        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil未安装，使用默认CPU配置")
            return cpu_info
        
        try:
            # 获取物理核心数和逻辑线程数
            physical_cores = psutil.cpu_count(logical=False) or 1
            logical_threads = psutil.cpu_count(logical=True) or 1
            
            cpu_info["cpu_cores"] = physical_cores
            cpu_info["cpu_threads"] = logical_threads
            
            # 尝试获取CPU名称和频率信息
            try:
                # 获取CPU频率信息
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    cpu_info["cpu_max_frequency"] = cpu_freq.max
                    
                # 在Windows上，尝试从注册表获取CPU名称
                import platform
                if platform.system() == "Windows":
                    try:
                        import winreg
                        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                          r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
                            cpu_info["cpu_name"] = cpu_name
                    except Exception as e:
                        self.logger.debug(f"无法从注册表获取CPU名称: {e}")
                        
                # 在Linux上，尝试从 /proc/cpuinfo 获取
                elif platform.system() == "Linux":
                    try:
                        with open("/proc/cpuinfo", "r") as f:
                            for line in f:
                                if line.startswith("model name"):
                                    cpu_name = line.split(":")[1].strip()
                                    cpu_info["cpu_name"] = cpu_name
                                    break
                    except Exception as e:
                        self.logger.debug(f"无法从 /proc/cpuinfo 获取CPU名称: {e}")
                        
            except Exception as e:
                self.logger.debug(f"获取CPU详细信息失败: {e}")
            
            self.logger.info(f"检测到CPU: {cpu_info.get('cpu_name', 'Unknown')}, "
                           f"{physical_cores}个物理核心, {logical_threads}个逻辑线程")
            
        except Exception as e:
            self.logger.error(f"CPU检测失败: {e}")
        
        return cpu_info
    
    def _detect_memory(self) -> Dict:
        """检测系统内存信息"""
        memory_info = {
            "memory_total_mb": 0,
            "memory_available_mb": 0
        }
        
        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil未安装，无法检测内存信息")
            return memory_info
        
        try:
            # 获取内存信息
            memory = psutil.virtual_memory()
            total_mb = memory.total // (1024 * 1024)
            available_mb = memory.available // (1024 * 1024)
            
            memory_info["memory_total_mb"] = total_mb
            memory_info["memory_available_mb"] = available_mb
            
            usage_percent = round((1 - available_mb / max(1, total_mb)) * 100, 1)
            self.logger.info(f"检测到内存: 总计{total_mb}MB, 可用{available_mb}MB, 使用率{usage_percent}%")
            
        except Exception as e:
            self.logger.error(f"内存检测失败: {e}")
        
        return memory_info
    
    def _detect_storage(self) -> Dict:
        """检测临时存储空间"""
        storage_info = {
            "temp_space_available_gb": 0
        }
        
        try:
            # 获取临时目录的可用空间
            temp_dir = tempfile.gettempdir()
            if os.path.exists(temp_dir):
                total, used, free = shutil.disk_usage(temp_dir)
                free_gb = free // (1024 * 1024 * 1024)
                storage_info["temp_space_available_gb"] = free_gb
                
                self.logger.info(f"检测到临时存储空间: {free_gb}GB 可用于 {temp_dir}")
            else:
                self.logger.warning("无法访问临时目录")
                
        except Exception as e:
            self.logger.error(f"存储检测失败: {e}")
        
        return storage_info
    
    def _get_fallback_hardware_info(self) -> HardwareInfo:
        """获取保守的硬件配置作为后备方案"""
        return HardwareInfo(
            gpu_count=0,
            gpu_memory_mb=[],
            cuda_available=False,
            cpu_cores=1,
            cpu_threads=1,
            memory_total_mb=4096,  # 假设4GB内存
            memory_available_mb=2048,
            temp_space_available_gb=10  # 假设10GB可用空间
        )


class CoreOptimizer:
    """基于硬件信息的核心优化决策器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_optimization_config(self, hardware: HardwareInfo) -> OptimizationConfig:
        """根据硬件信息生成优化配置"""
        config = OptimizationConfig()
        
        # 计算最优批处理大小
        config.batch_size = self._calculate_optimal_batch_size(hardware)
        
        # 计算最优并发数
        config.concurrency = self._calculate_optimal_concurrency(hardware)
        
        # 决定是否使用内存映射
        config.use_memory_mapping = self._should_use_memory_mapping(hardware)
        
        # 生成CPU亲和性核心列表
        config.cpu_affinity_cores = self._get_cpu_affinity_cores(hardware)
        
        # 推荐设备选择
        config.recommended_device = self._get_recommended_device(hardware)
        
        # 推荐模型选择
        config.recommended_model = self._get_recommended_model(hardware)
        
        self.logger.info(f"生成优化配置: 批处理={config.batch_size}, "
                        f"并发={config.concurrency}, 设备={config.recommended_device}, "
                        f"模型={config.recommended_model}, "
                        f"CPU绑定={len(config.cpu_affinity_cores)}核心")
        
        return config
    
    def _calculate_optimal_batch_size(self, hardware: HardwareInfo) -> int:
        """根据GPU显存计算最优批处理大小"""
        if not hardware.cuda_available or not hardware.gpu_memory_mb:
            return 8  # CPU模式保守批处理大小
        
        # 取最大GPU显存计算
        max_gpu_memory = max(hardware.gpu_memory_mb)
        
        # 每500MB显存支持batch_size=1的经验公式
        batch_size = min(64, max(4, max_gpu_memory // 500))
        
        return batch_size
    
    def _calculate_optimal_concurrency(self, hardware: HardwareInfo) -> int:
        """计算最优并发数"""
        # 基于CPU核心数和GPU显存综合计算
        cpu_based = max(1, hardware.cpu_cores // 2)
        
        if hardware.cuda_available and hardware.gpu_memory_mb:
            # GPU模式：显存越大，并发能力越强
            max_gpu_memory = max(hardware.gpu_memory_mb)
            gpu_based = max(1, max_gpu_memory // 2000)  # 每2GB显存支持1个并发
            return min(cpu_based, gpu_based, 16)  # 上限16个并发
        else:
            # CPU模式：保守并发
            return min(cpu_based, 4)
    
    def _should_use_memory_mapping(self, hardware: HardwareInfo) -> bool:
        """决定是否启用内存映射"""
        # 内存少于8GB时强制启用内存映射以节省内存
        return hardware.memory_total_mb < 8000
    
    def _get_cpu_affinity_cores(self, hardware: HardwareInfo) -> List[int]:
        """生成CPU亲和性核心列表"""
        # 使用前一半的CPU核心进行绑定
        core_count = min(hardware.cpu_cores // 2, 8)  # 最多绑定8个核心
        return list(range(core_count))
    
    def _get_recommended_device(self, hardware: HardwareInfo) -> str:
        """推荐最佳设备"""
        if hardware.cuda_available and hardware.gpu_memory_mb:
            max_gpu_memory = max(hardware.gpu_memory_mb)
            if max_gpu_memory >= 4000:  # 4GB以上显存推荐GPU
                return "cuda"
        
        return "cpu"
    
    def _get_recommended_model(self, hardware: HardwareInfo) -> str:
        """根据硬件配置推荐最佳模型"""
        if hardware.cuda_available and hardware.gpu_memory_mb:
            max_gpu_memory = max(hardware.gpu_memory_mb)
            # 根据GPU显存推荐模型
            if max_gpu_memory >= 12000:  # 12GB+显存
                return "large"
            elif max_gpu_memory >= 8000:  # 8GB+显存
                return "medium"
            elif max_gpu_memory >= 4000:  # 4GB+显存
                return "small"
            else:  # 4GB以下显存
                return "tiny"
        else:
            # CPU模式根据内存推荐
            if hardware.memory_total_mb >= 16000:  # 16GB+内存
                return "medium"
            elif hardware.memory_total_mb >= 8000:  # 8GB+内存
                return "small"
            else:  # 8GB以下内存
                return "tiny"


# 单例实例
_detector_instance: Optional[CoreHardwareDetector] = None
_optimizer_instance: Optional[CoreOptimizer] = None


def get_hardware_detector() -> CoreHardwareDetector:
    """获取硬件检测器实例"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = CoreHardwareDetector()
    return _detector_instance


def get_hardware_optimizer() -> CoreOptimizer:
    """获取硬件优化器实例"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = CoreOptimizer()
    return _optimizer_instance