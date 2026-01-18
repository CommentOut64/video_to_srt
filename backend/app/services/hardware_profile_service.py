"""
硬件能力提供者（HardwareProfileProvider）

统一封装硬件检测、优化决策、ONNX 线程预算与 CPU 亲和性管理，
对上层仅暴露稳定接口，避免直接依赖底层实现细节。
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import platform
import shutil
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple

from app.models.hardware_models import HardwareInfo, OptimizationConfig, CPUAffinityConfig

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

logger = logging.getLogger(__name__)


class CoreHardwareDetector:
    """核心硬件检测器，专注于影响转录性能的关键硬件信息"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def detect(self) -> HardwareInfo:
        """执行全面的硬件检测"""
        try:
            gpu_info = self._detect_gpu()
            cpu_info = self._detect_cpu()
            memory_info = self._detect_memory()
            storage_info = self._detect_storage()

            hardware = HardwareInfo(
                gpu_count=gpu_info.get("gpu_count", 0),
                gpu_memory_mb=gpu_info.get("gpu_memory_mb", []),
                cuda_available=gpu_info.get("cuda_available", False),
                gpu_name=gpu_info.get("gpu_name"),
                cpu_cores=cpu_info.get("cpu_cores", 1),
                cpu_threads=cpu_info.get("cpu_threads", 1),
                cpu_name=cpu_info.get("cpu_name"),
                cpu_max_frequency=cpu_info.get("cpu_max_frequency"),
                memory_total_mb=memory_info.get("memory_total_mb", 0),
                memory_available_mb=memory_info.get("memory_available_mb", 0),
                temp_space_available_gb=storage_info.get("temp_space_available_gb", 0),
            )

            self.logger.info(
                "硬件检测完成: GPU=%s个, CPU=%s核/%s线程, 内存=%sMB, 临时空间=%sGB",
                hardware.gpu_count,
                hardware.cpu_cores,
                hardware.cpu_threads,
                hardware.memory_total_mb,
                hardware.temp_space_available_gb,
            )
            return hardware

        except Exception as exc:
            self.logger.error("硬件检测失败: %s", exc)
            return self._get_fallback_hardware_info()

    def _detect_gpu(self) -> Dict[str, Any]:
        """检测GPU核心信息"""
        gpu_info: Dict[str, Any] = {
            "gpu_count": 0,
            "gpu_memory_mb": [],
            "cuda_available": False,
            "gpu_name": None,
        }

        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch未安装，跳过GPU检测")
            return gpu_info

        try:
            cuda_available = torch.cuda.is_available()
            gpu_info["cuda_available"] = cuda_available

            if cuda_available:
                gpu_count = torch.cuda.device_count()
                gpu_info["gpu_count"] = gpu_count

                gpu_memory_list = []
                gpu_names = []
                for i in range(gpu_count):
                    try:
                        device_props = torch.cuda.get_device_properties(i)
                        memory_mb = device_props.total_memory // (1024 * 1024)
                        gpu_memory_list.append(memory_mb)
                        gpu_names.append(device_props.name)
                        self.logger.info(
                            "检测到GPU %s: %s, %sMB显存",
                            i,
                            device_props.name,
                            memory_mb,
                        )
                    except Exception as exc:
                        self.logger.warning("检测GPU %s显存失败: %s", i, exc)
                        gpu_memory_list.append(0)
                        gpu_names.append("Unknown GPU")

                gpu_info["gpu_memory_mb"] = gpu_memory_list
                gpu_info["gpu_name"] = gpu_names[0] if gpu_names else None
            else:
                self.logger.info("CUDA不可用，将使用CPU模式")

        except Exception as exc:
            self.logger.error("GPU检测失败: %s", exc)

        return gpu_info

    def _detect_cpu(self) -> Dict[str, Any]:
        """检测CPU关键信息"""
        cpu_info: Dict[str, Any] = {
            "cpu_cores": 1,
            "cpu_threads": 1,
            "cpu_name": None,
            "cpu_max_frequency": None,
        }

        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil未安装，使用默认CPU配置")
            return cpu_info

        try:
            physical_cores = psutil.cpu_count(logical=False) or 1
            logical_threads = psutil.cpu_count(logical=True) or 1

            cpu_info["cpu_cores"] = physical_cores
            cpu_info["cpu_threads"] = logical_threads

            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    cpu_info["cpu_max_frequency"] = cpu_freq.max

                if platform.system() == "Windows":
                    try:
                        import winreg
                        with winreg.OpenKey(
                            winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                        ) as key:
                            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
                            cpu_info["cpu_name"] = cpu_name
                    except Exception as exc:
                        self.logger.debug("无法从注册表获取CPU名称: %s", exc)
                elif platform.system() == "Linux":
                    try:
                        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
                            for line in handle:
                                if line.startswith("model name"):
                                    cpu_name = line.split(":")[1].strip()
                                    cpu_info["cpu_name"] = cpu_name
                                    break
                    except Exception as exc:
                        self.logger.debug("无法从 /proc/cpuinfo 获取CPU名称: %s", exc)
            except Exception as exc:
                self.logger.debug("获取CPU详细信息失败: %s", exc)

            self.logger.info(
                "检测到CPU: %s, %s个物理核心, %s个逻辑线程",
                cpu_info.get("cpu_name", "Unknown"),
                physical_cores,
                logical_threads,
            )

        except Exception as exc:
            self.logger.error("CPU检测失败: %s", exc)

        return cpu_info

    def _detect_memory(self) -> Dict[str, Any]:
        """检测系统内存信息"""
        memory_info: Dict[str, Any] = {
            "memory_total_mb": 0,
            "memory_available_mb": 0,
        }

        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil未安装，无法检测内存信息")
            return memory_info

        try:
            memory = psutil.virtual_memory()
            total_mb = memory.total // (1024 * 1024)
            available_mb = memory.available // (1024 * 1024)

            memory_info["memory_total_mb"] = total_mb
            memory_info["memory_available_mb"] = available_mb

            usage_percent = round((1 - available_mb / max(1, total_mb)) * 100, 1)
            self.logger.info(
                "检测到内存: 总计%sMB, 可用%sMB, 使用率%s%%",
                total_mb,
                available_mb,
                usage_percent,
            )

        except Exception as exc:
            self.logger.error("内存检测失败: %s", exc)

        return memory_info

    def _detect_storage(self) -> Dict[str, Any]:
        """检测临时存储空间"""
        storage_info: Dict[str, Any] = {
            "temp_space_available_gb": 0,
        }

        try:
            temp_dir = tempfile.gettempdir()
            if os.path.exists(temp_dir):
                total, used, free = shutil.disk_usage(temp_dir)
                free_gb = free // (1024 * 1024 * 1024)
                storage_info["temp_space_available_gb"] = free_gb

                self.logger.info("检测到临时存储空间: %sGB 可用于 %s", free_gb, temp_dir)
            else:
                self.logger.warning("无法访问临时目录")

        except Exception as exc:
            self.logger.error("存储检测失败: %s", exc)

        return storage_info

    @staticmethod
    def _get_fallback_hardware_info() -> HardwareInfo:
        """获取保守的硬件配置作为后备方案"""
        return HardwareInfo(
            gpu_count=0,
            gpu_memory_mb=[],
            cuda_available=False,
            cpu_cores=1,
            cpu_threads=1,
            memory_total_mb=4096,
            memory_available_mb=2048,
            temp_space_available_gb=10,
        )


class CoreOptimizer:
    """基于硬件信息的核心优化决策器"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def get_optimization_config(self, hardware: HardwareInfo) -> OptimizationConfig:
        """根据硬件信息生成优化配置"""
        config = OptimizationConfig()

        config.batch_size = self._calculate_optimal_batch_size(hardware)
        config.concurrency = self._calculate_optimal_concurrency(hardware)
        config.use_memory_mapping = self._should_use_memory_mapping(hardware)
        config.cpu_affinity_cores = self._get_cpu_affinity_cores(hardware)
        config.recommended_device = self._get_recommended_device(hardware)
        config.recommended_model = self._get_recommended_model(hardware)

        self.logger.info(
            "生成优化配置: 批处理=%s, 并发=%s, 设备=%s, 模型=%s, CPU绑定=%s核心",
            config.batch_size,
            config.concurrency,
            config.recommended_device,
            config.recommended_model,
            len(config.cpu_affinity_cores),
        )

        return config

    @staticmethod
    def _calculate_optimal_batch_size(hardware: HardwareInfo) -> int:
        """根据GPU显存计算最优批处理大小"""
        if not hardware.cuda_available or not hardware.gpu_memory_mb:
            return 8

        max_gpu_memory = max(hardware.gpu_memory_mb)
        batch_size = min(64, max(4, max_gpu_memory // 500))

        return batch_size

    @staticmethod
    def _calculate_optimal_concurrency(hardware: HardwareInfo) -> int:
        """计算最优并发数"""
        cpu_based = max(1, hardware.cpu_cores // 2)

        if hardware.cuda_available and hardware.gpu_memory_mb:
            max_gpu_memory = max(hardware.gpu_memory_mb)
            gpu_based = max(1, max_gpu_memory // 2000)
            return min(cpu_based, gpu_based, 16)

        return min(cpu_based, 4)

    @staticmethod
    def _should_use_memory_mapping(hardware: HardwareInfo) -> bool:
        """决定是否启用内存映射"""
        return hardware.memory_total_mb < 8000

    @staticmethod
    def _get_cpu_affinity_cores(hardware: HardwareInfo) -> List[int]:
        """生成CPU亲和性核心列表"""
        core_count = min(hardware.cpu_cores // 2, 8)
        return list(range(core_count))

    @staticmethod
    def _get_recommended_device(hardware: HardwareInfo) -> str:
        """推荐最佳设备"""
        if hardware.cuda_available and hardware.gpu_memory_mb:
            max_gpu_memory = max(hardware.gpu_memory_mb)
            if max_gpu_memory >= 4000:
                return "cuda"

        return "cpu"

    @staticmethod
    def _get_recommended_model(hardware: HardwareInfo) -> str:
        """根据硬件配置推荐最佳模型"""
        if hardware.cuda_available and hardware.gpu_memory_mb:
            max_gpu_memory = max(hardware.gpu_memory_mb)
            if max_gpu_memory >= 12000:
                return "large"
            if max_gpu_memory >= 8000:
                return "medium"
            if max_gpu_memory >= 4000:
                return "small"
            return "tiny"

        if hardware.memory_total_mb >= 16000:
            return "medium"
        if hardware.memory_total_mb >= 8000:
            return "small"
        return "tiny"


class CPUAffinityManager:
    """
    CPU亲和性管理器

    设计模式：策略模式，封装不同策略的核心选择与绑定逻辑。
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.original_affinity: Optional[List[int]] = None
        self.is_supported = PSUTIL_AVAILABLE and hasattr(psutil.Process(), "cpu_affinity")

        if not self.is_supported:
            self.logger.warning("CPU亲和性功能不可用：psutil未安装或系统不支持")

    def get_system_info(self) -> Dict[str, Any]:
        """获取系统CPU信息"""
        if not self.is_supported:
            return {"supported": False, "reason": "psutil not available"}

        try:
            cpu_count = psutil.cpu_count(logical=True)
            physical_count = psutil.cpu_count(logical=False)
            current_affinity = psutil.Process().cpu_affinity()

            return {
                "supported": True,
                "logical_cores": cpu_count,
                "physical_cores": physical_count,
                "current_affinity": current_affinity,
                "platform": platform.system(),
            }
        except Exception as exc:
            return {"supported": False, "error": str(exc)}

    def calculate_optimal_cores(
        self,
        strategy: str = "auto",
        custom_cores: Optional[List[int]] = None,
        exclude_cores: Optional[List[int]] = None,
    ) -> List[int]:
        """计算最佳CPU核心分配"""
        if not self.is_supported:
            return []

        try:
            cpu_count = psutil.cpu_count(logical=True)
            available_cores = list(range(cpu_count))

            if exclude_cores:
                available_cores = [c for c in available_cores if c not in exclude_cores]

            if strategy == "custom" and custom_cores:
                return [c for c in custom_cores if c in available_cores]
            if strategy == "half":
                half_count = max(1, len(available_cores) // 2)
                return available_cores[:half_count]

            if cpu_count <= 4:
                return available_cores
            if cpu_count <= 8:
                return available_cores[:-1]
            use_count = max(1, int(cpu_count * 0.75))
            return available_cores[:use_count]

        except Exception as exc:
            self.logger.error("计算最佳核心失败: %s", exc)
            return []

    def apply_affinity_cores(self, cores: List[int], context: str = "") -> bool:
        """应用指定CPU核心列表"""
        if not self.is_supported:
            return False
        if not cores:
            self.logger.warning("未找到可用的CPU核心进行绑定")
            return False

        try:
            if self.original_affinity is None:
                self.original_affinity = psutil.Process().cpu_affinity()

            psutil.Process().cpu_affinity(cores)
            sys_info = self.get_system_info()
            self.logger.info(
                "CPU亲和性设置成功%s: 绑定核心=%s, 系统核心数=%s",
                f" ({context})" if context else "",
                cores,
                sys_info.get("logical_cores", "?"),
            )
            return True
        except Exception as exc:
            self.logger.error("CPU亲和性设置失败: %s", exc)
            return False

    def apply_cpu_affinity(self, config: CPUAffinityConfig, context: str = "") -> bool:
        """应用CPU亲和性设置"""
        if not getattr(config, "is_enabled", config.enabled) or not self.is_supported:
            return False

        target_cores = self.calculate_optimal_cores(
            strategy=config.strategy,
            custom_cores=config.custom_cores,
            exclude_cores=config.exclude_cores,
        )
        return self.apply_affinity_cores(target_cores, context=context)

    def restore_cpu_affinity(self, context: str = "") -> bool:
        """恢复原始CPU亲和性设置"""
        if not self.is_supported or self.original_affinity is None:
            return False

        try:
            psutil.Process().cpu_affinity(self.original_affinity)
            self.logger.info(
                "已恢复CPU亲和性设置%s: %s",
                f" ({context})" if context else "",
                self.original_affinity,
            )
            return True
        except Exception as exc:
            self.logger.error("恢复CPU亲和性失败: %s", exc)
            return False


class CPUArchitectureDetector:
    """CPU 架构检测器"""

    @staticmethod
    def detect_cpu_vendor(cpu_name: Optional[str] = None) -> str:
        """检测 CPU 厂商"""
        try:
            if cpu_name:
                cpu_name_lower = cpu_name.lower()
                if "intel" in cpu_name_lower:
                    return "intel"
                if "amd" in cpu_name_lower:
                    return "amd"

            cpu_info = platform.processor().lower()
            if "intel" in cpu_info:
                return "intel"
            if "amd" in cpu_info:
                return "amd"

            if platform.system() == "Windows":
                try:
                    import winreg
                    with winreg.OpenKey(
                        winreg.HKEY_LOCAL_MACHINE,
                        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                    ) as key:
                        cpu_name_reg = winreg.QueryValueEx(key, "ProcessorNameString")[0].lower()
                        if "intel" in cpu_name_reg:
                            return "intel"
                        if "amd" in cpu_name_reg:
                            return "amd"
                except Exception:
                    pass

            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
                    cpuinfo = handle.read().lower()
                    if "genuineintel" in cpuinfo:
                        return "intel"
                    if "authenticamd" in cpuinfo:
                        return "amd"
            except Exception:
                pass

            return "unknown"

        except Exception as exc:
            logger.warning("检测 CPU 厂商失败: %s", exc)
            return "unknown"

    @staticmethod
    def is_intel_hybrid_architecture(cpu_name: Optional[str] = None) -> bool:
        """检测是否为 Intel 混合架构（12代及以后）"""
        if not cpu_name:
            return False

        cpu_name_lower = cpu_name.lower()
        hybrid_indicators = [
            "12th gen",
            "13th gen",
            "14th gen",
            "15th gen",
            "core ultra",
            "-12",
            "-13",
            "-14",
            "-15",
            "12100",
            "12400",
            "12600",
            "12700",
            "12900",
            "13100",
            "13400",
            "13600",
            "13700",
            "13900",
            "14100",
            "14400",
            "14600",
            "14700",
            "14900",
        ]

        return any(indicator in cpu_name_lower for indicator in hybrid_indicators)

    @staticmethod
    def get_physical_cores(hardware_info: Optional[HardwareInfo] = None) -> int:
        """获取物理核心数（不含超线程）"""
        try:
            if hardware_info and hasattr(hardware_info, "cpu_cores"):
                return hardware_info.cpu_cores

            if PSUTIL_AVAILABLE:
                return psutil.cpu_count(logical=False) or 1
        except Exception:
            pass

        logical_cores = multiprocessing.cpu_count()
        return max(1, logical_cores // 2)

    @staticmethod
    def detect_intel_p_cores(cpu_name: Optional[str], physical_cores: Optional[int]) -> int:
        """检测 Intel P-Core 数量（性能核）"""
        if not cpu_name or not physical_cores:
            return physical_cores or 1

        if not CPUArchitectureDetector.is_intel_hybrid_architecture(cpu_name):
            return physical_cores

        cpu_name_lower = cpu_name.lower()
        if "i9" in cpu_name_lower:
            return min(8, physical_cores)
        if "i7" in cpu_name_lower:
            return min(6, physical_cores)
        if "i5" in cpu_name_lower:
            return min(4, physical_cores)
        if "i3" in cpu_name_lower:
            return min(4, physical_cores)

        return max(1, physical_cores // 2)


class ONNXThreadOptimizer:
    """ONNX Runtime 线程优化器"""

    @staticmethod
    def calculate_optimal_threads(
        vendor: Optional[str] = None,
        physical_cores: Optional[int] = None,
        cpu_name: Optional[str] = None,
        usage_ratio: float = 0.6,
        hardware_info: Optional[HardwareInfo] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """计算最优线程数"""
        detector = CPUArchitectureDetector()

        if hardware_info:
            if not cpu_name:
                cpu_name = hardware_info.cpu_name
            if not physical_cores:
                physical_cores = hardware_info.cpu_cores

        if vendor is None:
            vendor = detector.detect_cpu_vendor(cpu_name)

        if physical_cores is None:
            physical_cores = detector.get_physical_cores(hardware_info)

        logical_cores = multiprocessing.cpu_count()

        info: Dict[str, Any] = {
            "vendor": vendor,
            "physical_cores": physical_cores,
            "logical_cores": logical_cores,
            "cpu_name": cpu_name or "Unknown",
            "usage_ratio": usage_ratio,
            "strategy": "",
        }

        if vendor == "intel":
            is_hybrid = detector.is_intel_hybrid_architecture(cpu_name)

            if is_hybrid:
                p_cores = detector.detect_intel_p_cores(cpu_name, physical_cores)
                optimal_threads = max(1, int(p_cores * usage_ratio))
                info["strategy"] = f"Intel 混合架构：仅使用 {p_cores} 个 P-Core 的 {usage_ratio*100:.0f}%"
                info["p_cores"] = p_cores
            else:
                optimal_threads = max(1, int(physical_cores * usage_ratio))
                info["strategy"] = f"Intel 传统架构：使用物理核心数的 {usage_ratio*100:.0f}%"

        elif vendor == "amd":
            optimal_threads = max(1, int(physical_cores * usage_ratio))
            info["strategy"] = f"AMD 全大核：使用物理核心数的 {usage_ratio*100:.0f}%"

        else:
            optimal_threads = max(1, int(physical_cores * 0.5))
            info["strategy"] = "未知架构：保守策略，使用物理核心数的 50%"

        optimal_threads = max(1, min(optimal_threads, physical_cores))
        info["optimal_threads"] = optimal_threads

        return optimal_threads, info

    @staticmethod
    def get_onnx_session_options(
        vendor: Optional[str] = None,
        optimal_threads: Optional[int] = None,
        hardware_info: Optional[HardwareInfo] = None,
    ):
        """获取配置好的 ONNX Runtime SessionOptions"""
        import onnxruntime as ort

        if optimal_threads is None:
            optimal_threads, _ = ONNXThreadOptimizer.calculate_optimal_threads(
                vendor=vendor,
                hardware_info=hardware_info,
            )

        if vendor is None:
            cpu_name = hardware_info.cpu_name if hardware_info else None
            vendor = CPUArchitectureDetector.detect_cpu_vendor(cpu_name)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        sess_options.intra_op_num_threads = optimal_threads
        sess_options.inter_op_num_threads = 1

        try:
            sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            sess_options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            sess_options.add_session_config_entry("session.set_denormal_as_zero", "1")
        except Exception as exc:
            logger.debug("CPU 优化配置失败（可能不支持）: %s", exc)

        return sess_options


class HardwareProfileProvider:
    """
    硬件能力提供者。

    设计模式：适配器模式，用于隔离硬件检测/优化实现，确保上层只依赖稳定接口。
    """

    def __init__(
        self,
        detector: Optional[CoreHardwareDetector] = None,
        optimizer: Optional[CoreOptimizer] = None,
        affinity_manager: Optional[CPUAffinityManager] = None,
        onnx_optimizer: Optional[Any] = None,
    ) -> None:
        self._detector = detector or CoreHardwareDetector()
        self._optimizer = optimizer or CoreOptimizer()
        self._affinity_manager = affinity_manager or CPUAffinityManager()
        self._onnx_optimizer = onnx_optimizer or ONNXThreadOptimizer
        self._lock = threading.Lock()
        self._cached_hardware: Optional[HardwareInfo] = None
        self._last_onnx_info: Dict[str, Any] = {}

    def get_hardware_info(self, is_force_refresh: bool = False) -> HardwareInfo:
        """获取硬件信息（默认缓存，可强制刷新）。"""
        with self._lock:
            if self._cached_hardware is None or is_force_refresh:
                self._cached_hardware = self._detector.detect()
            return self._cached_hardware

    def get_optimization_config(
        self,
        hardware_info: Optional[HardwareInfo] = None,
    ) -> OptimizationConfig:
        """获取硬件优化配置（批处理/并发/设备推荐等）。"""
        info = hardware_info or self.get_hardware_info()
        return self._optimizer.get_optimization_config(info)

    def get_cpu_system_info(self) -> Dict[str, Any]:
        """获取CPU系统信息。"""
        return self._affinity_manager.get_system_info()

    def get_cpu_affinity_cores(
        self,
        strategy: str = "auto",
        custom_cores: Optional[List[int]] = None,
        exclude_cores: Optional[List[int]] = None,
    ) -> List[int]:
        """计算 CPU 亲和性核心列表（不做实际绑定）。"""
        return self._affinity_manager.calculate_optimal_cores(
            strategy=strategy,
            custom_cores=custom_cores,
            exclude_cores=exclude_cores,
        )

    def apply_cpu_affinity(self, config: CPUAffinityConfig, context: str = "") -> bool:
        """应用 CPU 亲和性配置。"""
        return self._affinity_manager.apply_cpu_affinity(config, context=context)

    def restore_cpu_affinity(self, context: str = "") -> bool:
        """恢复 CPU 亲和性配置。"""
        return self._affinity_manager.restore_cpu_affinity(context=context)

    def apply_pcore_affinity(self, context: str = "") -> bool:
        """仅对 Intel 混合架构绑定 P-Core。"""
        info = self.get_hardware_info()
        cpu_name = info.cpu_name
        if not CPUArchitectureDetector.is_intel_hybrid_architecture(cpu_name):
            logger.debug("非 Intel 混合架构，跳过 P-Core 亲和性设置")
            return False

        physical_cores = CPUArchitectureDetector.get_physical_cores(info)
        p_cores = CPUArchitectureDetector.detect_intel_p_cores(cpu_name, physical_cores)
        p_core_logical_ids = list(range(p_cores * 2))

        return self._affinity_manager.apply_affinity_cores(
            p_core_logical_ids,
            context=context or "P-Core",
        )

    def get_onnx_thread_budget(
        self,
        hardware_info: Optional[HardwareInfo] = None,
        usage_ratio: float = 0.6,
    ) -> int:
        """获取 ONNX 推理线程预算。"""
        info = hardware_info or self.get_hardware_info()
        try:
            threads, detail = self._onnx_optimizer.calculate_optimal_threads(
                usage_ratio=usage_ratio,
                hardware_info=info,
            )
        except TypeError:
            threads, detail = self._onnx_optimizer.calculate_optimal_threads(
                hardware_info=info,
            )
        self._last_onnx_info = detail or {}
        return max(1, int(threads))

    def get_last_onnx_info(self) -> Dict[str, Any]:
        """获取最近一次 ONNX 线程预算的详细信息。"""
        return dict(self._last_onnx_info)

    def build_onnx_session_options(
        self,
        hardware_info: Optional[HardwareInfo] = None,
        usage_ratio: float = 0.6,
    ) -> Tuple[Any, int, Dict[str, Any]]:
        """构建 ONNX SessionOptions 并返回线程信息。"""
        info = hardware_info or self.get_hardware_info()
        threads = self.get_onnx_thread_budget(info, usage_ratio=usage_ratio)
        try:
            sess_options = self._onnx_optimizer.get_onnx_session_options(
                optimal_threads=threads,
                hardware_info=info,
            )
        except Exception:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = threads
            sess_options.inter_op_num_threads = 1

        return sess_options, threads, self.get_last_onnx_info()

    def get_runtime_recommendation(
        self,
        hardware_info: Optional[HardwareInfo] = None,
    ) -> Dict[str, Any]:
        """
        输出运行参数推荐值（硬件优先）。

        供运行参数合并与 ModelManagerV2 初始化使用。
        """
        info = hardware_info or self.get_hardware_info()
        optim = self.get_optimization_config(info)
        onnx_threads = self.get_onnx_thread_budget(info)

        max_vram_mb = None
        if info.cuda_available and info.gpu_memory_mb:
            max_vram_mb = int(max(info.gpu_memory_mb) * 0.8)

        return {
            "device_preference": optim.recommended_device,
            "cpu_threads": onnx_threads,
            "onnx_intra_threads": onnx_threads,
            "onnx_inter_threads": 1,
            "max_vram_mb": max_vram_mb,
            "reserved_vram_mb": 500,
            "max_models": 3,
            "cpu_affinity_strategy": "auto",
        }


_provider_instance: Optional[HardwareProfileProvider] = None
_provider_lock = threading.Lock()


def get_hardware_profile_provider() -> HardwareProfileProvider:
    """硬件能力提供者单例入口。"""
    global _provider_instance
    if _provider_instance is None:
        with _provider_lock:
            if _provider_instance is None:
                _provider_instance = HardwareProfileProvider()
    return _provider_instance
