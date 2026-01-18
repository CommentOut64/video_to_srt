"""
硬件检测相关数据模型
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging


@dataclass
class HardwareInfo:
    """核心硬件信息结构"""
    # GPU关键信息
    gpu_count: int = 0
    gpu_memory_mb: List[int] = field(default_factory=list)  # 每个GPU显存容量
    cuda_available: bool = False
    gpu_name: Optional[str] = None  # GPU型号名称
    
    # CPU关键信息  
    cpu_cores: int = 1
    cpu_threads: int = 1
    cpu_name: Optional[str] = None  # CPU型号名称
    cpu_max_frequency: Optional[float] = None  # CPU最大频率(MHz)
    
    # 内存关键信息
    memory_total_mb: int = 0
    memory_available_mb: int = 0
    
    # 存储关键信息
    temp_space_available_gb: int = 0
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "gpu": {
                "count": self.gpu_count,
                "memory_mb": self.gpu_memory_mb,
                "cuda_available": self.cuda_available,
                "total_memory_mb": sum(self.gpu_memory_mb) if self.gpu_memory_mb else 0,
                "available_memory_mb": sum(self.gpu_memory_mb) if self.gpu_memory_mb else 0,  # 简化处理
                "memory_usage_percent": 0,  # 简化处理
                "name": self.gpu_name,
                "device_name": self.gpu_name  # 兼容前端的多种字段名
            },
            "cpu": {
                "cores": self.cpu_cores,
                "threads": self.cpu_threads,
                "name": self.cpu_name,
                "max_frequency": self.cpu_max_frequency,
                "usage_percent": 0  # 简化处理，前端可以不显示或者默认为0
            },
            "memory": {
                "total_mb": self.memory_total_mb,
                "available_mb": self.memory_available_mb,
                "used_mb": self.memory_total_mb - self.memory_available_mb,
                "usage_percent": round((1 - self.memory_available_mb / max(1, self.memory_total_mb)) * 100, 1)
            },
            "storage": {
                "temp_space_gb": self.temp_space_available_gb
            }
        }


@dataclass
class OptimizationConfig:
    """基于硬件的优化配置"""
    # 转录优化配置
    batch_size: int = 16
    concurrency: int = 1
    use_memory_mapping: bool = False
    cpu_affinity_cores: List[int] = field(default_factory=list)

    # 推荐设备选择
    recommended_device: str = "cpu"
    recommended_model: str = "medium"  # 添加推荐模型

    # SenseVoice 相关配置
    enable_sensevoice: bool = True
    enable_demucs: bool = True
    demucs_model: str = "htdemucs"  # htdemucs, mdx_extra
    sensevoice_device: str = "cuda"  # cuda 或 cpu
    sensevoice_quantize: bool = True  # 是否使用量化模型
    note: str = ""

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "transcription": {
                "batch_size": self.batch_size,
                "concurrency": self.concurrency,
                "device": self.recommended_device,
                "recommended_model": self.recommended_model
            },
            "sensevoice": {
                "enable": self.enable_sensevoice,
                "device": self.sensevoice_device,
                "quantize": self.sensevoice_quantize
            },
            "demucs": {
                "enable": self.enable_demucs,
                "model": self.demucs_model
            },
            "system": {
                "use_memory_mapping": self.use_memory_mapping,
                "cpu_affinity_cores": self.cpu_affinity_cores,
                "process_priority": "normal",  # 添加进程优先级字段
                "note": self.note
            }
        }


@dataclass
class CPUAffinityConfig:
    """CPU亲和性配置（统一硬件能力提供者使用）"""
    enabled: bool = True
    strategy: str = "auto"  # auto/half/custom
    custom_cores: Optional[List[int]] = None
    exclude_cores: Optional[List[int]] = None

    @property
    def is_enabled(self) -> bool:
        """兼容旧字段名，统一使用 is_enabled 判定。"""
        return self.enabled

    @is_enabled.setter
    def is_enabled(self, value: bool) -> None:
        self.enabled = value
