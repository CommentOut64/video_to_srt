"""
模型运行参数数据模型
用于前端读写运行参数与后端合并解析。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Literal, Any

DeviceType = Literal["auto", "cuda", "cpu"]


@dataclass
class ModelRuntimeOverride:
    """单模型运行参数覆盖配置。"""

    device: Optional[DeviceType] = None
    compute_type: Optional[str] = None
    cpu_threads: Optional[int] = None
    is_keep_resident: Optional[bool] = None
    evict_priority: Optional[int] = None
    max_concurrency: Optional[int] = None

    def to_dict(self) -> Dict[str, Optional[object]]:
        """转为字典，供API输出（保持前端字段命名）。"""
        return {
            "device": self.device,
            "compute_type": self.compute_type,
            "cpu_threads": self.cpu_threads,
            "keep_resident": self.is_keep_resident,
            "evict_priority": self.evict_priority,
            "max_concurrency": self.max_concurrency,
        }


@dataclass
class GlobalRuntimeConfig:
    """全局运行参数配置。"""

    device_preference: Optional[DeviceType] = None
    is_allow_download: Optional[bool] = None
    max_vram_mb: Optional[int] = None
    reserved_vram_mb: Optional[int] = None
    max_models: Optional[int] = None
    cpu_threads: Optional[int] = None
    cpu_affinity_strategy: Optional[str] = None
    onnx_intra_threads: Optional[int] = None
    onnx_inter_threads: Optional[int] = None

    def to_dict(self) -> Dict[str, Optional[object]]:
        """转为字典，供API输出（保持前端字段命名）。"""
        return {
            "device_preference": self.device_preference,
            "allow_download": self.is_allow_download,
            "max_vram_mb": self.max_vram_mb,
            "reserved_vram_mb": self.reserved_vram_mb,
            "max_models": self.max_models,
            "cpu_threads": self.cpu_threads,
            "cpu_affinity_strategy": self.cpu_affinity_strategy,
            "onnx_intra_threads": self.onnx_intra_threads,
            "onnx_inter_threads": self.onnx_inter_threads,
        }


@dataclass
class ModelRuntimeConfig:
    """运行参数配置集合（全局 + 单模型）。"""

    global_config: GlobalRuntimeConfig = field(default_factory=GlobalRuntimeConfig)
    per_model: Dict[str, ModelRuntimeOverride] = field(default_factory=dict)
    runtime: Dict[str, Dict[str, Any]] = field(default_factory=dict)
