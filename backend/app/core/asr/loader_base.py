"""
模型加载器抽象与加载计划
V3.2.0+dev.20260114.01
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from app.core.asr.model_spec import ModelSpec


@dataclass(frozen=True)
class LoadPlan:
    """加载计划，绑定设备、计算类型与本地路径。"""

    device: str
    compute_type: str
    local_path: str

    @property
    def key(self) -> str:
        """缓存键，确保同配置命中。"""
        return f"{self.device}:{self.compute_type}:{self.local_path}"


class ModelLoader(ABC):
    """加载器抽象，按框架实现子类。"""

    @abstractmethod
    def load(self, spec: ModelSpec, plan: LoadPlan) -> Any:
        """加载模型并返回句柄。"""
        raise NotImplementedError

    @abstractmethod
    def unload(self, handle: Any) -> None:
        """卸载模型句柄并释放资源。"""
        raise NotImplementedError

    def warmup(self, handle: Any, spec: ModelSpec) -> None:
        """可选预热，默认空实现。"""
        return

