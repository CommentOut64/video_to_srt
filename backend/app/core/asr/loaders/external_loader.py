"""
外部服务型 Loader
通过工厂方法创建实例，用于桥接已有服务。
"""

from __future__ import annotations

import importlib
from typing import Callable

from app.core.asr.loader_base import LoadPlan, ModelLoader
from app.core.asr.model_spec import ModelSpec


class ExternalLoader(ModelLoader):
    """使用 spec.env.factory 指定工厂函数，创建外部服务实例。"""

    FACTORY_KEY = "factory"

    def load(self, spec: ModelSpec, plan: LoadPlan):
        factory_path = spec.env.get(self.FACTORY_KEY) if spec.env else None
        if not factory_path:
            raise ValueError(f"external loader 需要 env.factory: {spec.id}")
        factory = self._import_factory(factory_path)
        return factory()

    def unload(self, handle) -> None:
        if hasattr(handle, "unload"):
            handle.unload()

    @staticmethod
    def _import_factory(path: str) -> Callable:
        module_name, attr = path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attr)
