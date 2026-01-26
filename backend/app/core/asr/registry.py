"""
模型注册表：加载 models.yaml / models.d/*.yaml，供 ModelManager V2 使用。
V3.2.0+dev.20260114.01
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from app.core.asr.model_spec import ModelSpec


class ModelRegistry:
    """模型注册表，提供查询与过滤。"""

    def __init__(self, config_path: Path, extra_dir: Optional[Path] = None, base_dir: Optional[Path] = None):
        """
        初始化模型注册表。

        Args:
            config_path: 主配置文件路径（models.yaml）
            extra_dir: 额外配置目录（models.d/）
            base_dir: 基准目录，用于解析相对路径（默认为 config_path 的父目录）
        """
        self.config_path = config_path
        self.extra_dir = extra_dir
        self.base_dir = base_dir or config_path.parent.parent.parent  # V3.2.0+dev.20260116.02: 默认为项目根目录
        self._specs: Dict[str, ModelSpec] = {}
        self._load_all()

    def _load_all(self) -> None:
        specs: Dict[str, ModelSpec] = {}

        def load_file(path: Path) -> None:
            if not path.exists():
                return
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or []
            for item in data:
                spec = ModelSpec.from_dict(item)
                # V3.2.0+dev.20260116.02: 转换相对路径为绝对路径
                if spec.source.local_path:
                    local_path = Path(spec.source.local_path)
                    if not local_path.is_absolute():
                        # 相对路径，转换为绝对路径（相对于 base_dir）
                        absolute_path = (self.base_dir / local_path).resolve()
                        spec.source.local_path = str(absolute_path)
                specs[spec.id] = spec

        load_file(self.config_path)

        if self.extra_dir and self.extra_dir.exists():
            for file in glob.glob(str(self.extra_dir / "*.yaml")):
                load_file(Path(file))

        self._specs = specs

    def get(self, model_id: str) -> ModelSpec:
        if model_id not in self._specs:
            raise KeyError(f"未找到模型配置: {model_id}")
        return self._specs[model_id]

    def list(self, kind: Optional[str] = None) -> List[ModelSpec]:
        if kind is None:
            return list(self._specs.values())
        return [s for s in self._specs.values() if s.kind == kind]

