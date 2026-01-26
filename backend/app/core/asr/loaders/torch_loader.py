"""
Torch 模型加载器
支持自动选择权重文件并按设备映射。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.core.asr.loader_base import LoadPlan, ModelLoader
from app.core.asr.model_spec import ModelSpec


class TorchLoader(ModelLoader):
    """使用 torch.load 加载权重（适用于简单权重场景）。"""

    def _select_weight_file(self, spec: ModelSpec, plan: LoadPlan) -> Path:
        base = Path(plan.local_path)
        filename = spec.features.get("weight_file") if spec.features else None
        if filename:
            candidate = base / filename
            if candidate.exists():
                return candidate
        exts = (".pt", ".pth", ".bin")
        for fname in spec.source.files:
            if fname.endswith(exts):
                candidate = base / fname
                if candidate.exists():
                    return candidate
        for ext in exts:
            for f in base.glob(f"*{ext}"):
                return f
        raise FileNotFoundError(f"未找到 Torch 权重文件: {base}")

    def load(self, spec: ModelSpec, plan: LoadPlan):
        import torch

        weight_file = self._select_weight_file(spec, plan)
        map_location = "cpu" if plan.device == "cpu" else "cuda"

        # 优先尝试常规 torch.load
        try:
            return torch.load(weight_file, map_location=map_location)
        except Exception:
            # 针对 pyannote/brouhaha 等 HuggingFace 格式，尝试 Model.from_pretrained
            try:
                from pyannote.audio import Model  # 延迟导入，避免无依赖环境报错

                model_dir = weight_file.parent
                return Model.from_pretrained(str(model_dir))
            except Exception as exc:
                raise RuntimeError(f"加载 Torch 模型失败: {weight_file} -> {exc}") from exc

    def unload(self, handle) -> None:
        """对 state_dict 无需特殊卸载。"""
        return
