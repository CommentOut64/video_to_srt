"""
Demucs 模型加载器
统一通过 demucs 库加载，供 ModelManager V2 使用。
"""

from __future__ import annotations

from typing import Optional

from app.core.asr.loader_base import LoadPlan, ModelLoader
from app.core.asr.model_spec import ModelSpec


class DemucsLoader(ModelLoader):
    """使用 demucs.pretrained.get_model 加载人声分离模型。"""

    def load(self, spec: ModelSpec, plan: LoadPlan):
        from demucs.pretrained import get_model
        import torch

        model_name: Optional[str] = None
        if spec.features:
            model_name = spec.features.get("model_name") or spec.features.get("name")
        if not model_name:
            # 从 id 解析，比如 demucs-htdemucs -> htdemucs
            model_name = spec.id.replace("demucs-", "")

        model = get_model(name=model_name)
        if plan.device != "cpu" and torch.cuda.is_available():
            model.cuda()
        else:
            model.cpu()
        model.eval()
        return model

    def unload(self, handle) -> None:
        import torch

        try:
            del handle
        except Exception:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
