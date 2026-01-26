"""
CTranslate2/Faster-Whisper 模型加载器
"""

from __future__ import annotations

from pathlib import Path

from app.core.asr.loader_base import LoadPlan, ModelLoader
from app.core.asr.model_spec import ModelSpec


class CT2Loader(ModelLoader):
    """基于 faster-whisper 的加载器。"""

    def load(self, spec: ModelSpec, plan: LoadPlan):
        from faster_whisper import WhisperModel

        model_path = Path(plan.local_path)
        # faster-whisper 接受目录或模型名；此处优先目录
        whisper = WhisperModel(
            model_path.as_posix(),
            device=plan.device,
            compute_type=plan.compute_type,
            download_root=model_path.parent.as_posix(),
            local_files_only=True,
        )
        return whisper

    def unload(self, handle) -> None:
        if hasattr(handle, "model"):
            del handle.model
