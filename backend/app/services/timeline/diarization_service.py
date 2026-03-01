"""
Pyannote 完整说话人分离服务。

设计模式：Adapter Pattern。
原因：封装 pyannote `Pipeline` 调用细节，向 Timeline 域输出稳定结构，
避免编排层直接耦合第三方返回对象。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np

from app.services.model_manager_v2 import get_model_manager_v2
from app.services.pyannote_compat import load_pyannote_pipeline


@dataclass(frozen=True)
class DiarizationSegment:
    """单段说话人区间。"""

    speaker_id: str
    start: float
    end: float
    confidence: float = 0.85
    is_overlap: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class DiarizationResult:
    """完整 diarization 结果。"""

    segments: list[DiarizationSegment]
    speaker_ids: list[str]


@dataclass(frozen=True)
class PyannoteDiarizationConfig:
    """pyannote diarization 配置。"""

    enabled: bool = False
    model_id: str = ""
    local_path: str = ""
    hf_token: Optional[str] = None
    prefer_device: str = "auto"
    min_segment_duration_sec: float = 0.15
    max_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    num_speakers: Optional[int] = None


class PyannoteDiarizationService:
    """使用 pyannote 完整 Pipeline 产出说话人区间。"""

    # V3.2.0+dev.20260216.03: 进程内缓存 Pipeline，避免每个 block 重复加载模型。
    _PIPELINE_CACHE: ClassVar[dict[tuple[str, str, str], Any]] = {}
    _PIPELINE_CACHE_LOCK: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        config: Optional[PyannoteDiarizationConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or PyannoteDiarizationConfig()
        self.logger = logger or logging.getLogger(__name__)

    @property
    def is_enabled(self) -> bool:
        return bool(self.config.enabled)

    def run(
        self,
        *,
        audio: np.ndarray,
        sample_rate: int,
    ) -> DiarizationResult:
        """执行完整 diarization 并返回标准化区间列表。"""
        if audio.size == 0:
            return DiarizationResult(segments=[], speaker_ids=[])

        pipeline = self._acquire_pipeline()
        if pipeline is None:
            return DiarizationResult(segments=[], speaker_ids=[])

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("未安装 torch，无法执行 pyannote diarization") from exc

        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        inference_input = {"waveform": waveform, "sample_rate": sample_rate}

        diarization_kwargs = {}
        if self.config.num_speakers is not None:
            # Why: 显式指定人数时应避免与 min/max 同时约束造成冲突。
            diarization_kwargs["num_speakers"] = int(self.config.num_speakers)
        else:
            if self.config.max_speakers is not None:
                diarization_kwargs["max_speakers"] = int(self.config.max_speakers)
            if self.config.min_speakers is not None:
                diarization_kwargs["min_speakers"] = int(self.config.min_speakers)

        raw_output = pipeline(inference_input, **diarization_kwargs)
        annotation = self._extract_annotation(raw_output)

        segments: list[DiarizationSegment] = []
        speaker_ids: set[str] = set()
        min_duration = max(0.0, float(self.config.min_segment_duration_sec))

        for segment, _, label in annotation.itertracks(yield_label=True):
            start = float(segment.start)
            end = float(segment.end)
            if end - start < min_duration:
                continue

            speaker_id = str(label or "unknown")
            if not speaker_id.strip():
                speaker_id = "unknown"

            segments.append(
                DiarizationSegment(
                    speaker_id=speaker_id,
                    start=start,
                    end=end,
                    confidence=0.85,
                    is_overlap=False,
                )
            )
            speaker_ids.add(speaker_id)

        segments.sort(key=lambda item: (item.start, item.end))
        return DiarizationResult(
            segments=segments,
            speaker_ids=sorted(speaker_ids),
        )

    def _extract_annotation(self, raw_output):
        """
        兼容 pyannote 3.x/4.x diarization 输出结构。

        3.x: 直接返回 `Annotation`；
        4.x community-1: 返回 `DiarizeOutput`，优先使用 `exclusive_speaker_diarization`。
        """
        if hasattr(raw_output, "itertracks"):
            return raw_output

        # V3.2.0+dev.20260212.01:
        # 优先使用 exclusive 通道，避免重叠说话人段在边界处放大歧义。
        exclusive_diarization = getattr(raw_output, "exclusive_speaker_diarization", None)
        if exclusive_diarization is not None and hasattr(exclusive_diarization, "itertracks"):
            return exclusive_diarization

        speaker_diarization = getattr(raw_output, "speaker_diarization", None)
        if speaker_diarization is not None and hasattr(speaker_diarization, "itertracks"):
            return speaker_diarization

        raise RuntimeError(
            f"diarization 输出类型不支持: {type(raw_output)!r}"
        )

    def _acquire_pipeline(self):
        """优先本地路径，其次模型管理器 ID，最终回退到直连 repo_id。"""
        checkpoint = self._resolve_checkpoint()
        if not checkpoint:
            self.logger.warning("diarization 未配置模型路径或 model_id，跳过")
            return None

        device = str(self.config.prefer_device or "auto")
        cache_key = (
            str(checkpoint),
            str(self.config.hf_token or ""),
            device,
        )

        with self._PIPELINE_CACHE_LOCK:
            cached_pipeline = self._PIPELINE_CACHE.get(cache_key)
        if cached_pipeline is not None:
            return cached_pipeline

        pipeline = load_pyannote_pipeline(
            checkpoint=checkpoint,
            token=self.config.hf_token,
            logger=self.logger,
        )

        try:
            if device.startswith("cuda"):
                import torch

                if torch.cuda.is_available() and hasattr(pipeline, "to"):
                    pipeline = pipeline.to(torch.device(device))
        except Exception as exc:
            self.logger.warning("diarization pipeline 切换设备失败，回退默认设备: %s", exc)

        with self._PIPELINE_CACHE_LOCK:
            self._PIPELINE_CACHE[cache_key] = pipeline
        return pipeline

    def _resolve_checkpoint(self) -> str:
        local_path = str(self.config.local_path or "").strip()
        if local_path:
            path = Path(local_path)
            if path.exists():
                return str(path)
            self.logger.warning("diarization local_path 不存在: %s", local_path)

        model_id = str(self.config.model_id or "").strip()
        if not model_id:
            return ""

        try:
            manager = get_model_manager_v2()
            model_path = manager.ensure_available(model_id)
            return str(Path(model_path))
        except Exception:
            # Why: 允许直接填写 HF repo_id（尚未注册到 models.yaml 时可用）。
            return model_id
