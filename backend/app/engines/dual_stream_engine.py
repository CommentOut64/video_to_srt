"""
双流引擎（SenseVoice + Whisper）。
V3.2.0+dev.20260119.02
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from app.core.asr.engine import ASREngine
from app.core.asr.enums import ASRCapability, TimestampPrecision
from app.core.asr.models import ASRResult
logger = logging.getLogger(__name__)


class DualStreamEngine(ASREngine):
    """双流混合引擎：SenseVoice 提供时间戳，Whisper 复核文本。"""

    def __init__(
        self,
        draft_engine: Optional[ASREngine] = None,
        patch_engine: Optional[ASREngine] = None,
        patch_threshold: float = 0.7,
    ) -> None:
        if draft_engine is None or patch_engine is None:
            raise ValueError("DualStreamEngine requires draft_engine and patch_engine")
        self.draft_engine = draft_engine
        self.patch_engine = patch_engine
        self.patch_threshold = patch_threshold

    def get_capabilities(self) -> List[ASRCapability]:
        return list(
            {
                *self.draft_engine.get_capabilities(),
                *self.patch_engine.get_capabilities(),
            }
        )

    def get_timestamp_precision(self) -> TimestampPrecision:
        return self.draft_engine.get_timestamp_precision()

    async def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        **kwargs: object,
    ) -> ASRResult:
        """执行双流转录，必要时触发复核。"""
        enable_patch = bool(kwargs.get("enable_patch", True))
        patch_threshold = float(kwargs.get("patch_threshold", self.patch_threshold))

        draft_result = await self.draft_engine.transcribe(audio, language=language, **kwargs)
        if not enable_patch or draft_result.confidence >= patch_threshold:
            return draft_result

        logger.info(
            "触发复核: confidence=%.3f < threshold=%.3f",
            draft_result.confidence,
            patch_threshold,
        )
        patch_result = await self.patch_engine.transcribe(audio, language=language, **kwargs)

        # 仅替换文本，保持时间戳权威来自 draft
        draft_result.text = patch_result.text
        draft_result.text_clean = patch_result.text_clean or patch_result.text
        draft_result.confidence = patch_result.confidence
        draft_result.metadata.source = "whisper_patch"
        draft_result.metadata.raw_tags["patch_text"] = patch_result.text

        return draft_result

    def estimate_confidence(self, raw_output: object) -> float:
        return self.draft_engine.estimate_confidence(raw_output)
