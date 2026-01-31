"""
SenseVoice 引擎适配器。
V3.2.0+dev.20260119.03
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.asr.engine import ASREngine
from app.core.asr.enums import ASRCapability, TimestampPrecision
from app.core.asr.models import ASRMetadata, ASRResult, Segment, WordTimestamp
from app.core.asr.normalizer import ConfidenceNormalizer
from app.services.sensevoice_onnx_service import get_sensevoice_service

logger = logging.getLogger(__name__)


class SenseVoiceEngine(ASREngine):
    """SenseVoice ONNX 引擎适配器。"""

    def __init__(self, language: str = "auto") -> None:
        self.language = language
        self.service = get_sensevoice_service()

    def get_capabilities(self) -> List[ASRCapability]:
        return [
            ASRCapability.WORD_TIMESTAMPS,
            ASRCapability.CONFIDENCE_SCORES,
            ASRCapability.LANGUAGE_DETECTION,
            ASRCapability.EMOTION_DETECTION,
            ASRCapability.EVENT_TAGGING,
        ]

    def get_timestamp_precision(self) -> TimestampPrecision:
        return TimestampPrecision.FRAME

    async def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> ASRResult:
        """转录音频并输出统一结果。"""
        sample_rate = int(kwargs.get("sample_rate", 16000))
        use_itn = kwargs.get("use_itn")
        ban_emo_unk = kwargs.get("ban_emo_unk")

        if not getattr(self.service, "is_loaded", False):
            self.service.load_model()

        try:
            result = self.service.transcribe_audio_array(
                audio,
                sample_rate=sample_rate,
                language=language or self.language,
                use_itn=use_itn,
                ban_emo_unk=ban_emo_unk,
            )
        except Exception as exc:
            logger.error("SenseVoice 转录失败: %s", exc, exc_info=True)
            raise

        words = self._build_words(result.get("words") or [])
        raw_tokens = result.get("raw_tokens") if isinstance(result, dict) else None
        segments = self._build_segments(result, words)
        event_tag = result.get("event")

        metadata = ASRMetadata(
            engine="sensevoice",
            source="sensevoice",
            timestamp_precision=TimestampPrecision.FRAME,
            capabilities=self.get_capabilities(),
            raw_tags={
                "emotion_tag": result.get("emotion"),
                "event_tag": event_tag,
                "text_with_tags": result.get("text"),
                "raw_tokens": raw_tokens,
            },
        )

        return ASRResult(
            text=result.get("text", ""),
            text_clean=result.get("text_clean"),
            segments=segments,
            words=words or None,
            raw_tokens=raw_tokens,
            confidence=float(result.get("confidence", 0.0)),
            language=result.get("language") or (language or self.language),
            emotion=result.get("emotion"),
            event_tags=event_tag.split() if event_tag else None,
            metadata=metadata,
        )

    def estimate_confidence(self, raw_output: Any) -> float:
        if isinstance(raw_output, dict) and "confidence" in raw_output:
            try:
                return float(raw_output["confidence"])
            except (TypeError, ValueError):
                return 0.0
        if isinstance(raw_output, np.ndarray):
            return ConfidenceNormalizer.normalize_sensevoice(raw_output)
        if isinstance(raw_output, list):
            return ConfidenceNormalizer.normalize_sensevoice(np.asarray(raw_output))
        return 0.0

    @staticmethod
    def _build_words(raw_words: List[Dict[str, Any]]) -> List[WordTimestamp]:
        words: List[WordTimestamp] = []
        for word in raw_words:
            words.append(
                WordTimestamp(
                    word=str(word.get("word", "")),
                    start=float(word.get("start", 0.0)),
                    end=float(word.get("end", 0.0)),
                    confidence=word.get("confidence"),
                    confidence_display_raw=word.get("confidence_display_raw"),
                    confidence_raw=word.get("confidence_raw"),
                    is_pseudo=bool(word.get("is_pseudo", False)),
                    token_type=word.get("token_type"),
                )
            )
        return words

    @staticmethod
    def _build_segments(result: Dict[str, Any], words: List[WordTimestamp]) -> List[Segment]:
        if words:
            start = words[0].start
            end = words[-1].end
        else:
            start = 0.0
            end = 0.0
        return [
            Segment(
                start=start,
                end=end,
                text=result.get("text", ""),
                confidence=result.get("confidence"),
                words=words,
            )
        ]
