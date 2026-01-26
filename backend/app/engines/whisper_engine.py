"""
Whisper 引擎适配器。
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
from app.services.whisper_service import get_whisper_service

logger = logging.getLogger(__name__)


class WhisperEngine(ASREngine):
    """Faster-Whisper 引擎适配器。"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cuda",
        compute_type: Optional[str] = None,
    ) -> None:
        self.service = get_whisper_service()
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type

        if model_name:
            self._ensure_model_loaded()

    def get_capabilities(self) -> List[ASRCapability]:
        return [
            ASRCapability.WORD_TIMESTAMPS,
            ASRCapability.CONFIDENCE_SCORES,
            ASRCapability.LANGUAGE_DETECTION,
        ]

    def get_timestamp_precision(self) -> TimestampPrecision:
        return TimestampPrecision.WORD

    async def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> ASRResult:
        """转录音频并输出统一结果。"""
        self._ensure_model_loaded()

        word_timestamps = kwargs.get("word_timestamps")
        initial_prompt = kwargs.get("initial_prompt")
        vad_filter = kwargs.get("vad_filter")
        vad_parameters = kwargs.get("vad_parameters")
        beam_size = kwargs.get("beam_size")
        temperature = kwargs.get("temperature")
        condition_on_previous_text = kwargs.get("condition_on_previous_text")
        suppress_tokens = kwargs.get("suppress_tokens")
        repetition_penalty = kwargs.get("repetition_penalty")
        no_repeat_ngram_size = kwargs.get("no_repeat_ngram_size")

        try:
            result = self.service.transcribe(
                audio,
                language=language,
                initial_prompt=initial_prompt,
                word_timestamps=word_timestamps,
                beam_size=beam_size,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters,
                temperature=temperature,
                condition_on_previous_text=condition_on_previous_text,
                suppress_tokens=suppress_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        except Exception as exc:
            logger.error("Whisper 转录失败: %s", exc, exc_info=True)
            raise

        segments = self._build_segments(result.get("segments") or [])
        words = [word for seg in segments for word in (seg.words or [])] or None

        precision = TimestampPrecision.WORD if words else TimestampPrecision.SEGMENT

        return ASRResult(
            text=result.get("text", ""),
            segments=segments,
            words=words,
            confidence=self._mean_confidence(segments),
            language=result.get("language", "auto"),
            metadata=ASRMetadata(
                engine="whisper",
                source="whisper",
                timestamp_precision=precision,
                capabilities=self.get_capabilities(),
                model_version=self.model_name,
                raw_tags={"raw_result": result},
            ),
        )

    def estimate_confidence(self, raw_output: Any) -> float:
        if isinstance(raw_output, dict):
            segments = raw_output.get("segments", [])
            if segments:
                return float(self._mean_confidence(self._build_segments(segments)))
        return 0.0

    def _ensure_model_loaded(self) -> None:
        if self.service.model:
            return
        try:
            if self.model_name:
                self.service.load_model(
                    model_name=self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            else:
                self.service.load_model(
                    device=self.device,
                    compute_type=self.compute_type,
                )
        except Exception as exc:
            logger.error("Whisper 模型加载失败: %s", exc, exc_info=True)
            raise

    def _build_segments(self, raw_segments: List[Dict[str, Any]]) -> List[Segment]:
        segments: List[Segment] = []
        for seg in raw_segments:
            words = [
                WordTimestamp(
                    word=str(w.get("word", "")),
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", 0.0)),
                    confidence=w.get("probability"),
                    probability=w.get("probability"),
                    is_pseudo=False,
                )
                for w in (seg.get("words") or [])
            ]
            segments.append(
                Segment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=str(seg.get("text", "")),
                    confidence=self._segment_confidence(seg),
                    words=words or None,
                )
            )
        return segments

    @staticmethod
    def _segment_confidence(segment: Dict[str, Any]) -> float:
        return ConfidenceNormalizer.normalize_whisper(
            float(segment.get("avg_logprob", -1.0)),
            float(segment.get("no_speech_prob", 0.0)),
        )

    @staticmethod
    def _mean_confidence(segments: List[Segment]) -> float:
        values = [seg.confidence for seg in segments if seg.confidence is not None]
        if not values:
            return 0.0
        return float(np.mean(values))
