"""Whisper 结果到 TextTruthPackage 的慢流适配器。"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from app.services.arbitration.hallucination_detector import HallucinationDetector
from app.services.timeanchored_alignment.contracts import (
    TextTruthPackage,
    TextTruthQuality,
    TextTruthUnit,
)
from app.services.whisper.whisper_text_sanitizer import WhisperTextSanitizer


class WhisperTextAdapter:
    """将旧 whisper_result(dict) 适配为新链 TextTruthPackage。"""

    def __init__(
        self,
        *,
        sanitizer: Optional[Any] = None,
        hallucination_detector: Optional[Any] = None,
    ) -> None:
        self._sanitizer = sanitizer or WhisperTextSanitizer()
        self._hallucination_detector = hallucination_detector or HallucinationDetector()

    def build_text_truth_package(
        self,
        *,
        whisper_result: Mapping[str, Any],
        default_language: str = "auto",
    ) -> TextTruthPackage:
        raw_text = self._pick_raw_text(whisper_result).strip()
        base_text = str(
            whisper_result.get("text")
            or whisper_result.get("text_clean")
            or whisper_result.get("min_clean_text")
            or raw_text
            or ""
        ).strip()
        prompt = str(whisper_result.get("prompt") or "") or None
        normalized_text = str(self._sanitizer.sanitize_minimal(base_text, prompt=prompt) or "").strip()
        language = str(whisper_result.get("language") or default_language or "auto").strip().lower() or "auto"

        confidence = self._clamp_probability(whisper_result.get("confidence"), default=0.0)
        is_hallucination = self._resolve_hallucination(whisper_result, prompt=prompt)
        repetition_ratio = self._estimate_repetition_ratio(normalized_text)
        length_ratio = max(float(len(normalized_text) or 0) / max(len(raw_text), 1), 1e-6)

        quality = TextTruthQuality(
            hallucination_risk=1.0 if is_hallucination else 0.0,
            repetition_ratio=repetition_ratio,
            length_ratio=length_ratio,
        )
        units, timestamp_mode = self._build_text_units(
            whisper_result=whisper_result,
            normalized_text=normalized_text,
            fallback_text=base_text,
            language=language,
            default_confidence=confidence,
            prompt=prompt,
        )

        return TextTruthPackage(
            units=units,
            quality=quality,
            language=language,
            raw_text=raw_text,
            normalized_text=normalized_text,
            is_hallucination=is_hallucination,
            quality_signals={
                "confidence": confidence,
                "repetition_ratio": repetition_ratio,
                "length_ratio": length_ratio,
            },
            source_metadata={
                "prompt": str(whisper_result.get("prompt") or ""),
                "raw_result": dict(whisper_result.get("raw_result") or {}),
                "text_itn_raw": str(whisper_result.get("text_itn_raw") or ""),
                "text_clean": str(whisper_result.get("text_clean") or ""),
                "timestamp_mode": timestamp_mode,
            },
            source="whisper",
            protected_spans=tuple(),
        )

    @staticmethod
    def _pick_raw_text(whisper_result: Mapping[str, Any]) -> str:
        return str(
            whisper_result.get("text_raw")
            or whisper_result.get("raw_text")
            or whisper_result.get("text")
            or ""
        )

    def _resolve_hallucination(
        self,
        whisper_result: Mapping[str, Any],
        *,
        prompt: Optional[str],
    ) -> bool:
        if "is_hallucination" in whisper_result:
            return bool(whisper_result.get("is_hallucination"))
        try:
            return bool(
                self._hallucination_detector.is_hallucination(
                    dict(whisper_result),
                    prompt,
                )
            )
        except Exception:
            return False

    @staticmethod
    def _clamp_probability(value: Any, *, default: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float(default)
        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _estimate_repetition_ratio(text: str) -> float:
        cleaned = str(text or "")
        if not cleaned:
            return 0.0
        repeated_count = 0
        for index in range(1, len(cleaned)):
            if cleaned[index] == cleaned[index - 1]:
                repeated_count += 1
        return max(0.0, min(1.0, repeated_count / max(len(cleaned), 1)))

    def _build_text_units(
        self,
        *,
        whisper_result: Mapping[str, Any],
        normalized_text: str,
        fallback_text: str,
        language: str,
        default_confidence: float,
        prompt: Optional[str],
    ) -> tuple[tuple[TextTruthUnit, ...], str]:
        word_units = self._build_units_from_word_timestamps(
            whisper_result=whisper_result,
            language=language,
            default_confidence=default_confidence,
            prompt=prompt,
        )
        if word_units:
            return tuple(word_units), "word"

        segment_units = self._build_units_from_segments(
            whisper_result=whisper_result,
            language=language,
            default_confidence=default_confidence,
            prompt=prompt,
        )
        if segment_units:
            return tuple(segment_units), "segment"

        synthetic_units = self._build_synthetic_units(
            text=normalized_text or fallback_text,
            language=language,
            confidence=default_confidence,
        )
        if synthetic_units:
            return tuple(synthetic_units), "synthetic"

        return tuple(), "none"

    def _build_units_from_word_timestamps(
        self,
        *,
        whisper_result: Mapping[str, Any],
        language: str,
        default_confidence: float,
        prompt: Optional[str],
    ) -> list[TextTruthUnit]:
        units: list[TextTruthUnit] = []
        time_offset = self._resolve_time_offset(whisper_result)
        for segment in self._iter_segments(whisper_result):
            words = segment.get("words")
            if not isinstance(words, list):
                continue
            for word in words:
                if not isinstance(word, Mapping):
                    continue
                normalized = self._sanitize_fragment(str(word.get("word") or ""), prompt=prompt)
                if not normalized:
                    continue
                span = self._resolve_time_span(
                    start=word.get("start"),
                    end=word.get("end"),
                    time_offset=time_offset,
                )
                if span is None:
                    continue
                confidence = self._clamp_probability(word.get("probability"), default=default_confidence)
                units.append(
                    TextTruthUnit(
                        text=normalized,
                        normalized_text=normalized,
                        confidence=confidence,
                        language=language,
                        start=span[0],
                        end=span[1],
                        source="whisper",
                    )
                )
        return units

    def _build_units_from_segments(
        self,
        *,
        whisper_result: Mapping[str, Any],
        language: str,
        default_confidence: float,
        prompt: Optional[str],
    ) -> list[TextTruthUnit]:
        units: list[TextTruthUnit] = []
        time_offset = self._resolve_time_offset(whisper_result)
        for segment in self._iter_segments(whisper_result):
            span = self._resolve_time_span(
                start=segment.get("start"),
                end=segment.get("end"),
                time_offset=time_offset,
            )
            if span is None:
                continue
            text = self._sanitize_fragment(str(segment.get("text") or ""), prompt=prompt, strip=False)
            if not text:
                continue
            tokens = self._split_tokens(text=text, language=language)
            if not tokens:
                continue
            confidence = self._clamp_probability(segment.get("confidence"), default=default_confidence)
            duration = max(1e-3, float(span[1] - span[0]))
            token_weights = [max(len(str(token).strip()), 1) for token in tokens]
            total_weight = max(sum(token_weights), 1)
            cursor = float(span[0])
            for index, token in enumerate(tokens):
                weight = token_weights[index]
                token_duration = duration * float(weight) / float(total_weight)
                token_end = float(span[1]) if index == len(tokens) - 1 else min(float(span[1]), cursor + token_duration)
                if token_end <= cursor:
                    token_end = cursor + 0.01
                units.append(
                    TextTruthUnit(
                        text=token,
                        normalized_text=token,
                        confidence=confidence,
                        language=language,
                        start=cursor,
                        end=token_end,
                        source="whisper",
                    )
                )
                cursor = token_end
        return units

    def _build_synthetic_units(
        self,
        *,
        text: str,
        language: str,
        confidence: float,
    ) -> list[TextTruthUnit]:
        normalized = self._sanitize_fragment(text, prompt=None, strip=False)
        tokens = self._split_tokens(text=normalized, language=language)
        if not tokens:
            return []
        units: list[TextTruthUnit] = []
        cursor = 0.0
        for token in tokens:
            end = cursor + 0.12
            units.append(
                TextTruthUnit(
                    text=token,
                    normalized_text=token,
                    confidence=confidence,
                    language=language,
                    start=cursor,
                    end=end,
                    source="whisper",
                )
            )
            cursor = end
        return units

    @staticmethod
    def _iter_segments(whisper_result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        raw_result = whisper_result.get("raw_result")
        segments_from_raw = raw_result.get("segments") if isinstance(raw_result, Mapping) else None
        segments = whisper_result.get("segments") or segments_from_raw or []
        if not isinstance(segments, list):
            return []
        return [item for item in segments if isinstance(item, Mapping)]

    @staticmethod
    def _resolve_time_offset(whisper_result: Mapping[str, Any]) -> float:
        if str(whisper_result.get("word_time_base") or "").strip().lower() != "batch_local":
            return 0.0
        try:
            return float(whisper_result.get("word_time_offset") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _resolve_time_span(
        *,
        start: Any,
        end: Any,
        time_offset: float = 0.0,
    ) -> tuple[float, float] | None:
        try:
            span_start = float(start) + float(time_offset)
            span_end = float(end) + float(time_offset)
        except (TypeError, ValueError):
            return None
        if span_end <= span_start:
            span_end = span_start + 0.01
        return span_start, span_end

    def _sanitize_fragment(
        self,
        text: str,
        *,
        prompt: Optional[str],
        strip: bool = True,
    ) -> str:
        cleaned = str(self._sanitizer.sanitize_minimal(str(text or ""), prompt=prompt) or "")
        return cleaned.strip() if strip else cleaned

    @staticmethod
    def _split_tokens(*, text: str, language: str) -> list[str]:
        normalized = str(text or "")
        if not normalized:
            return []
        if language in {"zh", "ja", "ko"}:
            return [char for char in normalized if not char.isspace()]
        token_pattern = re.compile(
            r"[A-Za-z0-9]+(?:[’'._-][A-Za-z0-9]+)*|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]|[^\s]",
            re.UNICODE,
        )
        return [token for token in token_pattern.findall(normalized) if str(token).strip()]
