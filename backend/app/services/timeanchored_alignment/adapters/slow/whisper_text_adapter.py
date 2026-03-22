"""Whisper 结果到 TextTruthPackage 的慢流适配器。"""

from __future__ import annotations

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
        unit = TextTruthUnit(
            text=base_text,
            normalized_text=normalized_text,
            confidence=confidence,
            language=language,
            source="whisper",
        )

        return TextTruthPackage(
            units=(unit,),
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
