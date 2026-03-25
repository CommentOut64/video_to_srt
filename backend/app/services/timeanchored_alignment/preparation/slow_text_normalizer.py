"""Preparation 慢流文本规范化。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from app.services.timeanchored_alignment.adapters.slow.whisper_text_adapter import WhisperTextAdapter
from app.services.timeanchored_alignment.contracts import TextTruthPackage


@dataclass(frozen=True)
class SlowTextNormalizationResult:
    text_truth: TextTruthPackage
    source_text: str
    language: str


class SlowTextNormalizer:
    """把 whisper result 规范化为 preparation 可消费的慢流文本。"""

    def __init__(
        self,
        *,
        sanitizer: Any | None = None,
        hallucination_detector: Any | None = None,
        whisper_adapter: WhisperTextAdapter | None = None,
    ) -> None:
        self._adapter = whisper_adapter or WhisperTextAdapter(
            sanitizer=sanitizer,
            hallucination_detector=hallucination_detector,
        )

    def normalize(
        self,
        *,
        whisper_result: Mapping[str, Any],
        default_language: str,
        fallback_text: str = "",
    ) -> SlowTextNormalizationResult:
        text_truth = self._adapter.build_text_truth_package(
            whisper_result=whisper_result,
            default_language=default_language,
        )
        source_text = str(text_truth.normalized_text or text_truth.raw_text or "").strip()
        if not source_text:
            source_text = str(fallback_text or "").strip()
            text_truth = replace(
                text_truth,
                raw_text=source_text,
                normalized_text=source_text,
            )
        return SlowTextNormalizationResult(
            text_truth=text_truth,
            source_text=source_text,
            language=str(text_truth.language or default_language or "auto"),
        )
