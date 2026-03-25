"""
慢流 Whisper 幻觉守卫。

职责：
1. 统一慢流结果的最小清洗与幻觉判定；
2. 提供快流兜底覆盖与非幻觉文本规范化工具。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional


@dataclass(frozen=True)
class SlowWhisperGuardResult:
    """慢流幻觉守卫输出。"""

    is_hallucination: bool
    raw_text: str
    text: str


class SlowWhisperHallucinationGuard:
    """复用型慢流幻觉处理器。"""

    def __init__(
        self,
        *,
        sanitizer: Any,
        hallucination_detector: Any,
        text_normalizer: Any,
    ) -> None:
        self._sanitizer = sanitizer
        self._hallucination_detector = hallucination_detector
        self._text_normalizer = text_normalizer

    def prepare_and_detect(
        self,
        *,
        whisper_result: MutableMapping[str, Any],
        prompt: Optional[str],
    ) -> SlowWhisperGuardResult:
        """补齐慢流字段、执行最小清洗并判定幻觉。"""
        raw_text = str(
            whisper_result.get("raw_text")
            or whisper_result.get("text_raw")
            or whisper_result.get("text")
            or ""
        )
        whisper_result["raw_text"] = raw_text
        whisper_result["text_raw"] = raw_text
        whisper_result["prompt"] = prompt

        base_text = whisper_result.get("text") or whisper_result.get("min_clean_text") or raw_text
        text = self._sanitizer.sanitize_minimal(str(base_text or ""), prompt=prompt)
        whisper_result["text"] = text

        is_hallucination = bool(
            self._hallucination_detector.is_hallucination(dict(whisper_result), prompt)
        )
        whisper_result["is_hallucination"] = is_hallucination
        return SlowWhisperGuardResult(
            is_hallucination=is_hallucination,
            raw_text=raw_text,
            text=str(whisper_result.get("text") or ""),
        )

    @staticmethod
    def apply_fast_fallback(
        *,
        whisper_result: MutableMapping[str, Any],
        sv_result: Mapping[str, Any],
    ) -> None:
        """当慢流幻觉时，使用快流文本覆盖慢流文本事实。"""
        fallback_text = str(sv_result.get("text_clean") or "")
        whisper_result["text"] = fallback_text
        whisper_result["text_itn_raw"] = str(sv_result.get("text_itn_raw") or fallback_text)
        whisper_result["text_clean"] = fallback_text
        whisper_result["language"] = str(sv_result.get("language") or whisper_result.get("language") or "auto")

    def normalize_non_hallucination(
        self,
        *,
        whisper_result: MutableMapping[str, Any],
        language: str,
    ) -> None:
        """对非幻觉慢流文本做标准规范化。"""
        normalized = self._text_normalizer.normalize(
            str(whisper_result.get("text", "")),
            str(language or "auto"),
        )
        whisper_result["text_itn_raw"] = normalized.text_itn_raw
        whisper_result["text_clean"] = normalized.text_clean
        whisper_result["text"] = normalized.text_clean or str(whisper_result.get("text", ""))
        whisper_result["language"] = str(language or whisper_result.get("language") or "auto")
