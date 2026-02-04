"""
L1 规范化处理器（Processor）。
V3.2.0+dev.20260204.02
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from app.core.logging import resolve_loguru_logger
from app.services.alignment.text_normalizer import (
    TextNormalizer,
    get_alignment_text_normalizer,
)
from app.services.alignment.types import L1Input, L1Output, NormalizationResult, TextTrack


class TextNormalizerProcessor:
    """处理器模式：统一输入输出，便于流水线接入。"""

    def __init__(
        self,
        *,
        normalizer: Optional[TextNormalizer] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self._logger = resolve_loguru_logger(
            logger,
            __name__,
            layer="L1",
            processor_name="text_normalizer_processor",
        )
        self._normalizer = normalizer or get_alignment_text_normalizer(logger=logger)

    def process(self, data: L1Input) -> L1Output:
        """执行 L1 规范化，输出 TextTrack。"""
        return L1Output(
            sv_track=self._normalize_single(data.sv_raw_result, source="sv", language_hint=data.language_hint),
            whisper_track=self._normalize_single(
                data.wh_raw_result,
                source="whisper",
                language_hint=data.language_hint,
            ),
        )

    def _normalize_single(
        self,
        raw_result: Optional[Dict[str, Any]],
        *,
        source: str,
        language_hint: Optional[str],
    ) -> Optional[TextTrack]:
        if not raw_result:
            return None
        raw_text = str(raw_result.get("raw_text") or "")
        input_text = raw_text
        if not input_text and source == "whisper":
            input_text = str(raw_result.get("min_clean_text") or "")
        if not input_text:
            self._logger.warning("L1 规范化缺失 raw_text/min_clean_text source={}", source)
            return None
        if not raw_text:
            raw_text = input_text
        language = str(raw_result.get("language") or language_hint or "auto")
        normalized = self._normalizer.normalize(input_text, language)
        return self._build_text_track(raw_text, normalized, source=source, language=language)

    @staticmethod
    def _build_text_track(
        raw_text: str,
        normalized: NormalizationResult,
        *,
        source: str,
        language: str,
    ) -> TextTrack:
        return TextTrack(
            raw_text=raw_text,
            text_itn_raw=normalized.text_itn_raw,
            text_clean=normalized.text_clean or normalized.text_itn_raw,
            char_mapping=normalized.char_mapping,
            raw_to_clean=normalized.raw_to_clean,
            clean_to_raw=normalized.clean_to_raw,
            language=language or "auto",
            source=source,
            itn_fallback=normalized.itn_fallback,
            itn_fallback_reason=normalized.itn_fallback_reason,
            mapping_coverage=normalized.mapping_coverage,
        )
