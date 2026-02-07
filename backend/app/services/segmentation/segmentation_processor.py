"""
L6 切分层处理器（SegmentationProcessor）。
V3.2.0+dev.20260207.02
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.logging import resolve_loguru_logger
from app.models.sensevoice_models import SentenceSegment, TextSource, WordTimestamp
from app.services.alignment.default_aligner import _strip_trailing_punct_smart
from app.services.alignment.types import L6Input, L6Output
from app.services.punctuation.final_splitter import FinalSplitter


class SegmentationProcessor:
    """L6 处理器：仅负责边界决策与句子切分。"""

    def __init__(
        self,
        *,
        final_splitter: FinalSplitter,
        logger: Optional[Any] = None,
        is_keep_sentence_end_punct: bool = False,
    ) -> None:
        self._logger = resolve_loguru_logger(
            logger,
            __name__,
            layer="L6",
            processor_name="segmentation_processor",
        )
        self._final_splitter = final_splitter
        self._is_keep_sentence_end_punct = is_keep_sentence_end_punct

    def process(self, data: L6Input) -> L6Output:
        """执行 L6 单路径切分。"""
        words_for_split = self._build_words_for_split(data.annotated_words)
        if not words_for_split:
            return L6Output(
                sentence_segments=[],
                words_for_split=[],
                segmentation_report={
                    "boundary_score_stats": {},
                    "forced_split_count": 0.0,
                    "error_code": "E_L6_SPLIT_EMPTY",
                },
            )

        sentence_segments = self._final_splitter.split(words_for_split)
        if not self._is_keep_sentence_end_punct:
            self._strip_sentence_end_punct(sentence_segments)
        self._finalize_sentence_metadata(sentence_segments)

        if not sentence_segments:
            fallback_sentence = self._build_single_sentence(words_for_split)
            sentence_segments = [fallback_sentence] if fallback_sentence else []
            error_code = "E_L6_SPLIT_EMPTY"
        else:
            error_code = ""

        split_stats = dict(self._final_splitter.last_split_stats or {})
        report: Dict[str, Any] = {
            "boundary_score_stats": split_stats,
            "forced_split_count": float(split_stats.get("force_split_count", 0.0)),
            "error_code": error_code,
        }
        self._logger.info(
            "L6 切分完成: input_words={} sentences={} error={}",
            len(words_for_split),
            len(sentence_segments),
            error_code or "none",
        )
        return L6Output(
            sentence_segments=sentence_segments,
            words_for_split=words_for_split,
            segmentation_report=report,
        )

    @staticmethod
    def _build_words_for_split(annotated_words: List[Any]) -> List[WordTimestamp]:
        words: List[WordTimestamp] = []
        for item in annotated_words:
            word_text = f"{item.word}{item.trailing_punct or ''}"
            words.append(
                WordTimestamp(
                    word=word_text,
                    start=float(item.start) if item.start is not None else 0.0,
                    end=float(item.end) if item.end is not None else 0.0,
                    confidence=item.confidence,
                    confidence_source=item.confidence_source,
                    is_pseudo=False,
                )
            )
        return words

    @staticmethod
    def _build_single_sentence(words: List[WordTimestamp]) -> Optional[SentenceSegment]:
        if not words:
            return None
        text = "".join((word.word or "") for word in words).strip()
        start = float(words[0].start)
        end = float(words[-1].end)
        confidence_values = [float(word.confidence) for word in words if word.confidence is not None]
        confidence = sum(confidence_values) / len(confidence_values) if confidence_values else None
        return SentenceSegment(
            text=text,
            text_clean=text,
            start=start,
            end=end,
            confidence=confidence,
            words=words,
            source=TextSource.WHISPER_PATCH,
            is_draft=False,
            is_finalized=True,
        )

    @staticmethod
    def _strip_sentence_end_punct(sentences: List[SentenceSegment]) -> None:
        for sentence in sentences:
            text_clean = sentence.text_clean or sentence.text or ""
            sentence.text_clean = _strip_trailing_punct_smart(text_clean)
            sentence.text = sentence.text_clean

    @staticmethod
    def _finalize_sentence_metadata(sentences: List[SentenceSegment]) -> None:
        for sentence in sentences:
            sentence.source = TextSource.WHISPER_PATCH
            sentence.is_draft = False
            sentence.is_finalized = True
