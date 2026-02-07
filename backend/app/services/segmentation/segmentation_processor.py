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
        """执行 L6 单路径切分（V3.2.0+dev.20260207.03: speaker run 强制边界）。"""
        annotated_words = data.annotated_words or []
        if not annotated_words:
            return L6Output(
                sentence_segments=[],
                words_for_split=[],
                segmentation_report={
                    "boundary_score_stats": {},
                    "forced_split_count": 0.0,
                    "speaker_run_count": 0,
                    "speaker_forced_split_count": 0,
                    "error_code": "E_L6_SPLIT_EMPTY",
                },
            )

        # V3.2.0+dev.20260207.03: 按 speaker_id 分组为连续 run
        speaker_runs = self._group_by_speaker_runs(annotated_words)
        all_sentence_segments: List[SentenceSegment] = []
        speaker_forced_split_count = len(speaker_runs) - 1 if len(speaker_runs) > 1 else 0

        # 对每个 speaker run 分别切分
        for run_annotated_words in speaker_runs:
            run_words = self._build_words_for_split(run_annotated_words)
            if not run_words:
                continue
            run_segments = self._final_splitter.split(run_words)
            if run_segments:
                all_sentence_segments.extend(run_segments)

        # 后处理
        if not self._is_keep_sentence_end_punct:
            self._strip_sentence_end_punct(all_sentence_segments)
        self._finalize_sentence_metadata(all_sentence_segments)

        # 降级处理：若所有 run 切分后仍为空
        if not all_sentence_segments:
            words_for_split = self._build_words_for_split(annotated_words)
            fallback_sentence = self._build_single_sentence(words_for_split)
            all_sentence_segments = [fallback_sentence] if fallback_sentence else []
            error_code = "E_L6_SPLIT_EMPTY"
        else:
            error_code = ""

        split_stats = dict(self._final_splitter.last_split_stats or {})
        report: Dict[str, Any] = {
            "boundary_score_stats": split_stats,
            "forced_split_count": float(split_stats.get("force_split_count", 0.0)),
            "speaker_run_count": len(speaker_runs),
            "speaker_forced_split_count": speaker_forced_split_count,
            "error_code": error_code,
        }
        self._logger.info(
            "L6 切分完成: input_words={} sentences={} speaker_runs={} speaker_forced={} error={}",
            len(annotated_words),
            len(all_sentence_segments),
            len(speaker_runs),
            speaker_forced_split_count,
            error_code or "none",
        )

        # words_for_split 返回完整列表（非分组）用于下游兼容
        all_words_for_split = self._build_words_for_split(annotated_words)
        return L6Output(
            sentence_segments=all_sentence_segments,
            words_for_split=all_words_for_split,
            segmentation_report=report,
        )

    @staticmethod
    def _group_by_speaker_runs(annotated_words: List[Any]) -> List[List[Any]]:
        """按 speaker_id 分组为连续 run（V3.2.0+dev.20260207.03）。

        降级策略：只有"前后都非空且不同"才触发硬边界。
        """
        if not annotated_words:
            return []

        runs: List[List[Any]] = []
        current_run: List[Any] = []
        last_speaker_id: Optional[str] = None

        for word in annotated_words:
            current_speaker_id = getattr(word, "speaker_id", None)  # 兼容旧测试 mock

            # 判断是否需要切换 run
            need_split = False
            if current_run:  # 非首词
                # 只有"前后都非空且不同"才强制断开
                if (last_speaker_id is not None
                    and current_speaker_id is not None
                    and last_speaker_id != current_speaker_id):
                    need_split = True

            if need_split:
                runs.append(current_run)
                current_run = [word]
            else:
                current_run.append(word)

            last_speaker_id = current_speaker_id

        if current_run:
            runs.append(current_run)

        return runs if runs else [[]]

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
