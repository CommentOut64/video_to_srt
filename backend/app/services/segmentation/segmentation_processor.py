"""
L6 切分层处理器（SegmentationProcessor）。
V3.2.0+dev.20260207.02
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.logging import resolve_loguru_logger
from app.models.sensevoice_models import SentenceSegment, TextSource, WordTimestamp
from app.services.alignment.default_aligner import _strip_trailing_punct_smart
from app.services.alignment.types import L6Input, L6Output
from app.services.punctuation.final_splitter import FinalSplitter


class SegmentationProcessor:
    """L6 处理器：仅负责边界决策与句子切分。"""

    # V3.2.0+dev.20260210.03: L6 内建跨 chunk 连续性处理（仅处理高风险残词）。
    _CROSS_CHUNK_CARRY_WORDS = {"a", "an", "the"}
    _SENTENCE_END_PUNCT = {"。", "！", "？", ".", "!", "?"}

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
        self._pending_prefix_words_by_stream: Dict[str, List[WordTimestamp]] = {}

    def reset_state(self) -> None:
        """重置 L6 跨 chunk 状态（新任务开始时调用）。"""
        self._pending_prefix_words_by_stream.clear()

    def process(
        self,
        data: L6Input,
        *,
        stream_id: str = "main",
        chunk_index: Optional[int] = None,
        is_last_chunk: bool = False,
    ) -> L6Output:
        """执行 L6 单路径切分（V3.2.0+dev.20260207.03: speaker run 强制边界）。"""
        annotated_words = data.annotated_words or []
        pending_prefix_words = self._consume_pending_prefix_words(stream_id)
        has_annotated_words = bool(annotated_words)
        if not has_annotated_words and not pending_prefix_words:
            return L6Output(
                sentence_segments=[],
                words_for_split=[],
                segmentation_report={
                    "boundary_score_stats": {},
                    "forced_split_count": 0.0,
                    "speaker_run_count": 0,
                    "speaker_forced_split_count": 0,
                    "cross_chunk_pending_in_word_count": 0,
                    "cross_chunk_pending_out_word_count": 0,
                    "cross_chunk_dangling_fix_count": 0,
                    "stream_id": stream_id,
                    "error_code": "E_L6_SPLIT_EMPTY",
                },
            )

        # V3.2.0+dev.20260207.03: 按 speaker_id 分组为连续 run
        speaker_runs = self._group_by_speaker_runs(annotated_words)
        if not speaker_runs and pending_prefix_words:
            speaker_runs = [[]]
        all_sentence_segments: List[SentenceSegment] = []
        speaker_forced_split_count = len(speaker_runs) - 1 if len(speaker_runs) > 1 else 0
        pending_in_word_count = len(pending_prefix_words)

        # 对每个 speaker run 分别切分
        for run_index, run_annotated_words in enumerate(speaker_runs):
            run_words = self._build_words_for_split(run_annotated_words)
            if run_index == 0 and pending_prefix_words:
                run_words = list(pending_prefix_words) + run_words
            if not run_words:
                continue
            run_segments = self._final_splitter.split(run_words)
            if run_segments:
                run_speaker_id, run_turn_id = self._extract_run_identity(run_annotated_words)
                self._apply_identity_to_sentences(
                    run_segments,
                    speaker_id=run_speaker_id,
                    turn_id=run_turn_id,
                )
                all_sentence_segments.extend(run_segments)

        pending_out_words: List[WordTimestamp] = []
        dangling_fix_count = 0
        if all_sentence_segments and not is_last_chunk:
            (
                all_sentence_segments,
                pending_out_words,
                dangling_fix_count,
            ) = self._extract_cross_chunk_pending_tail(all_sentence_segments)
            if pending_out_words:
                self._pending_prefix_words_by_stream[stream_id] = self._clone_words(pending_out_words)
        elif pending_prefix_words:
            # 没有产出时归还前缀，避免状态丢失。
            self._pending_prefix_words_by_stream[stream_id] = self._clone_words(pending_prefix_words)

        # 后处理
        if not self._is_keep_sentence_end_punct:
            self._strip_sentence_end_punct(all_sentence_segments)
        self._normalize_carried_article_sentence_case(
            sentence_segments=all_sentence_segments,
            pending_in_word_count=pending_in_word_count,
        )
        self._finalize_sentence_metadata(all_sentence_segments)

        # 降级处理：若所有 run 切分后仍为空
        if not all_sentence_segments:
            words_for_split = list(pending_prefix_words) + self._build_words_for_split(annotated_words)
            fallback_speaker_id, fallback_turn_id = self._extract_run_identity(annotated_words)
            fallback_sentence = self._build_single_sentence(
                words_for_split,
                speaker_id=fallback_speaker_id,
                turn_id=fallback_turn_id,
            )
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
            "cross_chunk_pending_in_word_count": pending_in_word_count,
            "cross_chunk_pending_out_word_count": len(pending_out_words),
            "cross_chunk_dangling_fix_count": dangling_fix_count,
            "stream_id": stream_id,
            "chunk_index": chunk_index,
            "error_code": error_code,
        }
        self._logger.info(
            "L6 切分完成: stream={} chunk={} input_words={} sentences={} "
            "speaker_runs={} speaker_forced={} pending_in={} pending_out={} error={}",
            stream_id,
            chunk_index,
            len(annotated_words),
            len(all_sentence_segments),
            len(speaker_runs),
            speaker_forced_split_count,
            pending_in_word_count,
            len(pending_out_words),
            error_code or "none",
        )

        # words_for_split 返回完整输入词流（包含跨 chunk 前缀），用于下游兼容。
        all_words_for_split = list(pending_prefix_words) + self._build_words_for_split(annotated_words)
        return L6Output(
            sentence_segments=all_sentence_segments,
            words_for_split=all_words_for_split,
            segmentation_report=report,
        )

    def _consume_pending_prefix_words(self, stream_id: str) -> List[WordTimestamp]:
        if not stream_id:
            return []
        cached_words = self._pending_prefix_words_by_stream.pop(stream_id, None)
        if not cached_words:
            return []
        return self._clone_words(cached_words)

    def _extract_cross_chunk_pending_tail(
        self,
        sentence_segments: List[SentenceSegment],
    ) -> Tuple[List[SentenceSegment], List[WordTimestamp], int]:
        if not sentence_segments:
            return sentence_segments, [], 0

        tail_words = self._detect_dangling_tail_words(sentence_segments[-1])
        if not tail_words:
            return sentence_segments, [], 0

        cut_word_count = len(tail_words)
        kept_words = sentence_segments[-1].words[:-cut_word_count]
        if not kept_words:
            return sentence_segments, [], 0

        rebuilt_last = self._final_splitter._build_sentence(kept_words, 0, len(kept_words) - 1)
        self._copy_sentence_metadata(sentence_segments[-1], rebuilt_last)
        sentence_segments[-1] = rebuilt_last
        return sentence_segments, self._clone_words(tail_words), 1

    def _detect_dangling_tail_words(self, sentence: SentenceSegment) -> List[WordTimestamp]:
        words = list(sentence.words or [])
        if len(words) <= 1:
            return []

        last_token = self._normalize_boundary_token(words[-1].word)
        if last_token not in self._CROSS_CHUNK_CARRY_WORDS:
            return []

        anchor_token_raw = str(words[-2].word or "").strip()
        has_sentence_end_anchor = any(
            anchor_token_raw.endswith(ch) for ch in self._SENTENCE_END_PUNCT
        )
        if not has_sentence_end_anchor:
            return []

        return [words[-1]]

    def _normalize_carried_article_sentence_case(
        self,
        *,
        sentence_segments: List[SentenceSegment],
        pending_in_word_count: int,
    ) -> None:
        """当跨 chunk 回放冠词时，规范句首为英文句式大小写。"""
        if pending_in_word_count <= 0 or not sentence_segments:
            return
        first_sentence = sentence_segments[0]
        words = list(first_sentence.words or [])
        if len(words) < 2:
            return

        carry_token = self._normalize_boundary_token(words[0].word)
        if carry_token not in self._CROSS_CHUNK_CARRY_WORDS:
            return

        words[0].word = words[0].word.capitalize()

        second_raw = str(words[1].word or "")
        second_trimmed = second_raw.strip()
        if second_trimmed and second_trimmed[:1].isupper() and second_trimmed[1:].islower():
            words[1].word = second_trimmed[:1].lower() + second_trimmed[1:]

        rebuilt = self._final_splitter._build_sentence(words, 0, len(words) - 1)
        self._copy_sentence_metadata(first_sentence, rebuilt)
        sentence_segments[0] = rebuilt

    @staticmethod
    def _normalize_boundary_token(token: Optional[str]) -> str:
        token_str = str(token or "")
        return token_str.strip().strip("\"'”’）)]}】」』，、,;；:：。？！.!?").lower()

    @staticmethod
    def _clone_words(words: Sequence[WordTimestamp]) -> List[WordTimestamp]:
        cloned_words: List[WordTimestamp] = []
        for word in words:
            cloned_words.append(
                WordTimestamp(
                    word=word.word,
                    start=word.start,
                    end=word.end,
                    confidence=word.confidence,
                    confidence_raw=word.confidence_raw,
                    confidence_display_raw=word.confidence_display_raw,
                    confidence_source=word.confidence_source,
                    token_type=word.token_type,
                    is_pseudo=word.is_pseudo,
                )
            )
        return cloned_words

    @staticmethod
    def _copy_sentence_metadata(source: SentenceSegment, target: SentenceSegment) -> None:
        target.source = source.source
        target.is_draft = source.is_draft
        target.is_finalized = source.is_finalized
        target.alignment_score = source.alignment_score
        target.matched_ratio = source.matched_ratio
        target.whisper_text = source.whisper_text
        target.sv_original_text = source.sv_original_text
        target.confidence_source = source.confidence_source
        target.warning_type = source.warning_type
        target.speaker_id = source.speaker_id
        target.turn_id = source.turn_id
        target.group_id = source.group_id
        target.is_soft_break = source.is_soft_break
        target.group_position = source.group_position

    @staticmethod
    def _extract_run_identity(annotated_words: List[Any]) -> Tuple[Optional[str], Optional[str]]:
        """从 run 中提取统一 speaker/turn 标识。"""
        run_speaker_id: Optional[str] = None
        run_turn_id: Optional[str] = None
        for word in annotated_words:
            if run_speaker_id is None:
                run_speaker_id = getattr(word, "speaker_id", None)
            if run_turn_id is None:
                run_turn_id = getattr(word, "turn_id", None)
            if run_speaker_id is not None and run_turn_id is not None:
                break
        return run_speaker_id, run_turn_id

    @staticmethod
    def _apply_identity_to_sentences(
        sentences: List[SentenceSegment],
        *,
        speaker_id: Optional[str],
        turn_id: Optional[str],
    ) -> None:
        """将 run 级 speaker/turn 标识写入句子。"""
        for sentence in sentences:
            sentence.speaker_id = speaker_id
            sentence.turn_id = turn_id

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
    def _build_single_sentence(
        words: List[WordTimestamp],
        *,
        speaker_id: Optional[str] = None,
        turn_id: Optional[str] = None,
    ) -> Optional[SentenceSegment]:
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
            speaker_id=speaker_id,
            turn_id=turn_id,
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
