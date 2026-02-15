"""
L6 切分层处理器（SegmentationProcessor）。
V3.2.0+dev.20260214.10
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.logging import resolve_loguru_logger
from app.models.sensevoice_models import SentenceSegment, TextSource, WordTimestamp
from app.services.alignment.default_aligner import _strip_trailing_punct_smart
from app.services.alignment.types import L6Input, L6Output
from app.services.segmentation.boundary_mapper import WordBoundaryMapper
from app.services.punctuation.final_splitter import FinalSplitter


class SegmentationProcessor:
    """L6 处理器：仅负责边界决策与句子切分。"""

    # V3.2.0+dev.20260210.03: L6 内建跨 chunk 连续性处理（仅处理高风险残词）。
    _CROSS_CHUNK_CARRY_WORDS = {"a", "an", "the"}
    _SENTENCE_END_PUNCT = {"。", "！", "？", ".", "!", "?"}
    _CUT_PLAN_SINGLETON_TAIL_MAX_GAP_SEC = 0.65
    _CUT_PLAN_SINGLETON_TAIL_MAX_DURATION_SEC = 0.95
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
        self._boundary_mapper = WordBoundaryMapper()

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
        """执行 L6 单路径切分（仅文本边界，不做 speaker/turn 硬切）。"""
        annotated_words = data.annotated_words or []
        pending_prefix_words = self._consume_pending_prefix_words(stream_id)
        cut_plan = data.cut_plan
        has_annotated_words = bool(annotated_words)
        if not has_annotated_words and not pending_prefix_words:
            return L6Output(
                sentence_segments=[],
                words_for_split=[],
                segmentation_report={
                    "boundary_score_stats": {},
                    "forced_split_count": 0.0,
                    "cross_chunk_pending_in_word_count": 0,
                    "cross_chunk_pending_out_word_count": 0,
                    "cross_chunk_dangling_fix_count": 0,
                    "soft_cut_stats": self._build_soft_cut_stats(
                        cut_plan=cut_plan,
                        applied_window_ids=[],
                    ),
                    "stream_id": stream_id,
                    "error_code": "E_L6_SPLIT_EMPTY",
                },
                applied_cut_plan=cut_plan,
            )

        words_for_split = list(pending_prefix_words) + self._build_words_for_split(annotated_words)
        pending_in_word_count = len(pending_prefix_words)
        if not words_for_split:
            return L6Output(
                sentence_segments=[],
                words_for_split=[],
                segmentation_report={
                    "boundary_score_stats": {},
                    "forced_split_count": 0.0,
                    "cross_chunk_pending_in_word_count": pending_in_word_count,
                    "cross_chunk_pending_out_word_count": 0,
                    "cross_chunk_dangling_fix_count": 0,
                    "soft_cut_stats": self._build_soft_cut_stats(
                        cut_plan=cut_plan,
                        applied_window_ids=[],
                    ),
                    "stream_id": stream_id,
                    "chunk_index": chunk_index,
                    "error_code": "E_L6_SPLIT_EMPTY",
                },
                applied_cut_plan=cut_plan,
            )

        applied_window_ids: List[str] = []
        if cut_plan is not None:
            all_sentence_segments, applied_window_ids = self._split_by_cut_plan(
                words_for_split=words_for_split,
                cut_plan=cut_plan,
            )
        else:
            all_sentence_segments = self._final_splitter.split(words_for_split)

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
            fallback_sentence = self._build_single_sentence(words_for_split)
            all_sentence_segments = [fallback_sentence] if fallback_sentence else []
            error_code = "E_L6_SPLIT_EMPTY"
        else:
            error_code = ""

        split_stats = dict(self._final_splitter.last_split_stats or {})
        report: Dict[str, Any] = {
            "boundary_score_stats": split_stats,
            "forced_split_count": float(split_stats.get("force_split_count", 0.0)),
            "cross_chunk_pending_in_word_count": pending_in_word_count,
            "cross_chunk_pending_out_word_count": len(pending_out_words),
            "cross_chunk_dangling_fix_count": dangling_fix_count,
            "soft_cut_stats": self._build_soft_cut_stats(
                cut_plan=cut_plan,
                applied_window_ids=applied_window_ids,
            ),
            "stream_id": stream_id,
            "chunk_index": chunk_index,
            "error_code": error_code,
        }
        self._logger.info(
            "L6 切分完成: stream={} chunk={} input_words={} sentences={} "
            "pending_in={} pending_out={} error={}",
            stream_id,
            chunk_index,
            len(words_for_split),
            len(all_sentence_segments),
            pending_in_word_count,
            len(pending_out_words),
            error_code or "none",
        )

        return L6Output(
            sentence_segments=all_sentence_segments,
            words_for_split=words_for_split,
            segmentation_report=report,
            applied_cut_plan=cut_plan,
        )

    def _split_by_cut_plan(
        self,
        *,
        words_for_split: List[WordTimestamp],
        cut_plan: Any,
    ) -> Tuple[List[SentenceSegment], List[str]]:
        """
        按 CutPlan 执行词流切分。

        说明：
        - 优先使用决策时间映射到词边界；
        - 若无可用边界，回退到 FinalSplitter 默认路径，确保不中断主流程。
        """
        decisions = list(getattr(cut_plan, "decisions", []) or [])
        if len(words_for_split) <= 1 or not decisions:
            return self._final_splitter.split(words_for_split), []

        split_points, split_to_window = self._resolve_cut_plan_split_points(
            words_for_split=words_for_split,
            decisions=decisions,
        )
        if not split_points:
            return self._final_splitter.split(words_for_split), []

        sentence_segments: List[SentenceSegment] = []
        start_idx = 0
        for split_idx in split_points:
            sentence_segments.append(
                self._final_splitter._build_sentence(words_for_split, start_idx, split_idx)
            )
            start_idx = split_idx + 1
        if start_idx < len(words_for_split):
            sentence_segments.append(
                self._final_splitter._build_sentence(
                    words_for_split,
                    start_idx,
                    len(words_for_split) - 1,
                )
            )
        sentence_segments = self._merge_cut_plan_singleton_tail_sentence(sentence_segments)
        applied_window_ids = sorted(set(split_to_window.values()))
        return sentence_segments, applied_window_ids

    def _resolve_cut_plan_split_points(
        self,
        *,
        words_for_split: Sequence[WordTimestamp],
        decisions: Sequence[Any],
    ) -> Tuple[List[int], Dict[int, str]]:
        """
        将 CutPlan 决策时间映射到词边界索引。

        Returns:
            split_points: 词边界左索引列表（i 表示在 i 和 i+1 之间切）。
            split_to_window: split_idx -> window_id
        """
        if len(words_for_split) <= 1:
            return [], {}

        split_to_window: Dict[int, str] = {}
        for decision in sorted(
            decisions,
            key=lambda item: (float(getattr(item, "time", 0.0)), str(getattr(item, "window_id", ""))),
        ):
            decision_time = float(getattr(decision, "time", 0.0))
            window_id = str(getattr(decision, "window_id", "") or "")
            selection = self._boundary_mapper.select_best_boundary(
                words=words_for_split,
                event_time=decision_time,
            )
            if selection is None:
                continue
            split_to_window[selection.split_idx] = window_id

        split_points = sorted(split_to_window.keys())
        return split_points, split_to_window

    def _merge_cut_plan_singleton_tail_sentence(
        self,
        sentence_segments: List[SentenceSegment],
    ) -> List[SentenceSegment]:
        """
        CutPlan 路径尾词守门：将明显误切的末尾 1 词句回并到前一句。

        Why:
        - soft-cut 在误报窗口下可能把末词前停顿当成切口（如 `best | god.`、`about to | be.`）。
        - 该守门只处理“最后一句=1词”的高风险场景，避免影响正常多词句切分。
        """
        if len(sentence_segments) < 2:
            return sentence_segments

        last_sentence = sentence_segments[-1]
        prev_sentence = sentence_segments[-2]
        last_words = list(last_sentence.words or [])
        prev_words = list(prev_sentence.words or [])
        if len(last_words) != 1 or not prev_words:
            return sentence_segments

        left_end = float(getattr(prev_words[-1], "end", 0.0) or 0.0)
        right_start = float(getattr(last_words[0], "start", left_end) or left_end)
        gap_sec = max(0.0, right_start - left_end)
        if gap_sec > self._CUT_PLAN_SINGLETON_TAIL_MAX_GAP_SEC:
            return sentence_segments

        last_start = float(getattr(last_sentence, "start", right_start) or right_start)
        last_end = float(getattr(last_sentence, "end", right_start) or right_start)
        last_duration = max(0.0, last_end - last_start)
        if last_duration > self._CUT_PLAN_SINGLETON_TAIL_MAX_DURATION_SEC:
            return sentence_segments

        prev_tail = str(getattr(prev_words[-1], "word", "") or "").strip()
        if prev_tail.endswith(tuple(self._SENTENCE_END_PUNCT)):
            return sentence_segments

        tail_token = str(getattr(last_words[0], "word", "") or "").strip()
        if not self._is_lowercase_continuation_token(tail_token):
            return sentence_segments

        merged_words = prev_words + last_words
        rebuilt_prev = self._final_splitter._build_sentence(
            merged_words,
            0,
            len(merged_words) - 1,
        )
        self._copy_sentence_metadata(prev_sentence, rebuilt_prev)
        sentence_segments[-2] = rebuilt_prev
        sentence_segments.pop()
        return sentence_segments

    @staticmethod
    def _is_lowercase_continuation_token(token: str) -> bool:
        text = str(token or "").strip().lstrip("\"'“”‘’([{")
        if not text:
            return False
        for ch in text:
            if ch.isalpha():
                return ch.islower()
        return False

    def _build_soft_cut_stats(
        self,
        *,
        cut_plan: Any,
        applied_window_ids: Sequence[str],
    ) -> Dict[str, Any]:
        if cut_plan is None:
            return {
                "enabled": False,
                "plan_id": "",
                "decision_count": 0,
                "deferred_count": 0,
                "applied_decision_count": 0,
                "applied_window_ids": [],
                "unapplied_window_ids": [],
                "reason_stats": {},
                "risk_stats": {},
            }

        decisions = list(getattr(cut_plan, "decisions", []) or [])
        deferred_cuts = list(getattr(cut_plan, "deferred_cuts", []) or [])
        decision_window_ids = [
            str(getattr(item, "window_id", "") or "")
            for item in decisions
            if str(getattr(item, "window_id", "") or "")
        ]
        applied_window_set = set(applied_window_ids)
        applied_decision_count = sum(1 for window_id in decision_window_ids if window_id in applied_window_set)
        unapplied_window_ids = sorted(set(decision_window_ids) - applied_window_set)

        reason_stats = dict(
            Counter(
                str(getattr(item, "reason", "") or "")
                for item in decisions
                if str(getattr(item, "reason", "") or "")
            )
        )
        risk_stats = dict(
            Counter(
                str(getattr(item, "risk", "") or "")
                for item in decisions
                if str(getattr(item, "risk", "") or "")
            )
        )
        return {
            "enabled": True,
            "plan_id": str(getattr(cut_plan, "plan_id", "") or ""),
            "decision_count": len(decisions),
            "deferred_count": len(deferred_cuts),
            "applied_decision_count": applied_decision_count,
            "applied_window_ids": sorted(applied_window_set),
            "unapplied_window_ids": unapplied_window_ids,
            "reason_stats": reason_stats,
            "risk_stats": risk_stats,
        }

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
            cloned = WordTimestamp(
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
            # Why: 词级 speaker/turn 信号用于后续跨 speaker 残留修复，克隆时必须保留。
            setattr(cloned, "speaker_id", getattr(word, "speaker_id", None))
            setattr(cloned, "turn_id", getattr(word, "turn_id", None))
            cloned_words.append(cloned)
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
    def _build_words_for_split(annotated_words: List[Any]) -> List[WordTimestamp]:
        words: List[WordTimestamp] = []
        for item in annotated_words:
            word_text = f"{item.word}{item.trailing_punct or ''}"
            word = WordTimestamp(
                word=word_text,
                start=float(item.start) if item.start is not None else 0.0,
                end=float(item.end) if item.end is not None else 0.0,
                confidence=item.confidence,
                confidence_source=item.confidence_source,
                is_pseudo=False,
            )
            # Why: 后续“单次闭环”修复需要词级 speaker/turn 信息判断跨 speaker 残留。
            setattr(word, "speaker_id", getattr(item, "speaker_id", None))
            setattr(word, "turn_id", getattr(item, "turn_id", None))
            words.append(word)
        return words

    @staticmethod
    def _build_single_sentence(
        words: List[WordTimestamp],
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
