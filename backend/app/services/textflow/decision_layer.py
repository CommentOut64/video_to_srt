"""
裁决层切分处理器（SegmentationProcessor）。
V3.2.0+dev.20260215.24
"""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.logging import resolve_loguru_logger
from app.models.sensevoice_models import SentenceSegment, TextSource, WordTimestamp
from app.services.alignment.default_aligner import _strip_trailing_punct_smart
from app.services.alignment.types import DecisionLayerInput, DecisionLayerOutput, OutputTrace
from app.services.segmentation.boundary_mapper import WordBoundaryMapper
from app.services.punctuation.final_splitter import FinalSplitter


class SegmentationProcessor:
    """裁决层处理器：仅负责边界决策与句子切分。"""

    # V3.2.0+dev.20260210.03: 裁决层内建跨 chunk 连续性处理（仅处理高风险残词）。
    _CROSS_CHUNK_CARRY_WORDS = {"a", "an", "the"}
    _SENTENCE_END_PUNCT = {"。", "！", "？", ".", "!", "?"}
    _CUT_PLAN_SINGLETON_TAIL_MAX_GAP_SEC = 0.65
    _CUT_PLAN_SINGLETON_TAIL_MAX_DURATION_SEC = 0.95
    _CUT_PLAN_SHORT_TAIL_MAX_WORDS = 2
    _CUT_PLAN_SHORT_TAIL_MAX_GAP_SEC = 0.45
    _CUT_PLAN_SHORT_TAIL_MAX_DURATION_SEC = 2.20
    _CUT_PLAN_SHORT_TAIL_GUARD_REASONS = {
        "speaker_change",
        "pause",
    }
    _CUT_PLAN_SINGLETON_MID_MAX_GAP_SEC = 0.60
    _CUT_PLAN_SINGLETON_MID_MAX_DURATION_SEC = 0.95
    _CUT_PLAN_SINGLETON_MID_GUARD_REASONS = {
        "speaker_change",
        "hard_limit_forced",
    }
    _SOFT_CUT_SHORT_TURN_CONTINUATION_SEC = 0.75
    _SOFT_CUT_SHORT_TURN_CONTINUATION_GAP_SEC = 0.80
    _SPEAKER_REPAIR_MIN_TURN_DURATION_SEC = 0.45
    _SPEAKER_REPAIR_STRONG_BREAK_MIN_PAUSE_SEC = 0.30
    _SPEAKER_REPAIR_SENTENCE_END_PUNCT = {"。", "！", "？", ".", "!", "?"}
    _SPEAKER_REPAIR_EDGE_GUARD_SEC = 0.15
    def __init__(
        self,
        *,
        final_splitter: FinalSplitter,
        logger: Optional[Any] = None,
        is_keep_sentence_end_punct: bool = False,
        is_enable_soft_cut_overlap_degrade: bool = False,
        is_enable_speaker_guided_split: bool = True,
    ) -> None:
        self._logger = resolve_loguru_logger(
            logger,
            __name__,
            layer="裁决层",
            processor_name="segmentation_processor",
        )
        self._final_splitter = final_splitter
        self._is_keep_sentence_end_punct = is_keep_sentence_end_punct
        self._is_enable_soft_cut_overlap_degrade = is_enable_soft_cut_overlap_degrade
        self._is_enable_speaker_guided_split = is_enable_speaker_guided_split
        self._pending_prefix_words_by_stream: Dict[str, List[WordTimestamp]] = {}
        self._boundary_mapper = WordBoundaryMapper()
        self._active_vad_intervals: List[Tuple[float, float]] = []

    def reset_state(self) -> None:
        """重置裁决层跨 chunk 状态（新任务开始时调用）。"""
        self._pending_prefix_words_by_stream.clear()

    def process(
        self,
        data: DecisionLayerInput,
        *,
        stream_id: str = "main",
        chunk_index: Optional[int] = None,
        is_last_chunk: bool = False,
    ) -> DecisionLayerOutput:
        """执行裁决层单路径切分，并在层内执行一次跨 speaker 残留修复。"""
        annotated_words = data.annotated_words or []
        pending_prefix_words = self._consume_pending_prefix_words(stream_id)
        cut_plan = self._normalize_cut_plan(
            data.cut_plan,
            stream_id=stream_id,
            chunk_index=chunk_index,
        )
        aligned_facts = data.aligned_facts
        fused_evidence = data.fused_evidence
        has_annotated_words = bool(annotated_words)
        if not has_annotated_words and not pending_prefix_words:
            return DecisionLayerOutput(
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
                    "aligned_facts_stats": self._build_aligned_facts_stats(aligned_facts),
                    "fused_evidence_stats": self._build_fused_evidence_stats(fused_evidence),
                    "stream_id": stream_id,
                    "error_code": "E_DECISION_SPLIT_EMPTY",
                },
                applied_cut_plan=cut_plan,
                output_traces=[],
            )

        words_for_split = list(pending_prefix_words) + self._build_words_for_split(annotated_words)
        pending_in_word_count = len(pending_prefix_words)
        if not words_for_split:
            return DecisionLayerOutput(
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
                    "aligned_facts_stats": self._build_aligned_facts_stats(aligned_facts),
                    "fused_evidence_stats": self._build_fused_evidence_stats(fused_evidence),
                    "stream_id": stream_id,
                    "chunk_index": chunk_index,
                    "error_code": "E_DECISION_SPLIT_EMPTY",
                },
                applied_cut_plan=cut_plan,
                output_traces=[],
            )

        applied_window_ids: List[str] = []
        output_traces: List[OutputTrace] = []
        all_sentence_segments, applied_window_ids, output_traces = self._split_by_cut_plan(
            words_for_split=words_for_split,
            cut_plan=cut_plan,
            fallback_clean_text_ref=str(getattr(data, "fallback_clean_text_ref", "") or ""),
            fallback_punctuation_positions=list(
                getattr(data, "fallback_punctuation_positions", []) or []
            ),
        )
        self._apply_output_traces_to_sentences(
            sentence_segments=all_sentence_segments,
            output_traces=output_traces,
        )
        self._active_vad_intervals = list(data.vad_intervals or [])
        speaker_repair_split_count = 0
        if self._is_enable_speaker_guided_split and isinstance(all_sentence_segments, list):
            speaker_repair_split_count = self._repair_cross_speaker_sentences_once(
                sentences=all_sentence_segments,
                turns=self._collect_speaker_turns_for_repair(aligned_facts),
            )
            if speaker_repair_split_count > 0:
                self._logger.info(
                    "跨speaker残留修复: repaired_splits={}",
                    speaker_repair_split_count,
                )
                output_traces = self._normalize_output_traces_for_sentences(
                    sentence_segments=all_sentence_segments,
                    output_traces=output_traces,
                    default_reason="post_identity_binding",
                )

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
        self._apply_output_traces_to_sentences(
            sentence_segments=all_sentence_segments,
            output_traces=output_traces,
        )

        # 降级处理：若所有 run 切分后仍为空
        if not all_sentence_segments:
            fallback_sentence = self._build_single_sentence(words_for_split)
            all_sentence_segments = [fallback_sentence] if fallback_sentence else []
            output_traces = self._build_default_output_traces(
                all_sentence_segments,
                default_reason="fallback_single_sentence",
            )
            self._apply_output_traces_to_sentences(
                sentence_segments=all_sentence_segments,
                output_traces=output_traces,
            )
            error_code = "E_DECISION_SPLIT_EMPTY"
        else:
            error_code = ""

        split_stats = dict(self._final_splitter.last_split_stats or {})
        report: Dict[str, Any] = {
            "boundary_score_stats": split_stats,
            "forced_split_count": float(split_stats.get("force_split_count", 0.0)),
            "cross_chunk_pending_in_word_count": pending_in_word_count,
            "cross_chunk_pending_out_word_count": len(pending_out_words),
            "cross_chunk_dangling_fix_count": dangling_fix_count,
            "speaker_repair_split_count": int(speaker_repair_split_count),
            "soft_cut_stats": self._build_soft_cut_stats(
                cut_plan=cut_plan,
                applied_window_ids=applied_window_ids,
            ),
            "aligned_facts_stats": self._build_aligned_facts_stats(aligned_facts),
            "fused_evidence_stats": self._build_fused_evidence_stats(fused_evidence),
            "stream_id": stream_id,
            "chunk_index": chunk_index,
            "error_code": error_code,
            "output_trace": [self._serialize_output_trace(item) for item in output_traces],
        }
        self._logger.info(
            "裁决层切分完成: stream={} chunk={} input_words={} sentences={} "
            "pending_in={} pending_out={} error={}",
            stream_id,
            chunk_index,
            len(words_for_split),
            len(all_sentence_segments),
            pending_in_word_count,
            len(pending_out_words),
            error_code or "none",
        )
        self._active_vad_intervals = []

        return DecisionLayerOutput(
            sentence_segments=all_sentence_segments,
            words_for_split=words_for_split,
            segmentation_report=report,
            applied_cut_plan=cut_plan,
            output_traces=output_traces,
        )

    @staticmethod
    def _build_aligned_facts_stats(aligned_facts: Any) -> Dict[str, Any]:
        """输出集合层事实统计，便于阶段3观测。"""
        if aligned_facts is None:
            return {
                "word_count": 0,
                "turn_count": 0,
                "fast_draft_cut_count": 0,
                "time_mapping_count": 0,
                "mapping_quality": {},
            }
        time_mappings = list(getattr(aligned_facts, "time_mappings", []) or [])
        quality_counter: Counter[str] = Counter()
        for item in time_mappings:
            if not isinstance(item, dict):
                continue
            quality = str(item.get("mapping_quality", "") or "unknown")
            quality_counter[quality] += 1
        return {
            "word_count": int(len(getattr(aligned_facts, "annotated_words", []) or [])),
            "turn_count": int(len(getattr(aligned_facts, "speaker_turns", []) or [])),
            "fast_draft_cut_count": int(len(getattr(aligned_facts, "fast_draft_cuts", []) or [])),
            "time_mapping_count": int(len(time_mappings)),
            "mapping_quality": dict(quality_counter),
        }

    @staticmethod
    def _normalize_cut_plan(
        cut_plan: Any,
        *,
        stream_id: str,
        chunk_index: Optional[int],
    ) -> Any:
        if cut_plan is not None:
            return cut_plan
        chunk_part = int(chunk_index) if chunk_index is not None else -1
        return SimpleNamespace(
            plan_id=f"{stream_id}-fallback-{chunk_part}",
            block_id=f"{stream_id}:{chunk_part}",
            decisions=[],
            deferred_cuts=[],
            generation_report={
                "fallback_reason": "missing_cut_plan",
                "generated_by": "segmentation_processor",
            },
        )

    @staticmethod
    def _build_fused_evidence_stats(fused_evidence: Any) -> Dict[str, Any]:
        """输出评分层统计，便于阶段4观测。"""
        if fused_evidence is None:
            return {
                "speaker_change_count": 0,
                "pause_anchor_count": 0,
                "semantic_anchor_count": 0,
                "punctuation_anchor_count": 0,
                "report_keys": [],
            }
        report = dict(getattr(fused_evidence, "evidence_report", {}) or {})
        return {
            "speaker_change_count": int(len(getattr(fused_evidence, "speaker_changes", []) or [])),
            "pause_anchor_count": int(len(getattr(fused_evidence, "pause_anchors", []) or [])),
            "semantic_anchor_count": int(len(getattr(fused_evidence, "semantic_anchors", []) or [])),
            "punctuation_anchor_count": int(len(getattr(fused_evidence, "punctuation_anchors", []) or [])),
            "report_keys": sorted([str(item) for item in report.keys()]),
        }

    def _split_by_cut_plan(
        self,
        *,
        words_for_split: List[WordTimestamp],
        cut_plan: Any,
        fallback_clean_text_ref: str = "",
        fallback_punctuation_positions: Optional[Sequence[Any]] = None,
    ) -> Tuple[List[SentenceSegment], List[str], List[OutputTrace]]:
        """
        按 CutPlan 执行词流切分。

        说明：
        - 优先使用决策时间映射到词边界；
        - 若无可用边界，回退到 FinalSplitter 默认路径，确保不中断主流程。
        """
        decisions = list(getattr(cut_plan, "decisions", []) or [])
        if len(words_for_split) <= 1 or not decisions:
            sentence_segments = self._final_splitter.split(
                words_for_split,
                clean_text=fallback_clean_text_ref or None,
                punctuation_positions=list(fallback_punctuation_positions or []),
            )
            output_traces = self._build_default_output_traces(sentence_segments)
            return sentence_segments, [], output_traces

        split_points, split_to_window, split_to_mapping = self._resolve_cut_plan_split_points(
            words_for_split=words_for_split,
            decisions=decisions,
        )
        decision_by_window = self._build_cut_plan_decision_map(decisions=decisions)
        split_points = self._filter_cut_plan_singleton_split_points(
            words_for_split=words_for_split,
            split_points=split_points,
            split_to_window=split_to_window,
            decision_by_window=decision_by_window,
        )
        split_points = self._augment_split_points_with_sentence_end_punct(
            words_for_split=words_for_split,
            split_points=split_points,
            split_to_mapping=split_to_mapping,
        )
        if not split_points:
            sentence_segments = self._final_splitter.split(
                words_for_split,
                clean_text=fallback_clean_text_ref or None,
                punctuation_positions=list(fallback_punctuation_positions or []),
            )
            output_traces = self._build_default_output_traces(sentence_segments)
            return sentence_segments, [], output_traces

        sentence_segments: List[SentenceSegment] = []
        output_traces: List[OutputTrace] = []
        start_idx = 0
        sentence_index = 0
        for split_idx in split_points:
            sentence = self._final_splitter._build_sentence(words_for_split, start_idx, split_idx)
            sentence_segments.append(sentence)
            output_traces.append(
                self._build_cut_plan_output_trace(
                    sentence_index=sentence_index,
                    sentence=sentence,
                    split_idx=split_idx,
                    split_to_window=split_to_window,
                    split_to_mapping=split_to_mapping,
                    decision_by_window=decision_by_window,
                )
            )
            start_idx = split_idx + 1
            sentence_index += 1
        if start_idx < len(words_for_split):
            tail_sentence = self._final_splitter._build_sentence(
                words_for_split,
                start_idx,
                len(words_for_split) - 1,
            )
            sentence_segments.append(tail_sentence)
            output_traces.append(
                OutputTrace(
                    sentence_index=sentence_index,
                    split_reason="tail_flush",
                    split_risk="",
                    window_id="",
                    pyannote_frame_time=None,
                    mapped_cut_time=float(tail_sentence.end),
                    mapping_quality="tail",
                    mapping_reason="tail_flush",
                    sentence_start=float(tail_sentence.start),
                    sentence_end=float(tail_sentence.end),
                )
            )
        sentence_segments, output_traces = self._merge_cut_plan_singleton_tail_sentence(
            sentence_segments=sentence_segments,
            output_traces=output_traces,
        )
        applied_window_ids = sorted(
            {
                str(split_to_window.get(split_idx, "") or "")
                for split_idx in split_points
                if str(split_to_window.get(split_idx, "") or "")
            }
        )
        return sentence_segments, applied_window_ids, output_traces

    @staticmethod
    def _build_cut_plan_decision_map(
        *,
        decisions: Sequence[Any],
    ) -> Dict[str, Any]:
        decision_by_window: Dict[str, Any] = {}
        for decision in decisions:
            window_id = str(getattr(decision, "window_id", "") or "")
            if not window_id:
                continue
            decision_by_window[window_id] = decision
        return decision_by_window

    def _filter_cut_plan_singleton_split_points(
        self,
        *,
        words_for_split: Sequence[WordTimestamp],
        split_points: Sequence[int],
        split_to_window: Dict[int, str],
        decision_by_window: Dict[str, Any],
    ) -> List[int]:
        if len(words_for_split) <= 1 or not split_points:
            return list(split_points)

        kept_split_points: List[int] = []
        for idx, split_idx in enumerate(split_points):
            next_split = split_points[idx + 1] if idx + 1 < len(split_points) else None
            window_id = str(split_to_window.get(split_idx, "") or "")
            decision = decision_by_window.get(window_id)
            is_drop = self._should_drop_split_for_singleton_guard(
                words_for_split=words_for_split,
                split_idx=split_idx,
                next_split=next_split,
                decision=decision,
            )
            if not is_drop:
                kept_split_points.append(int(split_idx))
        return kept_split_points

    def _should_drop_split_for_singleton_guard(
        self,
        *,
        words_for_split: Sequence[WordTimestamp],
        split_idx: int,
        next_split: Optional[int],
        decision: Optional[Any],
    ) -> bool:
        if split_idx < 0 or split_idx >= len(words_for_split) - 1:
            return False

        singleton_word_count = (
            (len(words_for_split) - split_idx - 1)
            if next_split is None
            else (next_split - split_idx)
        )
        if singleton_word_count <= 0:
            return False

        left_word = words_for_split[split_idx]
        left_end = float(getattr(left_word, "end", 0.0) or 0.0)
        first_tail_word = words_for_split[split_idx + 1]
        right_start = float(getattr(first_tail_word, "start", left_end) or left_end)
        gap_sec = max(0.0, right_start - left_end)

        prev_tail = str(getattr(left_word, "word", "") or "").strip()
        if prev_tail.endswith(tuple(self._SENTENCE_END_PUNCT)):
            return False

        if singleton_word_count == 1:
            singleton_start = float(getattr(first_tail_word, "start", right_start) or right_start)
            singleton_end = float(getattr(first_tail_word, "end", singleton_start) or singleton_start)
            singleton_duration = max(0.0, singleton_end - singleton_start)
            singleton_token = str(getattr(first_tail_word, "word", "") or "").strip()
            if not self._is_lowercase_continuation_token(singleton_token):
                return False

            if next_split is None:
                return (
                    gap_sec <= self._CUT_PLAN_SINGLETON_TAIL_MAX_GAP_SEC
                    and singleton_duration <= self._CUT_PLAN_SINGLETON_TAIL_MAX_DURATION_SEC
                )

            reason = str(getattr(decision, "reason", "") or "")
            if reason not in self._CUT_PLAN_SINGLETON_MID_GUARD_REASONS:
                return False
            return (
                gap_sec <= self._CUT_PLAN_SINGLETON_MID_MAX_GAP_SEC
                and singleton_duration <= self._CUT_PLAN_SINGLETON_MID_MAX_DURATION_SEC
            )

        # Why: 末尾仅剩 acronym+noun（如 `DDLC character?`）时，pause/speaker 触发切分常是不自然误切。
        if next_split is not None:
            return False
        if singleton_word_count > self._CUT_PLAN_SHORT_TAIL_MAX_WORDS:
            return False
        reason = str(getattr(decision, "reason", "") or "")
        if reason not in self._CUT_PLAN_SHORT_TAIL_GUARD_REASONS:
            return False
        tail_words = list(words_for_split[split_idx + 1 :])
        if len(tail_words) != singleton_word_count:
            return False
        head_tail_token = str(getattr(tail_words[0], "word", "") or "").strip()
        if not self._is_uppercase_acronym_token(head_tail_token):
            return False
        tail_start = float(getattr(tail_words[0], "start", right_start) or right_start)
        tail_end = float(getattr(tail_words[-1], "end", tail_start) or tail_start)
        tail_duration = max(0.0, tail_end - tail_start)
        return (
            gap_sec <= self._CUT_PLAN_SHORT_TAIL_MAX_GAP_SEC
            and tail_duration <= self._CUT_PLAN_SHORT_TAIL_MAX_DURATION_SEC
        )

    def _resolve_cut_plan_split_points(
        self,
        *,
        words_for_split: Sequence[WordTimestamp],
        decisions: Sequence[Any],
    ) -> Tuple[List[int], Dict[int, str], Dict[int, Dict[str, Any]]]:
        """
        将 CutPlan 决策时间映射到词边界索引。

        Returns:
            split_points: 词边界左索引列表（i 表示在 i 和 i+1 之间切）。
            split_to_window: split_idx -> window_id
        """
        if len(words_for_split) <= 1:
            return [], {}, {}

        split_to_window: Dict[int, str] = {}
        split_to_mapping: Dict[int, Dict[str, Any]] = {}
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
            split_to_mapping[selection.split_idx] = {
                "mapped_cut_time": self._resolve_split_boundary_time(
                    words_for_split=words_for_split,
                    split_idx=selection.split_idx,
                ),
                "mapping_quality": "gap" if selection.is_in_gap else "boundary",
                "mapping_reason": "word_boundary_mapper",
                "selection_score": float(selection.score),
            }

        split_points = sorted(split_to_window.keys())
        return split_points, split_to_window, split_to_mapping

    def _augment_split_points_with_sentence_end_punct(
        self,
        *,
        words_for_split: Sequence[WordTimestamp],
        split_points: Sequence[int],
        split_to_mapping: Dict[int, Dict[str, Any]],
    ) -> List[int]:
        """
        在 CutPlan 路径补齐句末标点边界，确保与 FinalSplitter 强切口径一致。
        """
        max_split_idx = len(words_for_split) - 2
        if max_split_idx < 0:
            return []
        merged_split_points = {
            int(split_idx)
            for split_idx in split_points
            if 0 <= int(split_idx) <= max_split_idx
        }
        if not self._is_force_split_on_sentence_end_punct_enabled():
            return sorted(merged_split_points)

        sentence_end_punct = tuple(self._SENTENCE_END_PUNCT)
        for split_idx in range(0, max_split_idx + 1):
            word_text = str(getattr(words_for_split[split_idx], "word", "") or "").strip()
            if not word_text.endswith(sentence_end_punct):
                continue
            if split_idx in merged_split_points:
                continue
            merged_split_points.add(split_idx)
            split_to_mapping[split_idx] = {
                "mapped_cut_time": self._resolve_split_boundary_time(
                    words_for_split=words_for_split,
                    split_idx=split_idx,
                ),
                "mapping_quality": "boundary",
                "mapping_reason": "sentence_end_punct_forced",
                "selection_score": 0.0,
            }
        return sorted(merged_split_points)

    def _is_force_split_on_sentence_end_punct_enabled(self) -> bool:
        splitter_config = getattr(self._final_splitter, "config", None)
        return bool(
            getattr(splitter_config, "is_force_split_on_sentence_end_punct", False)
        )

    def _build_cut_plan_output_trace(
        self,
        *,
        sentence_index: int,
        sentence: SentenceSegment,
        split_idx: int,
        split_to_window: Dict[int, str],
        split_to_mapping: Dict[int, Dict[str, Any]],
        decision_by_window: Dict[str, Any],
    ) -> OutputTrace:
        window_id = str(split_to_window.get(split_idx, "") or "")
        decision = decision_by_window.get(window_id)
        mapping_payload = dict(split_to_mapping.get(split_idx, {}) or {})

        split_reason = "soft_cut_plan"
        split_risk = ""
        pyannote_frame_time: Optional[float] = None
        mapped_cut_time: Optional[float] = None
        mapping_quality = str(mapping_payload.get("mapping_quality", "") or "")
        mapping_reason = str(mapping_payload.get("mapping_reason", "") or "")

        if decision is not None:
            split_reason = str(getattr(decision, "reason", "") or split_reason)
            split_risk = str(getattr(decision, "risk", "") or "")
            pyannote_frame_time = self._optional_float(
                getattr(decision, "pyannote_frame_time", None),
            )
            if pyannote_frame_time is None:
                pyannote_frame_time = self._optional_float(getattr(decision, "time", None))
            mapped_cut_time = self._optional_float(
                getattr(decision, "mapped_cut_time", None),
            )
            mapping_quality = str(
                getattr(decision, "mapping_quality", None) or mapping_quality
            )
            mapping_reason = str(
                getattr(decision, "mapping_reason", None) or mapping_reason
            )

        if mapped_cut_time is None:
            mapped_cut_time = self._optional_float(mapping_payload.get("mapped_cut_time"))
        if mapped_cut_time is None:
            mapped_cut_time = float(sentence.end)

        if not mapping_quality:
            mapping_quality = "boundary"
        if not mapping_reason:
            mapping_reason = "word_boundary_mapper"
        if decision is None and mapping_reason == "sentence_end_punct_forced":
            split_reason = "punctuation"

        return OutputTrace(
            sentence_index=sentence_index,
            split_reason=split_reason,
            split_risk=split_risk,
            window_id=window_id,
            pyannote_frame_time=pyannote_frame_time,
            mapped_cut_time=mapped_cut_time,
            mapping_quality=mapping_quality,
            mapping_reason=mapping_reason,
            sentence_start=float(sentence.start),
            sentence_end=float(sentence.end),
        )

    @staticmethod
    def _resolve_split_boundary_time(
        *,
        words_for_split: Sequence[WordTimestamp],
        split_idx: int,
    ) -> float:
        if split_idx < 0 or split_idx >= len(words_for_split) - 1:
            return 0.0
        left_end = float(getattr(words_for_split[split_idx], "end", 0.0) or 0.0)
        right_start = float(
            getattr(words_for_split[split_idx + 1], "start", left_end) or left_end
        )
        return (left_end + right_start) / 2.0

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _serialize_output_trace(trace: OutputTrace) -> Dict[str, Any]:
        return {
            "sentence_index": int(trace.sentence_index),
            "split_reason": str(trace.split_reason or ""),
            "split_risk": str(trace.split_risk or ""),
            "window_id": str(trace.window_id or ""),
            "pyannote_frame_time": trace.pyannote_frame_time,
            "mapped_cut_time": trace.mapped_cut_time,
            "mapping_quality": str(trace.mapping_quality or ""),
            "mapping_reason": str(trace.mapping_reason or ""),
            "sentence_start": trace.sentence_start,
            "sentence_end": trace.sentence_end,
        }

    def _build_default_output_traces(
        self,
        sentence_segments: Sequence[SentenceSegment],
        *,
        default_reason: str = "default_splitter",
    ) -> List[OutputTrace]:
        traces: List[OutputTrace] = []
        for sentence_index, sentence in enumerate(sentence_segments):
            traces.append(
                OutputTrace(
                    sentence_index=sentence_index,
                    split_reason=default_reason,
                    split_risk="",
                    window_id="",
                    pyannote_frame_time=None,
                    mapped_cut_time=float(sentence.end),
                    mapping_quality="default",
                    mapping_reason=default_reason,
                    sentence_start=float(sentence.start),
                    sentence_end=float(sentence.end),
                )
            )
        return traces

    @staticmethod
    def _normalize_output_traces_for_sentences(
        *,
        sentence_segments: Sequence[SentenceSegment],
        output_traces: Sequence[OutputTrace],
        default_reason: str,
    ) -> List[OutputTrace]:
        if len(sentence_segments) == len(output_traces):
            return list(output_traces)
        normalized: List[OutputTrace] = []
        for sentence_index, sentence in enumerate(sentence_segments):
            normalized.append(
                OutputTrace(
                    sentence_index=sentence_index,
                    split_reason=str(getattr(sentence, "split_reason", "") or default_reason),
                    split_risk=str(getattr(sentence, "split_risk", "") or ""),
                    window_id=str(getattr(sentence, "window_id", "") or ""),
                    pyannote_frame_time=getattr(sentence, "pyannote_frame_time", None),
                    mapped_cut_time=getattr(sentence, "mapped_cut_time", None),
                    mapping_quality=str(getattr(sentence, "mapping_quality", "") or "default"),
                    mapping_reason=str(getattr(sentence, "mapping_reason", "") or default_reason),
                    sentence_start=float(sentence.start),
                    sentence_end=float(sentence.end),
                )
            )
        return normalized

    def _apply_output_traces_to_sentences(
        self,
        *,
        sentence_segments: Sequence[SentenceSegment],
        output_traces: Sequence[OutputTrace],
    ) -> None:
        if not sentence_segments:
            return
        traces = list(output_traces or [])
        if not traces:
            traces = self._build_default_output_traces(sentence_segments)
        trace_by_index = {int(trace.sentence_index): trace for trace in traces}
        for sentence_index, sentence in enumerate(sentence_segments):
            trace = trace_by_index.get(sentence_index)
            if trace is None:
                trace = OutputTrace(
                    sentence_index=sentence_index,
                    split_reason="default_splitter",
                    split_risk="",
                    window_id="",
                    pyannote_frame_time=None,
                    mapped_cut_time=float(sentence.end),
                    mapping_quality="default",
                    mapping_reason="default_splitter",
                    sentence_start=float(sentence.start),
                    sentence_end=float(sentence.end),
                )
            sentence.split_reason = str(trace.split_reason or "")
            sentence.split_risk = str(trace.split_risk or "")
            sentence.window_id = str(trace.window_id or "")
            sentence.pyannote_frame_time = trace.pyannote_frame_time
            sentence.mapped_cut_time = trace.mapped_cut_time
            sentence.mapping_quality = str(trace.mapping_quality or "")
            sentence.mapping_reason = str(trace.mapping_reason or "")

    @staticmethod
    def _renumber_output_traces(traces: Sequence[OutputTrace]) -> List[OutputTrace]:
        return [
            OutputTrace(
                sentence_index=index,
                split_reason=str(trace.split_reason or ""),
                split_risk=str(trace.split_risk or ""),
                window_id=str(trace.window_id or ""),
                pyannote_frame_time=trace.pyannote_frame_time,
                mapped_cut_time=trace.mapped_cut_time,
                mapping_quality=str(trace.mapping_quality or ""),
                mapping_reason=str(trace.mapping_reason or ""),
                sentence_start=trace.sentence_start,
                sentence_end=trace.sentence_end,
            )
            for index, trace in enumerate(traces)
        ]

    def _merge_cut_plan_singleton_tail_sentence(
        self,
        *,
        sentence_segments: List[SentenceSegment],
        output_traces: List[OutputTrace],
    ) -> Tuple[List[SentenceSegment], List[OutputTrace]]:
        """
        CutPlan 路径尾词守门：将明显误切的末尾 1 词句回并到前一句。

        Why:
        - soft-cut 在误报窗口下可能把末词前停顿当成切口（如 `best | god.`、`about to | be.`）。
        - 该守门只处理“最后一句=1词”的高风险场景，避免影响正常多词句切分。
        """
        if len(sentence_segments) < 2:
            return sentence_segments, output_traces

        last_sentence = sentence_segments[-1]
        prev_sentence = sentence_segments[-2]
        last_words = list(last_sentence.words or [])
        prev_words = list(prev_sentence.words or [])
        if len(last_words) != 1 or not prev_words:
            return sentence_segments, output_traces

        left_end = float(getattr(prev_words[-1], "end", 0.0) or 0.0)
        right_start = float(getattr(last_words[0], "start", left_end) or left_end)
        gap_sec = max(0.0, right_start - left_end)
        if gap_sec > self._CUT_PLAN_SINGLETON_TAIL_MAX_GAP_SEC:
            return sentence_segments, output_traces

        last_start = float(getattr(last_sentence, "start", right_start) or right_start)
        last_end = float(getattr(last_sentence, "end", right_start) or right_start)
        last_duration = max(0.0, last_end - last_start)
        if last_duration > self._CUT_PLAN_SINGLETON_TAIL_MAX_DURATION_SEC:
            return sentence_segments, output_traces

        prev_tail = str(getattr(prev_words[-1], "word", "") or "").strip()
        if prev_tail.endswith(tuple(self._SENTENCE_END_PUNCT)):
            return sentence_segments, output_traces

        tail_token = str(getattr(last_words[0], "word", "") or "").strip()
        if not self._is_lowercase_continuation_token(tail_token):
            return sentence_segments, output_traces

        merged_words = prev_words + last_words
        rebuilt_prev = self._final_splitter._build_sentence(
            merged_words,
            0,
            len(merged_words) - 1,
        )
        self._copy_sentence_metadata(prev_sentence, rebuilt_prev)
        sentence_segments[-2] = rebuilt_prev
        sentence_segments.pop()
        if output_traces:
            output_traces[-2] = OutputTrace(
                sentence_index=max(0, len(sentence_segments) - 1),
                split_reason="singleton_tail_merge",
                split_risk="",
                window_id="",
                pyannote_frame_time=None,
                mapped_cut_time=float(rebuilt_prev.end),
                mapping_quality="merged",
                mapping_reason="singleton_tail_guard",
                sentence_start=float(rebuilt_prev.start),
                sentence_end=float(rebuilt_prev.end),
            )
            output_traces = output_traces[:-1]
            output_traces = self._renumber_output_traces(output_traces)
        return sentence_segments, output_traces

    @staticmethod
    def _is_lowercase_continuation_token(token: str) -> bool:
        text = str(token or "").strip().lstrip("\"'“”‘’([{")
        if not text:
            return False
        for ch in text:
            if ch.isalpha():
                return ch.islower()
        return False

    @staticmethod
    def _is_uppercase_acronym_token(token: str) -> bool:
        text = str(token or "").strip().lstrip("\"'“”‘’([{")
        if not text:
            return False
        letters = [ch for ch in text if ch.isalpha()]
        if len(letters) < 2:
            return False
        return all(ch.isupper() for ch in letters)

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
                "deferred_state_stats": {
                    "pending": 0,
                    "resolved": 0,
                    "forced": 0,
                    "expired": 0,
                },
                "applied_decision_count": 0,
                "applied_window_ids": [],
                "unapplied_window_ids": [],
                "reason_stats": {},
                "risk_stats": {},
                "source_stats": {},
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
        source_stats = dict(
            Counter(
                str(getattr(item, "source", "") or "")
                for item in decisions
                if str(getattr(item, "source", "") or "")
            )
        )
        deferred_state_stats = {
            "pending": 0,
            "resolved": 0,
            "forced": 0,
            "expired": 0,
        }
        for item in deferred_cuts:
            state_value = str(getattr(getattr(item, "state", None), "value", "") or "").lower()
            if state_value in deferred_state_stats:
                deferred_state_stats[state_value] += 1
        return {
            "enabled": True,
            "plan_id": str(getattr(cut_plan, "plan_id", "") or ""),
            "decision_count": len(decisions),
            "deferred_count": len(deferred_cuts),
            "deferred_state_stats": deferred_state_stats,
            "applied_decision_count": applied_decision_count,
            "applied_window_ids": sorted(applied_window_set),
            "unapplied_window_ids": unapplied_window_ids,
            "reason_stats": reason_stats,
            "risk_stats": risk_stats,
            "source_stats": source_stats,
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
        target.split_reason = source.split_reason
        target.split_risk = source.split_risk
        target.window_id = source.window_id
        target.pyannote_frame_time = source.pyannote_frame_time
        target.mapped_cut_time = source.mapped_cut_time
        target.mapping_quality = source.mapping_quality
        target.mapping_reason = source.mapping_reason

    @staticmethod
    def _resolve_turn_field(turn: Any, field_name: str, default: Any = None) -> Any:
        if isinstance(turn, dict):
            return turn.get(field_name, default)
        return getattr(turn, field_name, default)

    @classmethod
    def _resolve_turn_float(cls, turn: Any, field_name: str) -> float:
        value = cls._resolve_turn_field(turn, field_name, 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _resolve_turn_speaker(cls, turn: Any) -> str:
        return str(cls._resolve_turn_field(turn, "speaker_id", "") or "").strip()

    @classmethod
    def _resolve_turn_id(cls, turn: Any) -> str:
        return str(cls._resolve_turn_field(turn, "turn_id", "") or "").strip()

    def _collect_speaker_turns_for_repair(self, aligned_facts: Any) -> List[Dict[str, Any]]:
        turns = list(getattr(aligned_facts, "speaker_turns", []) or [])
        normalized: List[Dict[str, Any]] = []
        for turn in turns:
            start = self._resolve_turn_float(turn, "start")
            end = self._resolve_turn_float(turn, "end")
            if end <= start:
                continue
            normalized.append(
                {
                    "start": start,
                    "end": end,
                    "speaker_id": self._resolve_turn_speaker(turn),
                    "turn_id": self._resolve_turn_id(turn),
                }
            )
        return normalized

    def _repair_cross_speaker_sentences_once(
        self,
        *,
        sentences: List[SentenceSegment],
        turns: Sequence[Any],
    ) -> int:
        """
        单次闭环修复：仅对“句内跨 speaker”句子做一次局部重切。

        Why:
        - 触发点来自 turn 边界，但落点必须是词边界，禁止切断词。
        - 不做二次全局复跑，避免重复计算和结果抖动。
        """
        if not sentences or len(turns) <= 1:
            return 0

        repaired_sentences: List[SentenceSegment] = []
        repaired_split_count = 0
        ordered_turns = sorted(
            list(turns),
            key=lambda item: (
                self._resolve_turn_float(item, "start"),
                self._resolve_turn_float(item, "end"),
            ),
        )
        for sentence in sentences:
            repaired_parts = self._split_sentence_by_turn_boundaries_once(
                sentence=sentence,
                ordered_turns=ordered_turns,
            )
            repaired_sentences.extend(repaired_parts)
            if len(repaired_parts) > 1:
                repaired_split_count += len(repaired_parts) - 1

        if repaired_split_count > 0:
            sentences[:] = repaired_sentences
        return repaired_split_count

    def _split_sentence_by_turn_boundaries_once(
        self,
        *,
        sentence: SentenceSegment,
        ordered_turns: Sequence[Any],
    ) -> List[SentenceSegment]:
        words = list(sentence.words or [])
        if len(words) <= 1:
            return [sentence]

        sentence_start = float(getattr(sentence, "start", 0.0) or 0.0)
        sentence_end = float(getattr(sentence, "end", 0.0) or 0.0)
        if sentence_end <= sentence_start:
            return [sentence]

        overlap_turns = [
            turn
            for turn in ordered_turns
            if min(sentence_end, self._resolve_turn_float(turn, "end"))
            > max(sentence_start, self._resolve_turn_float(turn, "start"))
        ]
        speaker_set = {
            self._resolve_turn_speaker(turn)
            for turn in overlap_turns
            if self._resolve_turn_speaker(turn)
        }
        if len(speaker_set) <= 1:
            return [sentence]
        if (
            self._is_enable_soft_cut_overlap_degrade
            and self._is_overlap_turn_window(turns=overlap_turns)
        ):
            # Why: 重叠抢话场景的 diarization 边界噪声高，降级为“不做句级硬切”更稳妥。
            return [sentence]

        boundary_times = self._collect_sentence_turn_change_times(
            words=words,
            overlap_turns=overlap_turns,
            sentence_start=sentence_start,
            sentence_end=sentence_end,
        )
        if not boundary_times:
            return [sentence]

        boundary_by_split_index: Dict[int, Tuple[float, float]] = {}
        for boundary_time in boundary_times:
            split_idx, score = self._select_best_word_boundary_for_turn_change(
                words=words,
                boundary_time=boundary_time,
            )
            if split_idx is None:
                continue
            existing = boundary_by_split_index.get(split_idx)
            if existing is None or score > existing[1]:
                boundary_by_split_index[split_idx] = (boundary_time, score)

        if not boundary_by_split_index:
            return [sentence]

        selected_split_indices = sorted(
            split_idx
            for split_idx in boundary_by_split_index.keys()
            if 0 <= split_idx < len(words) - 1
        )
        if not selected_split_indices:
            return [sentence]

        rebuilt_sentences: List[SentenceSegment] = []
        start_idx = 0
        used_split_indices: List[int] = []
        for split_idx in selected_split_indices:
            is_has_room = split_idx >= start_idx and split_idx < len(words) - 1
            if not is_has_room:
                continue
            rebuilt_sentences.append(
                self._final_splitter._build_sentence(words, start_idx, split_idx)
            )
            used_split_indices.append(split_idx)
            start_idx = split_idx + 1
        if start_idx < len(words):
            rebuilt_sentences.append(
                self._final_splitter._build_sentence(words, start_idx, len(words) - 1)
            )

        if len(rebuilt_sentences) <= 1:
            return [sentence]

        # 用 turn 边界时间吸附相邻句子的 start/end，减少“跨speaker时间窗”残留。
        for idx, split_idx in enumerate(used_split_indices):
            if idx + 1 >= len(rebuilt_sentences):
                break
            boundary_time = float(boundary_by_split_index[split_idx][0])
            left_sentence = rebuilt_sentences[idx]
            right_sentence = rebuilt_sentences[idx + 1]
            left_start = float(getattr(left_sentence, "start", 0.0) or 0.0)
            right_end = float(getattr(right_sentence, "end", 0.0) or 0.0)
            if right_end <= left_start:
                continue
            snapped = max(left_start + 1e-3, min(right_end - 1e-3, boundary_time))
            left_sentence.end = max(left_start + 1e-3, min(float(left_sentence.end), snapped))
            right_sentence.start = min(right_end - 1e-3, max(float(right_sentence.start), snapped))
            if right_sentence.start <= left_sentence.end:
                right_sentence.start = min(right_end - 1e-3, left_sentence.end + 1e-3)
            if right_sentence.end <= right_sentence.start:
                right_sentence.end = right_sentence.start + 1e-3

        for rebuilt in rebuilt_sentences:
            self._copy_sentence_metadata(source=sentence, target=rebuilt)
            # 交给后续句级 timeline 绑定重新计算最终 speaker/turn。
            rebuilt.speaker_id = None
            rebuilt.turn_id = None
        return rebuilt_sentences

    def _collect_sentence_turn_change_times(
        self,
        *,
        words: Sequence[WordTimestamp],
        overlap_turns: Sequence[Any],
        sentence_start: float,
        sentence_end: float,
    ) -> List[float]:
        if len(overlap_turns) <= 1:
            return []

        ordered = sorted(
            list(overlap_turns),
            key=lambda item: (
                self._resolve_turn_float(item, "start"),
                self._resolve_turn_float(item, "end"),
            ),
        )
        change_times: List[float] = []
        for idx in range(1, len(ordered)):
            left = ordered[idx - 1]
            right = ordered[idx]
            left_speaker = self._resolve_turn_speaker(left)
            right_speaker = self._resolve_turn_speaker(right)
            if not left_speaker or not right_speaker or left_speaker == right_speaker:
                continue
            left_start = self._resolve_turn_float(left, "start")
            left_end = self._resolve_turn_float(left, "end")
            right_start = self._resolve_turn_float(right, "start")
            right_end = self._resolve_turn_float(right, "end")
            left_duration = max(0.0, left_end - left_start)
            right_duration = max(0.0, right_end - right_start)
            # Why: 极短 turn 常由分割抖动导致，不应直接触发句内重切。
            if min(left_duration, right_duration) < self._SPEAKER_REPAIR_MIN_TURN_DURATION_SEC:
                continue
            if self._is_short_turn_followed_by_same_speaker_continuation(
                turns=ordered,
                index=idx,
                right_speaker=right_speaker,
                right_end=right_end,
                right_duration=right_duration,
            ):
                # Why: 短 turn 若被同 speaker 续段承接，多为 pyannote 抖动；仅在强断句证据时放行。
                if not self._is_repair_boundary_strong_break(
                    words=words,
                    boundary_time=right_start,
                ):
                    continue
            boundary_time = right_start
            # Why: 句首/句尾附近的切点通常由 chunk 边缘时间误差触发，避免把首词单独切出。
            if (
                boundary_time <= sentence_start + self._SPEAKER_REPAIR_EDGE_GUARD_SEC
                or boundary_time >= sentence_end - self._SPEAKER_REPAIR_EDGE_GUARD_SEC
            ):
                continue
            if not (sentence_start < boundary_time < sentence_end):
                continue
            change_times.append(boundary_time)

        deduped: List[float] = []
        for value in sorted(change_times):
            if not deduped or abs(value - deduped[-1]) > 0.08:
                deduped.append(value)
        return deduped

    def _is_short_turn_followed_by_same_speaker_continuation(
        self,
        *,
        turns: Sequence[Any],
        index: int,
        right_speaker: str,
        right_end: float,
        right_duration: float,
    ) -> bool:
        if right_duration >= self._SOFT_CUT_SHORT_TURN_CONTINUATION_SEC:
            return False
        if index + 1 >= len(turns):
            return False

        next_turn = turns[index + 1]
        next_speaker = self._resolve_turn_speaker(next_turn)
        if not next_speaker or next_speaker != right_speaker:
            return False
        next_start = self._resolve_turn_float(next_turn, "start")
        continuation_gap = max(0.0, next_start - right_end)
        return continuation_gap <= self._SOFT_CUT_SHORT_TURN_CONTINUATION_GAP_SEC

    def _is_repair_boundary_strong_break(
        self,
        *,
        words: Sequence[WordTimestamp],
        boundary_time: float,
    ) -> bool:
        selection = self._boundary_mapper.select_best_boundary(
            words=words,
            event_time=boundary_time,
        )
        if selection is None:
            return False

        split_idx = int(selection.split_idx)
        if split_idx < 0 or split_idx >= len(words) - 1:
            return False

        left_word = words[split_idx]
        right_word = words[split_idx + 1]
        left_text = str(getattr(left_word, "word", "") or "").strip()
        if left_text.endswith(tuple(self._SPEAKER_REPAIR_SENTENCE_END_PUNCT)):
            return True

        left_end = float(getattr(left_word, "end", 0.0) or 0.0)
        right_start = float(getattr(right_word, "start", left_end) or left_end)
        pause_duration = max(0.0, right_start - left_end)
        return pause_duration >= self._SPEAKER_REPAIR_STRONG_BREAK_MIN_PAUSE_SEC

    def _select_best_word_boundary_for_turn_change(
        self,
        *,
        words: Sequence[WordTimestamp],
        boundary_time: float,
    ) -> Tuple[Optional[int], float]:
        if len(words) <= 1:
            return None, 0.0

        ranked = self._boundary_mapper.rank_boundaries(
            words=words,
            event_time=boundary_time,
        )
        if not ranked:
            return None, 0.0

        best_idx: Optional[int] = None
        best_score = float("-inf")
        for candidate in ranked[:3]:
            local_score = self._score_word_boundary_candidate(
                words=words,
                split_idx=candidate.split_idx,
                boundary_time=boundary_time,
            )
            score = (candidate.score * 1.2) + local_score
            if score > best_score:
                best_score = score
                best_idx = candidate.split_idx

        if best_idx is not None and best_score > -0.8:
            return best_idx, best_score
        return ranked[0].split_idx, ranked[0].score

    def _score_word_boundary_candidate(
        self,
        *,
        words: Sequence[WordTimestamp],
        split_idx: int,
        boundary_time: float,
    ) -> float:
        left_end = float(getattr(words[split_idx], "end", 0.0) or 0.0)
        right_start = float(getattr(words[split_idx + 1], "start", left_end) or left_end)
        gap_left = min(left_end, right_start)
        gap_right = max(left_end, right_start)
        is_in_gap = gap_left <= boundary_time <= gap_right
        if is_in_gap:
            delta = 0.0
        else:
            delta = min(
                abs(left_end - boundary_time),
                abs(right_start - boundary_time),
            )
        distance_score = max(0.0, 1.0 - (delta / 0.8))
        pause_duration = max(0.0, right_start - left_end)
        pause_score = min(1.0, pause_duration / 0.35)
        valley_score = self._score_boundary_vad_valley((left_end + right_start) / 2.0)

        left_ratio, left_speaker = self._resolve_word_speaker_purity(
            words=words,
            start_idx=0,
            end_idx=split_idx,
        )
        right_ratio, right_speaker = self._resolve_word_speaker_purity(
            words=words,
            start_idx=split_idx + 1,
            end_idx=len(words) - 1,
        )
        speaker_purity_score = (left_ratio + right_ratio) / 2.0
        is_has_explicit_change = (
            bool(left_speaker)
            and bool(right_speaker)
            and left_speaker != right_speaker
        )
        speaker_change_bonus = 1.0 if is_has_explicit_change else 0.0

        short_penalty = 0.0
        if split_idx + 1 < 2:
            short_penalty += 0.4
        if len(words) - (split_idx + 1) < 2:
            short_penalty += 0.4

        return (
            distance_score * 1.6
            + pause_score * 1.0
            + valley_score * 0.6
            + speaker_purity_score * 0.8
            + speaker_change_bonus * 1.2
            - short_penalty
        )

    def _score_boundary_vad_valley(self, boundary_time: float) -> float:
        if not self._active_vad_intervals:
            return 0.0
        for start, end in self._active_vad_intervals:
            if float(start) <= boundary_time <= float(end):
                return 0.0
        return 1.0

    @staticmethod
    def _resolve_word_speaker_purity(
        *,
        words: Sequence[WordTimestamp],
        start_idx: int,
        end_idx: int,
    ) -> Tuple[float, str]:
        if end_idx < start_idx:
            return 0.0, ""
        counts: Dict[str, int] = {}
        total = 0
        for idx in range(start_idx, end_idx + 1):
            speaker_id = str(getattr(words[idx], "speaker_id", "") or "").strip()
            if not speaker_id:
                continue
            counts[speaker_id] = counts.get(speaker_id, 0) + 1
            total += 1
        if total <= 0 or not counts:
            return 0.0, ""
        best_speaker, best_count = max(counts.items(), key=lambda item: item[1])
        return best_count / total, best_speaker

    @staticmethod
    def _is_overlap_turn_window(
        *,
        turns: Sequence[Any],
    ) -> bool:
        if len(turns) <= 1:
            return False
        ordered = sorted(
            list(turns),
            key=lambda item: (
                SegmentationProcessor._resolve_turn_float(item, "start"),
                SegmentationProcessor._resolve_turn_float(item, "end"),
            ),
        )
        for idx in range(1, len(ordered)):
            left = ordered[idx - 1]
            right = ordered[idx]
            left_end = SegmentationProcessor._resolve_turn_float(left, "end")
            right_start = SegmentationProcessor._resolve_turn_float(right, "start")
            if right_start < left_end - 1e-3:
                return True
        return False

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


# 兼容新旧命名。
DecisionSegmentationProcessor = SegmentationProcessor

__all__ = ["SegmentationProcessor", "DecisionSegmentationProcessor"]

