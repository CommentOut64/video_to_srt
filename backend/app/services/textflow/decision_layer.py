"""
裁决层切分处理器（SegmentationProcessor）。
V3.2.0+dev.20260220.01
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple

from app.core.logging import resolve_loguru_logger
from app.models.sensevoice_models import SentenceSegment, TextSource, WordTimestamp
from app.services.alignment.default_aligner import _strip_trailing_punct_smart
from app.services.alignment.types import (
    AlignedFacts,
    AnnotatedWord,
    CharMapping,
    DecisionLayerInput,
    DecisionLayerOutput,
    OutputTrace,
    TextTrack,
    TextTrackBundle,
)
from app.services.segmentation.soft_cut.types import AnchorType, CutDecision, CutPlan
from app.services.segmentation.boundary_mapper import WordBoundaryMapper
from app.services.punctuation.final_splitter import FinalSplitter
from app.services.textflow.canonical_text_stream_adapter import CanonicalTextStreamAdapter
from app.services.textflow.contracts import (
    ConsumedBoundaryPunct,
    SegmentPlan,
    SegmentationResult,
    SegmentationIngressContext,
)
from app.services.textflow.ingress_segment_planner import IngressSegmentPlanner
from app.services.textflow.render_core import RenderCore
from app.services.textflow.segmentation_core import SegmentationCore
from app.services.textflow.subtitle_delivery import SubtitleDelivery
from app.services.text_protection import (
    is_sentence_end_punct,
    merge_protected_word_tokens,
)

if TYPE_CHECKING:
    from app.services.language_policy.types import LanguagePolicySnapshot


class SegmentationProcessor:
    """裁决层处理器：仅负责边界决策与句子切分。"""

    # V3.2.0+dev.20260210.03: 裁决层内建跨 chunk 连续性处理（仅处理高风险残词）。
    _CROSS_CHUNK_CARRY_WORDS = {"a", "an", "the"}
    _CROSS_CHUNK_CJK_CARRY_SINGLE_CHARS = {"经", "直", "才", "因", "检", "警"}
    _CROSS_CHUNK_CJK_CARRY_WORDS = {"因为", "以及", "直到", "经过", "检测"}
    _CROSS_CHUNK_CJK_MAX_GAP_SEC = 0.10
    _CROSS_CHUNK_CJK_MAX_DURATION_SEC = 0.80
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
    _CUT_PLAN_CJK_FAST_DRAFT_SINGLETON_MIN_LEFT_WORDS = 3
    _CUT_PLAN_CJK_FAST_DRAFT_SHORT_PREFIX_MAX_WORDS = 3
    _CUT_PLAN_CJK_PUNCT_FRAGMENT_MAX_GAP_SEC = 0.65
    _CUT_PLAN_CJK_PUNCT_FRAGMENT_SINGLETON_MAX_DURATION_SEC = 1.20
    _CUT_PLAN_CJK_PUNCT_FRAGMENT_SHORT_MAX_DURATION_SEC = 0.95
    _CUT_PLAN_CJK_PUNCT_FRAGMENT_CONTINUATION_MAX_WORDS = 4
    _CUT_PLAN_CJK_PUNCT_FRAGMENT_CONTINUATION_MAX_DURATION_SEC = 2.40
    _CUT_PLAN_CJK_SHORT_PREFIX_MAX_WORDS = 3
    _CUT_PLAN_CJK_SHORT_PREFIX_MAX_DURATION_SEC = 1.10
    _CUT_PLAN_CJK_SHORT_PREFIX_MAX_GAP_SEC = 0.55
    _SOFT_CUT_SHORT_TURN_CONTINUATION_SEC = 0.75
    _SOFT_CUT_SHORT_TURN_CONTINUATION_GAP_SEC = 0.80
    _SPEAKER_REPAIR_MIN_TURN_DURATION_SEC = 0.45
    _SPEAKER_REPAIR_STRONG_BREAK_MIN_PAUSE_SEC = 0.30
    _SPEAKER_REPAIR_SENTENCE_END_PUNCT = {"。", "！", "？", ".", "!", "?"}
    _SPEAKER_REPAIR_EDGE_GUARD_SEC = 0.15
    _TIMELINE_BACKTRACK_DROP_TOLERANCE_SEC = 0.02
    _TIMELINE_NORMALIZE_MIN_DURATION_SEC = 0.01
    _INGRESS_LOW_QUALITY_MOUNT_STATUSES = {"inferred", "unresolved"}
    _INGRESS_FRAGMENT_MAX_LEFT_ALPHA_LEN = 4
    _INGRESS_FRAGMENT_MAX_RIGHT_ALPHA_LEN = 2
    _INGRESS_SCORING_DEFAULT_THRESHOLD = 0.74
    _INGRESS_SCORING_CJK_THRESHOLD = 0.66
    _INGRESS_SCORING_MIN_ADJACENT_BOUNDARY_GAP_SEC = 0.22
    _INGRESS_SCORING_MIN_ADJACENT_BOUNDARY_GAP_CJK_SEC = 0.16
    _INGRESS_SCORING_REASON_BONUS = {
        "punctuation_sentence_end": 0.20,
        "speaker_change": 0.14,
        "gap_pause": 0.10,
        "blank_valley": 0.05,
        "pause_long": 0.05,
        "pause": 0.04,
        "punctuation_soft": 0.03,
        "lexical_boundary": 0.00,
        "anchor_block_close": -0.02,
    }
    _INGRESS_SCORING_REASON_PRIORITY = {
        "punctuation_sentence_end": 100,
        "speaker_change": 90,
        "gap_pause": 80,
        "blank_valley": 70,
        "pause_long": 65,
        "pause": 60,
        "punctuation_soft": 55,
        "lexical_boundary": 45,
        "anchor_block_close": 40,
    }
    _UNKNOWN_PSEUDO_JUNK_PATTERN = re.compile(
        r"^[\s\|·•`~!@#$%^&*()_+\-=\[\]{};:'\",.<>/?\\，。！？：；、（）《》【】…—]+$"
    )
    _CUT_BOUNDARY_WEAK_PUNCT = ",，、;；:："
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
        self._segmentation_core = SegmentationCore(processor=self)
        self._canonical_text_stream_adapter = CanonicalTextStreamAdapter()
        self._ingress_segment_planner = IngressSegmentPlanner()
        self._render_core = RenderCore()
        self._subtitle_delivery = SubtitleDelivery()

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
        """统一切分主入口：固定走统一渲染主链。"""
        segmentation_output = self._segmentation_core.process(
            data=data,
            stream_id=stream_id,
            chunk_index=chunk_index,
            is_last_chunk=is_last_chunk,
        )
        return self._render_with_unified_pipeline(
            segmentation_output=segmentation_output,
            data=data,
            stream_id=stream_id,
            chunk_index=chunk_index,
        )

    def _render_with_unified_pipeline(
        self,
        *,
        segmentation_output: DecisionLayerOutput,
        data: DecisionLayerInput,
        stream_id: str,
        chunk_index: Optional[int],
    ) -> DecisionLayerOutput:
        if not segmentation_output.words_for_split:
            report = dict(segmentation_output.segmentation_report or {})
            report["pipeline_route"] = "canonical_segmentation_render"
            report["render_report"] = {
                "segment_count": 0,
                "input_segment_count": 0,
                "rendered_terminal_count": 0,
                "dropped_punct_fact_count": 0,
                "dropped_punct_facts": [],
            }
            report["render_output_trace"] = []
            report["output_trace"] = [
                self._serialize_output_trace(item)
                for item in segmentation_output.output_traces
            ]
            segmentation_output.segmentation_report = report
            return segmentation_output

        canonical_stream = self._build_canonical_stream_from_words(
            words=segmentation_output.words_for_split,
            data=data,
            stream_id=stream_id,
            chunk_index=chunk_index,
        )
        segmentation_result = segmentation_output.segmentation_result
        if segmentation_result is None:
            raise ValueError(
                "SegmentationCore 主路径必须直接产出 segmentation_result；"
                "legacy segmentation_result bridge 已停用。"
            )
        self._assert_segmentation_token_contract(
            canonical_stream=canonical_stream,
            segmentation_result=segmentation_result,
        )
        segmentation_result = self._hydrate_segmentation_result_with_canonical_stream(
            canonical_stream=canonical_stream,
            segmentation_result=segmentation_result,
        )
        render_result = self._render_core.render(
            canonical_stream=canonical_stream,
            segmentation_result=segmentation_result,
        )
        subtitle_batch = self._subtitle_delivery.build_batch(
            render_result=render_result,
            chunk_id=str(canonical_stream.chunk_ref),
            chunk_index=chunk_index,
            ingress_context=self._serialize_ingress_context(data.ingress_context),
        )
        rendered_sentences = self._build_compat_sentences_from_subtitle_batch(
            subtitle_batch=subtitle_batch,
            sentence_segments=segmentation_output.sentence_segments,
        )
        rendered_traces = self._build_output_traces_from_subtitle_batch(subtitle_batch)
        for sentence_index, sentence in enumerate(rendered_sentences):
            if sentence_index < len(segmentation_result.segments):
                sentence.words = self._build_sentence_words_from_segment(
                    canonical_stream=canonical_stream,
                    segment=segmentation_result.segments[sentence_index],
                )
        self._apply_output_traces_to_sentences(
            sentence_segments=rendered_sentences,
            output_traces=rendered_traces,
        )
        self._strip_boundary_residual_weak_punct(rendered_sentences)

        report = dict(segmentation_output.segmentation_report or {})
        report["pipeline_route"] = "canonical_segmentation_render"
        report["render_report"] = dict(render_result.render_report or {})
        report["render_output_trace"] = list(render_result.output_trace or ())
        report["output_trace"] = [self._serialize_output_trace(item) for item in rendered_traces]

        return DecisionLayerOutput(
            sentence_segments=rendered_sentences,
            words_for_split=segmentation_output.words_for_split,
            segmentation_report=report,
            applied_cut_plan=segmentation_output.applied_cut_plan,
            output_traces=rendered_traces,
            segmentation_result=segmentation_result,
            render_result=render_result,
            subtitle_batch=subtitle_batch,
        )

    @staticmethod
    def _serialize_ingress_context(
        ingress_context: Optional[SegmentationIngressContext],
    ) -> Optional[Dict[str, Any]]:
        if ingress_context is None:
            return None
        return ingress_context.to_dict()

    def _build_compat_sentences_from_subtitle_batch(
        self,
        *,
        subtitle_batch: Any,
        sentence_segments: Sequence[SentenceSegment],
    ) -> List[SentenceSegment]:
        rendered_sentences: List[SentenceSegment] = []
        legacy_sentences = list(sentence_segments or [])
        for sentence_index, item in enumerate(list(getattr(subtitle_batch, "items", ()) or ())):
            if sentence_index < len(legacy_sentences):
                sentence = legacy_sentences[sentence_index]
            else:
                sentence = SentenceSegment(
                    text=str(item.text),
                    text_clean=str(item.text),
                    start=float(item.start),
                    end=float(item.end),
                )
            sentence.text = str(item.text)
            sentence.text_clean = str(item.text)
            sentence.start = float(item.start)
            sentence.end = float(item.end)
            sentence.source = self._map_render_text_source(str(item.source or ""))
            sentence.is_draft = str(item.status or "").lower() == "draft"
            sentence.is_finalized = not sentence.is_draft
            sentence.speaker_id = item.speaker_id
            sentence.turn_id = item.turn_id
            sentence.segment_id = item.segment_id
            sentence.sentence_uid = item.segment_id
            trace_data = dict(item.trace or {})
            sentence.split_reason = str(trace_data.get("split_reason", "") or "")
            sentence.split_risk = str(trace_data.get("split_risk", "") or "")
            sentence.window_id = str(trace_data.get("window_id", "") or "")
            sentence.pyannote_frame_time = trace_data.get("pyannote_frame_time")
            sentence.mapped_cut_time = trace_data.get("mapped_cut_time", item.end)
            sentence.mapping_quality = str(trace_data.get("mapping_quality", "") or "")
            sentence.mapping_reason = str(trace_data.get("mapping_reason", "") or "")
            rendered_sentences.append(sentence)
        return rendered_sentences

    @staticmethod
    def _build_output_traces_from_subtitle_batch(subtitle_batch: Any) -> List[OutputTrace]:
        traces: List[OutputTrace] = []
        for sentence_index, item in enumerate(list(getattr(subtitle_batch, "items", ()) or ())):
            trace_data = dict(getattr(item, "trace", {}) or {})
            traces.append(
                OutputTrace(
                    sentence_index=sentence_index,
                    split_reason=str(trace_data.get("split_reason", "") or ""),
                    split_risk=str(trace_data.get("split_risk", "") or ""),
                    window_id=str(trace_data.get("window_id", "") or ""),
                    pyannote_frame_time=trace_data.get("pyannote_frame_time"),
                    mapped_cut_time=trace_data.get("mapped_cut_time", float(item.end)),
                    mapping_quality=str(trace_data.get("mapping_quality", "") or ""),
                    mapping_reason=str(trace_data.get("mapping_reason", "") or ""),
                    sentence_start=float(item.start),
                    sentence_end=float(item.end),
                )
            )
        return traces

    def _hydrate_segmentation_result_with_canonical_stream(
        self,
        *,
        canonical_stream: Any,
        segmentation_result: SegmentationResult,
    ) -> SegmentationResult:
        facts = list(getattr(canonical_stream, "punctuation_facts", ()) or ())
        if not facts or not segmentation_result.segments:
            return segmentation_result
        updated_segments: List[SegmentPlan] = []
        is_changed = False
        for segment in segmentation_result.segments:
            if segment.consumed_boundary_punct is not None:
                updated_segments.append(segment)
                continue
            consumed = self._resolve_consumed_boundary_punct(
                facts=facts,
                token_end=int(segment.token_end),
            )
            if consumed is None:
                updated_segments.append(segment)
                continue
            updated_segments.append(replace(segment, consumed_boundary_punct=consumed))
            is_changed = True
        if not is_changed:
            return segmentation_result
        return SegmentationResult(
            segments=tuple(updated_segments),
            boundary_traces=tuple(segmentation_result.boundary_traces),
            segmentation_report=dict(segmentation_result.segmentation_report or {}),
        )

    @staticmethod
    def _assert_segmentation_token_contract(
        *,
        canonical_stream: Any,
        segmentation_result: SegmentationResult,
    ) -> None:
        segments = list(getattr(segmentation_result, "segments", ()) or ())
        if not segments:
            return
        token_indices: Set[int] = {
            int(getattr(token, "index", -1))
            for token in list(getattr(canonical_stream, "tokens", ()) or ())
        }
        if not token_indices:
            raise ValueError("Unified render 缺少 canonical tokens，无法消费 segmentation_result。")

        invalid_segments: List[Dict[str, Any]] = []
        for segment in segments:
            token_start = int(segment.token_start)
            token_end = int(segment.token_end)
            missing = [idx for idx in range(token_start, token_end + 1) if idx not in token_indices]
            if not missing:
                continue
            invalid_segments.append(
                {
                    "segment_id": str(getattr(segment, "segment_id", "") or ""),
                    "token_start": token_start,
                    "token_end": token_end,
                    "missing": missing[:8],
                }
            )
        if invalid_segments:
            raise ValueError(
                "Unified render token contract mismatch: "
                f"canonical_token_count={len(token_indices)}, invalid_segments={invalid_segments}"
            )

    @staticmethod
    def _map_render_text_source(text_source: str) -> TextSource:
        normalized = str(text_source or "").strip().lower()
        if normalized == "fast":
            return TextSource.SENSEVOICE
        if normalized in {"slow", "aligned"}:
            return TextSource.WHISPER_PATCH
        return TextSource.WHISPER_PATCH

    def _build_canonical_stream_from_words(
        self,
        *,
        words: Sequence[WordTimestamp],
        data: DecisionLayerInput,
        stream_id: str,
        chunk_index: Optional[int],
    ):
        language = self._resolve_language_from_decision_input(data)
        ingress_context = data.ingress_context or SegmentationIngressContext(
            unit_kind="chunk",
            unit_id=str(chunk_index if chunk_index is not None else "chunk-unknown"),
            chunk_id=str(chunk_index if chunk_index is not None else "chunk-unknown"),
            chunk_index=chunk_index,
        )
        chunk_ref = str(
            getattr(ingress_context, "chunk_id", None)
            or (chunk_index if chunk_index is not None else "chunk-unknown")
        )
        synthetic_text, clean_to_word = self._build_synthetic_text_and_word_mapping(words=words)
        char_mapping = [
            CharMapping(raw_idx=index, clean_idx=index, punct=None)
            for index in range(len(synthetic_text))
        ]
        chosen_track = TextTrack(
            raw_text=synthetic_text,
            text_itn_raw=synthetic_text,
            text_clean=synthetic_text,
            char_mapping=char_mapping,
            raw_to_clean=list(range(len(synthetic_text))),
            clean_to_raw=list(range(len(synthetic_text))),
            language=language,
            source="aligned",
            clean_to_word=clean_to_word,
            punct_positions=list(data.fallback_punctuation_positions or []),
            mapping_coverage=1.0 if synthetic_text else 0.0,
        )
        tracks = TextTrackBundle(chosen_track=chosen_track)
        aligned_facts = data.aligned_facts or self._build_synthetic_aligned_facts(words=words)
        canonical_stream = self._canonical_text_stream_adapter.build(
            stream_id=stream_id,
            chunk_ref=chunk_ref,
            ingress_context=ingress_context,
            tracks=tracks,
            text_source=self._resolve_text_source_from_words(words=words),
            language=language,
            aligned_facts=aligned_facts,
            metadata={
                "builder": "decision_layer",
                "phase": "6",
            },
        )
        if data.canonical_punctuation_facts or data.canonical_candidate_boundaries:
            return replace(
                canonical_stream,
                punctuation_facts=(
                    tuple(data.canonical_punctuation_facts)
                    if data.canonical_punctuation_facts
                    else tuple(canonical_stream.punctuation_facts)
                ),
                candidate_boundaries=(
                    tuple(data.canonical_candidate_boundaries)
                    if data.canonical_candidate_boundaries
                    else tuple(canonical_stream.candidate_boundaries)
                ),
            )
        return canonical_stream

    @staticmethod
    def _build_synthetic_text_and_word_mapping(
        *,
        words: Sequence[WordTimestamp],
    ) -> Tuple[str, List[Optional[int]]]:
        chars: List[str] = []
        clean_to_word: List[Optional[int]] = []
        previous_char: str = ""
        for word_index, word in enumerate(list(words or [])):
            token = str(getattr(word, "word", "") or "")
            if not token:
                continue
            if chars:
                first_char = token[0]
                if (
                    previous_char
                    and previous_char.isascii()
                    and first_char.isascii()
                    and previous_char.isalnum()
                    and first_char.isalnum()
                ):
                    chars.append(" ")
                    clean_to_word.append(None)
            for char in token:
                chars.append(char)
                clean_to_word.append(word_index)
            previous_char = token[-1]
        return "".join(chars), clean_to_word

    @staticmethod
    def _build_synthetic_aligned_facts(
        *,
        words: Sequence[WordTimestamp],
    ) -> AlignedFacts:
        annotated_words = [
            AnnotatedWord(
                word=str(getattr(item, "word", "") or ""),
                start=float(getattr(item, "start", 0.0) or 0.0),
                end=float(getattr(item, "end", 0.0) or 0.0),
                confidence=getattr(item, "confidence", None),
                confidence_source=str(getattr(item, "confidence_source", "") or "unknown"),
                is_pseudo=bool(getattr(item, "is_pseudo", False)),
                speaker_id=getattr(item, "speaker_id", None),
                turn_id=getattr(item, "turn_id", None),
                track_id="decision_synthetic",
            )
            for item in list(words or [])
        ]
        return AlignedFacts(annotated_words=annotated_words)

    @staticmethod
    def _resolve_text_source_from_words(*, words: Sequence[WordTimestamp]) -> str:
        source_count: Counter[str] = Counter()
        for item in list(words or []):
            source = str(getattr(item, "confidence_source", "") or "").strip().lower()
            if source in {"fast", "slow", "aligned"}:
                source_count[source] += 1
            elif source == "merged":
                source_count["aligned"] += 1
        if not source_count:
            return "aligned"
        return source_count.most_common(1)[0][0]

    @staticmethod
    def _resolve_language_from_decision_input(data: DecisionLayerInput) -> str:
        snapshot = getattr(data, "policy_snapshot", None)
        language = str(getattr(snapshot, "language_tag", "") or getattr(snapshot, "language", "") or "")
        if language:
            return language
        return "auto"

    @staticmethod
    def _resolve_segment_token_range(
        *,
        tokens: Sequence[Any],
        sentence: SentenceSegment,
        cursor: int,
        tolerance_sec: float = 0.001,
    ) -> Tuple[Optional[int], Optional[int], int]:
        if not tokens:
            return None, None, cursor
        sentence_start = float(getattr(sentence, "start", 0.0) or 0.0)
        sentence_end = float(getattr(sentence, "end", sentence_start) or sentence_start)
        matched_indices = [
            idx
            for idx, token in enumerate(tokens)
            if idx >= cursor
            and float(getattr(token, "end", 0.0) or 0.0) > sentence_start + 1e-4
            and float(getattr(token, "start", 0.0) or 0.0) < sentence_end - 1e-4
        ]
        if matched_indices:
            return matched_indices[0], matched_indices[-1], matched_indices[-1] + 1
        if cursor < len(tokens):
            return cursor, cursor, cursor + 1
        last_idx = len(tokens) - 1
        return last_idx, last_idx, len(tokens)

    @staticmethod
    def _resolve_consumed_boundary_punct(
        *,
        facts: Sequence[Any],
        token_end: int,
    ) -> Optional[ConsumedBoundaryPunct]:
        candidates = [
            fact
            for fact in facts
            if str(getattr(fact, "punct_class", "") or "") == "sentence_end"
            and (
                getattr(fact, "left_token_index", None) == token_end
                or getattr(fact, "right_token_index", None) == token_end + 1
            )
        ]
        if not candidates:
            return None
        chosen = sorted(
            candidates,
            key=lambda item: int(getattr(item, "priority", 0) or 0),
            reverse=True,
        )[0]
        normalized_text = str(getattr(chosen, "normalized_text", "") or "")
        render_hint = "drop_period_default" if normalized_text in {".", "。"} else "keep"
        return ConsumedBoundaryPunct(
            fact_id=str(getattr(chosen, "fact_id", "") or ""),
            raw_text=str(getattr(chosen, "raw_text", "") or normalized_text),
            normalized_text=normalized_text,
            punct_class=str(getattr(chosen, "punct_class", "") or "sentence_end"),
            source=str(getattr(chosen, "source", "") or "aligned"),
            render_hint=render_hint,
        )

    @staticmethod
    def _build_trace_payload(
        *,
        sentence_index: int,
        sentence: SentenceSegment,
        output_traces: Sequence[OutputTrace],
    ) -> Dict[str, Any]:
        if sentence_index < len(output_traces):
            trace = output_traces[sentence_index]
            return {
                "split_reason": str(getattr(trace, "split_reason", "") or ""),
                "split_risk": str(getattr(trace, "split_risk", "") or ""),
                "window_id": str(getattr(trace, "window_id", "") or ""),
                "pyannote_frame_time": getattr(trace, "pyannote_frame_time", None),
                "mapped_cut_time": getattr(trace, "mapped_cut_time", None),
                "mapping_quality": str(getattr(trace, "mapping_quality", "") or ""),
                "mapping_reason": str(getattr(trace, "mapping_reason", "") or ""),
            }
        return {
            "split_reason": str(getattr(sentence, "split_reason", "") or "segmentation_core"),
            "split_risk": str(getattr(sentence, "split_risk", "") or ""),
            "window_id": str(getattr(sentence, "window_id", "") or ""),
            "pyannote_frame_time": getattr(sentence, "pyannote_frame_time", None),
            "mapped_cut_time": getattr(sentence, "mapped_cut_time", None),
            "mapping_quality": str(getattr(sentence, "mapping_quality", "") or ""),
            "mapping_reason": str(getattr(sentence, "mapping_reason", "") or ""),
        }

    def _build_segmentation_result_from_sentences(
        self,
        *,
        stream_id: str,
        chunk_ref: Any,
        words_for_split: Sequence[WordTimestamp],
        sentence_segments: Sequence[SentenceSegment],
        output_traces: Sequence[OutputTrace],
    ) -> SegmentationResult:
        tokens = list(words_for_split or ())
        if not tokens:
            return SegmentationResult(segments=tuple())
        plans: List[SegmentPlan] = []
        boundary_traces: List[Dict[str, Any]] = []
        cursor = 0
        chunk_ref_text = str(chunk_ref if chunk_ref is not None else "chunk-unknown")
        for sentence_index, sentence in enumerate(list(sentence_segments or [])):
            token_start, token_end, cursor = self._resolve_segment_token_range(
                tokens=tokens,
                sentence=sentence,
                cursor=cursor,
            )
            if token_start is None or token_end is None:
                continue
            trace_payload = self._build_trace_payload(
                sentence_index=sentence_index,
                sentence=sentence,
                output_traces=output_traces,
            )
            plan = SegmentPlan(
                segment_id=f"{stream_id}:{chunk_ref_text}:seg:{sentence_index}",
                token_start=token_start,
                token_end=token_end,
                start=float(getattr(sentence, "start", 0.0) or 0.0),
                end=float(getattr(sentence, "end", 0.0) or 0.0),
                boundary_reason=str(trace_payload.get("split_reason", "") or "segmentation_core"),
                boundary_score=1.0 if trace_payload.get("split_reason") else 0.0,
                hard_boundary=False,
                consumed_boundary_punct=self._infer_consumed_boundary_punct_from_sentence(sentence),
                trace=trace_payload,
            )
            plans.append(plan)
            boundary_traces.append(trace_payload)
        return SegmentationResult(
            segments=tuple(plans),
            boundary_traces=tuple(boundary_traces),
        )

    def _build_split_exit_segmentation_result(
        self,
        *,
        stream_id: str,
        chunk_ref: Any,
        words_for_split: Sequence[WordTimestamp],
        sentence_segments: Sequence[SentenceSegment],
        output_traces: Sequence[OutputTrace],
        source: str,
    ) -> SegmentationResult:
        result = self._build_segmentation_result_from_sentences(
            stream_id=stream_id,
            chunk_ref=chunk_ref,
            words_for_split=words_for_split,
            sentence_segments=sentence_segments,
            output_traces=output_traces,
        )
        return SegmentationResult(
            segments=tuple(result.segments),
            boundary_traces=tuple(result.boundary_traces),
            segmentation_report={
                "source": str(source or "split_exit"),
                "segment_count": int(len(result.segments)),
            },
        )

    @staticmethod
    def _infer_consumed_boundary_punct_from_sentence(
        sentence: SentenceSegment,
    ) -> Optional[ConsumedBoundaryPunct]:
        candidate = ""
        words = list(getattr(sentence, "words", []) or [])
        if words:
            candidate = str(getattr(words[-1], "word", "") or "")
        if not candidate:
            candidate = str(getattr(sentence, "text", "") or getattr(sentence, "text_clean", "") or "")
        candidate = candidate.rstrip()
        if not candidate:
            return None
        punct = candidate[-1]
        if punct not in {"。", "！", "？", ".", "!", "?"}:
            return None
        render_hint = "drop_period_default" if punct in {".", "。"} else "keep"
        return ConsumedBoundaryPunct(
            fact_id=f"inferred:{punct}:{int(float(getattr(sentence, 'end', 0.0) or 0.0) * 1000)}",
            raw_text=punct,
            normalized_text=punct,
            punct_class="sentence_end",
            source="aligned",
            render_hint=render_hint,
        )

    @staticmethod
    def _build_sentence_words_from_segment(
        *,
        canonical_stream: Any,
        segment: SegmentPlan,
    ) -> List[WordTimestamp]:
        token_by_index = {
            int(getattr(token, "index", -1)): token
            for token in list(getattr(canonical_stream, "tokens", ()) or ())
        }
        words: List[WordTimestamp] = []
        for token_index in range(int(segment.token_start), int(segment.token_end) + 1):
            token = token_by_index.get(token_index)
            if token is None:
                continue
            word = WordTimestamp(
                word=str(getattr(token, "text_core", "") or ""),
                start=float(getattr(token, "start", 0.0) or 0.0),
                end=float(getattr(token, "end", 0.0) or 0.0),
                confidence=getattr(token, "confidence", None),
                confidence_source=str(getattr(token, "source", "") or None),
                is_pseudo=bool(getattr(token, "is_pseudo", False)),
            )
            SegmentationProcessor._copy_word_runtime_metadata(source=token, target=word)
            words.append(word)
        return words

    def _run_segmentation_core(
        self,
        data: DecisionLayerInput,
        *,
        stream_id: str = "main",
        chunk_index: Optional[int] = None,
        is_last_chunk: bool = False,
    ) -> DecisionLayerOutput:
        """执行统一切分核心主路径（由 SegmentationCore 调用）。"""
        annotated_words = data.annotated_words or []
        pending_prefix_words = self._consume_pending_prefix_words(stream_id)
        input_cut_plan = data.cut_plan
        aligned_facts = data.aligned_facts
        fused_evidence = data.fused_evidence
        empty_cut_plan = self._normalize_cut_plan(
            input_cut_plan,
            stream_id=stream_id,
            chunk_index=chunk_index,
        )
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
                        cut_plan=empty_cut_plan,
                        applied_window_ids=[],
                    ),
                    "aligned_facts_stats": self._build_aligned_facts_stats(aligned_facts),
                    "fused_evidence_stats": self._build_fused_evidence_stats(fused_evidence),
                    "stream_id": stream_id,
                    "error_code": "E_DECISION_SPLIT_EMPTY",
                },
                applied_cut_plan=empty_cut_plan,
                output_traces=[],
            )

        raw_words_for_split = list(pending_prefix_words) + self._build_words_for_split(annotated_words)
        (
            words_for_split,
            dropped_unknown_pseudo_count,
            is_unknown_pseudo_filter_fallback,
            degraded_unknown_pseudo_count,
        ) = self._filter_unknown_pseudo_words(raw_words_for_split)
        words_for_split = merge_protected_word_tokens(words_for_split)
        words_for_split, timestamp_backtrack_fix_count = self._normalize_words_for_split_timestamps(
            words_for_split
        )
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
                    "unknown_pseudo_drop_count": int(dropped_unknown_pseudo_count),
                    "unknown_pseudo_degrade_count": int(degraded_unknown_pseudo_count),
                    "unknown_pseudo_filter_fallback": bool(is_unknown_pseudo_filter_fallback),
                    "timestamp_backtrack_fix_count": int(timestamp_backtrack_fix_count),
                    "soft_cut_stats": self._build_soft_cut_stats(
                        cut_plan=empty_cut_plan,
                        applied_window_ids=[],
                    ),
                    "aligned_facts_stats": self._build_aligned_facts_stats(aligned_facts),
                    "fused_evidence_stats": self._build_fused_evidence_stats(fused_evidence),
                    "stream_id": stream_id,
                    "chunk_index": chunk_index,
                    "error_code": "E_DECISION_SPLIT_EMPTY",
                },
                applied_cut_plan=empty_cut_plan,
                output_traces=[],
            )

        cut_plan = self._resolve_cut_plan_for_split(
            data=data,
            stream_id=stream_id,
            chunk_index=chunk_index,
            words_for_split=words_for_split,
        )

        applied_window_ids: List[str] = []
        output_traces: List[OutputTrace] = []
        segmentation_result: Optional[SegmentationResult] = None
        (
            all_sentence_segments,
            applied_window_ids,
            output_traces,
            segmentation_result,
        ) = self._split_by_cut_plan(
            stream_id=stream_id,
            chunk_ref=chunk_index,
            words_for_split=words_for_split,
            cut_plan=cut_plan,
            aligned_facts=aligned_facts,
            policy_snapshot=data.policy_snapshot,
            fallback_clean_text_ref=str(getattr(data, "fallback_clean_text_ref", "") or ""),
            fallback_punctuation_positions=list(
                getattr(data, "fallback_punctuation_positions", []) or []
            ),
            allow_fast_draft_fallback=bool(getattr(data, "allow_fast_draft_fallback", True)),
        )
        self._apply_output_traces_to_sentences(
            sentence_segments=all_sentence_segments,
            output_traces=output_traces,
        )
        self._active_vad_intervals = list(data.vad_intervals or [])
        speaker_repair_split_count = 0
        segmentation_result_needs_refresh = segmentation_result is None
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
                segmentation_result_needs_refresh = True

        pending_out_words: List[WordTimestamp] = []
        dangling_fix_count = 0
        if all_sentence_segments and not is_last_chunk:
            (
                all_sentence_segments,
                pending_out_words,
                dangling_fix_count,
            ) = self._extract_cross_chunk_pending_tail(
                all_sentence_segments,
                policy_snapshot=data.policy_snapshot,
            )
            if pending_out_words:
                self._pending_prefix_words_by_stream[stream_id] = self._clone_words(pending_out_words)
                segmentation_result_needs_refresh = True
            if dangling_fix_count > 0:
                segmentation_result_needs_refresh = True
        elif pending_prefix_words:
            # 没有产出时归还前缀，避免状态丢失。
            self._pending_prefix_words_by_stream[stream_id] = self._clone_words(pending_prefix_words)

        # 后处理
        self._normalize_carried_article_sentence_case(
            sentence_segments=all_sentence_segments,
            pending_in_word_count=pending_in_word_count,
            policy_snapshot=data.policy_snapshot,
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
            segmentation_result_needs_refresh = True
        else:
            error_code = ""

        split_stats = dict(self._final_splitter.last_split_stats or {})
        report: Dict[str, Any] = {
            "boundary_score_stats": split_stats,
            "forced_split_count": float(split_stats.get("force_split_count", 0.0)),
            "cross_chunk_pending_in_word_count": pending_in_word_count,
            "cross_chunk_pending_out_word_count": len(pending_out_words),
            "cross_chunk_dangling_fix_count": dangling_fix_count,
            "unknown_pseudo_drop_count": int(dropped_unknown_pseudo_count),
            "unknown_pseudo_degrade_count": int(degraded_unknown_pseudo_count),
            "unknown_pseudo_filter_fallback": bool(is_unknown_pseudo_filter_fallback),
            "timestamp_backtrack_fix_count": int(timestamp_backtrack_fix_count),
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
            "裁决层切分完成: stream={} chunk={} input_words={} raw_words={} "
            "sentences={} pending_in={} pending_out={} dropped_unknown_pseudo={} "
            "degraded_unknown_pseudo={} filter_fallback={} backtrack_fix={} error={}",
            stream_id,
            chunk_index,
            len(words_for_split),
            len(raw_words_for_split),
            len(all_sentence_segments),
            pending_in_word_count,
            len(pending_out_words),
            int(dropped_unknown_pseudo_count),
            int(degraded_unknown_pseudo_count),
            int(is_unknown_pseudo_filter_fallback),
            int(timestamp_backtrack_fix_count),
            error_code or "none",
        )
        self._active_vad_intervals = []
        if words_for_split and (
            segmentation_result_needs_refresh
            or segmentation_result is None
            or len(segmentation_result.segments) != len(all_sentence_segments)
        ):
            segmentation_result = self._build_split_exit_segmentation_result(
                stream_id=stream_id,
                chunk_ref=chunk_index,
                words_for_split=words_for_split,
                sentence_segments=all_sentence_segments,
                output_traces=output_traces,
                source="segmentation_core_postprocess_refresh",
            )

        return DecisionLayerOutput(
            sentence_segments=[],
            words_for_split=words_for_split,
            segmentation_report=report,
            applied_cut_plan=cut_plan,
            output_traces=output_traces,
            segmentation_result=segmentation_result,
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

    def _resolve_cut_plan_for_split(
        self,
        *,
        data: DecisionLayerInput,
        stream_id: str,
        chunk_index: Optional[int],
        words_for_split: Sequence[WordTimestamp],
    ) -> Any:
        if data.cut_plan is not None:
            return data.cut_plan
        if self._is_timeanchored_ingress(stream_id=stream_id, ingress_context=data.ingress_context):
            return self._ingress_segment_planner.build_plan(
                processor=self,
                data=data,
                stream_id=stream_id,
                chunk_index=chunk_index,
                words_for_split=words_for_split,
            )
        scored_plan = self._build_decision_scored_cut_plan(
            data=data,
            stream_id=stream_id,
            chunk_index=chunk_index,
            words_for_split=words_for_split,
        )
        if scored_plan is not None:
            return scored_plan
        return self._normalize_cut_plan(
            None,
            stream_id=stream_id,
            chunk_index=chunk_index,
        )

    def _build_decision_scored_cut_plan(
        self,
        *,
        data: DecisionLayerInput,
        stream_id: str,
        chunk_index: Optional[int],
        words_for_split: Sequence[WordTimestamp],
    ) -> Optional[CutPlan]:
        if len(words_for_split) <= 1:
            return None

        boundaries = self._collect_boundary_evidences_for_scoring(
            data=data,
            words_for_split=words_for_split,
        )
        language_is_cjk = self._is_cjk_policy_language(data.policy_snapshot)
        score_threshold = (
            self._INGRESS_SCORING_CJK_THRESHOLD
            if language_is_cjk
            else self._INGRESS_SCORING_DEFAULT_THRESHOLD
        )
        min_adjacent_gap_sec = (
            self._INGRESS_SCORING_MIN_ADJACENT_BOUNDARY_GAP_CJK_SEC
            if language_is_cjk
            else self._INGRESS_SCORING_MIN_ADJACENT_BOUNDARY_GAP_SEC
        )

        best_by_split_idx: Dict[int, Dict[str, Any]] = {}
        eligible_reason_stats: Counter[str] = Counter()
        rejected_boundary_stats: Counter[str] = Counter()
        for boundary in boundaries:
            reject_reason = self._resolve_ingress_boundary_reject_reason(
                boundary=boundary,
                words_for_split=words_for_split,
            )
            if reject_reason:
                rejected_boundary_stats[reject_reason] += 1
                continue
            candidate = self._build_ingress_boundary_candidate(
                boundary=boundary,
                words_for_split=words_for_split,
            )
            if candidate is None:
                continue
            is_force = bool(candidate["hard_flag"]) and str(candidate["reason"]) in {
                "punctuation_sentence_end",
                "speaker_change",
            }
            if not is_force and float(candidate["scored"]) < float(score_threshold):
                continue
            candidate["force"] = bool(is_force)
            eligible_reason_stats[str(candidate["reason"])] += 1

            split_idx = int(candidate["split_idx"])
            previous = best_by_split_idx.get(split_idx)
            if previous is None or self._is_better_ingress_candidate(candidate, previous):
                best_by_split_idx[split_idx] = candidate

        selected_candidates = [
            dict(item)
            for item in sorted(
                best_by_split_idx.values(),
                key=lambda value: (int(value["split_idx"]), float(value["event_time"])),
            )
        ]
        compressed_candidates: List[Dict[str, Any]] = []
        for candidate in selected_candidates:
            if not compressed_candidates:
                compressed_candidates.append(candidate)
                continue
            previous = compressed_candidates[-1]
            is_adjacent = int(candidate["split_idx"]) - int(previous["split_idx"]) <= 1
            boundary_gap = float(candidate["event_time"]) - float(previous["event_time"])
            if not is_adjacent or boundary_gap >= float(min_adjacent_gap_sec):
                compressed_candidates.append(candidate)
                continue
            if self._is_better_ingress_candidate(candidate, previous):
                compressed_candidates[-1] = candidate

        decisions = [
            self._build_ingress_cut_decision(
                stream_id=stream_id,
                candidate=item,
                words_for_split=words_for_split,
            )
            for item in compressed_candidates
        ]
        decision_reason_stats = Counter(str(item.reason) for item in decisions)
        fallback_reason = ""
        if not decisions:
            fallback_reason = "no_candidate_boundaries" if not boundaries else "no_boundary_passed_scoring"
        chunk_part = int(chunk_index) if chunk_index is not None else -1
        return CutPlan(
            plan_id=f"{stream_id}-evidence-scored-{chunk_part}",
            block_id=f"{stream_id}:{chunk_part}",
            decisions=decisions,
            deferred_cuts=[],
            generation_report={
                "generated_by": "decision_layer_boundary_scoring",
                "fallback_reason": fallback_reason,
                "input_boundary_count": int(len(boundaries)),
                "eligible_boundary_count": int(sum(eligible_reason_stats.values())),
                "decision_count": int(len(decisions)),
                "rejected_boundary_count": int(sum(rejected_boundary_stats.values())),
                "rejected_boundary_stats": dict(rejected_boundary_stats),
                "score_threshold": float(score_threshold),
                "reason_stats": dict(decision_reason_stats),
                "eligible_reason_stats": dict(eligible_reason_stats),
            },
        )

    def _collect_boundary_evidences_for_scoring(
        self,
        *,
        data: DecisionLayerInput,
        words_for_split: Sequence[WordTimestamp],
    ) -> List[Any]:
        boundaries: List[Any] = list(data.canonical_candidate_boundaries or ())
        boundaries.extend(
            self._build_punctuation_fact_boundaries(
                punctuation_facts=list(getattr(data, "canonical_punctuation_facts", ()) or ()),
                words_for_split=words_for_split,
            )
        )
        boundaries.extend(self._build_gap_boundaries(words_for_split=words_for_split))
        boundaries.extend(
            self._build_turn_change_boundaries(
                turns=list(getattr(getattr(data, "aligned_facts", None), "speaker_turns", ()) or ()),
                words_for_split=words_for_split,
            )
        )
        boundaries.extend(
            self._build_duration_guard_boundaries(
                words_for_split=words_for_split,
                policy_snapshot=data.policy_snapshot,
            )
        )
        return boundaries

    def _build_punctuation_fact_boundaries(
        self,
        *,
        punctuation_facts: Sequence[Any],
        words_for_split: Sequence[WordTimestamp],
    ) -> List[Any]:
        boundaries: List[Any] = []
        if len(words_for_split) <= 1:
            return boundaries
        for fact in punctuation_facts:
            left_idx_raw = getattr(fact, "left_token_index", None)
            right_idx_raw = getattr(fact, "right_token_index", None)
            split_idx: Optional[int] = None
            if left_idx_raw is not None:
                try:
                    split_idx = int(left_idx_raw)
                except (TypeError, ValueError):
                    split_idx = None
            elif right_idx_raw is not None:
                try:
                    split_idx = int(right_idx_raw) - 1
                except (TypeError, ValueError):
                    split_idx = None
            if split_idx is None or split_idx < 0 or split_idx >= len(words_for_split) - 1:
                continue
            punct_class = str(getattr(fact, "punct_class", "") or "").strip().lower()
            is_sentence_end = punct_class == "sentence_end"
            reason = "punctuation_sentence_end" if is_sentence_end else "punctuation_soft"
            left_end = float(getattr(words_for_split[split_idx], "end", 0.0) or 0.0)
            right_start = float(
                getattr(words_for_split[split_idx + 1], "start", left_end) or left_end
            )
            boundaries.append(
                SimpleNamespace(
                    split_idx=int(split_idx),
                    event_time=(left_end + right_start) / 2.0,
                    left_end=left_end,
                    right_start=right_start,
                    reason=reason,
                    score=1.0 if is_sentence_end else 0.65,
                    hard_flag=is_sentence_end,
                    metadata={
                        "evidence_source": "canonical_punctuation_fact",
                        "fact_id": str(getattr(fact, "fact_id", "") or ""),
                        "left_token_index": left_idx_raw,
                        "right_token_index": right_idx_raw,
                    },
                )
            )
        return boundaries

    def _build_gap_boundaries(
        self,
        *,
        words_for_split: Sequence[WordTimestamp],
    ) -> List[Any]:
        boundaries: List[Any] = []
        if len(words_for_split) <= 1:
            return boundaries
        for split_idx in range(len(words_for_split) - 1):
            left_word = words_for_split[split_idx]
            right_word = words_for_split[split_idx + 1]
            left_end = float(getattr(left_word, "end", 0.0) or 0.0)
            right_start = float(getattr(right_word, "start", left_end) or left_end)
            gap = max(0.0, right_start - left_end)
            if gap < 0.12:
                continue
            if gap >= 0.25:
                reason = "gap_pause"
                score = min(1.0, 0.45 + gap)
                hard_flag = gap >= 0.45
            else:
                reason = "blank_valley"
                score = min(0.8, 0.32 + gap)
                hard_flag = False
            boundaries.append(
                SimpleNamespace(
                    split_idx=int(split_idx),
                    event_time=(left_end + right_start) / 2.0,
                    left_end=left_end,
                    right_start=right_start,
                    reason=reason,
                    score=float(score),
                    hard_flag=bool(hard_flag),
                    metadata={"evidence_source": "decision_layer_gap_scan"},
                )
            )
        return boundaries

    def _build_turn_change_boundaries(
        self,
        *,
        turns: Sequence[Any],
        words_for_split: Sequence[WordTimestamp],
    ) -> List[Any]:
        boundaries: List[Any] = []
        if len(words_for_split) <= 1 or len(turns) <= 1:
            return boundaries
        ordered_turns = sorted(
            list(turns),
            key=lambda item: (
                self._resolve_turn_float(item, "start"),
                self._resolve_turn_float(item, "end"),
            ),
        )
        for index in range(1, len(ordered_turns)):
            left_turn = ordered_turns[index - 1]
            right_turn = ordered_turns[index]
            left_speaker = self._resolve_turn_speaker(left_turn)
            right_speaker = self._resolve_turn_speaker(right_turn)
            if not left_speaker or not right_speaker or left_speaker == right_speaker:
                continue
            left_start = self._resolve_turn_float(left_turn, "start")
            left_end = self._resolve_turn_float(left_turn, "end")
            right_start = self._resolve_turn_float(right_turn, "start")
            right_end = self._resolve_turn_float(right_turn, "end")
            if min(max(0.0, left_end - left_start), max(0.0, right_end - right_start)) < float(
                self._SPEAKER_REPAIR_MIN_TURN_DURATION_SEC
            ):
                continue
            selection = self._boundary_mapper.select_best_boundary(
                words=words_for_split,
                event_time=right_start,
            )
            if selection is None:
                continue
            split_idx = int(selection.split_idx)
            if split_idx < 0 or split_idx >= len(words_for_split) - 1:
                continue
            boundary_confidence = max(
                float(getattr(right_turn, "boundary_confidence", 0.0) or 0.0),
                float(getattr(left_turn, "boundary_confidence", 0.0) or 0.0),
            )
            if boundary_confidence <= 0.0:
                boundary_confidence = 0.8
            left_word_end = float(getattr(words_for_split[split_idx], "end", 0.0) or 0.0)
            right_word_start = float(
                getattr(words_for_split[split_idx + 1], "start", left_word_end) or left_word_end
            )
            boundaries.append(
                SimpleNamespace(
                    split_idx=split_idx,
                    event_time=float(right_start),
                    left_end=left_word_end,
                    right_start=right_word_start,
                    reason="speaker_change",
                    score=max(0.72, min(0.99, boundary_confidence)),
                    hard_flag=True,
                    metadata={
                        "evidence_source": "aligned_facts_speaker_turns",
                        "from_speaker": left_speaker,
                        "to_speaker": right_speaker,
                        "left_turn_id": self._resolve_turn_id(left_turn),
                        "right_turn_id": self._resolve_turn_id(right_turn),
                    },
                )
            )
        return boundaries

    def _build_duration_guard_boundaries(
        self,
        *,
        words_for_split: Sequence[WordTimestamp],
        policy_snapshot: Optional["LanguagePolicySnapshot"],
    ) -> List[Any]:
        boundaries: List[Any] = []
        if len(words_for_split) <= 2:
            return boundaries
        max_segment_sec = 6.2 if self._is_cjk_policy_language(policy_snapshot) else 5.8
        segment_start_idx = 0
        for index in range(1, len(words_for_split)):
            segment_start = float(getattr(words_for_split[segment_start_idx], "start", 0.0) or 0.0)
            current_end = float(getattr(words_for_split[index], "end", segment_start) or segment_start)
            if current_end - segment_start < float(max_segment_sec):
                continue
            split_idx = max(segment_start_idx, index - 1)
            if split_idx >= len(words_for_split) - 1:
                break
            left_word_end = float(getattr(words_for_split[split_idx], "end", 0.0) or 0.0)
            right_word_start = float(
                getattr(words_for_split[split_idx + 1], "start", left_word_end) or left_word_end
            )
            boundaries.append(
                SimpleNamespace(
                    split_idx=int(split_idx),
                    event_time=(left_word_end + right_word_start) / 2.0,
                    left_end=left_word_end,
                    right_start=right_word_start,
                    reason="hard_limit_forced",
                    score=1.0,
                    hard_flag=True,
                    metadata={
                        "evidence_source": "decision_layer_duration_guard",
                        "max_segment_sec": float(max_segment_sec),
                    },
                )
            )
            segment_start_idx = split_idx + 1
            if segment_start_idx >= len(words_for_split) - 1:
                break
        return boundaries

    def _resolve_ingress_boundary_reject_reason(
        self,
        *,
        boundary: Any,
        words_for_split: Sequence[WordTimestamp],
    ) -> str:
        try:
            split_idx = int(getattr(boundary, "split_idx", -1))
        except (TypeError, ValueError):
            return "invalid_split_idx"
        if split_idx < 0 or split_idx >= len(words_for_split) - 1:
            return "invalid_split_idx"

        metadata = dict(getattr(boundary, "metadata", {}) or {})
        if bool(metadata.get("blocked_by_lock")):
            return "blocked_by_lock"

        reason = str(getattr(boundary, "reason", "") or "").strip().lower()
        left_status = str(metadata.get("left_mount_status", "") or "").strip().lower()
        right_status = str(metadata.get("right_mount_status", "") or "").strip().lower()
        if (
            left_status in self._INGRESS_LOW_QUALITY_MOUNT_STATUSES
            or right_status in self._INGRESS_LOW_QUALITY_MOUNT_STATUSES
        ):
            return "low_quality_mount_status"

        left_word = str(getattr(words_for_split[split_idx], "word", "") or "")
        right_word = str(getattr(words_for_split[split_idx + 1], "word", "") or "")
        if self._is_ascii_fragment_boundary(
            left_token=left_word,
            right_token=right_word,
            reason=reason,
        ):
            return "alpha_fragment"

        return ""

    @classmethod
    def _is_ascii_fragment_boundary(
        cls,
        *,
        left_token: str,
        right_token: str,
        reason: str,
    ) -> bool:
        if reason in {"punctuation_sentence_end", "punctuation_soft"}:
            return False
        if is_sentence_end_punct(
            left_token,
            right_token,
            sentence_end_chars=tuple(cls._SENTENCE_END_PUNCT),
        ):
            return False

        left_tail = cls._extract_ascii_alpha_tail(left_token)
        right_head = cls._extract_ascii_alpha_head(right_token)
        if not left_tail or not right_head:
            return False
        if not (left_tail[-1].islower() and right_head[0].islower()):
            return False
        return (
            len(left_tail) <= cls._INGRESS_FRAGMENT_MAX_LEFT_ALPHA_LEN
            and len(right_head) <= cls._INGRESS_FRAGMENT_MAX_RIGHT_ALPHA_LEN
        )

    @staticmethod
    def _extract_ascii_alpha_tail(token: str) -> str:
        text = str(token or "").strip().lstrip("\"'“”‘’([{")
        if not text:
            return ""
        match = re.search(r"([A-Za-z]+)[^A-Za-z]*$", text)
        if match is None:
            return ""
        return str(match.group(1) or "")

    @staticmethod
    def _extract_ascii_alpha_head(token: str) -> str:
        text = str(token or "").strip().lstrip("\"'“”‘’([{")
        if not text:
            return ""
        match = re.match(r"([A-Za-z]+)", text)
        if match is None:
            return ""
        return str(match.group(1) or "")

    def _build_ingress_boundary_candidate(
        self,
        *,
        boundary: Any,
        words_for_split: Sequence[WordTimestamp],
    ) -> Optional[Dict[str, Any]]:
        if len(words_for_split) <= 1:
            return None
        try:
            split_idx = int(getattr(boundary, "split_idx", -1))
        except (TypeError, ValueError):
            return None
        if split_idx < 0 or split_idx >= len(words_for_split) - 1:
            return None

        reason = str(getattr(boundary, "reason", "") or "boundary_hint").strip().lower()
        if not reason:
            reason = "boundary_hint"
        try:
            base_score = float(getattr(boundary, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            base_score = 0.0
        base_score = max(0.0, min(1.0, base_score))
        hard_flag = bool(getattr(boundary, "hard_flag", False))

        left_end = self._optional_float(getattr(boundary, "left_end", None))
        right_start = self._optional_float(getattr(boundary, "right_start", None))
        if left_end is None:
            left_end = float(getattr(words_for_split[split_idx], "end", 0.0) or 0.0)
        if right_start is None:
            right_start = float(
                getattr(words_for_split[split_idx + 1], "start", left_end) or left_end
            )
        event_time = self._optional_float(getattr(boundary, "event_time", None))
        if event_time is None:
            event_time = (float(left_end) + float(right_start)) / 2.0

        reason_bonus = float(self._INGRESS_SCORING_REASON_BONUS.get(reason, 0.0))
        hard_bonus = 0.20 if hard_flag else 0.0
        scored = max(0.0, min(1.0, base_score + reason_bonus + hard_bonus))
        reason_priority = int(self._INGRESS_SCORING_REASON_PRIORITY.get(reason, 10))
        return {
            "split_idx": split_idx,
            "reason": reason,
            "base_score": base_score,
            "scored": scored,
            "hard_flag": hard_flag,
            "event_time": float(event_time),
            "left_end": float(left_end),
            "right_start": float(right_start),
            "reason_priority": reason_priority,
        }

    @staticmethod
    def _is_better_ingress_candidate(current: Dict[str, Any], previous: Dict[str, Any]) -> bool:
        if bool(current.get("force", False)) != bool(previous.get("force", False)):
            return bool(current.get("force", False))
        current_scored = float(current.get("scored", 0.0) or 0.0)
        previous_scored = float(previous.get("scored", 0.0) or 0.0)
        if abs(current_scored - previous_scored) > 1e-6:
            return current_scored > previous_scored
        current_priority = int(current.get("reason_priority", 0) or 0)
        previous_priority = int(previous.get("reason_priority", 0) or 0)
        if current_priority != previous_priority:
            return current_priority > previous_priority
        current_base_score = float(current.get("base_score", 0.0) or 0.0)
        previous_base_score = float(previous.get("base_score", 0.0) or 0.0)
        if abs(current_base_score - previous_base_score) > 1e-6:
            return current_base_score > previous_base_score
        return float(current.get("event_time", 0.0) or 0.0) <= float(
            previous.get("event_time", 0.0) or 0.0
        )

    def _build_ingress_cut_decision(
        self,
        *,
        stream_id: str,
        candidate: Dict[str, Any],
        words_for_split: Sequence[WordTimestamp],
    ) -> CutDecision:
        split_idx = int(candidate["split_idx"])
        mapped_cut_time = self._resolve_split_boundary_time(
            words_for_split=words_for_split,
            split_idx=split_idx,
        )
        reason = str(candidate["reason"] or "boundary_hint")
        return CutDecision(
            time=float(candidate["event_time"]),
            window_id=f"{stream_id}:slot-{split_idx}",
            reason=reason,
            risk=None,
            anchor_type=self._resolve_ingress_anchor_type(reason),
            anchor_score=float(candidate["scored"]),
            depends_on_fast_draft=reason == "fast_draft",
            time_range=(
                min(float(candidate["left_end"]), float(candidate["right_start"])),
                max(float(candidate["left_end"]), float(candidate["right_start"])),
            ),
            source="decision_layer_boundary_scoring",
            pyannote_frame_time=float(candidate["event_time"]),
            mapped_cut_time=float(mapped_cut_time),
            mapping_quality="boundary",
            mapping_reason="decision_layer_boundary_scoring",
        )

    @staticmethod
    def _resolve_ingress_anchor_type(reason: str) -> AnchorType:
        normalized_reason = str(reason or "").strip().lower()
        if "pause" in normalized_reason:
            return AnchorType.PAUSE_ANCHOR
        if "punct" in normalized_reason:
            return AnchorType.PUNCTUATION_ANCHOR
        if normalized_reason in {"lexical_boundary", "anchor_block_close", "semantic", "llm_semantic"}:
            return AnchorType.SEMANTIC_ANCHOR
        if normalized_reason in {"hard_limit_forced", "speaker_change"}:
            return AnchorType.WORD_BOUNDARY
        return AnchorType.WORD_BOUNDARY

    @staticmethod
    def _is_timeanchored_ingress(
        *,
        stream_id: str,
        ingress_context: Optional[SegmentationIngressContext],
    ) -> bool:
        if str(stream_id or "").strip().lower().startswith("timeanchored:"):
            return True
        unit_kind = str(getattr(ingress_context, "unit_kind", "") or "").strip().lower()
        return unit_kind == "slow_window"

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
        stream_id: str,
        chunk_ref: Any,
        words_for_split: List[WordTimestamp],
        cut_plan: Any,
        aligned_facts: Optional[Any] = None,
        policy_snapshot: Optional["LanguagePolicySnapshot"] = None,
        fallback_clean_text_ref: str = "",
        fallback_punctuation_positions: Optional[Sequence[Any]] = None,
        allow_fast_draft_fallback: bool = True,
    ) -> Tuple[List[SentenceSegment], List[str], List[OutputTrace], SegmentationResult]:
        """
        按 CutPlan 执行词流切分。

        说明：
        - 优先使用决策时间映射到词边界；
        - 若无可用边界，回退到 FinalSplitter 默认路径，确保不中断主流程。
        """
        decisions = list(getattr(cut_plan, "decisions", []) or [])
        if len(words_for_split) <= 1 or not decisions:
            if allow_fast_draft_fallback:
                fast_draft_fallback = self._try_split_by_fast_draft_cuts(
                    stream_id=stream_id,
                    chunk_ref=chunk_ref,
                    words_for_split=words_for_split,
                    aligned_facts=aligned_facts,
                    policy_snapshot=policy_snapshot,
                )
                if fast_draft_fallback is not None:
                    return fast_draft_fallback
            sentence_segments = self._final_splitter.split(
                words_for_split,
                clean_text=fallback_clean_text_ref or None,
                punctuation_positions=list(fallback_punctuation_positions or []),
            )
            output_traces = self._build_default_output_traces(sentence_segments)
            segmentation_result = self._build_split_exit_segmentation_result(
                stream_id=stream_id,
                chunk_ref=chunk_ref,
                words_for_split=words_for_split,
                sentence_segments=sentence_segments,
                output_traces=output_traces,
                source="default_splitter",
            )
            return sentence_segments, [], output_traces, segmentation_result

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
            policy_snapshot=policy_snapshot,
        )
        split_points = self._augment_split_points_with_sentence_end_punct(
            words_for_split=words_for_split,
            split_points=split_points,
            split_to_mapping=split_to_mapping,
        )
        if not split_points:
            if allow_fast_draft_fallback:
                fast_draft_decision_fallback = self._try_split_by_fast_draft_decisions(
                    stream_id=stream_id,
                    chunk_ref=chunk_ref,
                    words_for_split=words_for_split,
                    decisions=decisions,
                    policy_snapshot=policy_snapshot,
                )
                if fast_draft_decision_fallback is not None:
                    return fast_draft_decision_fallback
                fast_draft_fallback = self._try_split_by_fast_draft_cuts(
                    stream_id=stream_id,
                    chunk_ref=chunk_ref,
                    words_for_split=words_for_split,
                    aligned_facts=aligned_facts,
                    policy_snapshot=policy_snapshot,
                )
                if fast_draft_fallback is not None:
                    return fast_draft_fallback
            sentence_segments = self._final_splitter.split(
                words_for_split,
                clean_text=fallback_clean_text_ref or None,
                punctuation_positions=list(fallback_punctuation_positions or []),
            )
            output_traces = self._build_default_output_traces(sentence_segments)
            segmentation_result = self._build_split_exit_segmentation_result(
                stream_id=stream_id,
                chunk_ref=chunk_ref,
                words_for_split=words_for_split,
                sentence_segments=sentence_segments,
                output_traces=output_traces,
                source="default_splitter",
            )
            return sentence_segments, [], output_traces, segmentation_result

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
        sentence_segments, output_traces = self._merge_cjk_punctuation_fragments(
            sentence_segments=sentence_segments,
            output_traces=output_traces,
            policy_snapshot=policy_snapshot,
        )
        applied_window_ids = sorted(
            {
                str(split_to_window.get(split_idx, "") or "")
                for split_idx in split_points
                if str(split_to_window.get(split_idx, "") or "")
            }
        )
        segmentation_result = self._build_split_exit_segmentation_result(
            stream_id=stream_id,
            chunk_ref=chunk_ref,
            words_for_split=words_for_split,
            sentence_segments=sentence_segments,
            output_traces=output_traces,
            source="cut_plan",
        )
        return sentence_segments, applied_window_ids, output_traces, segmentation_result

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
        policy_snapshot: Optional["LanguagePolicySnapshot"] = None,
    ) -> List[int]:
        if len(words_for_split) <= 1 or not split_points:
            return list(split_points)

        kept_split_points: List[int] = []
        for idx, split_idx in enumerate(split_points):
            previous_split = split_points[idx - 1] if idx > 0 else None
            next_split = split_points[idx + 1] if idx + 1 < len(split_points) else None
            window_id = str(split_to_window.get(split_idx, "") or "")
            decision = decision_by_window.get(window_id)
            is_drop = self._is_temporal_backtrack_boundary(
                words_for_split=words_for_split,
                split_idx=split_idx,
            )
            if not is_drop:
                is_drop = self._should_drop_split_for_singleton_guard(
                    words_for_split=words_for_split,
                    split_idx=split_idx,
                    next_split=next_split,
                    decision=decision,
                )
            if not is_drop:
                is_drop = self._should_drop_split_for_short_prefix_guard(
                    words_for_split=words_for_split,
                    split_idx=split_idx,
                    previous_split=previous_split,
                    decision=decision,
                    policy_snapshot=policy_snapshot,
                )
            if not is_drop:
                kept_split_points.append(int(split_idx))
        return kept_split_points

    def _is_temporal_backtrack_boundary(
        self,
        *,
        words_for_split: Sequence[WordTimestamp],
        split_idx: int,
    ) -> bool:
        if split_idx < 0 or split_idx >= len(words_for_split) - 1:
            return False
        left_end = float(getattr(words_for_split[split_idx], "end", 0.0) or 0.0)
        right_start = float(getattr(words_for_split[split_idx + 1], "start", left_end) or left_end)
        return right_start < (left_end - self._TIMELINE_BACKTRACK_DROP_TOLERANCE_SEC)

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
        next_token = str(getattr(first_tail_word, "word", "") or "").strip()

        if singleton_word_count == 1:
            singleton_start = float(getattr(first_tail_word, "start", right_start) or right_start)
            singleton_end = float(getattr(first_tail_word, "end", singleton_start) or singleton_start)
            singleton_duration = max(0.0, singleton_end - singleton_start)
            singleton_token = str(getattr(first_tail_word, "word", "") or "").strip()
            is_lowercase_continuation = self._is_lowercase_continuation_token(singleton_token)
            is_cjk_fast_draft_tail = self._is_cjk_fast_draft_singleton_tail(
                words_for_split=words_for_split,
                split_idx=split_idx,
                singleton_token=singleton_token,
                decision=decision,
                next_split=next_split,
            )
            if is_cjk_fast_draft_tail:
                return True
            if is_sentence_end_punct(
                prev_tail,
                next_token,
                sentence_end_chars=tuple(self._SENTENCE_END_PUNCT),
            ):
                return False
            if not is_lowercase_continuation and not is_cjk_fast_draft_tail:
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
        if is_sentence_end_punct(
            prev_tail,
            next_token,
            sentence_end_chars=tuple(self._SENTENCE_END_PUNCT),
        ):
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

    def _should_drop_split_for_short_prefix_guard(
        self,
        *,
        words_for_split: Sequence[WordTimestamp],
        split_idx: int,
        previous_split: Optional[int],
        decision: Optional[Any],
        policy_snapshot: Optional["LanguagePolicySnapshot"],
    ) -> bool:
        """
        CJK 前缀短句守门：抑制 chunk 开头被 pause/speaker 误切成“短前缀句”。

        Why:
        - 双流 CJK 下，常见首句被错误切成 `如` / `其实` / `这起案件`；
        - 这些切点多由弱 pause 触发，且边界处并无句末强标点。
        """
        if not self._is_cjk_policy_language(policy_snapshot):
            return False
        if previous_split is not None:
            return False
        if split_idx < 0 or split_idx >= len(words_for_split) - 1:
            return False

        reason = str(getattr(decision, "reason", "") or "")
        if reason not in {"pause", "speaker_change", "fast_draft"}:
            return False

        left_word_count = split_idx + 1
        if left_word_count <= 0 or left_word_count > self._CUT_PLAN_CJK_SHORT_PREFIX_MAX_WORDS:
            return False

        left_words = list(words_for_split[:left_word_count])
        if not left_words:
            return False
        left_start = float(getattr(left_words[0], "start", 0.0) or 0.0)
        left_end = float(getattr(left_words[-1], "end", left_start) or left_start)
        prefix_duration = max(0.0, left_end - left_start)
        if prefix_duration > self._CUT_PLAN_CJK_SHORT_PREFIX_MAX_DURATION_SEC:
            return False

        right_word = words_for_split[split_idx + 1]
        right_start = float(getattr(right_word, "start", left_end) or left_end)
        boundary_gap_sec = max(0.0, right_start - left_end)
        if boundary_gap_sec > self._CUT_PLAN_CJK_SHORT_PREFIX_MAX_GAP_SEC:
            return False

        left_tail_token = str(getattr(left_words[-1], "word", "") or "").strip()
        right_head_token = str(getattr(right_word, "word", "") or "").strip()
        if is_sentence_end_punct(
            left_tail_token,
            right_head_token,
            sentence_end_chars=tuple(self._SENTENCE_END_PUNCT),
        ):
            return False

        normalized_prefix_tokens = [
            self._normalize_boundary_token(str(getattr(item, "word", "") or ""))
            for item in left_words
        ]
        normalized_prefix_tokens = [item for item in normalized_prefix_tokens if item]
        if not normalized_prefix_tokens:
            return False
        if any(any(char.isalpha() and char.isascii() for char in token) for token in normalized_prefix_tokens):
            return False
        single_cjk_count = sum(
            1 for token in normalized_prefix_tokens if self._is_single_cjk_token(token)
        )
        total_char_count = sum(len(token) for token in normalized_prefix_tokens)
        if left_word_count == 1:
            return True
        if left_word_count == 2:
            return bool(single_cjk_count >= 2 or total_char_count <= 3)
        if left_word_count >= 3 and single_cjk_count < 2:
            return False
        return True

    def _is_cjk_fast_draft_singleton_tail(
        self,
        *,
        words_for_split: Sequence[WordTimestamp],
        split_idx: int,
        singleton_token: str,
        decision: Optional[Any],
        next_split: Optional[int],
    ) -> bool:
        """
        CJK + fast_draft 特殊守门：避免“最后一字被切出去”。

        Why:
        - 中文/日文在字级时间轴下，fast_draft 切点若落在词尾前，常形成 `XX...男 | 子`。
        - 该场景在业务上几乎总是误切，应在切点阶段直接丢弃。
        """
        if next_split is not None:
            return False
        reason = str(getattr(decision, "reason", "") or "")
        source = str(getattr(decision, "source", "") or "")
        if reason != "fast_draft" and source != "fast_draft":
            return False
        if split_idx + 1 < self._CUT_PLAN_CJK_FAST_DRAFT_SINGLETON_MIN_LEFT_WORDS:
            return False
        if not self._is_single_cjk_token(singleton_token):
            return False
        return True

    @staticmethod
    def _is_single_cjk_token(token: str) -> bool:
        # V3.2.0+dev.20260221.01: 先归一化去尾标点，避免 `件。/人。` 误判为“非单字”。
        text = SegmentationProcessor._normalize_boundary_token(token)
        if len(text) != 1:
            return False
        return SegmentationProcessor._is_cjk_text(text)

    @staticmethod
    def _is_cjk_text(text: str) -> bool:
        for char in str(text or ""):
            if "\u4e00" <= char <= "\u9fff":
                return True
            if "\u3040" <= char <= "\u30ff":
                return True
            if "\uac00" <= char <= "\ud7af":
                return True
        return False

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

    def _try_split_by_fast_draft_decisions(
        self,
        *,
        stream_id: str,
        chunk_ref: Any,
        words_for_split: Sequence[WordTimestamp],
        decisions: Sequence[Any],
        policy_snapshot: Optional["LanguagePolicySnapshot"],
    ) -> Optional[Tuple[List[SentenceSegment], List[str], List[OutputTrace], SegmentationResult]]:
        """
        CutPlan 已有决策但切点全部失效时，优先回写 fast_draft 决策本身。

        Why:
        - 线上出现 `decision_count>0` 且 `split_points=0` 后直接走 default splitter；
        - fast_draft 决策仍然携带可用的事件时间，不应在这一步被完全丢弃。
        """
        if len(words_for_split) <= 1 or not decisions:
            return None
        if not self._is_cjk_policy_language(policy_snapshot):
            return None

        split_payload_by_idx: Dict[int, Dict[str, Any]] = {}
        for decision in sorted(
            decisions,
            key=lambda item: (
                float(getattr(item, "time", 0.0) or 0.0),
                str(getattr(item, "window_id", "") or ""),
            ),
        ):
            reason = str(getattr(decision, "reason", "") or "")
            source = str(getattr(decision, "source", "") or "")
            depends_on_fast_draft = bool(getattr(decision, "depends_on_fast_draft", False))
            if reason != "fast_draft" and source != "fast_draft" and not depends_on_fast_draft:
                continue

            decision_time = self._optional_float(getattr(decision, "time", None))
            if decision_time is None:
                continue
            selection = self._boundary_mapper.select_best_boundary(
                words=words_for_split,
                event_time=decision_time,
            )
            if selection is None:
                continue
            split_idx = int(selection.split_idx)
            if split_idx < 0 or split_idx >= len(words_for_split) - 1:
                continue
            if self._is_fast_draft_fallback_short_prefix_boundary(
                words_for_split=words_for_split,
                split_idx=split_idx,
            ):
                continue
            if self._is_fast_draft_fallback_singleton_tail_boundary(
                words_for_split=words_for_split,
                split_idx=split_idx,
            ):
                continue

            payload = {
                "window_id": str(getattr(decision, "window_id", "") or ""),
                "split_reason": reason or "fast_draft",
                "split_risk": str(getattr(decision, "risk", "") or ""),
                "pyannote_frame_time": self._optional_float(
                    getattr(decision, "pyannote_frame_time", None),
                ),
                "decision_time": float(decision_time),
                "mapped_cut_time": self._resolve_split_boundary_time(
                    words_for_split=words_for_split,
                    split_idx=split_idx,
                ),
                "mapping_quality": "gap" if selection.is_in_gap else "boundary",
                "mapping_reason": "fast_draft_decision_fallback",
                "selection_score": float(selection.score),
                "anchor_score": float(getattr(decision, "anchor_score", 0.0) or 0.0),
            }
            existing_payload = split_payload_by_idx.get(split_idx)
            if existing_payload is None:
                split_payload_by_idx[split_idx] = payload
                continue
            if float(payload["anchor_score"]) >= float(existing_payload.get("anchor_score", 0.0) or 0.0):
                split_payload_by_idx[split_idx] = payload

        split_points = sorted(split_payload_by_idx.keys())
        if not split_points:
            return None

        sentence_segments: List[SentenceSegment] = []
        output_traces: List[OutputTrace] = []
        start_idx = 0
        sentence_index = 0
        for split_idx in split_points:
            sentence = self._final_splitter._build_sentence(words_for_split, start_idx, split_idx)
            sentence_segments.append(sentence)
            payload = split_payload_by_idx.get(split_idx, {})
            pyannote_frame_time = self._optional_float(payload.get("pyannote_frame_time"))
            if pyannote_frame_time is None:
                pyannote_frame_time = self._optional_float(payload.get("decision_time"))
            output_traces.append(
                OutputTrace(
                    sentence_index=sentence_index,
                    split_reason=str(payload.get("split_reason", "fast_draft")),
                    split_risk=str(payload.get("split_risk", "")),
                    window_id=str(payload.get("window_id", "")),
                    pyannote_frame_time=pyannote_frame_time,
                    mapped_cut_time=float(payload.get("mapped_cut_time", sentence.end)),
                    mapping_quality=str(payload.get("mapping_quality", "boundary")),
                    mapping_reason=str(
                        payload.get("mapping_reason", "fast_draft_decision_fallback")
                    ),
                    sentence_start=float(sentence.start),
                    sentence_end=float(sentence.end),
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
        applied_window_ids = sorted(
            {
                str(payload.get("window_id", "") or "")
                for payload in split_payload_by_idx.values()
                if str(payload.get("window_id", "") or "")
            }
        )
        segmentation_result = self._build_split_exit_segmentation_result(
            stream_id=stream_id,
            chunk_ref=chunk_ref,
            words_for_split=words_for_split,
            sentence_segments=sentence_segments,
            output_traces=output_traces,
            source="fast_draft_decision_fallback",
        )
        return sentence_segments, applied_window_ids, output_traces, segmentation_result

    def _try_split_by_fast_draft_cuts(
        self,
        *,
        stream_id: str,
        chunk_ref: Any,
        words_for_split: Sequence[WordTimestamp],
        aligned_facts: Optional[Any],
        policy_snapshot: Optional["LanguagePolicySnapshot"],
    ) -> Optional[Tuple[List[SentenceSegment], List[str], List[OutputTrace], SegmentationResult]]:
        """
        CutPlan 无法产出有效切点时，优先尝试快流切点回写。

        Why:
        - CJK 双流在慢流弱标点场景常出现“有窗口无决策”，直接回退 default_splitter 会漏切。
        - fast_draft 切点时间更稳定，作为 fallback 可显著降低错切/漏切。
        """
        if len(words_for_split) <= 1:
            return None
        if not self._is_cjk_policy_language(policy_snapshot):
            return None
        fast_draft_cuts = list(getattr(aligned_facts, "fast_draft_cuts", []) or [])
        if not fast_draft_cuts:
            return None

        split_to_mapping: Dict[int, Dict[str, Any]] = {}
        for raw_cut in fast_draft_cuts:
            try:
                cut_time = float(raw_cut)
            except (TypeError, ValueError):
                continue
            selection = self._boundary_mapper.select_best_boundary(
                words=words_for_split,
                event_time=cut_time,
            )
            if selection is None:
                continue
            split_idx = int(selection.split_idx)
            if split_idx < 0 or split_idx >= len(words_for_split) - 1:
                continue
            if self._is_fast_draft_fallback_short_prefix_boundary(
                words_for_split=words_for_split,
                split_idx=split_idx,
            ):
                continue
            if self._is_fast_draft_fallback_singleton_tail_boundary(
                words_for_split=words_for_split,
                split_idx=split_idx,
            ):
                continue
            split_to_mapping[split_idx] = {
                "mapped_cut_time": self._resolve_split_boundary_time(
                    words_for_split=words_for_split,
                    split_idx=split_idx,
                ),
                "mapping_quality": "gap" if selection.is_in_gap else "boundary",
                "mapping_reason": "fast_draft_boundary_fallback",
                "selection_score": float(selection.score),
            }
        split_points = sorted(split_to_mapping.keys())
        if not split_points:
            return None

        sentence_segments: List[SentenceSegment] = []
        output_traces: List[OutputTrace] = []
        start_idx = 0
        sentence_index = 0
        for split_idx in split_points:
            sentence = self._final_splitter._build_sentence(words_for_split, start_idx, split_idx)
            sentence_segments.append(sentence)
            mapped_cut_time = self._optional_float(
                split_to_mapping.get(split_idx, {}).get("mapped_cut_time")
            )
            output_traces.append(
                OutputTrace(
                    sentence_index=sentence_index,
                    split_reason="fast_draft",
                    split_risk="fallback",
                    window_id="",
                    pyannote_frame_time=None,
                    mapped_cut_time=float(mapped_cut_time if mapped_cut_time is not None else sentence.end),
                    mapping_quality=str(
                        split_to_mapping.get(split_idx, {}).get("mapping_quality", "boundary")
                    ),
                    mapping_reason="fast_draft_boundary_fallback",
                    sentence_start=float(sentence.start),
                    sentence_end=float(sentence.end),
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
        segmentation_result = self._build_split_exit_segmentation_result(
            stream_id=stream_id,
            chunk_ref=chunk_ref,
            words_for_split=words_for_split,
            sentence_segments=sentence_segments,
            output_traces=output_traces,
            source="fast_draft_boundary_fallback",
        )
        return sentence_segments, [], output_traces, segmentation_result

    def _is_fast_draft_fallback_singleton_tail_boundary(
        self,
        *,
        words_for_split: Sequence[WordTimestamp],
        split_idx: int,
    ) -> bool:
        if split_idx < 0 or split_idx >= len(words_for_split) - 1:
            return False
        tail_count = len(words_for_split) - split_idx - 1
        if tail_count != 1:
            return False
        if split_idx + 1 < self._CUT_PLAN_CJK_FAST_DRAFT_SINGLETON_MIN_LEFT_WORDS:
            return False

        left_word = words_for_split[split_idx]
        tail_word = words_for_split[split_idx + 1]

        left_token = str(getattr(left_word, "word", "") or "").strip()
        tail_token = str(getattr(tail_word, "word", "") or "").strip()
        return self._is_single_cjk_token(tail_token)

    def _is_fast_draft_fallback_short_prefix_boundary(
        self,
        *,
        words_for_split: Sequence[WordTimestamp],
        split_idx: int,
    ) -> bool:
        if split_idx < 0 or split_idx >= len(words_for_split) - 1:
            return False
        left_word_count = split_idx + 1
        if left_word_count <= 0 or left_word_count > self._CUT_PLAN_CJK_FAST_DRAFT_SHORT_PREFIX_MAX_WORDS:
            return False
        left_start = float(getattr(words_for_split[0], "start", 0.0) or 0.0)
        left_end = float(getattr(words_for_split[split_idx], "end", left_start) or left_start)
        prefix_duration_sec = max(0.0, left_end - left_start)
        if prefix_duration_sec > self._CUT_PLAN_CJK_SHORT_PREFIX_MAX_DURATION_SEC:
            return False
        right_word = words_for_split[split_idx + 1]
        right_start = float(getattr(right_word, "start", left_end) or left_end)
        boundary_gap_sec = max(0.0, right_start - left_end)
        if boundary_gap_sec > self._CUT_PLAN_CJK_SHORT_PREFIX_MAX_GAP_SEC:
            return False
        left_tail = str(getattr(words_for_split[split_idx], "word", "") or "").strip()
        right_head = str(getattr(right_word, "word", "") or "").strip()
        if is_sentence_end_punct(
            left_tail,
            right_head,
            sentence_end_chars=tuple(self._SENTENCE_END_PUNCT),
        ):
            return False
        normalized_prefix_tokens = [
            self._normalize_boundary_token(str(getattr(item, "word", "") or ""))
            for item in list(words_for_split[:left_word_count])
        ]
        normalized_prefix_tokens = [item for item in normalized_prefix_tokens if item]
        if not normalized_prefix_tokens:
            return False
        single_cjk_count = sum(
            1 for token in normalized_prefix_tokens if self._is_single_cjk_token(token)
        )
        total_char_count = sum(len(token) for token in normalized_prefix_tokens)
        if left_word_count == 1:
            return True
        if left_word_count == 2:
            return bool(single_cjk_count >= 2 or total_char_count <= 3)
        if left_word_count >= 3 and single_cjk_count < 2:
            return False
        return True

    @staticmethod
    def _is_cjk_policy_language(
        policy_snapshot: Optional["LanguagePolicySnapshot"],
    ) -> bool:
        language_tag = str(getattr(policy_snapshot, "language_tag", "") or "").strip().lower()
        if not language_tag:
            return False
        return language_tag.startswith(("zh", "yue", "ja", "jp", "ko"))

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
            next_token = str(getattr(words_for_split[split_idx + 1], "word", "") or "").strip()
            if not is_sentence_end_punct(
                word_text,
                next_token,
                sentence_end_chars=sentence_end_punct,
            ):
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
        tail_token = str(getattr(last_words[0], "word", "") or "").strip()
        if is_sentence_end_punct(
            prev_tail,
            tail_token,
            sentence_end_chars=tuple(self._SENTENCE_END_PUNCT),
        ):
            return sentence_segments, output_traces

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

    def _merge_cjk_punctuation_fragments(
        self,
        *,
        sentence_segments: List[SentenceSegment],
        output_traces: List[OutputTrace],
        policy_snapshot: Optional["LanguagePolicySnapshot"] = None,
    ) -> Tuple[List[SentenceSegment], List[OutputTrace]]:
        """
        CJK + CutPlan 标点守门：回并逗号等弱断句导致的短碎片句。

        Why:
        - 双流 CJK 在慢流弱标点密集场景下，常出现 `punctuation` 连续落切导致的“短右句碎片”。
        - 该守门仅在 CJK 生效，且仅处理非强句末边界，避免影响英文与正常句末切分。
        """
        if len(sentence_segments) < 2 or not self._is_cjk_policy_language(policy_snapshot):
            return sentence_segments, output_traces

        sentence_end_chars = self._resolve_sentence_end_chars(policy_snapshot)
        continuation_words = {
            self._normalize_boundary_token(str(item or ""))
            for item in list(getattr(policy_snapshot, "continuation_words", frozenset()) or [])
            if self._normalize_boundary_token(str(item or ""))
        }
        traces = list(output_traces or [])
        merged_sentences: List[SentenceSegment] = []
        merged_traces: List[OutputTrace] = []
        index = 0
        while index < len(sentence_segments):
            if index >= len(sentence_segments) - 1:
                merged_sentences.append(sentence_segments[index])
                if index < len(traces):
                    merged_traces.append(traces[index])
                index += 1
                continue

            left_sentence = sentence_segments[index]
            right_sentence = sentence_segments[index + 1]
            left_trace = traces[index] if index < len(traces) else None
            if not self._should_merge_cjk_punctuation_boundary(
                left_sentence=left_sentence,
                right_sentence=right_sentence,
                left_trace=left_trace,
                continuation_words=continuation_words,
                sentence_end_chars=sentence_end_chars,
            ):
                merged_sentences.append(left_sentence)
                if left_trace is not None:
                    merged_traces.append(left_trace)
                index += 1
                continue

            left_words = list(left_sentence.words or [])
            right_words = list(right_sentence.words or [])
            if not left_words or not right_words:
                merged_sentences.append(left_sentence)
                if left_trace is not None:
                    merged_traces.append(left_trace)
                index += 1
                continue

            merged_words = left_words + right_words
            rebuilt_sentence = self._final_splitter._build_sentence(
                merged_words,
                0,
                len(merged_words) - 1,
            )
            self._copy_sentence_metadata(left_sentence, rebuilt_sentence)
            merged_sentences.append(rebuilt_sentence)

            next_trace = traces[index + 1] if (index + 1) < len(traces) else None
            if next_trace is None:
                next_trace = OutputTrace(
                    sentence_index=len(merged_sentences) - 1,
                    split_reason="tail_flush",
                    split_risk="",
                    window_id="",
                    pyannote_frame_time=None,
                    mapped_cut_time=float(rebuilt_sentence.end),
                    mapping_quality="merged",
                    mapping_reason="cjk_punctuation_guard",
                    sentence_start=float(rebuilt_sentence.start),
                    sentence_end=float(rebuilt_sentence.end),
                )
            else:
                next_trace = OutputTrace(
                    sentence_index=len(merged_sentences) - 1,
                    split_reason=str(next_trace.split_reason or ""),
                    split_risk=str(next_trace.split_risk or ""),
                    window_id=str(next_trace.window_id or ""),
                    pyannote_frame_time=next_trace.pyannote_frame_time,
                    mapped_cut_time=next_trace.mapped_cut_time,
                    mapping_quality=str(next_trace.mapping_quality or ""),
                    mapping_reason=str(next_trace.mapping_reason or ""),
                    sentence_start=float(rebuilt_sentence.start),
                    sentence_end=float(rebuilt_sentence.end),
                )
            merged_traces.append(next_trace)
            index += 2

        merged_traces = self._renumber_output_traces(merged_traces)
        return merged_sentences, merged_traces

    def _should_merge_cjk_punctuation_boundary(
        self,
        *,
        left_sentence: SentenceSegment,
        right_sentence: SentenceSegment,
        left_trace: Optional[OutputTrace],
        continuation_words: Set[str],
        sentence_end_chars: Tuple[str, ...],
    ) -> bool:
        if left_trace is None:
            return False
        if str(left_trace.split_reason or "") != "punctuation":
            return False

        left_words = list(left_sentence.words or [])
        right_words = list(right_sentence.words or [])
        if not left_words or not right_words:
            return False

        left_tail_raw = str(getattr(left_words[-1], "word", "") or "").strip()
        right_head_raw = str(getattr(right_words[0], "word", "") or "").strip()
        if is_sentence_end_punct(
            left_tail_raw,
            right_head_raw,
            sentence_end_chars=sentence_end_chars,
        ):
            return False

        left_end = float(getattr(left_words[-1], "end", 0.0) or 0.0)
        right_start = float(getattr(right_words[0], "start", left_end) or left_end)
        gap_sec = max(0.0, right_start - left_end)
        if gap_sec > self._CUT_PLAN_CJK_PUNCT_FRAGMENT_MAX_GAP_SEC:
            return False

        right_sentence_start = float(getattr(right_sentence, "start", right_start) or right_start)
        right_sentence_end = float(getattr(right_sentence, "end", right_sentence_start) or right_sentence_start)
        right_duration_sec = max(0.0, right_sentence_end - right_sentence_start)
        right_word_count = len(right_words)
        right_head_norm = self._normalize_boundary_token(right_head_raw)

        # Why: 单字/单词右句在 CJK 标点切分下高概率是碎片，优先回并。
        if right_word_count == 1:
            return right_duration_sec <= self._CUT_PLAN_CJK_PUNCT_FRAGMENT_SINGLETON_MAX_DURATION_SEC

        # Why: 连接词起句通常是句内延续，不应被弱标点直接截断。
        if right_head_norm in continuation_words:
            return (
                right_word_count <= self._CUT_PLAN_CJK_PUNCT_FRAGMENT_CONTINUATION_MAX_WORDS
                and right_duration_sec
                <= self._CUT_PLAN_CJK_PUNCT_FRAGMENT_CONTINUATION_MAX_DURATION_SEC
            )

        # Why: 短右句且首词为单个 CJK 字，常见于“字级词流 + 逗号切分”误切。
        if self._is_single_cjk_token(right_head_norm):
            return right_duration_sec <= self._CUT_PLAN_CJK_PUNCT_FRAGMENT_SHORT_MAX_DURATION_SEC

        return False

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
                "planner_diagnostics": {
                    "feature_stats": {},
                    "constraint_stats": {},
                    "rejection_stats": {},
                    "candidate_diagnostics": [],
                },
            }

        decisions = list(getattr(cut_plan, "decisions", []) or [])
        deferred_cuts = list(getattr(cut_plan, "deferred_cuts", []) or [])
        generation_report = dict(getattr(cut_plan, "generation_report", {}) or {})
        fusion_output_window_count = int(
            generation_report.get(
                "fusion_output_window_count",
                generation_report.get("output_window_count", 0),
            )
            or 0
        )
        fallback_reason = str(generation_report.get("fallback_reason", "") or "")
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
        diagnostic_code = ""
        if fusion_output_window_count > 0 and len(decisions) <= 0:
            diagnostic_code = "E_SOFT_CUT_WINDOWS_WITHOUT_DECISIONS"
        planner_diagnostics = {
            "feature_stats": dict(generation_report.get("feature_stats", {}) or {}),
            "constraint_stats": dict(generation_report.get("constraint_stats", {}) or {}),
            "rejection_stats": dict(generation_report.get("rejection_stats", {}) or {}),
            "candidate_diagnostics": list(generation_report.get("candidate_diagnostics", []) or []),
        }
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
            "fusion_output_window_count": fusion_output_window_count,
            "fallback_reason": fallback_reason,
            "diagnostic_code": diagnostic_code,
            "planner_diagnostics": planner_diagnostics,
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
        *,
        policy_snapshot: Optional["LanguagePolicySnapshot"] = None,
    ) -> Tuple[List[SentenceSegment], List[WordTimestamp], int]:
        if not sentence_segments:
            return sentence_segments, [], 0

        tail_words = self._detect_dangling_tail_words(
            sentence_segments[-1],
            policy_snapshot=policy_snapshot,
        )
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

    def _detect_dangling_tail_words(
        self,
        sentence: SentenceSegment,
        *,
        policy_snapshot: Optional["LanguagePolicySnapshot"] = None,
    ) -> List[WordTimestamp]:
        words = list(sentence.words or [])
        if len(words) <= 1:
            return []

        last_token = self._normalize_boundary_token(words[-1].word)
        is_english_carry = last_token in self._resolve_english_carry_words(policy_snapshot)
        is_cjk_carry = self._is_cjk_carry_token(last_token, policy_snapshot=policy_snapshot)
        if not is_english_carry and not is_cjk_carry:
            return []

        anchor_token_raw = str(words[-2].word or "").strip()
        has_sentence_end_anchor = is_sentence_end_punct(
            anchor_token_raw,
            str(words[-1].word or "").strip(),
            sentence_end_chars=self._resolve_sentence_end_chars(policy_snapshot),
        )
        if is_english_carry:
            if has_sentence_end_anchor:
                return [words[-1]]
            return []

        if has_sentence_end_anchor:
            return []
        cjk_tail_words = self._detect_cjk_dangling_tail_words(
            words,
            policy_snapshot=policy_snapshot,
        )
        if cjk_tail_words:
            return cjk_tail_words
        return []

    def _detect_cjk_dangling_tail_words(
        self,
        words: Sequence[WordTimestamp],
        *,
        policy_snapshot: Optional["LanguagePolicySnapshot"] = None,
    ) -> List[WordTimestamp]:
        if len(words) <= 1:
            return []

        last_word = words[-1]
        last_token = self._normalize_boundary_token(last_word.word)
        if not self._is_cjk_carry_token(last_token, policy_snapshot=policy_snapshot):
            return []

        prev_word = words[-2]
        prev_raw = str(prev_word.word or "").strip()
        last_raw = str(last_word.word or "").strip()
        if is_sentence_end_punct(
            prev_raw,
            last_raw,
            sentence_end_chars=self._resolve_sentence_end_chars(policy_snapshot),
        ):
            return []

        prev_end = float(getattr(prev_word, "end", 0.0) or 0.0)
        last_start = float(getattr(last_word, "start", prev_end) or prev_end)
        last_end = float(getattr(last_word, "end", last_start) or last_start)
        if last_end < last_start:
            last_end = last_start

        max_gap_sec, max_duration_sec = self._resolve_cjk_carry_limits(policy_snapshot)
        # Why: 仅兜底“紧贴 chunk 边界的短残词”，避免把真实句尾误判为跨 chunk 续接。
        if max(0.0, last_start - prev_end) > max_gap_sec:
            return []
        if (last_end - last_start) > max_duration_sec:
            return []
        return [last_word]

    @classmethod
    def _is_cjk_carry_token(
        cls,
        token: str,
        *,
        policy_snapshot: Optional["LanguagePolicySnapshot"] = None,
    ) -> bool:
        text = str(token or "").strip()
        if not text:
            return False
        return text in cls._resolve_cjk_carry_words(policy_snapshot)

    def _normalize_carried_article_sentence_case(
        self,
        *,
        sentence_segments: List[SentenceSegment],
        pending_in_word_count: int,
        policy_snapshot: Optional["LanguagePolicySnapshot"] = None,
    ) -> None:
        """当跨 chunk 回放冠词时，规范句首为英文句式大小写。"""
        if pending_in_word_count <= 0 or not sentence_segments:
            return
        first_sentence = sentence_segments[0]
        words = list(first_sentence.words or [])
        if len(words) < 2:
            return

        carry_token = self._normalize_boundary_token(words[0].word)
        if carry_token not in self._resolve_english_carry_words(policy_snapshot):
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
    def _resolve_sentence_end_chars(
        policy_snapshot: Optional["LanguagePolicySnapshot"],
    ) -> Tuple[str, ...]:
        if policy_snapshot and policy_snapshot.sentence_end_chars:
            ordered_chars = sorted(
                {
                    str(char).strip()
                    for char in policy_snapshot.sentence_end_chars
                    if str(char).strip()
                }
            )
            if ordered_chars:
                return tuple(ordered_chars)
        return tuple(SegmentationProcessor._SENTENCE_END_PUNCT)

    @classmethod
    def _resolve_english_carry_words(
        cls,
        policy_snapshot: Optional["LanguagePolicySnapshot"],
    ) -> Set[str]:
        words = cls._resolve_cross_chunk_word_set(
            policy_snapshot=policy_snapshot,
            key="english_carry_words",
        )
        if words is not None:
            return words
        return set(cls._CROSS_CHUNK_CARRY_WORDS)

    @classmethod
    def _resolve_cjk_carry_words(
        cls,
        policy_snapshot: Optional["LanguagePolicySnapshot"],
    ) -> Set[str]:
        words = cls._resolve_cross_chunk_word_set(
            policy_snapshot=policy_snapshot,
            key="cjk_carry_words",
        )
        if words is not None:
            return words
        return set(cls._CROSS_CHUNK_CJK_CARRY_WORDS).union(
            set(cls._CROSS_CHUNK_CJK_CARRY_SINGLE_CHARS)
        )

    @classmethod
    def _resolve_cjk_carry_limits(
        cls,
        policy_snapshot: Optional["LanguagePolicySnapshot"],
    ) -> Tuple[float, float]:
        cross_chunk_config = cls._resolve_cross_chunk_config(policy_snapshot)
        max_gap_sec = cls._parse_float_with_default(
            cross_chunk_config.get("cjk_max_gap_sec"),
            cls._CROSS_CHUNK_CJK_MAX_GAP_SEC,
        )
        max_duration_sec = cls._parse_float_with_default(
            cross_chunk_config.get("cjk_max_duration_sec"),
            cls._CROSS_CHUNK_CJK_MAX_DURATION_SEC,
        )
        return max_gap_sec, max_duration_sec

    @staticmethod
    def _parse_float_with_default(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _resolve_cross_chunk_config(
        policy_snapshot: Optional["LanguagePolicySnapshot"],
    ) -> Dict[str, Any]:
        if not policy_snapshot:
            return {}
        metadata = getattr(policy_snapshot, "metadata", {})
        if not isinstance(metadata, dict):
            return {}
        cross_chunk = metadata.get("cross_chunk")
        if isinstance(cross_chunk, dict):
            return cross_chunk
        return {}

    @classmethod
    def _resolve_cross_chunk_word_set(
        cls,
        *,
        policy_snapshot: Optional["LanguagePolicySnapshot"],
        key: str,
    ) -> Optional[Set[str]]:
        cross_chunk_config = cls._resolve_cross_chunk_config(policy_snapshot)
        if key not in cross_chunk_config:
            return None
        raw_value = cross_chunk_config.get(key)
        if isinstance(raw_value, str):
            values = [raw_value]
        elif isinstance(raw_value, (list, tuple, set, frozenset)):
            values = list(raw_value)
        else:
            return set()
        normalized: Set[str] = set()
        for item in values:
            token = cls._normalize_boundary_token(str(item or ""))
            if token:
                normalized.add(token)
        return normalized

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
            SegmentationProcessor._copy_word_runtime_metadata(source=word, target=cloned)
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
        if is_sentence_end_punct(
            left_text,
            str(getattr(right_word, "word", "") or "").strip(),
            sentence_end_chars=tuple(self._SPEAKER_REPAIR_SENTENCE_END_PUNCT),
        ):
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

    @classmethod
    def _filter_unknown_pseudo_words(
        cls,
        words_for_split: Sequence[WordTimestamp],
    ) -> Tuple[List[WordTimestamp], int, bool, int]:
        """
        在切分前处理低质量伪词：confidence_source=unknown 且 is_pseudo=True。

        处理策略：
        1. 仅剔除“明显噪声”伪词（纯分隔符/模型残留符号），避免边界污染；
        2. 对非噪声伪词不删词，仅降权保留，避免数字或句首句尾文本被吞；
        3. 若全部被剔除，触发降权回退，避免整句误判为空。
        """
        normalized_words = list(words_for_split or [])
        if not normalized_words:
            return [], 0, False, 0

        filtered_words: List[WordTimestamp] = []
        dropped_count = 0
        degraded_count = 0
        for word in normalized_words:
            if not cls._is_unknown_pseudo_word(word):
                filtered_words.append(word)
                continue
            if cls._is_unknown_pseudo_junk(word):
                dropped_count += 1
                continue
            filtered_words.append(cls._degrade_unknown_pseudo_word(word))
            degraded_count += 1
        if filtered_words:
            return filtered_words, dropped_count, False, degraded_count

        fallback_words = [
            cls._degrade_unknown_pseudo_word(word)
            if cls._is_unknown_pseudo_word(word)
            else word
            for word in normalized_words
        ]
        fallback_degraded_count = sum(1 for word in normalized_words if cls._is_unknown_pseudo_word(word))
        return fallback_words, dropped_count, dropped_count > 0, fallback_degraded_count

    @staticmethod
    def _is_unknown_pseudo_word(word: WordTimestamp) -> bool:
        confidence_source = str(getattr(word, "confidence_source", "") or "").strip().lower()
        return bool(getattr(word, "is_pseudo", False)) and confidence_source == "unknown"

    @classmethod
    def _is_unknown_pseudo_junk(cls, word: WordTimestamp) -> bool:
        token = str(getattr(word, "word", "") or "")
        token_stripped = token.strip()
        if not token_stripped:
            return True
        if "▁" in token_stripped:
            return True
        if "<|" in token_stripped or "|>" in token_stripped:
            return True
        return bool(cls._UNKNOWN_PSEUDO_JUNK_PATTERN.fullmatch(token_stripped))

    @staticmethod
    def _degrade_unknown_pseudo_word(word: WordTimestamp) -> WordTimestamp:
        degraded = WordTimestamp(
            word=word.word,
            start=word.start,
            end=word.end,
            confidence=0.0,
            confidence_raw=word.confidence_raw,
            confidence_display_raw=word.confidence_display_raw,
            confidence_source=word.confidence_source,
            token_type=word.token_type,
            is_pseudo=word.is_pseudo,
        )
        SegmentationProcessor._copy_word_runtime_metadata(source=word, target=degraded)
        return degraded

    @classmethod
    def _normalize_words_for_split_timestamps(
        cls,
        words_for_split: Sequence[WordTimestamp],
    ) -> Tuple[List[WordTimestamp], int]:
        """
        词流时间单调修正：避免时间回退词触发错误边界映射与拆句。
        """
        normalized_words: List[WordTimestamp] = []
        fix_count = 0
        previous_end: Optional[float] = None

        for source_word in list(words_for_split or []):
            start = float(getattr(source_word, "start", 0.0) or 0.0)
            end = float(getattr(source_word, "end", start) or start)

            if end < start:
                end = start
                fix_count += 1

            if previous_end is not None and start < previous_end:
                start = previous_end
                end = max(end, start + cls._TIMELINE_NORMALIZE_MIN_DURATION_SEC)
                fix_count += 1
            elif end <= start:
                end = start + cls._TIMELINE_NORMALIZE_MIN_DURATION_SEC
                fix_count += 1

            normalized = WordTimestamp(
                word=source_word.word,
                start=start,
                end=end,
                confidence=source_word.confidence,
                confidence_raw=source_word.confidence_raw,
                confidence_display_raw=source_word.confidence_display_raw,
                confidence_source=source_word.confidence_source,
                token_type=source_word.token_type,
                is_pseudo=source_word.is_pseudo,
            )
            cls._copy_word_runtime_metadata(source=source_word, target=normalized)
            normalized_words.append(normalized)
            previous_end = float(normalized.end)

        return normalized_words, fix_count

    @staticmethod
    def _build_words_for_split(annotated_words: List[Any]) -> List[WordTimestamp]:
        words: List[WordTimestamp] = []
        for item in annotated_words:
            trailing_punct = str(item.trailing_punct or "")
            word_core = str(item.word or "")
            if trailing_punct and word_core.endswith(trailing_punct):
                word_text = word_core
            else:
                word_text = f"{word_core}{trailing_punct}"
            word = WordTimestamp(
                word=word_text,
                start=float(item.start) if item.start is not None else 0.0,
                end=float(item.end) if item.end is not None else 0.0,
                confidence=item.confidence,
                confidence_source=item.confidence_source,
                is_pseudo=bool(getattr(item, "is_pseudo", False)),
            )
            SegmentationProcessor._copy_word_runtime_metadata(source=item, target=word)
            words.append(word)
        return words

    @staticmethod
    def _copy_word_runtime_metadata(*, source: Any, target: WordTimestamp) -> None:
        for attr in ("speaker_id", "turn_id"):
            setattr(target, attr, getattr(source, attr, None))

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

    @classmethod
    def _strip_boundary_residual_weak_punct(cls, sentences: Sequence[SentenceSegment]) -> None:
        sentence_list = list(sentences or [])
        if len(sentence_list) < 2:
            return
        for index in range(len(sentence_list) - 1):
            cls._strip_sentence_trailing_weak_punct(sentence_list[index])
            cls._strip_sentence_leading_weak_punct(sentence_list[index + 1])

    @classmethod
    def _strip_sentence_trailing_weak_punct(cls, sentence: SentenceSegment) -> None:
        sentence.text = cls._rstrip_boundary_weak_punct(str(sentence.text or ""))
        sentence.text_clean = cls._rstrip_boundary_weak_punct(
            str(sentence.text_clean or sentence.text or "")
        )
        if sentence.words:
            sentence.words[-1].word = cls._rstrip_boundary_weak_punct(
                str(sentence.words[-1].word or "")
            )

    @classmethod
    def _strip_sentence_leading_weak_punct(cls, sentence: SentenceSegment) -> None:
        sentence.text = cls._lstrip_boundary_weak_punct(str(sentence.text or ""))
        sentence.text_clean = cls._lstrip_boundary_weak_punct(
            str(sentence.text_clean or sentence.text or "")
        )
        if sentence.words:
            sentence.words[0].word = cls._lstrip_boundary_weak_punct(
                str(sentence.words[0].word or "")
            )

    @classmethod
    def _rstrip_boundary_weak_punct(cls, text: str) -> str:
        normalized = str(text or "").rstrip()
        while normalized and normalized[-1] in cls._CUT_BOUNDARY_WEAK_PUNCT:
            normalized = normalized[:-1].rstrip()
        return normalized

    @classmethod
    def _lstrip_boundary_weak_punct(cls, text: str) -> str:
        normalized = str(text or "").lstrip()
        while normalized and normalized[:1] in cls._CUT_BOUNDARY_WEAK_PUNCT:
            normalized = normalized[1:].lstrip()
        return normalized

    @staticmethod
    def _finalize_sentence_metadata(sentences: List[SentenceSegment]) -> None:
        for sentence in sentences:
            sentence.source = TextSource.WHISPER_PATCH
            sentence.is_draft = False
            sentence.is_finalized = True


# 兼容新旧命名。
DecisionSegmentationProcessor = SegmentationProcessor

__all__ = ["SegmentationProcessor", "DecisionSegmentationProcessor"]

