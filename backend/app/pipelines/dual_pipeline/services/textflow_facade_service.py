"""
四层文本链路门面服务。

设计模式：Facade Pattern。
原因：把 collection/scoring/decision/output 之间的串联流程从流水线主类中抽离，
让实现层更聚焦于调度与依赖注入，降低单类复杂度。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Sequence, Tuple

from app.models.sensevoice_models import SentenceSegment, TextSource, WordTimestamp
from app.schemas.pipeline_context import ProcessingContext
from app.services.language_policy import build_language_policy_snapshot
from app.services.alignment.types import (
    AlignedFacts,
    AnnotatedWord,
    AlignmentResult,
    CollectionLayerInput,
    DecisionLayerInput,
    CharMapping,
    FusedEvidence,
    OutputTrace,
    ScoringLayerInput,
    PuncPosition,
    PunctTrack,
    TextTrack,
    TextTrackBundle,
)
from app.services.textflow.contracts import SubtitleBatch
from app.services.text_protection import is_sentence_end_punct

if TYPE_CHECKING:
    from app.services.language_policy.types import LanguagePolicySnapshot


@dataclass
class Layer456RunResult:
    alignment_result: AlignmentResult
    aligned_facts: AlignedFacts
    fused_evidence: FusedEvidence
    words_for_split: List[WordTimestamp]
    injection_stats: Dict[str, Any]
    split_stats: Dict[str, Any]
    final_sentences: List[SentenceSegment]
    output_traces: List[OutputTrace]
    subtitle_batch: Optional[SubtitleBatch] = None
    alignment_time_source: str = "sv"
    alignment_time_word_count: int = 0
    detected_language: str = "auto"


class _TextflowFacadeHost(Protocol):
    """Textflow 门面依赖的最小主机协议。"""

    job_id: str
    logger: Any
    _vad_intervals: Optional[List[Tuple[float, float]]]
    _collection_processor: Any
    _scoring_processor: Any
    _decision_processor: Any
    _fact_builder: Any
    _final_splitter: Any
    _final_grouper: Any
    _timeline_turns: List[Any]

    def _bind_word_identity_by_timeline_overlap(
        self,
        *,
        annotated_words: Sequence[AnnotatedWord],
        fallback_speaker_id: Optional[str],
        fallback_turn_id: Optional[str],
    ) -> None:
        ...

    def _build_fused_evidence_for_decision(
        self,
        *,
        words: Sequence[AnnotatedWord],
        stream_id: str,
        aligned_facts: Optional[AlignedFacts],
        policy_snapshot: Optional["LanguagePolicySnapshot"] = None,
        is_fast_only_mode: bool = False,
    ) -> FusedEvidence:
        ...

    def _resolve_chunk_index_from_words(
        self,
        *,
        words: Sequence[WordTimestamp],
    ) -> Optional[int]:
        ...

    def _is_last_chunk_for_words(
        self,
        *,
        words: Sequence[WordTimestamp],
    ) -> bool:
        ...

    def _build_soft_cut_plan_for_decision(
        self,
        *,
        annotated_words: Sequence[AnnotatedWord],
        stream_id: str,
        block_id: str,
        is_last_chunk: bool,
        aligned_facts: Optional[AlignedFacts] = None,
        fused_evidence: Optional[FusedEvidence] = None,
        policy_snapshot: Optional["LanguagePolicySnapshot"] = None,
        is_fast_only_mode: bool = False,
    ) -> Optional[Any]:
        ...

    def _compute_matched_ratio(self, aligned_words: Sequence[Any]) -> float:
        ...

    def _resolve_sentence_confidence_source(
        self,
        words: Sequence[WordTimestamp],
    ) -> str:
        ...

    def _build_sv_word_timestamps(
        self,
        sv_result: Dict[str, Any],
        chunk: Any,
    ) -> List[WordTimestamp]:
        ...

    def _resolve_speaker_id_for_chunk(self, chunk: Any) -> Optional[str]:
        ...

    def _resolve_turn_id_for_chunk(self, chunk: Any) -> Optional[str]:
        ...

    def _is_last_chunk_index(self, chunk_index: int) -> bool:
        ...

    def _build_final_fallback_sentence(
        self,
        words_for_split: Sequence[WordTimestamp],
    ) -> Optional[SentenceSegment]:
        ...


class TextflowFacadeService:
    """四层文本链路门面。"""

    def __init__(self, *, host: _TextflowFacadeHost) -> None:
        self._host = host

    def run_collection_scoring_decision_once(
        self,
        *,
        tracks: TextTrackBundle,
        sv_result: Dict[str, Any],
        whisper_result: Dict[str, Any],
        sv_words: List[WordTimestamp],
        punctuation_positions: Optional[List[PuncPosition]],
        punctuation_clean_text: Optional[str],
        variant: str = "legacy",
        speaker_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        policy_snapshot: Optional["LanguagePolicySnapshot"] = None,
        is_fast_only_mode: bool = False,
    ) -> Layer456RunResult:
        """执行一次集合层→评分层→裁决层主路径（内部主入口）。"""
        time_words, time_source = self._resolve_alignment_time_words(
            variant=variant,
            sv_words=sv_words,
            whisper_result=whisper_result,
        )
        collection_output = self._host._collection_processor.process(
            CollectionLayerInput(
                chosen_text_track=tracks.chosen_track,
                sv_words=time_words,
                vad_intervals=self._host._vad_intervals,
                policy_snapshot=policy_snapshot,
                is_fast_only_mode=is_fast_only_mode,
            )
        )
        alignment_result = collection_output.alignment_result

        detected_language = whisper_result.get("language") or (
            tracks.chosen_track.language if tracks.chosen_track else "auto"
        )
        self._host._final_splitter.set_language(detected_language)
        is_cjk_language = self._is_cjk_language_tag(detected_language)
        # Why: 纯快流是当前稳定基线，双流 CJK 仅保留语义断点，不启用弱标点扩展切分。
        is_enable_cjk_weak_punct = (not is_fast_only_mode) and (not is_cjk_language)
        is_enable_cjk_semantic = (not is_fast_only_mode)
        self._host._final_splitter.set_cjk_split_mode(
            is_enable_weak_punct=is_enable_cjk_weak_punct,
            is_enable_semantic=is_enable_cjk_semantic,
        )
        if is_fast_only_mode:
            # Why: fast-only 以“稳定快流基线”为优先，避免 CJK 语义词表触发额外碎切。
            self._host._final_splitter.set_semantic_anchor_words([])
        else:
            self._host._final_splitter.set_semantic_anchor_words(
                list(getattr(policy_snapshot, "semantic_anchor_words", []) or [])
            )
        injection_stats: Dict[str, Any] = {
            "injection_positions_total": len(punctuation_positions or []),
            "injection_unmatched_total": 0,
            "injection_miss_ratio": 0.0,
            "injection_mapping_coverage": 0.0,
            "injection_blocked": 0.0,
            "injection_error_code": "",
        }
        l5_punct_track = PunctTrack(
            clean_text_ref=punctuation_clean_text or "",
            positions=list(punctuation_positions or []),
            source="punctuation_pre",
            confidence_stats={},
        )
        scoring_output = self._host._scoring_processor.process(
            ScoringLayerInput(
                alignment_result=alignment_result,
                punct_track=l5_punct_track,
                language=detected_language,
                speaker_id=speaker_id,
                turn_id=turn_id,
                policy_snapshot=policy_snapshot,
            )
        )
        injection_report = dict(scoring_output.injection_report or {})
        injection_stats["injection_unmatched_total"] = int(
            injection_report.get("mismatch_count", 0.0)
        )
        injection_stats["injection_miss_ratio"] = (
            injection_stats["injection_unmatched_total"]
            / max(len(punctuation_positions or []), 1)
        )
        injection_stats["injection_mapping_coverage"] = float(
            injection_report.get("mapping_coverage", 0.0)
        )
        injection_stats["injection_blocked"] = float(injection_report.get("blocked", 0.0))
        injection_stats["injection_error_code"] = str(
            injection_report.get("error_code", "") or ""
        )

        self._host._bind_word_identity_by_timeline_overlap(
            annotated_words=scoring_output.annotated_words,
            fallback_speaker_id=speaker_id,
            fallback_turn_id=turn_id,
        )
        speaker_turn_facts = self._collect_speaker_turn_facts_for_words(time_words)
        fast_draft_cuts = self._collect_fast_draft_cuts(
            sv_result,
            sv_words=time_words,
        )
        pyannote_frame_times = self._collect_pyannote_frame_times_for_words(time_words)
        aligned_facts = self._host._fact_builder.build(
            annotated_words=scoring_output.annotated_words,
            alignment_result=alignment_result,
            speaker_turns=speaker_turn_facts,
            fast_draft_cuts=fast_draft_cuts,
            pyannote_frame_times=pyannote_frame_times,
        )
        decision_stream_id = f"{variant}:{speaker_id or 'main'}:{turn_id or 'none'}"
        fused_evidence = self._host._build_fused_evidence_for_decision(
            words=scoring_output.annotated_words,
            stream_id=decision_stream_id,
            aligned_facts=aligned_facts,
            policy_snapshot=policy_snapshot,
            is_fast_only_mode=is_fast_only_mode,
        )

        decision_chunk_index = self._host._resolve_chunk_index_from_words(words=time_words)
        decision_is_last_chunk = self._host._is_last_chunk_for_words(words=time_words)
        soft_cut_plan = self._host._build_soft_cut_plan_for_decision(
            annotated_words=scoring_output.annotated_words,
            stream_id=decision_stream_id,
            block_id=(
                f"{self._host.job_id}:{decision_stream_id}:"
                f"{decision_chunk_index if decision_chunk_index is not None else -1}"
            ),
            is_last_chunk=decision_is_last_chunk,
            aligned_facts=aligned_facts,
            fused_evidence=fused_evidence,
            policy_snapshot=policy_snapshot,
            is_fast_only_mode=is_fast_only_mode,
        )
        decision_output = self._host._decision_processor.process(
            DecisionLayerInput(
                annotated_words=scoring_output.annotated_words,
                vad_intervals=self._host._vad_intervals,
                cut_plan=soft_cut_plan,
                aligned_facts=aligned_facts,
                fused_evidence=fused_evidence,
                fallback_clean_text_ref=str(punctuation_clean_text or ""),
                fallback_punctuation_positions=list(punctuation_positions or []),
                policy_snapshot=policy_snapshot,
            ),
            stream_id=decision_stream_id,
            chunk_index=decision_chunk_index,
            is_last_chunk=decision_is_last_chunk,
        )
        words_for_split = list(decision_output.words_for_split)
        final_sentences = list(decision_output.sentence_segments)
        output_traces = list(decision_output.output_traces or [])
        split_stats = dict(self._host._final_splitter.last_split_stats or {})
        split_stats.update(dict(decision_output.segmentation_report.get("boundary_score_stats", {})))
        split_stats.update(dict(decision_output.segmentation_report.get("soft_cut_stats", {})))
        fused_evidence_stats = dict(decision_output.segmentation_report.get("fused_evidence_stats", {}))
        split_stats["unknown_pseudo_drop_count"] = int(
            decision_output.segmentation_report.get("unknown_pseudo_drop_count", 0) or 0
        )
        split_stats["unknown_pseudo_degrade_count"] = int(
            decision_output.segmentation_report.get("unknown_pseudo_degrade_count", 0) or 0
        )
        split_stats["unknown_pseudo_filter_fallback"] = bool(
            decision_output.segmentation_report.get("unknown_pseudo_filter_fallback", False)
        )
        split_stats["output_trace_count"] = int(len(output_traces))
        split_stats["fused_punctuation_anchor_count"] = int(
            fused_evidence_stats.get("punctuation_anchor_count", 0) or 0
        )
        split_stats["fused_semantic_anchor_count"] = int(
            fused_evidence_stats.get("semantic_anchor_count", 0) or 0
        )
        source_stats = split_stats.get("source_stats", {})
        source_keys = [
            str(key).lower()
            for key in (source_stats.keys() if isinstance(source_stats, dict) else [])
        ]
        split_stats["soft_cut_punctuation_anchor_hit"] = (
            1 if any("punct" in key for key in source_keys) else 0
        )
        split_stats["soft_cut_no_fused_window"] = (
            1 if int(split_stats.get("fusion_output_window_count", 0) or 0) <= 0 else 0
        )
        default_splitter_trace_count = sum(
            1
            for item in output_traces
            if str(getattr(item, "split_reason", "") or "") == "default_splitter"
        )
        split_stats["soft_cut_used_default_splitter"] = (
            1
            if output_traces and default_splitter_trace_count == len(output_traces)
            else 0
        )
        segmentation_error = str(decision_output.segmentation_report.get("error_code", "") or "")
        if segmentation_error:
            split_stats["error_code"] = segmentation_error
        soft_cut_diagnostic_code = str(split_stats.get("diagnostic_code", "") or "")
        if soft_cut_diagnostic_code:
            split_stats["soft_cut_diagnostic_code"] = soft_cut_diagnostic_code

        if self._host._final_grouper:
            final_sentences = self._host._final_grouper.group(final_sentences)

        matched_ratio = self._host._compute_matched_ratio(alignment_result.aligned_words)
        for sentence in final_sentences:
            sentence.source = TextSource.WHISPER_PATCH
            sentence.is_finalized = True
            sentence.is_draft = False
            sentence.alignment_score = alignment_result.alignment_score
            sentence.matched_ratio = matched_ratio
            sentence.whisper_text = whisper_result.get("text", "")
            sentence.sv_original_text = sv_result.get("text_clean")
            sentence.confidence_source = self._host._resolve_sentence_confidence_source(
                sentence.words
            )
            if sentence.speaker_id is None:
                sentence.speaker_id = speaker_id
            if sentence.turn_id is None:
                sentence.turn_id = turn_id

        return Layer456RunResult(
            alignment_result=alignment_result,
            aligned_facts=aligned_facts,
            fused_evidence=fused_evidence,
            words_for_split=words_for_split,
            injection_stats=injection_stats,
            split_stats=dict(split_stats),
            final_sentences=final_sentences,
            output_traces=output_traces,
            subtitle_batch=decision_output.subtitle_batch,
            alignment_time_source=time_source,
            alignment_time_word_count=len(time_words),
            detected_language=str(detected_language or "auto"),
        )

    def finalize_sensevoice_only(self, ctx: ProcessingContext) -> Layer456RunResult:
        """Whisper 跳过时的定稿输出（仅使用 SenseVoice 结果，统一走四层主链）。"""
        if not ctx.sv_result or not ctx.audio_chunk:
            raise ValueError("Whisper 跳过路径缺少 SenseVoice 结果或音频块")

        tracks = ctx.text_tracks or TextTrackBundle()
        ctx.text_tracks = tracks

        normalize_sensevoice_result = getattr(self._host, "_normalize_sensevoice_result", None)
        if tracks.sv_track is None and callable(normalize_sensevoice_result):
            normalize_sensevoice_result(ctx)
            tracks = ctx.text_tracks or tracks

        if tracks.sv_track is None:
            fallback_text = str(
                ctx.sv_result.get("text_clean")
                or ctx.sv_result.get("text_itn_raw")
                or ctx.sv_result.get("text")
                or ctx.sv_result.get("raw_text")
                or ""
            ).strip()
            if not fallback_text:
                raise ValueError("Whisper 跳过路径缺少 SenseVoice 文本轨道")
            text_length = len(fallback_text)
            mapping = list(range(text_length))
            tracks.sv_track = TextTrack(
                raw_text=fallback_text,
                text_itn_raw=fallback_text,
                text_clean=fallback_text,
                char_mapping=[
                    CharMapping(raw_idx=index, clean_idx=index, punct=None)
                    for index in range(text_length)
                ],
                raw_to_clean=list(mapping),
                clean_to_raw=list(mapping),
                language=str(ctx.audio_chunk.language or ctx.sv_result.get("language") or "auto"),
                source="sensevoice",
                mapping_coverage=1.0 if text_length > 0 else 0.0,
            )

        if tracks.chosen_track is None:
            clone_text_track = getattr(self._host, "_clone_text_track", None)
            if callable(clone_text_track):
                tracks.chosen_track = clone_text_track(tracks.sv_track, source="chosen")
            else:
                tracks.chosen_track = tracks.sv_track

        sv_words = self._host._build_sv_word_timestamps(ctx.sv_result, ctx.audio_chunk)
        speaker_id = self._host._resolve_speaker_id_for_chunk(ctx.audio_chunk)
        turn_id = self._host._resolve_turn_id_for_chunk(ctx.audio_chunk)
        punct_track = ctx.punct_track
        punctuation_positions = list(punct_track.positions) if punct_track and punct_track.positions else None
        punctuation_clean_text = punct_track.clean_text_ref if punct_track else None
        policy_snapshot: Optional["LanguagePolicySnapshot"] = None
        language_hint = str(
            (tracks.chosen_track.language if tracks.chosen_track else "")
            or getattr(ctx.audio_chunk, "language", "")
            or ctx.sv_result.get("language")
            or "auto"
        )
        try:
            policy_snapshot = build_language_policy_snapshot(language_hint=language_hint)
        except Exception:
            self._host.logger.exception(
                "Whisper 跳过路径语言策略快照编译失败: language_hint={}",
                language_hint,
            )
        run_result = self.run_collection_scoring_decision_once(
            tracks=tracks,
            sv_result=ctx.sv_result,
            whisper_result=ctx.whisper_result or {},
            sv_words=sv_words,
            punctuation_positions=punctuation_positions,
            punctuation_clean_text=punctuation_clean_text,
            variant="legacy",
            speaker_id=speaker_id,
            turn_id=turn_id,
            policy_snapshot=policy_snapshot,
            is_fast_only_mode=True,
        )

        final_sentences = list(run_result.final_sentences)
        split_stats = dict(run_result.split_stats)
        if not final_sentences:
            fallback = self._host._build_final_fallback_sentence(run_result.words_for_split)
            if fallback is not None:
                fallback.speaker_id = speaker_id
                fallback.turn_id = turn_id
                final_sentences = [fallback]
                split_stats["error_code"] = "E_DECISION_SPLIT_EMPTY"
                self._host.logger.warning("Whisper 跳过路径触发裁决层单句兜底")

        for sentence in final_sentences:
            sentence.source = TextSource.SENSEVOICE
            sentence.is_finalized = True
            sentence.is_draft = False
            if not sentence.confidence_source:
                sentence.confidence_source = "fast"
            if sentence.speaker_id is None:
                sentence.speaker_id = speaker_id
            if sentence.turn_id is None:
                sentence.turn_id = turn_id

        return Layer456RunResult(
            alignment_result=run_result.alignment_result,
            aligned_facts=run_result.aligned_facts,
            fused_evidence=run_result.fused_evidence,
            words_for_split=run_result.words_for_split,
            injection_stats=dict(run_result.injection_stats),
            split_stats=split_stats,
            final_sentences=final_sentences,
            output_traces=list(run_result.output_traces),
            subtitle_batch=run_result.subtitle_batch,
            alignment_time_source=run_result.alignment_time_source,
            alignment_time_word_count=run_result.alignment_time_word_count,
            detected_language=run_result.detected_language,
        )

    def _resolve_alignment_time_words(
        self,
        *,
        variant: str,
        sv_words: List[WordTimestamp],
        whisper_result: Dict[str, Any],
    ) -> Tuple[List[WordTimestamp], str]:
        """根据轨道选择集合层时间锚点词流。"""
        if variant != "experiment":
            return list(sv_words), "sv"

        slow_words = self._build_whisper_word_timestamps(whisper_result)
        if not slow_words:
            return list(sv_words), "sv_fallback_no_slow_words"

        resolved_source = "whisper_slow_words"
        if sv_words:
            sv_start = float(sv_words[0].start)
            sv_end = float(sv_words[-1].end)
            guard_start = sv_start - 0.5
            guard_end = sv_end + 0.5
            bounded = [
                word
                for word in slow_words
                if guard_start <= float(word.start) <= guard_end
                and guard_start <= float(word.end) <= guard_end
            ]
            if bounded:
                slow_words = bounded
            else:
                # 兜底：若慢流词仍未落入快流时间窗，按首词差值推断并平移时间基。
                inferred_offset = sv_start - float(slow_words[0].start)
                is_large_offset = abs(inferred_offset) >= 1.0
                if is_large_offset:
                    shifted = self._shift_word_timestamps(slow_words, inferred_offset)
                    bounded_shifted = [
                        word
                        for word in shifted
                        if guard_start <= float(word.start) <= guard_end
                        and guard_start <= float(word.end) <= guard_end
                    ]
                    if bounded_shifted:
                        slow_words = bounded_shifted
                        resolved_source = "whisper_slow_words_shifted"

        min_required = max(3, int(len(sv_words) * 0.4)) if sv_words else 3
        if len(slow_words) < min_required:
            return list(sv_words), "sv_fallback_insufficient_slow_words"

        return slow_words, resolved_source

    def _collect_speaker_turn_facts_for_words(
        self,
        words: Sequence[WordTimestamp],
    ) -> List[Dict[str, Any]]:
        """收集与当前词时间窗重叠的 turn 事实。"""
        if not words:
            return []
        window_start = float(words[0].start)
        window_end = float(words[-1].end)
        if window_end <= window_start:
            return []
        rows: List[Dict[str, Any]] = []
        for turn in list(self._host._timeline_turns or []):
            turn_start = float(getattr(turn, "start", 0.0) or 0.0)
            turn_end = float(getattr(turn, "end", 0.0) or 0.0)
            if turn_end <= turn_start:
                continue
            overlap_start = max(window_start, turn_start)
            overlap_end = min(window_end, turn_end)
            if overlap_end <= overlap_start:
                continue
            rows.append(
                {
                    "turn_id": str(getattr(turn, "turn_id", "") or ""),
                    "speaker_id": str(getattr(turn, "speaker_id", "unknown") or "unknown"),
                    "start": turn_start,
                    "end": turn_end,
                    "source": str(getattr(turn, "source", "") or ""),
                    "boundary_confidence": float(
                        getattr(turn, "boundary_confidence", 0.0) or 0.0
                    ),
                }
            )
        return rows

    def _collect_pyannote_frame_times_for_words(
        self,
        words: Sequence[WordTimestamp],
    ) -> List[float]:
        """收集当前词窗内的 pyannote 边界时间（使用 turn end 作为帧候选）。"""
        if not words:
            return []
        window_start = float(words[0].start)
        window_end = float(words[-1].end)
        if window_end <= window_start:
            return []
        candidates: List[float] = []
        for turn in list(self._host._timeline_turns or []):
            turn_end = float(getattr(turn, "end", 0.0) or 0.0)
            if window_start <= turn_end <= window_end:
                candidates.append(turn_end)
        return sorted(set(candidates))

    @staticmethod
    def _is_cjk_language_tag(language: Optional[str]) -> bool:
        tag = str(language or "").strip().lower()
        if not tag:
            return False
        return tag.startswith(("zh", "yue", "ja", "jp", "ko"))

    @staticmethod
    def _collect_fast_draft_cuts(
        sv_result: Dict[str, Any],
        *,
        sv_words: Sequence[WordTimestamp],
    ) -> List[float]:
        """
        从快流结果提取句级边界时间。

        Why:
        - 双流 CJK 任务中，`sv_result` 常缺少 `sentences/sentence_segments`，旧逻辑会退化成“仅尾部一个 cut”；
        - 单尾部 cut 会让 fast_draft 证据高度尾部化，最终在 Decision 守门后大量失效。
        """
        raw_sentences = sv_result.get("sentences")
        if raw_sentences is None:
            raw_sentences = sv_result.get("sentence_segments")
        cuts: List[float] = []
        if isinstance(raw_sentences, Sequence) and not isinstance(raw_sentences, (str, bytes)):
            for item in raw_sentences:
                if isinstance(item, dict):
                    end_value = item.get("end")
                else:
                    end_value = getattr(item, "end", None)
                normalized_cut_time = TextflowFacadeService._normalize_fast_draft_cut_time(end_value)
                if normalized_cut_time is not None:
                    cuts.append(normalized_cut_time)
        cuts.extend(
            TextflowFacadeService._collect_fast_draft_word_boundary_cuts(
                sv_words=sv_words,
            )
        )
        if cuts:
            return TextflowFacadeService._dedupe_sorted_fast_draft_cut_times(cuts)
        if sv_words:
            return [float(sv_words[-1].end)]
        return []

    @staticmethod
    def _normalize_fast_draft_cut_time(raw_value: Any) -> Optional[float]:
        if raw_value is None:
            return None
        try:
            cut_time = float(raw_value)
        except (TypeError, ValueError):
            return None
        if cut_time < 0.0:
            return None
        return cut_time

    @staticmethod
    def _dedupe_sorted_fast_draft_cut_times(cuts: Sequence[float]) -> List[float]:
        unique_by_ms: Dict[int, float] = {}
        for item in cuts:
            try:
                value = float(item)
            except (TypeError, ValueError):
                continue
            unique_by_ms[int(round(value * 1000.0))] = value
        return [unique_by_ms[key] for key in sorted(unique_by_ms.keys())]

    @staticmethod
    def _collect_fast_draft_word_boundary_cuts(
        *,
        sv_words: Sequence[WordTimestamp],
    ) -> List[float]:
        """
        从快流词边界提取补充 cut。

        Why:
        - 当快流缺失句级结构时，仍需要若干“句中候选边界”支撑 soft-cut；
        - 仅依赖尾部 cut 会导致 fast_draft 触发位置长期靠近 chunk 末端。
        """
        if len(sv_words) <= 3:
            return []
        if TextflowFacadeService._is_dense_cjk_char_stream(sv_words):
            return []
        boundary_rows: List[Tuple[int, float, float, float]] = []
        for index in range(1, len(sv_words)):
            left = sv_words[index - 1]
            right = sv_words[index]
            left_end = float(getattr(left, "end", 0.0) or 0.0)
            right_start = float(getattr(right, "start", left_end) or left_end)
            gap_sec = max(0.0, right_start - left_end)
            boundary_time = (left_end + right_start) / 2.0

            left_token = str(getattr(left, "word", "") or "").strip()
            right_token = str(getattr(right, "word", "") or "").strip()
            punct_score = 0.0
            if TextflowFacadeService._has_fast_draft_sentence_end_punct(
                left_token=left_token,
                right_token=right_token,
            ):
                punct_score = 1.35
            elif TextflowFacadeService._has_fast_draft_weak_punct(left_token):
                punct_score = 0.70
            boundary_rows.append((index, boundary_time, gap_sec, punct_score))

        if not boundary_rows:
            return []
        positive_gaps = sorted(item[2] for item in boundary_rows if item[2] > 1e-6)
        if positive_gaps:
            pivot_index = min(
                len(positive_gaps) - 1,
                max(0, int(round((len(positive_gaps) - 1) * 0.75))),
            )
            pause_trigger_sec = max(0.18, min(0.52, float(positive_gaps[pivot_index])))
        else:
            pause_trigger_sec = 0.32

        max_candidate_count = 1
        if len(sv_words) >= 6:
            max_candidate_count = 2
        max_candidate_count = min(4, max(max_candidate_count, len(sv_words) // 14))
        scored_rows: List[Tuple[float, float, int, float]] = []
        for index, boundary_time, gap_sec, punct_score in boundary_rows:
            # Why: 两端边界通常对应 chunk 前后沿，噪声较大，不作为句中候选。
            if index <= 1 or index >= len(sv_words) - 1:
                continue
            is_pause_hit = gap_sec >= pause_trigger_sec
            if not is_pause_hit and punct_score <= 0.0:
                continue
            score = punct_score + min(1.2, gap_sec * 1.8)
            scored_rows.append((score, gap_sec, index, boundary_time))
        if not scored_rows:
            return []

        scored_rows.sort(
            key=lambda item: (
                float(item[0]),
                float(item[1]),
            ),
            reverse=True,
        )
        selected_times: List[float] = []
        selected_indices: List[int] = []
        for _score, _gap_sec, index, boundary_time in scored_rows:
            # Why: 防止在相邻字边界上重复入选，造成过密切点。
            if any(abs(index - picked_index) < 2 for picked_index in selected_indices):
                continue
            selected_indices.append(index)
            selected_times.append(boundary_time)
            if len(selected_times) >= max_candidate_count:
                break
        return selected_times

    @staticmethod
    def _is_dense_cjk_char_stream(
        sv_words: Sequence[WordTimestamp],
    ) -> bool:
        normalized_tokens: List[str] = []
        for item in sv_words:
            token = TextflowFacadeService._normalize_fast_draft_token(
                str(getattr(item, "word", "") or ""),
            )
            if token:
                normalized_tokens.append(token)
        if len(normalized_tokens) < 8:
            return False
        single_cjk_count = sum(
            1
            for token in normalized_tokens
            if len(token) == 1 and TextflowFacadeService._is_all_cjk_text(token)
        )
        if single_cjk_count <= 0:
            return False
        return (single_cjk_count / float(len(normalized_tokens))) >= 0.72

    @staticmethod
    def _normalize_fast_draft_token(token: str) -> str:
        text = str(token or "").strip()
        if not text:
            return ""
        normalized_chars = [
            char
            for char in text
            if char.isalnum() or TextflowFacadeService._is_cjk_char(char)
        ]
        return "".join(normalized_chars)

    @staticmethod
    def _is_all_cjk_text(text: str) -> bool:
        normalized = str(text or "")
        if not normalized:
            return False
        return all(TextflowFacadeService._is_cjk_char(char) for char in normalized)

    @staticmethod
    def _is_cjk_char(char: str) -> bool:
        code_point = ord(char)
        if 0x4E00 <= code_point <= 0x9FFF:
            return True
        if 0x3400 <= code_point <= 0x4DBF:
            return True
        if 0x3040 <= code_point <= 0x30FF:
            return True
        if 0xAC00 <= code_point <= 0xD7AF:
            return True
        return False

    @staticmethod
    def _has_fast_draft_sentence_end_punct(
        *,
        left_token: str,
        right_token: str,
    ) -> bool:
        return is_sentence_end_punct(
            left_token,
            right_token,
            sentence_end_chars=("。", "！", "？", ".", "!", "?"),
        )

    @staticmethod
    def _has_fast_draft_weak_punct(token: str) -> bool:
        weak_chars = ("，", "、", ",", ";", "；", ":", "：")
        return any(char in token for char in weak_chars)

    @staticmethod
    def _shift_word_timestamps(
        words: Sequence[WordTimestamp],
        offset: float,
    ) -> List[WordTimestamp]:
        """平移词时间戳，用于统一时间基。"""
        has_effective_offset = abs(offset) >= 1e-6
        if not has_effective_offset:
            return list(words)
        shifted: List[WordTimestamp] = []
        for item in words:
            shifted.append(
                WordTimestamp(
                    word=item.word,
                    start=item.start + offset,
                    end=item.end + offset,
                    confidence=item.confidence,
                    confidence_source=item.confidence_source,
                )
            )
        return shifted

    @staticmethod
    def _build_whisper_word_timestamps(
        whisper_result: Dict[str, Any],
    ) -> List[WordTimestamp]:
        """从 Whisper 结果解析词级时间戳。"""
        words: List[WordTimestamp] = []
        segments = whisper_result.get("segments", [])
        if not segments and isinstance(whisper_result.get("raw_result"), dict):
            segments = whisper_result.get("raw_result", {}).get("segments", [])
        if not isinstance(segments, Sequence):
            return words
        time_base = str(whisper_result.get("word_time_base", "") or "").strip().lower()
        time_offset = 0.0
        if time_base == "batch_local":
            try:
                time_offset = float(whisper_result.get("word_time_offset", 0.0) or 0.0)
            except (TypeError, ValueError):
                time_offset = 0.0

        for segment in segments:
            if not isinstance(segment, dict):
                continue
            segment_words = segment.get("words")
            if not isinstance(segment_words, Sequence):
                continue
            for item in segment_words:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("word", "") or "").strip()
                if not text:
                    continue
                try:
                    start = float(item.get("start", 0.0)) + time_offset
                    end = float(item.get("end", item.get("start", 0.0))) + time_offset
                except (TypeError, ValueError):
                    continue
                confidence = float(item.get("probability", item.get("confidence", 0.0)) or 0.0)
                words.append(
                    WordTimestamp(
                        word=text,
                        start=start,
                        end=max(start, end),
                        confidence=confidence,
                        confidence_source="slow",
                    )
                )

        words.sort(key=lambda item: (item.start, item.end))
        return words




