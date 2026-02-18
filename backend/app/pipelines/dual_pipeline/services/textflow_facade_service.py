"""
四层文本链路门面服务。

设计模式：Facade Pattern。
原因：把 collection/scoring/decision/output 之间的串联流程从流水线主类中抽离，
让实现层更聚焦于调度与依赖注入，降低单类复杂度。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from app.models.sensevoice_models import SentenceSegment, TextSource, WordTimestamp
from app.schemas.pipeline_context import ProcessingContext
from app.services.alignment.types import (
    AlignedFacts,
    AnnotatedWord,
    AlignmentResult,
    CharMapping,
    FusedEvidence,
    L4Input,
    L5Input,
    L6Input,
    OutputTrace,
    PuncPosition,
    PunctTrack,
    TextTrack,
    TextTrackBundle,
)


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
    alignment_time_source: str = "sv"
    alignment_time_word_count: int = 0
    detected_language: str = "auto"


class _TextflowFacadeHost(Protocol):
    """Textflow 门面依赖的最小主机协议。"""

    job_id: str
    logger: Any
    _vad_intervals: Optional[List[Tuple[float, float]]]
    _l4_processor: Any
    _l5_processor: Any
    _l6_processor: Any
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

    def _build_fused_evidence_for_l6(
        self,
        *,
        words: Sequence[AnnotatedWord],
        stream_id: str,
        aligned_facts: Optional[AlignedFacts],
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

    def _build_soft_cut_plan_for_l6(
        self,
        *,
        annotated_words: Sequence[AnnotatedWord],
        stream_id: str,
        block_id: str,
        is_last_chunk: bool,
        aligned_facts: Optional[AlignedFacts] = None,
        fused_evidence: Optional[FusedEvidence] = None,
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

    def run_l4_to_l6_once(
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
    ) -> Layer456RunResult:
        """执行一次 collection/scoring/decision 主路径。"""
        time_words, time_source = self._resolve_alignment_time_words(
            variant=variant,
            sv_words=sv_words,
            whisper_result=whisper_result,
        )
        l4_output = self._host._l4_processor.process(
            L4Input(
                chosen_text_track=tracks.chosen_track,
                sv_words=time_words,
                vad_intervals=self._host._vad_intervals,
            )
        )
        alignment_result = l4_output.alignment_result

        detected_language = whisper_result.get("language") or (
            tracks.chosen_track.language if tracks.chosen_track else "auto"
        )
        self._host._final_splitter.set_language(detected_language)
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
            source="l3",
            confidence_stats={},
        )
        l5_output = self._host._l5_processor.process(
            L5Input(
                alignment_result=alignment_result,
                punct_track=l5_punct_track,
                language=detected_language,
                speaker_id=speaker_id,
                turn_id=turn_id,
            )
        )
        injection_report = dict(l5_output.injection_report or {})
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
            annotated_words=l5_output.annotated_words,
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
            annotated_words=l5_output.annotated_words,
            alignment_result=alignment_result,
            speaker_turns=speaker_turn_facts,
            fast_draft_cuts=fast_draft_cuts,
            pyannote_frame_times=pyannote_frame_times,
        )
        l6_stream_id = f"{variant}:{speaker_id or 'main'}:{turn_id or 'none'}"
        fused_evidence = self._host._build_fused_evidence_for_l6(
            words=l5_output.annotated_words,
            stream_id=l6_stream_id,
            aligned_facts=aligned_facts,
        )

        l6_chunk_index = self._host._resolve_chunk_index_from_words(words=time_words)
        l6_is_last_chunk = self._host._is_last_chunk_for_words(words=time_words)
        soft_cut_plan = self._host._build_soft_cut_plan_for_l6(
            annotated_words=l5_output.annotated_words,
            stream_id=l6_stream_id,
            block_id=(
                f"{self._host.job_id}:{l6_stream_id}:"
                f"{l6_chunk_index if l6_chunk_index is not None else -1}"
            ),
            is_last_chunk=l6_is_last_chunk,
            aligned_facts=aligned_facts,
            fused_evidence=fused_evidence,
        )
        l6_output = self._host._l6_processor.process(
            L6Input(
                annotated_words=l5_output.annotated_words,
                vad_intervals=self._host._vad_intervals,
                cut_plan=soft_cut_plan,
                aligned_facts=aligned_facts,
                fused_evidence=fused_evidence,
                fallback_clean_text_ref=str(punctuation_clean_text or ""),
                fallback_punctuation_positions=list(punctuation_positions or []),
            ),
            stream_id=l6_stream_id,
            chunk_index=l6_chunk_index,
            is_last_chunk=l6_is_last_chunk,
        )
        words_for_split = list(l6_output.words_for_split)
        final_sentences = list(l6_output.sentence_segments)
        output_traces = list(l6_output.output_traces or [])
        split_stats = dict(self._host._final_splitter.last_split_stats or {})
        split_stats.update(dict(l6_output.segmentation_report.get("boundary_score_stats", {})))
        split_stats.update(dict(l6_output.segmentation_report.get("soft_cut_stats", {})))
        split_stats["output_trace_count"] = int(len(output_traces))
        segmentation_error = str(l6_output.segmentation_report.get("error_code", "") or "")
        if segmentation_error:
            split_stats["error_code"] = segmentation_error

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
        run_result = self.run_l4_to_l6_once(
            tracks=tracks,
            sv_result=ctx.sv_result,
            whisper_result=ctx.whisper_result or {},
            sv_words=sv_words,
            punctuation_positions=punctuation_positions,
            punctuation_clean_text=punctuation_clean_text,
            variant="legacy",
            speaker_id=speaker_id,
            turn_id=turn_id,
        )

        final_sentences = list(run_result.final_sentences)
        split_stats = dict(run_result.split_stats)
        if not final_sentences:
            fallback = self._host._build_final_fallback_sentence(run_result.words_for_split)
            if fallback is not None:
                fallback.speaker_id = speaker_id
                fallback.turn_id = turn_id
                final_sentences = [fallback]
                split_stats["error_code"] = "E_L6_SPLIT_EMPTY"
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
        """根据轨道选择 L4 时间锚点词流。"""
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
    def _collect_fast_draft_cuts(
        sv_result: Dict[str, Any],
        *,
        sv_words: Sequence[WordTimestamp],
    ) -> List[float]:
        """从快流结果提取句级边界时间。"""
        raw_sentences = sv_result.get("sentences")
        if raw_sentences is None:
            raw_sentences = sv_result.get("sentence_segments")
        cuts: List[float] = []
        if isinstance(raw_sentences, Sequence):
            for item in raw_sentences:
                if isinstance(item, dict):
                    end_value = item.get("end")
                else:
                    end_value = getattr(item, "end", None)
                if end_value is None:
                    continue
                try:
                    cuts.append(float(end_value))
                except (TypeError, ValueError):
                    continue
        if cuts:
            return sorted(set(cuts))
        if sv_words:
            return [float(sv_words[-1].end)]
        return []

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
        if not isinstance(segments, Sequence):
            return words

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
                    start = float(item.get("start", 0.0))
                    end = float(item.get("end", start))
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
