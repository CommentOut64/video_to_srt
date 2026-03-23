"""
诊断追踪服务。

职责：
- 统一封装 Whisper/层级诊断输出
- 隔离 `AsyncDualPipelineKernel` 中的大体量调试序列化逻辑
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.services.alignment.types import (
    AlignedFacts,
    AlignmentResult,
    FusedEvidence,
    L2Output,
    OutputTrace,
    PunctTrack,
    TextTrack,
    TextTrackBundle,
)
from app.models.confidence_models import AlignmentStatus
from app.models.sensevoice_models import SentenceSegment
from app.services.punctuation.base import PuncPosition


class DiagnosticTraceService:
    """
    诊断追踪领域服务。
    """

    def __init__(self, *, logger: Any) -> None:
        self.logger = logger

    def emit_whisper_debug(
        self,
        job_dir: Optional[Path],
        *,
        group_id: str,
        flush_reason: str,
        whisper_result: Dict[str, Any],
        chunk_indices: List[int],
        append_debug_whisper_line: Any,
    ) -> None:
        """
        记录 Whisper 批次调试信息。
        """
        raw_result = whisper_result.get("raw_result", {}) if isinstance(whisper_result, dict) else {}
        raw_segments = raw_result.get("segments", []) if isinstance(raw_result, dict) else []
        seg_count = len(raw_segments)
        avg_logprob = None
        avg_no_speech = None
        if seg_count > 0:
            avg_logprob = sum(
                float(seg.get("avg_logprob", 0.0) or 0.0) for seg in raw_segments
            ) / seg_count
            avg_no_speech = sum(
                float(seg.get("no_speech_prob", 0.0) or 0.0) for seg in raw_segments
            ) / seg_count
        raw_text = str(whisper_result.get("text_raw", "") or "")
        sanitized_text = str(whisper_result.get("text", "") or "")
        seg_text = "".join(str(seg.get("text", "")) for seg in raw_segments).strip()
        payload = {
            "group_id": group_id,
            "flush_reason": str(flush_reason or ""),
            "chunk_indices": chunk_indices,
            "text_len": len(sanitized_text),
            "raw_text_len": len(raw_text),
            "seg_text_len": len(seg_text),
            "segments_count": seg_count,
            "avg_logprob": avg_logprob,
            "avg_no_speech_prob": avg_no_speech,
            "language": whisper_result.get("language", "auto"),
        }
        append_debug_whisper_line(job_dir, payload, logger=self.logger)

    def emit_layer_diagnostics(
        self,
        ctx: Any,
        *,
        tracks: TextTrackBundle,
        punct_track: Optional[PunctTrack],
        alignment_result: AlignmentResult,
        injection_stats: Dict[str, Any],
        split_stats: Dict[str, Any],
        final_sentences: List[SentenceSegment],
        append_debug_layer_diag_line: Any,
    ) -> None:
        """输出分层诊断到独立文件（四层口径）。"""
        chosen_clean = tracks.chosen_track.text_clean if tracks.chosen_track else ""
        punct_ref = punct_track.clean_text_ref if punct_track else ""
        chosen_compact = str(chosen_clean or "").replace("\n", " ").strip()
        punct_compact = str(punct_ref or "").replace("\n", " ").strip()

        unmatched_prefix_len = 0
        for aligned_word in alignment_result.aligned_words:
            status = aligned_word.alignment_status
            is_prefix_unmatched = bool(aligned_word.is_pseudo) or status in {
                AlignmentStatus.INSERTED,
                AlignmentStatus.PSEUDO,
            }
            if not is_prefix_unmatched:
                break
            unmatched_prefix_len += len(str(aligned_word.word or ""))

        is_ref_mismatch = bool(chosen_compact and punct_compact and chosen_compact != punct_compact)
        is_slow_chosen = bool(
            ctx.arbitration_result and ctx.arbitration_result.chosen_source == "slow"
        )
        punct_positions_count = len(punct_track.positions) if punct_track and punct_track.positions else 0
        is_cross_chunk_boundary_suspected = bool(
            is_slow_chosen
            and punct_positions_count == 0
            and (
                unmatched_prefix_len > 0
                or float(alignment_result.gap_ratio) >= 0.25
                or float(alignment_result.coverage) <= 0.70
            )
        )
        split_reason_stats: Dict[str, int] = {}
        split_risk_stats: Dict[str, int] = {}
        for sentence in final_sentences:
            reason = str(getattr(sentence, "split_reason", "") or "")
            risk = str(getattr(sentence, "split_risk", "") or "")
            if reason:
                split_reason_stats[reason] = int(split_reason_stats.get(reason, 0) + 1)
            if risk:
                split_risk_stats[risk] = int(split_risk_stats.get(risk, 0) + 1)

        payload: Dict[str, Any] = {
            "job_id": ctx.job_id,
            "chunk_index": int(ctx.chunk_index),
            "layer": "选文到裁决",
            "chosen_source": ctx.arbitration_result.chosen_source if ctx.arbitration_result else "",
            "arbitration_reason": ctx.arbitration_result.reason if ctx.arbitration_result else "",
            "chosen_clean_len": len(chosen_clean or ""),
            "l2_chosen_text_head": self._clip_head(chosen_compact),
            "l2_chosen_text_tail": self._clip_tail(chosen_compact),
            "punctuation_pre_positions_total": len(punct_track.positions) if punct_track and punct_track.positions else 0,
            "punctuation_pre_source": punct_track.source if punct_track else "",
            "punctuation_pre_clean_text_match": bool(chosen_clean and punct_ref and chosen_clean == punct_ref),
            "punctuation_pre_clean_ref_head": self._clip_head(punct_compact),
            "punctuation_pre_clean_ref_tail": self._clip_tail(punct_compact),
            "is_punctuation_pre_match_blocked_by_ref_mismatch": is_ref_mismatch,
            "collection_alignment_score": float(alignment_result.alignment_score),
            "collection_gap_ratio": float(alignment_result.gap_ratio),
            "collection_coverage": float(alignment_result.coverage),
            "collection_gap_positions": list(alignment_result.gap_positions),
            "collection_unmatched_prefix_len": int(unmatched_prefix_len),
            "is_cross_chunk_boundary_suspected": is_cross_chunk_boundary_suspected,
            "scoring_injection_positions_total": int(injection_stats.get("injection_positions_total", 0) or 0),
            "scoring_injection_unmatched_total": int(injection_stats.get("injection_unmatched_total", 0) or 0),
            "scoring_injection_mapping_coverage": float(injection_stats.get("injection_mapping_coverage", 0.0) or 0.0),
            "scoring_injection_blocked": bool(injection_stats.get("injection_blocked", 0.0)),
            "decision_sentence_count": len(final_sentences),
            "decision_split_mapping_coverage": float(split_stats.get("mapping_coverage", 0.0) or 0.0),
            "decision_split_writeback_ratio": float(split_stats.get("writeback_ratio", 0.0) or 0.0),
            "decision_split_writeback_used": bool(split_stats.get("writeback_used", 0.0)),
            "decision_split_writeback_blocked": bool(split_stats.get("writeback_blocked", 0.0)),
            "decision_split_reason_stats": split_reason_stats,
            "decision_split_risk_stats": split_risk_stats,
            "decision_sentence_texts": [str(sentence.text or "") for sentence in final_sentences],
            "punctuation_pre": {
                "positions_total": len(punct_track.positions) if punct_track and punct_track.positions else 0,
                "source": punct_track.source if punct_track else "",
                "clean_text_match": bool(chosen_clean and punct_ref and chosen_clean == punct_ref),
                "clean_ref_head": self._clip_head(punct_compact),
                "clean_ref_tail": self._clip_tail(punct_compact),
                "is_match_blocked_by_ref_mismatch": is_ref_mismatch,
            },
            "collection": {
                "alignment_score": float(alignment_result.alignment_score),
                "gap_ratio": float(alignment_result.gap_ratio),
                "coverage": float(alignment_result.coverage),
                "gap_positions": list(alignment_result.gap_positions),
                "unmatched_prefix_len": int(unmatched_prefix_len),
                "is_cross_chunk_boundary_suspected": is_cross_chunk_boundary_suspected,
            },
            "scoring": {
                "injection_positions_total": int(injection_stats.get("injection_positions_total", 0) or 0),
                "injection_unmatched_total": int(injection_stats.get("injection_unmatched_total", 0) or 0),
                "injection_mapping_coverage": float(
                    injection_stats.get("injection_mapping_coverage", 0.0) or 0.0
                ),
                "injection_blocked": bool(injection_stats.get("injection_blocked", 0.0)),
            },
            "decision": {
                "sentence_count": len(final_sentences),
                "split_mapping_coverage": float(split_stats.get("mapping_coverage", 0.0) or 0.0),
                "split_writeback_ratio": float(split_stats.get("writeback_ratio", 0.0) or 0.0),
                "split_writeback_used": bool(split_stats.get("writeback_used", 0.0)),
                "split_writeback_blocked": bool(split_stats.get("writeback_blocked", 0.0)),
                "split_reason_stats": split_reason_stats,
                "split_risk_stats": split_risk_stats,
                "sentence_texts": [str(sentence.text or "") for sentence in final_sentences],
            },
        }
        append_debug_layer_diag_line(ctx.job_dir, payload, logger=self.logger)

    def emit_layer_trace_full(
        self,
        ctx: Any,
        *,
        tracks: TextTrackBundle,
        sv_result: Dict[str, Any],
        whisper_result: Dict[str, Any],
        arbitration_output: L2Output,
        punct_track: Optional[PunctTrack],
        alignment_result: AlignmentResult,
        aligned_facts: AlignedFacts,
        fused_evidence: FusedEvidence,
        words_for_split: Sequence[Any],
        injection_stats: Dict[str, Any],
        split_stats: Dict[str, Any],
        final_sentences: Sequence[SentenceSegment],
        output_traces: Sequence[OutputTrace],
        append_debug_layer_trace_line: Any,
    ) -> None:
        """输出全量分层追踪（逐层 + 逐 token）到独立文件。"""
        punctuation_pre_input = getattr(ctx, "_trace_punctuation_pre_input", {}) or {}
        punctuation_pre_output = getattr(ctx, "_trace_punctuation_pre_output", None)
        slow_raw = self._serialize_word_timestamps(
            [
                word
                for seg in (whisper_result.get("raw_result", {}) or {}).get("segments", []) or []
                for word in (seg.get("words", []) or [])
                if isinstance(word, dict)
            ]
        )
        l0_batch_trace = getattr(ctx, "_trace_l0_batch_whisper", {}) or {}
        hetero_alignment_report = getattr(ctx, "hetero_alignment_report", None)
        timeanchored_alignment = hetero_alignment_report if isinstance(hetero_alignment_report, dict) else {}
        timeanchored_alignment_report = timeanchored_alignment.get("alignment_report", {})
        if not isinstance(timeanchored_alignment_report, dict):
            timeanchored_alignment_report = {}
        timeanchored_alignment_metrics = timeanchored_alignment_report.get("metrics", {})
        if not isinstance(timeanchored_alignment_metrics, dict):
            timeanchored_alignment_metrics = {}
        payload: Dict[str, Any] = {
            "job_id": ctx.job_id,
            "chunk_index": int(ctx.chunk_index),
            "layer": "全链路追踪",
            "arbitration": {
                "chosen_source": arbitration_output.arbitration_result.chosen_source,
                "reason": arbitration_output.arbitration_result.reason,
                "coverage": float(arbitration_output.arbitration_result.coverage),
                "sv_score": float(arbitration_output.arbitration_result.sv_score),
                "wh_score": float(arbitration_output.arbitration_result.wh_score),
                "gap_positions": list(arbitration_output.arbitration_result.gap_positions or []),
            },
            "l0": {
                "bridge_batch": {
                    "batch_id": str(l0_batch_trace.get("batch_id", "")),
                    "chunk_indices": list(l0_batch_trace.get("chunk_indices", []) or []),
                    "prompt": str(l0_batch_trace.get("prompt", "")),
                    "flush_reason": str(l0_batch_trace.get("flush_reason", "")),
                    "whisper_text_raw": str(l0_batch_trace.get("whisper_text_raw", "")),
                    "whisper_text_clean": str(l0_batch_trace.get("whisper_text_clean", "")),
                    "raw_result": dict(l0_batch_trace.get("raw_result", {}) or {}),
                },
                "sv_result": {
                    "raw_text": str(sv_result.get("raw_text") or ""),
                    "text_clean": str(sv_result.get("text_clean") or ""),
                    "words": self._serialize_word_timestamps(sv_result.get("words") or []),
                },
                "whisper_result": {
                    "text_raw": str(whisper_result.get("text_raw") or whisper_result.get("raw_text") or ""),
                    "min_clean_text": str(whisper_result.get("min_clean_text") or ""),
                    "text_clean": str(whisper_result.get("text_clean") or whisper_result.get("text") or ""),
                    "prompt": str(whisper_result.get("prompt") or ""),
                    "segments": list((whisper_result.get("raw_result", {}) or {}).get("segments", []) or []),
                    "words_from_segments": slow_raw,
                },
            },
            "l1": {
                "sv_track": self._serialize_text_track(tracks.sv_track),
                "whisper_track": self._serialize_text_track(tracks.whisper_track),
                "chosen_track": self._serialize_text_track(tracks.chosen_track),
            },
            "punctuation_pre": {
                "input": {
                    "chosen_source": punctuation_pre_input.get("chosen_source"),
                    "word_timestamps": self._serialize_word_timestamps(punctuation_pre_input.get("word_timestamps") or []),
                    "sv_punct_source": {
                        "source": str(getattr(punctuation_pre_input.get("sv_punct_source"), "source", "") or ""),
                        "clean_text_ref": str(getattr(punctuation_pre_input.get("sv_punct_source"), "clean_text_ref", "") or ""),
                        "positions": self._serialize_punc_positions(getattr(punctuation_pre_input.get("sv_punct_source"), "positions", []) or []),
                    },
                    "wh_punct_source": {
                        "source": str(getattr(punctuation_pre_input.get("wh_punct_source"), "source", "") or ""),
                        "clean_text_ref": str(getattr(punctuation_pre_input.get("wh_punct_source"), "clean_text_ref", "") or ""),
                        "positions": self._serialize_punc_positions(getattr(punctuation_pre_input.get("wh_punct_source"), "positions", []) or []),
                    },
                },
                "output": {
                    "source": str(getattr(punctuation_pre_output, "source", "") or ""),
                    "clean_text_ref": str(getattr(punctuation_pre_output, "clean_text_ref", "") or ""),
                    "positions": self._serialize_punc_positions(getattr(punctuation_pre_output, "positions", []) or []),
                    "confidence_stats": dict(getattr(punctuation_pre_output, "confidence_stats", {}) or {}),
                },
            },
            "collection": {
                "alignment_score": float(alignment_result.alignment_score),
                "gap_ratio": float(alignment_result.gap_ratio),
                "coverage": float(alignment_result.coverage),
                "gap_positions": list(alignment_result.gap_positions),
                "aligned_words": self._serialize_aligned_words(alignment_result.aligned_words),
            },
            "scoring": {
                "injection_stats": dict(injection_stats),
                "words_for_split": self._serialize_word_timestamps(words_for_split),
                "aligned_facts": self._serialize_aligned_facts(aligned_facts),
                "fused_evidence": self._serialize_fused_evidence(fused_evidence),
            },
            "decision": {
                "split_stats": dict(split_stats),
                "final_sentences": self._serialize_sentences(final_sentences),
                "output_trace": self._serialize_output_traces(output_traces),
            },
            "timeanchored_alignment": {
                "report": timeanchored_alignment,
                "raw_mount_trace": dict(timeanchored_alignment_metrics.get("raw_mount_trace", {}) or {}),
            },
            "current_punct_track": {
                "source": str(punct_track.source if punct_track else ""),
                "clean_text_ref": str(punct_track.clean_text_ref if punct_track else ""),
                "positions": self._serialize_punc_positions(punct_track.positions if punct_track else []),
            },
        }
        append_debug_layer_trace_line(ctx.job_dir, payload, logger=self.logger)

    @staticmethod
    def _clip_head(text: str, limit: int = 40) -> str:
        if len(text) <= limit:
            return text
        return text[:limit]

    @staticmethod
    def _clip_tail(text: str, limit: int = 40) -> str:
        if len(text) <= limit:
            return text
        return text[-limit:]

    @staticmethod
    def _serialize_punc_positions(positions: Optional[Sequence[PuncPosition]]) -> List[Dict[str, Any]]:
        if not positions:
            return []
        return [
            {
                "char_index": int(pos.char_index),
                "punctuation": str(pos.punctuation),
                "confidence": float(pos.confidence),
            }
            for pos in positions
        ]

    @staticmethod
    def _serialize_word_timestamps(words: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
        if not words:
            return []
        items: List[Dict[str, Any]] = []
        for word in words:
            if isinstance(word, dict):
                items.append(
                    {
                        "word": str(word.get("word", "") or ""),
                        "start": float(word.get("start", 0.0) or 0.0),
                        "end": float(word.get("end", 0.0) or 0.0),
                        "confidence": (
                            float(word.get("confidence"))
                            if word.get("confidence") is not None
                            else (
                                float(word.get("probability"))
                                if word.get("probability") is not None
                                else None
                            )
                        ),
                    }
                )
                continue
            items.append(
                {
                    "word": str(getattr(word, "word", "") or ""),
                    "start": float(getattr(word, "start", 0.0) or 0.0),
                    "end": float(getattr(word, "end", 0.0) or 0.0),
                    "confidence": (
                        float(getattr(word, "confidence"))
                        if getattr(word, "confidence", None) is not None
                        else None
                    ),
                    "confidence_source": str(getattr(word, "confidence_source", "") or ""),
                    "is_pseudo": bool(getattr(word, "is_pseudo", False)),
                }
            )
        return items

    @staticmethod
    def _serialize_text_track(track: Optional[TextTrack]) -> Dict[str, Any]:
        if not track:
            return {}
        return {
            "source": str(track.source or ""),
            "language": str(track.language or ""),
            "raw_text": str(track.raw_text or ""),
            "text_itn_raw": str(track.text_itn_raw or ""),
            "text_clean": str(track.text_clean or ""),
            "mapping_coverage": float(track.mapping_coverage),
            "itn_fallback": bool(track.itn_fallback),
            "itn_fallback_reason": str(track.itn_fallback_reason or ""),
            "punct_positions": DiagnosticTraceService._serialize_punc_positions(track.punct_positions),
            "clean_to_word": list(track.clean_to_word or []),
            "word_confidences": list(track.word_confidences or []),
        }

    @staticmethod
    def _serialize_aligned_words(words: Sequence[Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for word in words:
            rows.append(
                {
                    "word": str(getattr(word, "word", "") or ""),
                    "start": float(getattr(word, "start", 0.0) or 0.0),
                    "end": float(getattr(word, "end", 0.0) or 0.0),
                    "alignment_status": str(getattr(getattr(word, "alignment_status", None), "value", "")),
                    "is_pseudo": bool(getattr(word, "is_pseudo", False)),
                    "sv_confidence": (
                        float(getattr(word, "sv_confidence"))
                        if getattr(word, "sv_confidence", None) is not None
                        else None
                    ),
                    "whisper_confidence": (
                        float(getattr(word, "whisper_confidence"))
                        if getattr(word, "whisper_confidence", None) is not None
                        else None
                    ),
                    "final_confidence": (
                        float(getattr(word, "final_confidence"))
                        if getattr(word, "final_confidence", None) is not None
                        else None
                    ),
                    "confidence_source": str(getattr(word, "confidence_source", "") or ""),
                }
            )
        return rows

    @staticmethod
    def _serialize_sentences(sentences: Sequence[SentenceSegment]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for sentence in sentences:
            rows.append(
                {
                    "text": str(sentence.text or ""),
                    "text_clean": str(sentence.text_clean or ""),
                    "start": float(sentence.start),
                    "end": float(sentence.end),
                    "word_count": len(sentence.words or []),
                    "words": DiagnosticTraceService._serialize_word_timestamps(sentence.words),
                    "alignment_score": (
                        float(sentence.alignment_score)
                        if sentence.alignment_score is not None
                        else None
                    ),
                    "matched_ratio": (
                        float(sentence.matched_ratio)
                        if sentence.matched_ratio is not None
                        else None
                    ),
                    "confidence_source": str(sentence.confidence_source or ""),
                    "split_reason": str(getattr(sentence, "split_reason", "") or ""),
                    "split_risk": str(getattr(sentence, "split_risk", "") or ""),
                    "window_id": str(getattr(sentence, "window_id", "") or ""),
                    "pyannote_frame_time": getattr(sentence, "pyannote_frame_time", None),
                    "mapped_cut_time": getattr(sentence, "mapped_cut_time", None),
                    "mapping_quality": str(getattr(sentence, "mapping_quality", "") or ""),
                    "mapping_reason": str(getattr(sentence, "mapping_reason", "") or ""),
                }
            )
        return rows

    @staticmethod
    def _serialize_output_traces(traces: Sequence[OutputTrace]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for trace in traces:
            rows.append(
                {
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
            )
        return rows

    @staticmethod
    def _serialize_aligned_facts(facts: AlignedFacts) -> Dict[str, Any]:
        mapping_quality_stats: Dict[str, int] = {}
        for item in list(facts.time_mappings or []):
            quality = str(item.get("mapping_quality", "") or "unknown")
            mapping_quality_stats[quality] = mapping_quality_stats.get(quality, 0) + 1
        return {
            "alignment_score": float(facts.alignment_score),
            "gap_ratio": float(facts.gap_ratio),
            "gap_positions": list(facts.gap_positions or []),
            "annotated_word_count": int(len(facts.annotated_words or [])),
            "speaker_turn_count": int(len(facts.speaker_turns or [])),
            "fast_draft_cut_count": int(len(facts.fast_draft_cuts or [])),
            "time_axis_version": str(facts.time_axis_version or ""),
            "time_mappings": list(facts.time_mappings or []),
            "mapping_quality_stats": mapping_quality_stats,
        }

    @staticmethod
    def _serialize_fused_evidence(evidence: FusedEvidence) -> Dict[str, Any]:
        return {
            "speaker_change_count": int(len(evidence.speaker_changes or [])),
            "pause_anchor_count": int(len(evidence.pause_anchors or [])),
            "semantic_anchor_count": int(len(evidence.semantic_anchors or [])),
            "punctuation_anchor_count": int(len(evidence.punctuation_anchors or [])),
            "evidence_report": dict(evidence.evidence_report or {}),
            "speaker_changes": list(evidence.speaker_changes or []),
            "pause_anchors": list(evidence.pause_anchors or []),
            "semantic_anchors": list(evidence.semantic_anchors or []),
            "punctuation_anchors": list(evidence.punctuation_anchors or []),
        }


