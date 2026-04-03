from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from app.core.logging import resolve_loguru_logger
from app.schemas.pipeline_context import ProcessingContext
from app.services.alignment.types import L2Output, QualitySignals, TextTrack, TextTrackBundle
from app.services.timeanchored_alignment.contracts import SelectedTextTruth, SelectionDecision
from app.services.timeanchored_alignment.selection.contracts import (
    FastObservationSummary,
    SelectionInputs,
    SelectionOutcome,
    SlowQualitySignals,
    SlowTextCandidateSet,
    WindowSelectionScope,
)
from app.services.timeanchored_alignment.selection.reporting import (
    build_rejection_reasons,
    build_selection_report,
    parse_reason_codes,
)


class TextSelectionService:
    """Phase1 选择层正式服务。"""

    def __init__(self, *, logger: Optional[Any] = None) -> None:
        self._logger = resolve_loguru_logger(logger, __name__, layer="selection")

    def build_selection_inputs(
        self,
        *,
        ctx: ProcessingContext,
        tracks: TextTrackBundle,
        quality_signals: QualitySignals,
        sv_result: Dict[str, Any],
        whisper_result: Dict[str, Any],
    ) -> SelectionInputs:
        selection_scope = self._build_scope(ctx)
        return SelectionInputs(
            fast_summary=FastObservationSummary(
                text_track=tracks.sv_track,
                confidence=float(quality_signals.confidence_fast),
                language_hint=str(
                    getattr(tracks.sv_track, "language", "")
                    or sv_result.get("language")
                    or "auto"
                ),
                metadata={
                    "mapping_coverage": float(quality_signals.mapping_coverage),
                    "raw_text": str(sv_result.get("raw_text") or ""),
                },
            ),
            slow_candidates=SlowTextCandidateSet(
                primary_track=tracks.whisper_track,
                confidence=float(quality_signals.confidence_slow),
                language_hint=str(
                    getattr(tracks.whisper_track, "language", "")
                    or whisper_result.get("language")
                    or "auto"
                ),
                metadata={
                    "raw_text": str(
                        whisper_result.get("text_raw")
                        or whisper_result.get("raw_text")
                        or ""
                    ),
                },
            ),
            selection_scope=selection_scope,
            slow_quality_signals=SlowQualitySignals(
                quality_signals=quality_signals,
                metadata={
                    "job_id": str(ctx.job_id),
                    "chunk_index": int(ctx.chunk_index),
                },
            ),
        )

    def select(
        self,
        *,
        ctx: ProcessingContext,
        selection_inputs: SelectionInputs,
        arbitration_processor: Any,
        clone_text_track: Callable[[TextTrack, str], TextTrack],
    ) -> SelectionOutcome:
        quality_signals = selection_inputs.slow_quality_signals.to_l2_quality_signals()
        arbitration_output = self._run_arbitration(
            selection_inputs=selection_inputs,
            arbitration_processor=arbitration_processor,
        )
        arbitration_result = arbitration_output.arbitration_result
        tracks = TextTrackBundle(
            sv_track=selection_inputs.fast_summary.text_track,
            whisper_track=selection_inputs.slow_candidates.primary_track,
        )
        chosen_track = self._resolve_chosen_track(
            ctx=ctx,
            tracks=tracks,
            arbitration_output=arbitration_output,
            clone_text_track=clone_text_track,
        )
        selected_text_truth = self._build_selected_text_truth(
            ctx=ctx,
            chosen_track=chosen_track,
            quality_signals=quality_signals,
            arbitration_reason=str(getattr(arbitration_result, "reason", "") or ""),
            chosen_source=str(getattr(arbitration_result, "chosen_source", "") or "fast"),
        )
        selection_decision = self._build_selection_decision(arbitration_result)
        selection_report = build_selection_report(
            scope=selection_inputs.selection_scope,
            quality_signals=quality_signals,
            arbitration_result=arbitration_result,
            selected_text_truth=selected_text_truth,
            selection_decision=selection_decision,
        )
        return SelectionOutcome(
            selected_text_truth=selected_text_truth,
            selection_decision=selection_decision,
            selection_report=selection_report,
            selection_inputs=selection_inputs,
            arbitration_output=arbitration_output,
            arbitration_result=arbitration_result,
            chosen_track=chosen_track,
            quality_signals=quality_signals,
        )

    def apply_runtime_selection(
        self,
        *,
        tracks: TextTrackBundle,
        whisper_result: Dict[str, Any],
        sv_result: Dict[str, Any],
        selection_outcome: SelectionOutcome,
    ) -> str:
        chosen_track = selection_outcome.chosen_track
        if chosen_track is not None:
            tracks.chosen_track = chosen_track
        chosen_text_clean = selection_outcome.chosen_text_clean
        chosen_source = selection_outcome.selection_decision.chosen_source
        if chosen_source == "fast":
            whisper_result["text"] = chosen_text_clean
            whisper_result["text_clean"] = chosen_text_clean
            whisper_result["text_itn_raw"] = sv_result.get("text_itn_raw") or chosen_text_clean
        else:
            whisper_result["text"] = chosen_text_clean
            whisper_result["text_clean"] = chosen_text_clean
        return chosen_text_clean

    @staticmethod
    def _run_arbitration(
        *,
        selection_inputs: SelectionInputs,
        arbitration_processor: Any,
    ) -> L2Output:
        processor = getattr(arbitration_processor, "process", arbitration_processor)
        if not callable(processor):
            raise TypeError("选择层缺少可调用的 arbitration_processor")
        return processor(selection_inputs.to_l2_input())

    def _build_scope(self, ctx: ProcessingContext) -> WindowSelectionScope:
        ready_window = getattr(ctx, "ready_slow_window", None)
        source_chunk_ids = tuple(getattr(ready_window, "source_chunk_ids", ()) or ())
        source_chunk_indices = tuple(getattr(ready_window, "source_chunk_indices", ()) or ())
        if not source_chunk_ids and getattr(ctx, "audio_chunk", None) is not None:
            source_chunk_ids = (str(getattr(ctx.audio_chunk, "chunk_id", "") or f"chunk-{ctx.chunk_index}"),)
        if not source_chunk_indices:
            source_chunk_indices = (int(ctx.chunk_index),)
        return WindowSelectionScope(
            job_id=str(ctx.job_id),
            chunk_index=int(ctx.chunk_index),
            window_id=str(getattr(ready_window, "window_id", "") or ""),
            source_chunk_ids=source_chunk_ids,
            source_chunk_indices=source_chunk_indices,
            edge_selection_mode=str(getattr(ctx, "edge_selection_mode", "auto") or "auto"),
        )

    def _resolve_chosen_track(
        self,
        *,
        ctx: ProcessingContext,
        tracks: TextTrackBundle,
        arbitration_output: L2Output,
        clone_text_track: Callable[[TextTrack, str], TextTrack],
    ) -> Optional[TextTrack]:
        arbitration_result = getattr(arbitration_output, "arbitration_result", None)
        if arbitration_output.chosen_text_track is not None:
            return arbitration_output.chosen_text_track
        if (
            arbitration_result is not None
            and str(getattr(arbitration_result, "error_code", "") or "")
            == "E_L2_ARBITRATION_FORCED_SOURCE_MISSING"
        ):
            forced_source = str(getattr(arbitration_result, "forced_source", "") or "")
            raise ValueError(
                f"Chunk {ctx.chunk_index}: 强制选边源缺失，终止本 chunk 定稿。forced_source={forced_source}"
            )
        fallback_track = tracks.sv_track or tracks.whisper_track
        if fallback_track is None:
            return None
        return clone_text_track(fallback_track, source="chosen")

    def _build_selected_text_truth(
        self,
        *,
        ctx: ProcessingContext,
        chosen_track: Optional[TextTrack],
        quality_signals: QualitySignals,
        arbitration_reason: str,
        chosen_source: str,
    ) -> SelectedTextTruth:
        if chosen_track is None:
            raise ValueError(f"Chunk {ctx.chunk_index}: 选择层缺少可用 chosen_track")
        selected_text = str(
            chosen_track.text_clean
            or chosen_track.text_itn_raw
            or chosen_track.raw_text
            or ""
        ).strip()
        if not selected_text:
            raise ValueError(f"Chunk {ctx.chunk_index}: 选择层缺少可用文本真相")
        scope = self._build_scope(ctx)
        reason_codes = parse_reason_codes(arbitration_reason)
        return SelectedTextTruth(
            text=selected_text,
            text_source=chosen_source,
            language_hint=str(getattr(chosen_track, "language", "") or "auto"),
            source_chunk_ids=scope.source_chunk_ids,
            quality={
                "confidence_fast": float(quality_signals.confidence_fast),
                "confidence_slow": float(quality_signals.confidence_slow),
                "length_ratio": float(quality_signals.length_ratio),
                "mapping_coverage": float(quality_signals.mapping_coverage),
                "is_hallucination": 1.0 if quality_signals.is_hallucination else 0.0,
                "is_repetition": 1.0 if quality_signals.is_repetition else 0.0,
            },
            rejection_reasons=build_rejection_reasons(reason_codes),
            metadata={
                "raw_text": str(getattr(chosen_track, "raw_text", "") or selected_text),
                "text_itn_raw": str(getattr(chosen_track, "text_itn_raw", "") or selected_text),
                "edge_selection_mode": scope.edge_selection_mode,
                "source_chunk_indices": list(scope.source_chunk_indices),
            },
        )

    def _build_selection_decision(self, arbitration_result: Any) -> SelectionDecision:
        chosen_source = str(getattr(arbitration_result, "chosen_source", "") or "fast")
        forced_source = str(getattr(arbitration_result, "forced_source", "") or "")
        reason_codes = parse_reason_codes(str(getattr(arbitration_result, "reason", "") or ""))
        if forced_source == "fast":
            decision = "force_fast"
        elif forced_source == "slow":
            decision = "force_slow"
        elif chosen_source == "mixed":
            decision = "mixed"
        elif chosen_source == "slow":
            decision = "accept_slow"
        else:
            decision = "accept_fast"
        return SelectionDecision(
            decision=decision,
            accepted_text_source=chosen_source,
            reason_codes=reason_codes,
            metadata={
                "coverage": float(getattr(arbitration_result, "coverage", 0.0) or 0.0),
                "sv_score": float(getattr(arbitration_result, "sv_score", 0.0) or 0.0),
                "wh_score": float(getattr(arbitration_result, "wh_score", 0.0) or 0.0),
                "error_code": str(getattr(arbitration_result, "error_code", "") or ""),
            },
        )
