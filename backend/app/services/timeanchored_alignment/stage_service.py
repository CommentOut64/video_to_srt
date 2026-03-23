"""TimeanchoredAlignmentStageService：timeanchored 主链编排入口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.models.sensevoice_models import SentenceSegment
from app.services.alignment.types import OutputLayerInput
from app.services.timeanchored_alignment.chunk_projector import (
    ChunkProjection,
    ChunkProjector,
    ChunkWindow,
)
from app.services.timeanchored_alignment.contracts import (
    AlignmentItem,
    FinalAlignmentResult,
    LanguageRunPackage,
    LayerReport,
    PipelineReport,
    PronunciationPackage,
    TextTruthPackage,
    TimeBasePackage,
)
from app.services.timeanchored_alignment.edge_selector import EdgeSelector, FailedSpan
from app.services.timeanchored_alignment.output_adapter import OutputAdapter
from app.services.timeanchored_alignment.sentence_segmenter import SentenceSegmenter
from app.services.timeanchored_alignment.subtitle_assembler import SubtitleAssembler
from app.services.timeanchored_alignment.text_aligner import TextAligner


@dataclass(frozen=True)
class TimeanchoredStageResult:
    """timeanchored 主链单窗口输出。"""

    text_result: FinalAlignmentResult
    edge_result: FinalAlignmentResult
    base_result: FinalAlignmentResult
    failed_spans: tuple[FailedSpan, ...]
    final_stream: tuple[AlignmentItem, ...]
    sentence_segments: tuple[SentenceSegment, ...]
    projections: tuple[ChunkProjection, ...]
    output_inputs: tuple[OutputLayerInput, ...]
    pipeline_report: PipelineReport


class TimeanchoredAlignmentStageService:
    """timeanchored 选边、切分、投影、输出适配编排入口。"""

    def __init__(
        self,
        *,
        text_aligner: TextAligner | None = None,
        edge_selector: EdgeSelector | None = None,
        subtitle_assembler: SubtitleAssembler | None = None,
        sentence_segmenter: SentenceSegmenter | None = None,
        chunk_projector: ChunkProjector | None = None,
        output_adapter: OutputAdapter | None = None,
    ) -> None:
        self._text_aligner = text_aligner or TextAligner()
        self._edge_selector = edge_selector or EdgeSelector()
        self._subtitle_assembler = subtitle_assembler or SubtitleAssembler()
        self._sentence_segmenter = sentence_segmenter or SentenceSegmenter()
        self._chunk_projector = chunk_projector or ChunkProjector()
        self._output_adapter = output_adapter or OutputAdapter()

    def execute(
        self,
        *,
        time_base: TimeBasePackage,
        text_truth: TextTruthPackage,
        language_runs: LanguageRunPackage,
        pronunciation: PronunciationPackage,
        chunk_window: ChunkWindow,
        language: str = "auto",
        edge_selection_mode: str = "auto",
        speaker_id: str | None = None,
        turn_id: str | None = None,
    ) -> TimeanchoredStageResult:
        text_result = self._text_aligner.align_window(
            time_base=time_base,
            text_truth=text_truth,
            language_runs=language_runs,
            pronunciation=pronunciation,
        )
        failed_spans = self._extract_failed_spans(text_result.items)
        edge_result = self._edge_selector.select(
            time_base=time_base,
            text_truth=text_truth,
            failed_spans=failed_spans,
            edge_selection_mode=edge_selection_mode,
        )
        base_result = self._choose_base_result(text_result=text_result, edge_result=edge_result)
        fallback_result = edge_result if base_result is text_result else None
        final_stream = self._subtitle_assembler.assemble(
            base_result=base_result,
            fallback_result=fallback_result,
            failed_spans=failed_spans,
        )
        if not final_stream and edge_result.items:
            final_stream = tuple(edge_result.items)
        final_stream = self._apply_chunk_global_offset(
            stream=final_stream,
            chunk_window=chunk_window,
        )

        sentence_segments = tuple(
            self._sentence_segmenter.segment(
                stream=final_stream,
                language=language,
                protected_spans=text_truth.protected_spans,
            )
        )
        if speaker_id or turn_id:
            for sentence in sentence_segments:
                if speaker_id and getattr(sentence, "speaker_id", None) is None:
                    sentence.speaker_id = speaker_id
                if turn_id and getattr(sentence, "turn_id", None) is None:
                    sentence.turn_id = turn_id

        projections = self._chunk_projector.project(
            sentence_segments=sentence_segments,
            chunk_windows=(chunk_window,),
        )
        output_inputs = tuple(
            self._output_adapter.to_output_layer_inputs(
                projections=projections,
                language=language,
                injection_report={
                    "mapping_coverage": float(base_result.metrics.coverage),
                    "error_code": str(base_result.error_code or ""),
                },
                segmentation_report={
                    "route": "timeanchored",
                    "text_route": text_result.route,
                    "edge_route": edge_result.route,
                    "error_code": str(base_result.error_code or ""),
                },
            )
        )

        return TimeanchoredStageResult(
            text_result=text_result,
            edge_result=edge_result,
            base_result=base_result,
            failed_spans=failed_spans,
            final_stream=tuple(final_stream),
            sentence_segments=sentence_segments,
            projections=projections,
            output_inputs=output_inputs,
            pipeline_report=self._build_pipeline_report(
                time_base=time_base,
                text_truth=text_truth,
                text_result=text_result,
                edge_result=edge_result,
                base_result=base_result,
                failed_spans=failed_spans,
                sentence_count=len(sentence_segments),
                output_count=len(output_inputs),
            ),
        )

    @staticmethod
    def _extract_failed_spans(items: Sequence[AlignmentItem]) -> tuple[FailedSpan, ...]:
        spans: list[FailedSpan] = []
        start: int | None = None
        for index, item in enumerate(items):
            if item.status == "failed":
                if start is None:
                    start = index
                continue
            if start is not None:
                spans.append(FailedSpan(start=start, end=index - 1))
                start = None
        if start is not None:
            spans.append(FailedSpan(start=start, end=len(items) - 1))
        return tuple(spans)

    @staticmethod
    def _choose_base_result(
        *,
        text_result: FinalAlignmentResult,
        edge_result: FinalAlignmentResult,
    ) -> FinalAlignmentResult:
        if text_result.route in {"text", "phonetic"}:
            return text_result
        if edge_result.route != "error":
            return edge_result
        return text_result

    def _build_pipeline_report(
        self,
        *,
        time_base: TimeBasePackage,
        text_truth: TextTruthPackage,
        text_result: FinalAlignmentResult,
        edge_result: FinalAlignmentResult,
        base_result: FinalAlignmentResult,
        failed_spans: Sequence[FailedSpan],
        sentence_count: int,
        output_count: int,
    ) -> PipelineReport:
        warnings: list[str] = []
        if text_result.route == "error" and text_result.error_code:
            warnings.append(f"text:{text_result.error_code}")
        if edge_result.route == "error" and edge_result.error_code:
            warnings.append(f"edge:{edge_result.error_code}")
        if base_result.route == "error" and base_result.error_code:
            warnings.append(f"base:{base_result.error_code}")

        phonetic_report = dict(getattr(self._text_aligner, "last_phonetic_report", {}) or {})
        phonetic_report.setdefault("trace_count", len(getattr(self._text_aligner, "last_phonetic_traces", ())))

        return PipelineReport(
            time_base_report=LayerReport(
                name="time_base",
                status="ok",
                metrics={
                    "raw_unit_count": len(time_base.raw_units),
                    "word_unit_count": len(time_base.word_units),
                    "blank_ratio": float(time_base.quality.blank_ratio),
                    "avg_max_prob": float(time_base.quality.avg_max_prob),
                    "low_prob_ratio": float(time_base.quality.low_prob_ratio),
                },
            ),
            text_truth_report=LayerReport(
                name="text_truth",
                status="ok",
                metrics={
                    "unit_count": len(text_truth.units),
                    "is_hallucination": bool(text_truth.is_hallucination),
                    "hallucination_risk": float(text_truth.quality.hallucination_risk),
                    "protected_span_count": len(text_truth.protected_spans),
                },
            ),
            phonetic_report=LayerReport(
                name="phonetic",
                status="ok",
                metrics=phonetic_report,
            ),
            alignment_report=LayerReport(
                name="alignment",
                status="ok" if base_result.route != "error" else "error",
                metrics={
                    "text_route": text_result.route,
                    "edge_route": edge_result.route,
                    "base_route": base_result.route,
                    "failed_span_count": len(failed_spans),
                    "coverage": float(base_result.metrics.coverage),
                    "route_confidence": float(base_result.metrics.route_confidence),
                },
                errors=tuple(warnings),
            ),
            segmentation_report=LayerReport(
                name="segmentation",
                status="ok",
                metrics={
                    "sentence_count": int(sentence_count),
                },
            ),
            output_report=LayerReport(
                name="output",
                status="ok",
                metrics={
                    "output_chunk_count": int(output_count),
                },
            ),
            warnings=tuple(warnings),
        )

    @staticmethod
    def _apply_chunk_global_offset(
        *,
        stream: Sequence[AlignmentItem],
        chunk_window: ChunkWindow,
    ) -> tuple[AlignmentItem, ...]:
        if not stream:
            return tuple()
        offset = float(chunk_window.start)
        if abs(offset) <= 1e-9:
            return tuple(stream)
        if TimeanchoredAlignmentStageService._looks_like_global_timeline(stream=stream, chunk_window=chunk_window):
            return tuple(stream)

        shifted: list[AlignmentItem] = []
        for item in stream:
            shifted_start = float(item.start) + offset
            shifted_end = float(item.end) + offset
            if shifted_end <= shifted_start:
                shifted_end = shifted_start + 0.01
            shifted.append(
                AlignmentItem(
                    text=str(item.text or ""),
                    start=shifted_start,
                    end=shifted_end,
                    status=item.status,
                    source=item.source,
                    confidence=item.confidence,
                    reason=item.reason,
                )
            )
        return tuple(shifted)

    @staticmethod
    def _looks_like_global_timeline(
        *,
        stream: Sequence[AlignmentItem],
        chunk_window: ChunkWindow,
    ) -> bool:
        starts = [float(item.start) for item in stream]
        ends = [float(item.end) for item in stream]
        min_start = min(starts)
        max_end = max(ends)
        window_start = float(chunk_window.start)
        window_end = float(chunk_window.end)
        window_span = max(window_end - window_start, 0.01)
        lower_bound = window_start - 0.25
        upper_bound = window_end + max(1.0, window_span)
        return min_start >= lower_bound and max_end <= upper_bound


__all__ = ["TimeanchoredAlignmentStageService", "TimeanchoredStageResult"]
