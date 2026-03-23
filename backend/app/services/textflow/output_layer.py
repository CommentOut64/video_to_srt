"""
输出层统一入口（真实实现）。
V3.2.0+dev.20260215.24
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from app.services.alignment.types import OutputLayerInput, OutputLayerOutput, OutputTrace
from app.services.textflow.output_dispatch_adapter import OutputDispatchAdapter
from app.services.subtitle_visibility import is_hidden_unknown_sentence


class OutputLayerProcessor:
    """输出层处理器：负责过滤、补齐 trace 并委托分发适配层。"""

    def __init__(
        self,
        *,
        subtitle_manager: Any,
        speaker_store_service_getter: Optional[Callable[[], Any]] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self._output_dispatch_adapter = OutputDispatchAdapter(
            subtitle_manager=subtitle_manager,
            speaker_store_service_getter=speaker_store_service_getter,
            logger=logger,
        )

    def process(self, data: OutputLayerInput) -> OutputLayerOutput:
        """执行输出层分发。"""
        sentence_segments = list(data.sentence_segments or [])
        raw_output_traces = list(data.output_traces or [])
        filtered_sentences: list[Any] = []
        filtered_traces: list[OutputTrace] = []
        unknown_sentence_filtered_count = 0
        for sentence_index, sentence in enumerate(sentence_segments):
            if is_hidden_unknown_sentence(sentence):
                unknown_sentence_filtered_count += 1
                continue
            filtered_sentences.append(sentence)
            if sentence_index < len(raw_output_traces):
                filtered_traces.append(raw_output_traces[sentence_index])
        sentence_segments = filtered_sentences

        output_traces = self._resolve_output_traces(
            sentence_segments=sentence_segments,
            output_traces=filtered_traces,
        )
        self._apply_output_traces_to_sentences(
            sentence_segments=sentence_segments,
            output_traces=output_traces,
        )
        payload = self._output_dispatch_adapter.dispatch(
            chunk_index=data.chunk_index,
            sentence_segments=sentence_segments,
            output_traces=output_traces,
            injection_report=dict(data.injection_report or {}),
            segmentation_report=dict(data.segmentation_report or {}),
            unknown_sentence_filtered_count=int(unknown_sentence_filtered_count),
        )
        return OutputLayerOutput(
            output_payload=payload,
            output_traces=output_traces,
        )

    @staticmethod
    def _resolve_output_traces(
        *,
        sentence_segments: Sequence[Any],
        output_traces: Optional[Sequence[OutputTrace]],
    ) -> list[OutputTrace]:
        traces = list(output_traces or [])
        if traces:
            return traces
        fallback: list[OutputTrace] = []
        for sentence_index, sentence in enumerate(sentence_segments):
            fallback.append(
                OutputTrace(
                    sentence_index=sentence_index,
                    split_reason=str(getattr(sentence, "split_reason", "") or "default_splitter"),
                    split_risk=str(getattr(sentence, "split_risk", "") or ""),
                    window_id=str(getattr(sentence, "window_id", "") or ""),
                    pyannote_frame_time=getattr(sentence, "pyannote_frame_time", None),
                    mapped_cut_time=getattr(sentence, "mapped_cut_time", None),
                    mapping_quality=str(getattr(sentence, "mapping_quality", "") or "default"),
                    mapping_reason=str(getattr(sentence, "mapping_reason", "") or "default_splitter"),
                    sentence_start=float(getattr(sentence, "start", 0.0) or 0.0),
                    sentence_end=float(getattr(sentence, "end", 0.0) or 0.0),
                )
            )
        return fallback

    @staticmethod
    def _apply_output_traces_to_sentences(
        *,
        sentence_segments: Sequence[Any],
        output_traces: Sequence[OutputTrace],
    ) -> None:
        trace_by_index = {int(trace.sentence_index): trace for trace in output_traces}
        for sentence_index, sentence in enumerate(sentence_segments):
            trace = trace_by_index.get(sentence_index)
            if trace is None:
                continue
            setattr(sentence, "split_reason", str(trace.split_reason or ""))
            setattr(sentence, "split_risk", str(trace.split_risk or ""))
            setattr(sentence, "window_id", str(trace.window_id or ""))
            setattr(sentence, "pyannote_frame_time", trace.pyannote_frame_time)
            setattr(sentence, "mapped_cut_time", trace.mapped_cut_time)
            setattr(sentence, "mapping_quality", str(trace.mapping_quality or ""))
            setattr(sentence, "mapping_reason", str(trace.mapping_reason or ""))


# 兼容旧命名（用于过渡期）。
OutputProcessor = OutputLayerProcessor

__all__ = ["OutputLayerProcessor", "OutputProcessor"]

