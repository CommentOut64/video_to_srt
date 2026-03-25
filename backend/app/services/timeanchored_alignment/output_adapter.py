"""OutputAdapter：新主链输出到旧 OutputLayerInput 的一次性契约转换。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

from app.services.alignment.types import OutputLayerInput, OutputTrace
from app.services.timeanchored_alignment.chunk_projector import ChunkProjection


@dataclass(frozen=True)
class OutputAdapter:
    """输出适配器。"""

    def to_output_layer_inputs(
        self,
        *,
        projections: Sequence[ChunkProjection],
        language: str = "auto",
        injection_report: dict[str, Any] | None = None,
        segmentation_report: dict[str, Any] | None = None,
    ) -> list[OutputLayerInput]:
        from app.services.textflow.subtitle_delivery import SubtitleDelivery

        subtitle_delivery = SubtitleDelivery()
        outputs: list[OutputLayerInput] = []
        for projection in projections:
            sentences = [deepcopy(sentence) for sentence in projection.sentence_segments]
            traces = self._build_output_traces(sentences)
            chunk_ref = projection.chunk_window.chunk_ref
            outputs.append(
                OutputLayerInput(
                    chunk_index=chunk_ref,
                    sentence_segments=sentences,
                    language=language,
                    injection_report=dict(injection_report or {}),
                    segmentation_report=dict(segmentation_report or {}),
                    output_traces=traces,
                    subtitle_batch=subtitle_delivery.build_batch_from_sentences(
                        chunk_id=str(chunk_ref),
                        chunk_index=self._try_parse_chunk_index(chunk_ref),
                        sentence_segments=sentences,
                        output_traces=traces,
                        source="timeanchored_output_adapter",
                    ),
                )
            )
        return outputs

    @staticmethod
    def _build_output_traces(sentences: Sequence[Any]) -> list[OutputTrace]:
        traces: list[OutputTrace] = []
        for idx, sentence in enumerate(sentences):
            traces.append(
                OutputTrace(
                    sentence_index=idx,
                    split_reason=str(getattr(sentence, "split_reason", "") or ""),
                    split_risk=str(getattr(sentence, "split_risk", "") or ""),
                    window_id=str(getattr(sentence, "window_id", "") or ""),
                    pyannote_frame_time=getattr(sentence, "pyannote_frame_time", None),
                    mapped_cut_time=getattr(sentence, "mapped_cut_time", None),
                    mapping_quality=str(getattr(sentence, "mapping_quality", "") or ""),
                    mapping_reason=str(getattr(sentence, "mapping_reason", "") or ""),
                    sentence_start=float(getattr(sentence, "start", 0.0) or 0.0),
                    sentence_end=float(getattr(sentence, "end", 0.0) or 0.0),
                )
            )
        return traces

    @staticmethod
    def _try_parse_chunk_index(chunk_ref: Any) -> int | None:
        if isinstance(chunk_ref, int):
            return chunk_ref
        chunk_text = str(chunk_ref or "").strip()
        if not chunk_text:
            return None
        if chunk_text.lstrip("-").isdigit():
            try:
                return int(chunk_text)
            except ValueError:
                return None
        if chunk_text.startswith("chunk-"):
            try:
                return int(chunk_text.split("-")[-1])
            except ValueError:
                return None
        return None


__all__ = ["OutputAdapter"]
