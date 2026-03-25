"""SubtitleDelivery：统一南向字幕 DTO 投递层。"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Sequence

from app.services.alignment.types import OutputTrace

from app.services.textflow.contracts import (
    RenderResult,
    SubtitleBatch,
    SubtitleItem,
)


class SubtitleDelivery:
    """将 RenderResult 收口为统一字幕批次 DTO。"""

    @staticmethod
    def _build_output_trace(items: tuple[SubtitleItem, ...]) -> list[dict[str, Any]]:
        traces: list[dict[str, Any]] = []
        for sentence_index, item in enumerate(items):
            trace = dict(item.trace or {})
            traces.append(
                {
                    "sentence_index": sentence_index,
                    "split_reason": str(trace.get("split_reason", "") or ""),
                    "split_risk": str(trace.get("split_risk", "") or ""),
                    "window_id": str(trace.get("window_id", "") or ""),
                    "pyannote_frame_time": trace.get("pyannote_frame_time"),
                    "mapped_cut_time": trace.get("mapped_cut_time", item.end),
                    "mapping_quality": str(trace.get("mapping_quality", "") or ""),
                    "mapping_reason": str(trace.get("mapping_reason", "") or ""),
                    "sentence_start": float(item.start),
                    "sentence_end": float(item.end),
                }
            )
        return traces

    def build_batch(
        self,
        *,
        render_result: RenderResult,
        chunk_id: str,
        chunk_index: Optional[int] = None,
        ingress_context: Optional[Dict[str, Any]] = None,
        source: str = "render_core",
    ) -> SubtitleBatch:
        items = tuple(
            SubtitleItem(
                segment_id=item.segment_id,
                chunk_id=chunk_id,
                start=float(item.start),
                end=float(item.end),
                text=str(item.text_display),
                status="final",
                source=str(item.text_source or "unknown"),
                speaker_id=item.speaker_id,
                turn_id=item.turn_id,
                trace=dict(item.trace or {}),
            )
            for item in render_result.subtitles
        )
        diagnostics = {
            "source": str(source or "render_core"),
            "render_output_trace": [dict(entry) for entry in render_result.output_trace],
            "output_trace": self._build_output_trace(items),
        }
        if ingress_context:
            diagnostics["ingress_context"] = dict(ingress_context)
        return SubtitleBatch(
            chunk_id=str(chunk_id),
            chunk_index=chunk_index,
            items=items,
            render_report=dict(render_result.render_report or {}),
            diagnostics=diagnostics,
        )

    def build_batch_from_sentences(
        self,
        *,
        chunk_id: str,
        sentence_segments: Sequence[Any],
        output_traces: Optional[Sequence[OutputTrace]] = None,
        chunk_index: Optional[int] = None,
        ingress_context: Optional[Dict[str, Any]] = None,
        source: str = "sentence_segment_adapter",
    ) -> SubtitleBatch:
        traces = self._normalize_output_traces(
            sentence_segments=sentence_segments,
            output_traces=output_traces,
        )
        items: list[SubtitleItem] = []
        trace_payloads: list[dict[str, Any]] = []
        for sentence_index, sentence in enumerate(list(sentence_segments or [])):
            trace = traces[sentence_index] if sentence_index < len(traces) else OutputTrace(
                sentence_index=sentence_index,
                split_reason="default_splitter",
                split_risk="",
                window_id="",
                pyannote_frame_time=None,
                mapped_cut_time=float(getattr(sentence, "end", 0.0) or 0.0),
                mapping_quality="default",
                mapping_reason="default_splitter",
                sentence_start=float(getattr(sentence, "start", 0.0) or 0.0),
                sentence_end=float(getattr(sentence, "end", 0.0) or 0.0),
            )
            trace_payload = self._serialize_output_trace(trace)
            items.append(
                SubtitleItem(
                    segment_id=self._resolve_segment_id(str(chunk_id), sentence_index, sentence),
                    chunk_id=str(chunk_id),
                    start=float(getattr(sentence, "start", 0.0) or 0.0),
                    end=float(getattr(sentence, "end", 0.0) or 0.0),
                    text=self._resolve_sentence_text(sentence),
                    status=self._resolve_sentence_status(sentence),
                    source=self._resolve_sentence_source(sentence),
                    speaker_id=getattr(sentence, "speaker_id", None),
                    turn_id=getattr(sentence, "turn_id", None),
                    trace={
                        "split_reason": str(trace.split_reason or ""),
                        "split_risk": str(trace.split_risk or ""),
                        "window_id": str(trace.window_id or ""),
                        "pyannote_frame_time": trace.pyannote_frame_time,
                        "mapped_cut_time": trace.mapped_cut_time,
                        "mapping_quality": str(trace.mapping_quality or ""),
                        "mapping_reason": str(trace.mapping_reason or ""),
                    },
                )
            )
            trace_payloads.append(trace_payload)

        diagnostics = {
            "source": str(source or "sentence_segment_adapter"),
            "output_trace": trace_payloads,
        }
        if ingress_context:
            diagnostics["ingress_context"] = dict(ingress_context)
        return SubtitleBatch(
            chunk_id=str(chunk_id),
            chunk_index=chunk_index,
            items=tuple(items),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _normalize_output_traces(
        *,
        sentence_segments: Sequence[Any],
        output_traces: Optional[Sequence[OutputTrace]],
    ) -> list[OutputTrace]:
        traces = list(output_traces or [])
        if len(traces) == len(list(sentence_segments or [])):
            return traces
        normalized: list[OutputTrace] = []
        for sentence_index, sentence in enumerate(list(sentence_segments or [])):
            normalized.append(
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
        return normalized

    @staticmethod
    def _resolve_sentence_text(sentence: Any) -> str:
        return str(getattr(sentence, "text_clean", "") or getattr(sentence, "text", "") or "")

    @staticmethod
    def _resolve_sentence_status(sentence: Any) -> str:
        if getattr(sentence, "is_draft", False) and not getattr(sentence, "is_finalized", False):
            return "draft"
        return "final"

    @staticmethod
    def _resolve_sentence_source(sentence: Any) -> str:
        source = getattr(sentence, "source", None)
        return str(getattr(source, "value", source) or "unknown")

    @classmethod
    def _resolve_segment_id(cls, chunk_id: str, sentence_index: int, sentence: Any) -> str:
        existing = str(getattr(sentence, "segment_id", "") or getattr(sentence, "sentence_uid", "") or "")
        if existing:
            return existing
        seed = (
            f"{chunk_id}|{sentence_index}|"
            f"{float(getattr(sentence, 'start', 0.0) or 0.0):.3f}|"
            f"{float(getattr(sentence, 'end', 0.0) or 0.0):.3f}|"
            f"{cls._resolve_sentence_text(sentence)}"
        )
        return f"seg-{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:16]}"

    @staticmethod
    def _serialize_output_trace(trace: OutputTrace) -> dict[str, Any]:
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
