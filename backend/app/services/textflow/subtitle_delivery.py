"""SubtitleDelivery：统一南向字幕 DTO 投递层。"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Sequence

from app.services.alignment.types import OutputTrace

from app.services.textflow.contracts import (
    ChunkSentenceIndex,
    RenderResult,
    SentenceRecord,
    SubtitleBatch,
    SubtitleItem,
)


class SubtitleDelivery:
    """将 RenderResult 收口为统一字幕批次 DTO。"""

    def build_sentence_records(
        self,
        *,
        render_result: RenderResult,
        source_chunk_ids: Sequence[Any],
        overlap_chunk_ids: Optional[Sequence[Any]] = None,
        replace_scope_chunk_ids: Optional[Sequence[Any]] = None,
        route: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[SentenceRecord, ...]:
        base_source_chunk_ids = self._normalize_chunk_ids(source_chunk_ids)
        base_overlap_chunk_ids = self._normalize_chunk_ids(
            overlap_chunk_ids if overlap_chunk_ids is not None else base_source_chunk_ids
        )
        base_replace_scope_chunk_ids = self._normalize_chunk_ids(
            replace_scope_chunk_ids if replace_scope_chunk_ids is not None else base_source_chunk_ids
        )
        shared_metadata = dict(metadata or {})
        records: list[SentenceRecord] = []
        for item in render_result.subtitles:
            trace = dict(item.trace or {})
            record_metadata = dict(shared_metadata)
            record_metadata.update(
                {
                    "status": "final",
                    "source": str(item.text_source or "unknown"),
                    "speaker_id": item.speaker_id,
                    "turn_id": item.turn_id,
                }
            )
            records.append(
                SentenceRecord(
                    sentence_id=str(item.segment_id),
                    text=str(item.text_display),
                    start=float(item.start),
                    end=float(item.end),
                    source_chunk_ids=self._normalize_chunk_ids(
                        trace.get("source_chunk_ids") or base_source_chunk_ids
                    ),
                    overlap_chunk_ids=self._normalize_chunk_ids(
                        trace.get("overlap_chunk_ids") or base_overlap_chunk_ids
                    ),
                    replace_scope_chunk_ids=self._normalize_chunk_ids(
                        trace.get("replace_scope_chunk_ids") or base_replace_scope_chunk_ids
                    ),
                    route=str(trace.get("route") or route or ""),
                    trace=trace,
                    metadata=record_metadata,
                )
            )
        return tuple(records)

    def build_sentence_records_from_sentences(
        self,
        *,
        sentence_segments: Sequence[Any],
        output_traces: Optional[Sequence[OutputTrace]] = None,
        source_chunk_ids: Optional[Sequence[Any]] = None,
        overlap_chunk_ids: Optional[Sequence[Any]] = None,
        replace_scope_chunk_ids: Optional[Sequence[Any]] = None,
        route: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[SentenceRecord, ...]:
        traces = self._normalize_output_traces(
            sentence_segments=sentence_segments,
            output_traces=output_traces,
        )
        base_source_chunk_ids = self._normalize_chunk_ids(source_chunk_ids or ())
        base_overlap_chunk_ids = self._normalize_chunk_ids(
            overlap_chunk_ids if overlap_chunk_ids is not None else base_source_chunk_ids
        )
        base_replace_scope_chunk_ids = self._normalize_chunk_ids(
            replace_scope_chunk_ids if replace_scope_chunk_ids is not None else base_source_chunk_ids
        )
        shared_metadata = dict(metadata or {})
        records: list[SentenceRecord] = []
        for sentence_index, sentence in enumerate(list(sentence_segments or [])):
            trace = self._serialize_output_trace(traces[sentence_index])
            record_metadata = dict(shared_metadata)
            record_metadata.update(
                {
                    "status": self._resolve_sentence_status(sentence),
                    "source": self._resolve_sentence_source(sentence),
                    "speaker_id": getattr(sentence, "speaker_id", None),
                    "turn_id": getattr(sentence, "turn_id", None),
                }
            )
            records.append(
                SentenceRecord(
                    sentence_id=self._resolve_segment_id("sentence-record", sentence_index, sentence),
                    text=self._resolve_sentence_text(sentence),
                    start=float(getattr(sentence, "start", 0.0) or 0.0),
                    end=float(getattr(sentence, "end", 0.0) or 0.0),
                    source_chunk_ids=base_source_chunk_ids,
                    overlap_chunk_ids=base_overlap_chunk_ids,
                    replace_scope_chunk_ids=base_replace_scope_chunk_ids,
                    route=str(route or ""),
                    trace=trace,
                    metadata=record_metadata,
                )
            )
        return tuple(records)

    @staticmethod
    def build_chunk_sentence_indices(
        *,
        chunk_id: str,
        sentence_records: Sequence[SentenceRecord],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[ChunkSentenceIndex, ...]:
        return (
            ChunkSentenceIndex(
                chunk_id=str(chunk_id),
                sentence_ids=tuple(str(record.sentence_id) for record in sentence_records),
                metadata=dict(metadata or {}),
            ),
        )

    def build_batch_from_sentence_records(
        self,
        *,
        chunk_id: str,
        sentence_records: Sequence[SentenceRecord],
        chunk_index: Optional[int] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
        render_report: Optional[Dict[str, Any]] = None,
    ) -> SubtitleBatch:
        items = tuple(
            SubtitleItem(
                segment_id=str(record.sentence_id),
                chunk_id=str(chunk_id),
                start=float(record.start),
                end=float(record.end),
                text=str(record.text),
                status=str(record.metadata.get("status", "final") or "final"),
                source=str(record.metadata.get("source", "unknown") or "unknown"),
                speaker_id=record.metadata.get("speaker_id"),
                turn_id=record.metadata.get("turn_id"),
                trace=dict(record.trace or {}),
            )
            for record in sentence_records
        )
        normalized_diagnostics = dict(diagnostics or {})
        if "output_trace" not in normalized_diagnostics:
            normalized_diagnostics["output_trace"] = self._build_output_trace(items)
        computed_projection = self._build_projection_payload_from_sentence_records(
            sentence_records=sentence_records
        )
        existing_projection = dict(normalized_diagnostics.get("projection") or {})
        if existing_projection or computed_projection:
            merged_projection = dict(computed_projection)
            merged_projection.update(existing_projection)
            merged_projection.setdefault(
                "replace_scope_chunk_ids",
                computed_projection.get("replace_scope_chunk_ids", []),
            )
            merged_projection.setdefault(
                "source_chunk_ids",
                computed_projection.get("source_chunk_ids", []),
            )
            merged_projection.setdefault(
                "overlap_chunk_ids",
                computed_projection.get("overlap_chunk_ids", []),
            )
            normalized_diagnostics["projection"] = merged_projection
        return SubtitleBatch(
            chunk_id=str(chunk_id),
            chunk_index=chunk_index,
            items=items,
            render_report=dict(render_report or {}),
            diagnostics=normalized_diagnostics,
        )

    def build_sentence_records_from_subtitle_batch(
        self,
        *,
        subtitle_batch: SubtitleBatch,
    ) -> tuple[SentenceRecord, ...]:
        projection = dict(subtitle_batch.diagnostics.get("projection") or {})
        source_chunk_ids = self._normalize_chunk_ids(
            projection.get("source_chunk_ids")
            or projection.get("replace_scope_chunk_ids")
            or ()
        )
        overlap_chunk_ids = self._normalize_chunk_ids(
            projection.get("overlap_chunk_ids") or source_chunk_ids
        )
        replace_scope_chunk_ids = self._normalize_chunk_ids(
            projection.get("replace_scope_chunk_ids") or source_chunk_ids
        )
        records: list[SentenceRecord] = []
        for item in subtitle_batch.items:
            trace = dict(item.trace or {})
            records.append(
                SentenceRecord(
                    sentence_id=str(item.segment_id),
                    text=str(item.text),
                    start=float(item.start),
                    end=float(item.end),
                    source_chunk_ids=self._normalize_chunk_ids(
                        trace.get("source_chunk_ids") or source_chunk_ids
                    ),
                    overlap_chunk_ids=self._normalize_chunk_ids(
                        trace.get("overlap_chunk_ids") or overlap_chunk_ids
                    ),
                    replace_scope_chunk_ids=self._normalize_chunk_ids(
                        trace.get("replace_scope_chunk_ids") or replace_scope_chunk_ids
                    ),
                    route=str(trace.get("route") or ""),
                    trace=trace,
                    metadata={
                        "status": str(item.status or "final"),
                        "source": str(item.source or "unknown"),
                        "speaker_id": item.speaker_id,
                        "turn_id": item.turn_id,
                    },
                )
            )
        return tuple(records)

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
        ingress_payload = dict(ingress_context or {})
        source_chunk_ids = ingress_payload.get("source_chunk_ids") or (str(chunk_id),)
        overlap_chunk_ids = (
            ingress_payload.get("projection_chunk_ids")
            or ingress_payload.get("source_chunk_ids")
            or source_chunk_ids
        )
        sentence_records = self.build_sentence_records(
            render_result=render_result,
            source_chunk_ids=source_chunk_ids,
            overlap_chunk_ids=overlap_chunk_ids,
            replace_scope_chunk_ids=source_chunk_ids,
            metadata={"chunk_id": str(chunk_id)},
        )
        diagnostics = {
            "source": str(source or "render_core"),
            "render_output_trace": [dict(entry) for entry in render_result.output_trace],
        }
        if ingress_context:
            diagnostics["ingress_context"] = ingress_payload
        return self.build_batch_from_sentence_records(
            chunk_id=str(chunk_id),
            chunk_index=chunk_index,
            sentence_records=sentence_records,
            diagnostics=diagnostics,
            render_report=dict(render_result.render_report or {}),
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
        sentence_records = tuple(
            SentenceRecord(
                sentence_id=str(item.segment_id),
                text=str(item.text),
                start=float(item.start),
                end=float(item.end),
                route=str(source or ""),
                trace=dict(item.trace or {}),
                metadata={
                    "status": str(item.status or "final"),
                    "source": str(item.source or "unknown"),
                    "speaker_id": item.speaker_id,
                    "turn_id": item.turn_id,
                },
            )
            for item in items
        )
        return self.build_batch_from_sentence_records(
            chunk_id=str(chunk_id),
            chunk_index=chunk_index,
            sentence_records=sentence_records,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _build_projection_payload_from_sentence_records(
        *,
        sentence_records: Sequence[SentenceRecord],
    ) -> dict[str, Any]:
        replace_scope_chunk_ids = SubtitleDelivery._merge_chunk_id_sequences(
            record.replace_scope_chunk_ids for record in sentence_records
        )
        source_chunk_ids = SubtitleDelivery._merge_chunk_id_sequences(
            record.source_chunk_ids for record in sentence_records
        )
        overlap_chunk_ids = SubtitleDelivery._merge_chunk_id_sequences(
            record.overlap_chunk_ids for record in sentence_records
        )
        return {
            "source_chunk_ids": source_chunk_ids,
            "overlap_chunk_ids": overlap_chunk_ids,
            "replace_scope_chunk_ids": replace_scope_chunk_ids or source_chunk_ids,
        }

    @staticmethod
    def _merge_chunk_id_sequences(sequences: Sequence[Sequence[Any]]) -> list[str]:
        merged: list[str] = []
        for sequence in sequences:
            for item in sequence:
                text = str(item or "").strip()
                if text and text not in merged:
                    merged.append(text)
        return merged

    @staticmethod
    def _normalize_chunk_ids(values: Sequence[Any]) -> tuple[str, ...]:
        normalized: list[str] = []
        for item in values:
            text = str(item or "").strip()
            if text and text not in normalized:
                normalized.append(text)
        return tuple(normalized)

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
