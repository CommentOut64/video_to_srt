"""输出分发与统一输出处理器。"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, Optional, Sequence

from app.core.logging import resolve_loguru_logger
from app.services.alignment.types import OutputLayerInput, OutputLayerOutput, OutputTrace
from app.services.textflow.contracts import ChunkSentenceIndex, SentenceRecord, SubtitleBatch, SubtitleItem
from app.services.textflow.subtitle_delivery import SubtitleDelivery


class OutputLayerProcessor:
    """统一输出处理器：只接受统一 SubtitleBatch 并分发。"""

    def __init__(
        self,
        *,
        subtitle_manager: Any,
        speaker_store_service_getter: Optional[Callable[[], Any]] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self._subtitle_delivery = SubtitleDelivery()
        self._output_dispatch_adapter = OutputDispatchAdapter(
            subtitle_manager=subtitle_manager,
            speaker_store_service_getter=speaker_store_service_getter,
            logger=logger,
        )

    def process(self, data: OutputLayerInput) -> OutputLayerOutput:
        """执行统一输出分发。"""
        sentence_records = self._resolve_sentence_records(data)
        chunk_sentence_indices = self._resolve_chunk_sentence_indices(
            data=data,
            sentence_records=sentence_records,
        )
        subtitle_batch = self._resolve_subtitle_batch_compat(
            data=data,
            sentence_records=sentence_records,
            chunk_sentence_indices=chunk_sentence_indices,
        )
        output_traces = self._resolve_output_traces(
            sentence_records=sentence_records,
            subtitle_batch=subtitle_batch,
        )
        payload = self._output_dispatch_adapter.dispatch(
            sentence_records=sentence_records,
            chunk_sentence_indices=chunk_sentence_indices,
            subtitle_batch=subtitle_batch,
            injection_report=dict(data.injection_report or {}),
            segmentation_report=dict(data.segmentation_report or {}),
            unknown_sentence_filtered_count=0,
        )
        return OutputLayerOutput(
            output_payload=payload,
            output_traces=output_traces,
        )

    @staticmethod
    def _resolve_output_traces(
        *,
        sentence_records: Sequence[SentenceRecord],
        subtitle_batch: SubtitleBatch,
    ) -> list[OutputTrace]:
        if sentence_records:
            traces: list[OutputTrace] = []
            for sentence_index, record in enumerate(sentence_records):
                trace = dict(record.trace or {})
                traces.append(
                    OutputTrace(
                        sentence_index=sentence_index,
                        split_reason=str(trace.get("split_reason", "") or ""),
                        split_risk=str(trace.get("split_risk", "") or ""),
                        window_id=str(trace.get("window_id", "") or ""),
                        pyannote_frame_time=trace.get("pyannote_frame_time"),
                        mapped_cut_time=trace.get("mapped_cut_time", float(record.end)),
                        mapping_quality=str(trace.get("mapping_quality", "") or ""),
                        mapping_reason=str(trace.get("mapping_reason", "") or ""),
                        sentence_start=float(record.start),
                        sentence_end=float(record.end),
                    )
                )
            return traces
        return OutputLayerProcessor._resolve_output_traces_from_batch(subtitle_batch)

    @staticmethod
    def _resolve_output_traces_from_batch(subtitle_batch: SubtitleBatch) -> list[OutputTrace]:
        raw_traces = list(subtitle_batch.diagnostics.get("output_trace") or [])
        if raw_traces:
            traces: list[OutputTrace] = []
            for item in raw_traces:
                traces.append(
                    OutputTrace(
                        sentence_index=int(item.get("sentence_index", 0) or 0),
                        split_reason=str(item.get("split_reason", "") or ""),
                        split_risk=str(item.get("split_risk", "") or ""),
                        window_id=str(item.get("window_id", "") or ""),
                        pyannote_frame_time=item.get("pyannote_frame_time"),
                        mapped_cut_time=item.get("mapped_cut_time"),
                        mapping_quality=str(item.get("mapping_quality", "") or ""),
                        mapping_reason=str(item.get("mapping_reason", "") or ""),
                        sentence_start=item.get("sentence_start"),
                        sentence_end=item.get("sentence_end"),
                    )
                )
            return traces
        traces: list[OutputTrace] = []
        for sentence_index, item in enumerate(subtitle_batch.items):
            trace = dict(item.trace or {})
            traces.append(
                OutputTrace(
                    sentence_index=sentence_index,
                    split_reason=str(trace.get("split_reason", "") or ""),
                    split_risk=str(trace.get("split_risk", "") or ""),
                    window_id=str(trace.get("window_id", "") or ""),
                    pyannote_frame_time=trace.get("pyannote_frame_time"),
                    mapped_cut_time=trace.get("mapped_cut_time", float(item.end)),
                    mapping_quality=str(trace.get("mapping_quality", "") or ""),
                    mapping_reason=str(trace.get("mapping_reason", "") or ""),
                    sentence_start=float(item.start),
                    sentence_end=float(item.end),
                )
            )
        return traces

    def _resolve_sentence_records(self, data: OutputLayerInput) -> list[SentenceRecord]:
        if data.sentence_records:
            return list(data.sentence_records)
        if data.subtitle_batch is not None and not data.sentence_segments:
            return []
        raise ValueError(
            "OutputLayerProcessor 需要 sentence_records；"
            "subtitle_batch 仅作为南向 compat 边界保留。"
        )

    def _resolve_chunk_sentence_indices(
        self,
        *,
        data: OutputLayerInput,
        sentence_records: Sequence[SentenceRecord],
    ) -> list[ChunkSentenceIndex]:
        if data.chunk_sentence_indices:
            return list(data.chunk_sentence_indices)
        chunk_id = self._resolve_target_chunk_id(data=data)
        return list(
            self._subtitle_delivery.build_chunk_sentence_indices(
                chunk_id=chunk_id,
                sentence_records=sentence_records,
            )
        )

    def _resolve_subtitle_batch_compat(
        self,
        *,
        data: OutputLayerInput,
        sentence_records: Sequence[SentenceRecord],
        chunk_sentence_indices: Sequence[ChunkSentenceIndex],
    ) -> SubtitleBatch:
        if data.subtitle_batch is not None:
            return data.subtitle_batch
        chunk_id = self._resolve_target_chunk_id(data=data, chunk_sentence_indices=chunk_sentence_indices)
        chunk_index = self._resolve_target_chunk_index(data=data, chunk_id=chunk_id)
        diagnostics: Dict[str, Any] = {}
        if not diagnostics and chunk_sentence_indices:
            first_metadata = dict(chunk_sentence_indices[0].metadata or {})
            projection_payload = dict(first_metadata.get("projection") or {})
            if projection_payload:
                diagnostics["projection"] = projection_payload
            projection_mode = str(first_metadata.get("projection_mode", "") or "")
            if projection_mode and "projection" not in diagnostics:
                diagnostics["projection"] = {"projection_mode": projection_mode}
        return self._subtitle_delivery.build_batch_from_sentence_records(
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            sentence_records=sentence_records,
            diagnostics=diagnostics,
            render_report={},
        )

    @staticmethod
    def _resolve_target_chunk_id(
        *,
        data: OutputLayerInput,
        chunk_sentence_indices: Sequence[ChunkSentenceIndex] = (),
    ) -> str:
        if chunk_sentence_indices:
            chunk_id = str(chunk_sentence_indices[0].chunk_id or "").strip()
            if chunk_id:
                return chunk_id
        if data.subtitle_batch is not None:
            chunk_id = str(data.subtitle_batch.chunk_id or "").strip()
            if chunk_id:
                return chunk_id
        chunk_text = str(data.chunk_index or "").strip()
        if chunk_text:
            return chunk_text
        raise ValueError("OutputLayerProcessor 无法解析目标 chunk_id")

    @staticmethod
    def _resolve_target_chunk_index(
        *,
        data: OutputLayerInput,
        chunk_id: str,
    ) -> int | None:
        if data.subtitle_batch is not None and data.subtitle_batch.chunk_index is not None:
            return int(data.subtitle_batch.chunk_index)
        if isinstance(data.chunk_index, int):
            return int(data.chunk_index)
        return OutputDispatchAdapter._try_parse_chunk_index(chunk_id)

class OutputDispatchAdapter:
    """负责输出层南向分发与 payload 封装。"""

    def __init__(
        self,
        *,
        subtitle_manager: Any,
        speaker_store_service_getter: Optional[Callable[[], Any]] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self._subtitle_manager = subtitle_manager
        self._speaker_store_service_getter = speaker_store_service_getter
        self._logger = resolve_loguru_logger(
            logger,
            __name__,
            layer="输出层",
            processor_name="output_dispatch_adapter",
        )

    def dispatch(
        self,
        *,
        sentence_records: Sequence[SentenceRecord],
        chunk_sentence_indices: Sequence[ChunkSentenceIndex],
        subtitle_batch: SubtitleBatch,
        injection_report: Optional[Dict[str, Any]] = None,
        segmentation_report: Optional[Dict[str, Any]] = None,
        unknown_sentence_filtered_count: int = 0,
    ) -> Dict[str, Any]:
        output_errors: list[str] = []
        replaced_sentence_indices: list[int] = []
        subtitle_channel_status = "ok"
        speaker_store_channel_status = "skipped"
        speaker_link_count = 0

        try:
            replaced_sentence_indices = list(
                self._subtitle_manager.replace_chunk_batch(subtitle_batch) or []
            )
        except Exception:
            output_errors.append("E_OUTPUT_CHANNEL_FAIL")
            subtitle_channel_status = "failed"
            speaker_store_channel_status = "skipped_upstream_failed"
            self._logger.exception(
                "输出层分发失败: chunk_id={} sentences={}",
                subtitle_batch.chunk_id,
                len(subtitle_batch.items),
            )

        if "E_OUTPUT_CHANNEL_FAIL" not in output_errors:
            try:
                speaker_store_channel_status, speaker_link_count = self._write_speaker_links(
                    sentence_records=sentence_records,
                    sentence_indices=replaced_sentence_indices,
                )
            except Exception:
                output_errors.append("E_OUTPUT_SPEAKER_STORE_FAIL")
                speaker_store_channel_status = "failed"
                self._logger.exception(
                    "输出层 speaker_store 写入失败: chunk_id={} sentences={}",
                    subtitle_batch.chunk_id,
                    len(subtitle_batch.items),
                )

        segmentation_payload = dict(segmentation_report or {})
        segmentation_payload["unknown_sentence_filtered_count"] = int(unknown_sentence_filtered_count)
        projection_payload = dict(subtitle_batch.diagnostics.get("projection") or {})
        source_chunk_ids = [
            str(item)
            for item in list(
                projection_payload.get("replace_scope_chunk_ids")
                or projection_payload.get("source_chunk_ids")
                or []
            )
            if item is not None
        ]
        payload: Dict[str, Any] = {
            "chunk_index": self._resolve_chunk_index(
                subtitle_batch=subtitle_batch,
                chunk_sentence_indices=chunk_sentence_indices,
            ),
            "chunk_uid": str(self._resolve_chunk_uid(subtitle_batch, chunk_sentence_indices)),
            "source_chunk_ids": source_chunk_ids,
            "sentence_count": int(len(sentence_records)),
            "sentence_segments": [
                self._serialize_sentence_record(
                    record=record,
                    chunk_id=str(self._resolve_chunk_uid(subtitle_batch, chunk_sentence_indices)),
                )
                for record in sentence_records
            ],
            "transport_meta": {
                "channels": [
                    "streaming_subtitle.replace_chunk",
                    "speaker_store.subtitle_speaker_links_upsert",
                ],
                "subtitle_channel": {
                    "status": subtitle_channel_status,
                    "replaced_sentence_count": int(len(replaced_sentence_indices)),
                },
                "speaker_store_channel": {
                    "status": speaker_store_channel_status,
                    "upsert_count": int(speaker_link_count),
                },
            },
            "injection_report": dict(injection_report or {}),
            "segmentation_report": segmentation_payload,
            "output_trace": [
                self._serialize_trace_dict(item)
                for item in list(subtitle_batch.diagnostics.get("output_trace") or [])
            ],
            "errors": output_errors,
        }
        self._logger.info(
            "输出分发完成: chunk_id={} sentences={} filtered_unknown={} errors={}",
            subtitle_batch.chunk_id,
            len(subtitle_batch.items),
            int(unknown_sentence_filtered_count),
            len(output_errors),
        )
        return payload

    @staticmethod
    def _compute_sentence_text_hash(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _resolve_speaker_store_service(self) -> Optional[Any]:
        if self._speaker_store_service_getter is None:
            return None
        try:
            return self._speaker_store_service_getter()
        except Exception:
            self._logger.exception("输出层获取 speaker_store_service 失败")
            return None

    def _write_speaker_links(
        self,
        *,
        sentence_records: Sequence[SentenceRecord],
        sentence_indices: Sequence[int],
    ) -> tuple[str, int]:
        speaker_store_service = self._resolve_speaker_store_service()
        if speaker_store_service is None:
            return "skipped_no_service", 0
        if not sentence_records or not sentence_indices:
            return "skipped_empty", 0
        if len(sentence_indices) < len(sentence_records):
            self._logger.warning(
                "输出层 speaker_store 写入索引不足: indices={} sentences={}",
                len(sentence_indices),
                len(sentence_records),
            )

        payload_items = []
        for idx, record in enumerate(sentence_records):
            if idx >= len(sentence_indices):
                break
            sentence_index = int(sentence_indices[idx])
            text = str(record.text or "")
            payload_items.append(
                {
                    "sentence_index": sentence_index,
                    "turn_id": record.metadata.get("turn_id"),
                    "speaker_id": str(record.metadata.get("speaker_id") or "unknown"),
                    "start": float(record.start or 0.0),
                    "end": float(record.end or 0.0),
                    "text_hash": self._compute_sentence_text_hash(text),
                    "binding_source": "auto",
                }
            )

        if payload_items:
            speaker_store_service.upsert_subtitle_speaker_links(payload_items)
            return "ok", int(len(payload_items))
        return "skipped_empty", 0

    @staticmethod
    def _try_parse_chunk_index(chunk_ref: Any) -> Optional[int]:
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

    def _resolve_chunk_uid(
        self,
        subtitle_batch: SubtitleBatch,
        chunk_sentence_indices: Sequence[ChunkSentenceIndex],
    ) -> str:
        chunk_id = str(subtitle_batch.chunk_id or "").strip()
        if chunk_id:
            return chunk_id
        if chunk_sentence_indices:
            chunk_id = str(chunk_sentence_indices[0].chunk_id or "").strip()
            if chunk_id:
                return chunk_id
        return str(subtitle_batch.chunk_id)

    def _resolve_chunk_index(
        self,
        *,
        subtitle_batch: SubtitleBatch,
        chunk_sentence_indices: Sequence[ChunkSentenceIndex],
    ) -> Optional[int]:
        if subtitle_batch.chunk_index is not None:
            return int(subtitle_batch.chunk_index)
        chunk_uid = self._resolve_chunk_uid(subtitle_batch, chunk_sentence_indices)
        return self._try_parse_chunk_index(chunk_uid)

    @staticmethod
    def _serialize_trace_dict(trace: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "sentence_index": int(trace.get("sentence_index", 0) or 0),
            "split_reason": str(trace.get("split_reason", "") or ""),
            "split_risk": str(trace.get("split_risk", "") or ""),
            "window_id": str(trace.get("window_id", "") or ""),
            "pyannote_frame_time": trace.get("pyannote_frame_time"),
            "mapped_cut_time": trace.get("mapped_cut_time"),
            "mapping_quality": str(trace.get("mapping_quality", "") or ""),
            "mapping_reason": str(trace.get("mapping_reason", "") or ""),
            "sentence_start": trace.get("sentence_start"),
            "sentence_end": trace.get("sentence_end"),
        }

    @staticmethod
    def _serialize_subtitle_item(item: SubtitleItem) -> Dict[str, Any]:
        trace = dict(item.trace or {})
        return {
            "segment_id": str(item.segment_id),
            "chunk_id": str(item.chunk_id),
            "text": str(item.text or ""),
            "text_clean": str(item.text or ""),
            "start": float(item.start or 0.0),
            "end": float(item.end or 0.0),
            "status": str(item.status or ""),
            "source": str(item.source or ""),
            "speaker_id": str(item.speaker_id or ""),
            "turn_id": str(item.turn_id or ""),
            "split_reason": str(trace.get("split_reason", "") or ""),
            "split_risk": str(trace.get("split_risk", "") or ""),
            "window_id": str(trace.get("window_id", "") or ""),
            "mapped_cut_time": trace.get("mapped_cut_time"),
            "mapping_quality": str(trace.get("mapping_quality", "") or ""),
            "mapping_reason": str(trace.get("mapping_reason", "") or ""),
        }

    @staticmethod
    def _serialize_sentence_record(*, record: SentenceRecord, chunk_id: str) -> Dict[str, Any]:
        trace = dict(record.trace or {})
        return {
            "segment_id": str(record.sentence_id),
            "chunk_id": str(chunk_id),
            "text": str(record.text or ""),
            "text_clean": str(record.text or ""),
            "start": float(record.start or 0.0),
            "end": float(record.end or 0.0),
            "status": str(record.metadata.get("status", "final") or "final"),
            "source": str(record.metadata.get("source", "") or ""),
            "speaker_id": str(record.metadata.get("speaker_id") or ""),
            "turn_id": str(record.metadata.get("turn_id") or ""),
            "source_chunk_ids": [str(item) for item in record.source_chunk_ids],
            "overlap_chunk_ids": [str(item) for item in record.overlap_chunk_ids],
            "replace_scope_chunk_ids": [str(item) for item in record.replace_scope_chunk_ids],
            "split_reason": str(trace.get("split_reason", "") or ""),
            "split_risk": str(trace.get("split_risk", "") or ""),
            "window_id": str(trace.get("window_id", "") or ""),
            "mapped_cut_time": trace.get("mapped_cut_time"),
            "mapping_quality": str(trace.get("mapping_quality", "") or ""),
            "mapping_reason": str(trace.get("mapping_reason", "") or ""),
        }


OutputProcessor = OutputLayerProcessor

__all__ = ["OutputLayerProcessor", "OutputProcessor", "OutputDispatchAdapter"]
