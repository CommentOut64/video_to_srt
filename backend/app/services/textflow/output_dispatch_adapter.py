"""输出分发与统一输出处理器。"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, Optional, Sequence

from app.core.logging import resolve_loguru_logger
from app.services.alignment.types import OutputLayerInput, OutputLayerOutput, OutputTrace
from app.services.textflow.contracts import SubtitleBatch, SubtitleItem


class OutputLayerProcessor:
    """统一输出处理器：只接受统一 SubtitleBatch 并分发。"""

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
        """执行统一输出分发。"""
        subtitle_batch = data.subtitle_batch
        if subtitle_batch is None:
            raise ValueError(
                "OutputLayerProcessor 需要 subtitle_batch；"
                "sentence_segments -> SubtitleBatch 兼容回退已停用。"
            )
        output_traces = self._resolve_output_traces_from_batch(subtitle_batch)
        payload = self._output_dispatch_adapter.dispatch(
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
                    subtitle_items=subtitle_batch.items,
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
        payload: Dict[str, Any] = {
            "chunk_index": self._resolve_chunk_index(subtitle_batch),
            "chunk_uid": str(subtitle_batch.chunk_id),
            "sentence_count": int(len(subtitle_batch.items)),
            "sentence_segments": [
                self._serialize_subtitle_item(item) for item in subtitle_batch.items
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
        subtitle_items: Sequence[SubtitleItem],
        sentence_indices: Sequence[int],
    ) -> tuple[str, int]:
        speaker_store_service = self._resolve_speaker_store_service()
        if speaker_store_service is None:
            return "skipped_no_service", 0
        if not subtitle_items or not sentence_indices:
            return "skipped_empty", 0
        if len(sentence_indices) < len(subtitle_items):
            self._logger.warning(
                "输出层 speaker_store 写入索引不足: indices={} sentences={}",
                len(sentence_indices),
                len(subtitle_items),
            )

        payload_items = []
        for idx, subtitle_item in enumerate(subtitle_items):
            if idx >= len(sentence_indices):
                break
            sentence_index = int(sentence_indices[idx])
            text = str(subtitle_item.text or "")
            payload_items.append(
                {
                    "sentence_index": sentence_index,
                    "turn_id": subtitle_item.turn_id,
                    "speaker_id": str(subtitle_item.speaker_id or "unknown"),
                    "start": float(subtitle_item.start or 0.0),
                    "end": float(subtitle_item.end or 0.0),
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

    def _resolve_chunk_index(self, subtitle_batch: SubtitleBatch) -> Optional[int]:
        if subtitle_batch.chunk_index is not None:
            return int(subtitle_batch.chunk_index)
        return self._try_parse_chunk_index(subtitle_batch.chunk_id)

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


OutputProcessor = OutputLayerProcessor

__all__ = ["OutputLayerProcessor", "OutputProcessor", "OutputDispatchAdapter"]
