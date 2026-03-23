"""OutputDispatchAdapter（Phase 5）。"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Callable, Dict, Optional, Sequence

from app.core.logging import resolve_loguru_logger
from app.services.alignment.types import OutputTrace


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
        chunk_index: Any,
        sentence_segments: Sequence[Any],
        output_traces: Sequence[OutputTrace],
        injection_report: Optional[Dict[str, Any]] = None,
        segmentation_report: Optional[Dict[str, Any]] = None,
        unknown_sentence_filtered_count: int = 0,
    ) -> Dict[str, Any]:
        sentences_for_manager = copy.deepcopy(list(sentence_segments or []))
        output_errors: list[str] = []
        replaced_sentence_indices: list[int] = []
        subtitle_channel_status = "ok"
        speaker_store_channel_status = "skipped"
        speaker_link_count = 0

        try:
            replaced_sentence_indices = list(
                self._subtitle_manager.replace_chunk(chunk_index, sentences_for_manager) or []
            )
        except Exception:
            output_errors.append("E_OUTPUT_CHANNEL_FAIL")
            subtitle_channel_status = "failed"
            speaker_store_channel_status = "skipped_upstream_failed"
            self._logger.exception(
                "输出层分发失败: chunk_index={} sentences={}",
                chunk_index,
                len(sentence_segments),
            )

        if "E_OUTPUT_CHANNEL_FAIL" not in output_errors:
            try:
                speaker_store_channel_status, speaker_link_count = self._write_speaker_links(
                    sentence_segments=sentence_segments,
                    sentence_indices=replaced_sentence_indices,
                )
            except Exception:
                output_errors.append("E_OUTPUT_SPEAKER_STORE_FAIL")
                speaker_store_channel_status = "failed"
                self._logger.exception(
                    "输出层 speaker_store 写入失败: chunk_index={} sentences={}",
                    chunk_index,
                    len(sentence_segments),
                )

        segmentation_payload = dict(segmentation_report or {})
        segmentation_payload["unknown_sentence_filtered_count"] = int(unknown_sentence_filtered_count)
        payload: Dict[str, Any] = {
            "chunk_index": self._try_parse_chunk_index(chunk_index),
            "chunk_uid": str(chunk_index),
            "sentence_count": int(len(sentence_segments)),
            "sentence_segments": [
                self._serialize_sentence_segment(item) for item in sentence_segments
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
            "output_trace": [self._serialize_output_trace(item) for item in output_traces],
            "errors": output_errors,
        }
        self._logger.info(
            "输出分发完成: chunk_index={} sentences={} filtered_unknown={} errors={}",
            chunk_index,
            len(sentence_segments),
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
        sentence_segments: Sequence[Any],
        sentence_indices: Sequence[int],
    ) -> tuple[str, int]:
        speaker_store_service = self._resolve_speaker_store_service()
        if speaker_store_service is None:
            return "skipped_no_service", 0
        if not sentence_segments or not sentence_indices:
            return "skipped_empty", 0
        if len(sentence_indices) < len(sentence_segments):
            self._logger.warning(
                "输出层 speaker_store 写入索引不足: indices={} sentences={}",
                len(sentence_indices),
                len(sentence_segments),
            )

        payload_items = []
        for idx, sentence in enumerate(sentence_segments):
            if idx >= len(sentence_indices):
                break
            sentence_index = int(sentence_indices[idx])
            text = str(getattr(sentence, "text_clean", "") or getattr(sentence, "text", "") or "")
            payload_items.append(
                {
                    "sentence_index": sentence_index,
                    "turn_id": getattr(sentence, "turn_id", None),
                    "speaker_id": str(getattr(sentence, "speaker_id", None) or "unknown"),
                    "start": float(getattr(sentence, "start", 0.0) or 0.0),
                    "end": float(getattr(sentence, "end", 0.0) or 0.0),
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

    @staticmethod
    def _serialize_output_trace(trace: OutputTrace) -> Dict[str, Any]:
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

    @staticmethod
    def _serialize_sentence_segment(sentence: Any) -> Dict[str, Any]:
        return {
            "text": str(getattr(sentence, "text", "") or ""),
            "text_clean": str(getattr(sentence, "text_clean", "") or ""),
            "start": float(getattr(sentence, "start", 0.0) or 0.0),
            "end": float(getattr(sentence, "end", 0.0) or 0.0),
            "speaker_id": str(getattr(sentence, "speaker_id", "") or ""),
            "turn_id": str(getattr(sentence, "turn_id", "") or ""),
            "split_reason": str(getattr(sentence, "split_reason", "") or ""),
            "split_risk": str(getattr(sentence, "split_risk", "") or ""),
            "window_id": str(getattr(sentence, "window_id", "") or ""),
            "mapped_cut_time": getattr(sentence, "mapped_cut_time", None),
            "mapping_quality": str(getattr(sentence, "mapping_quality", "") or ""),
            "mapping_reason": str(getattr(sentence, "mapping_reason", "") or ""),
        }

