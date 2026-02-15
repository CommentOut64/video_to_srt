"""
L7 输出层处理器（OutputProcessor）。
V3.2.0+dev.20260207.02
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any, Callable, Dict, Optional, Sequence

from app.core.logging import resolve_loguru_logger
from app.services.alignment.types import L7Input, L7Output, OutputTrace


class OutputProcessor:
    """L7 处理器：仅负责格式化与分发，不改文本逻辑。"""

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
            layer="L7",
            processor_name="output_processor",
        )

    @staticmethod
    def _compute_sentence_text_hash(text: str) -> str:
        """计算句子文本哈希，用于 subtitle_speaker_links 追溯。"""
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _resolve_speaker_store_service(self) -> Optional[Any]:
        """延迟解析 speaker_store 依赖，避免初始化顺序耦合。"""
        if self._speaker_store_service_getter is None:
            return None
        try:
            return self._speaker_store_service_getter()
        except Exception:
            self._logger.exception("L7 获取 speaker_store_service 失败")
            return None

    def _write_speaker_links(
        self,
        *,
        sentence_segments: Sequence[Any],
        sentence_indices: Sequence[int],
    ) -> tuple[str, int]:
        """将句级 speaker/turn 绑定写入 speaker_store。"""
        speaker_store_service = self._resolve_speaker_store_service()
        if speaker_store_service is None:
            return "skipped_no_service", 0
        if not sentence_segments or not sentence_indices:
            return "skipped_empty", 0

        if len(sentence_indices) < len(sentence_segments):
            self._logger.warning(
                "L7 speaker_store 写入索引不足: indices={} sentences={}",
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

    def process(self, data: L7Input) -> L7Output:
        """执行 L7 输出分发。"""
        sentence_segments = list(data.sentence_segments or [])
        output_traces = self._resolve_output_traces(
            sentence_segments=sentence_segments,
            output_traces=data.output_traces,
        )
        self._apply_output_traces_to_sentences(
            sentence_segments=sentence_segments,
            output_traces=output_traces,
        )
        sentences_for_manager = copy.deepcopy(sentence_segments)
        output_errors = []
        replaced_sentence_indices = []
        subtitle_channel_status = "ok"
        speaker_store_channel_status = "skipped"
        speaker_link_count = 0

        try:
            replaced_sentence_indices = list(
                self._subtitle_manager.replace_chunk(data.chunk_index, sentences_for_manager)
                or []
            )
        except Exception:
            output_errors.append("E_L7_OUTPUT_CHANNEL_FAIL")
            subtitle_channel_status = "failed"
            speaker_store_channel_status = "skipped_upstream_failed"
            self._logger.exception(
                "L7 输出失败: chunk_index={} sentences={}",
                data.chunk_index,
                len(sentence_segments),
            )

        if "E_L7_OUTPUT_CHANNEL_FAIL" not in output_errors:
            try:
                speaker_store_channel_status, speaker_link_count = self._write_speaker_links(
                    sentence_segments=sentence_segments,
                    sentence_indices=replaced_sentence_indices,
                )
            except Exception:
                output_errors.append("E_L7_SPEAKER_STORE_FAIL")
                speaker_store_channel_status = "failed"
                self._logger.exception(
                    "L7 speaker_store 写入失败: chunk_index={} sentences={}",
                    data.chunk_index,
                    len(sentence_segments),
                )

        payload: Dict[str, Any] = {
            "chunk_index": int(data.chunk_index),
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
            "injection_report": dict(data.injection_report or {}),
            "segmentation_report": dict(data.segmentation_report or {}),
            "output_trace": [self._serialize_output_trace(item) for item in output_traces],
            "errors": output_errors,
        }
        self._logger.info(
            "L7 输出完成: chunk_index={} sentences={} errors={}",
            data.chunk_index,
            len(sentence_segments),
            len(output_errors),
        )
        return L7Output(
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

