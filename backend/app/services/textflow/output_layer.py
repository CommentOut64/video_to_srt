"""
输出层统一入口（真实实现）。
V3.2.0+dev.20260215.24
"""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Any, Callable, Dict, Optional, Sequence

from app.core.logging import resolve_loguru_logger
from app.services.alignment.types import L7Input, L7Output, OutputTrace


class OutputLayerProcessor:
    """输出层处理器：负责最终分发，并在定稿出口做标点风格标准化。"""

    _HALF_TO_FULL_PUNCT = {
        ",": "，",
        ".": "。",
        "!": "！",
        "?": "？",
        ";": "；",
        ":": "：",
        "(": "（",
        ")": "）",
        "[": "【",
        "]": "】",
        "{": "｛",
        "}": "｝",
    }
    _FULL_TO_HALF_PUNCT = {
        value: key for key, value in _HALF_TO_FULL_PUNCT.items()
    }
    _FULL_TO_HALF_PUNCT.update(
        {
            "．": ".",
            "、": ",",
            "“": "\"",
            "”": "\"",
            "‘": "'",
            "’": "'",
        }
    )
    _EN_RIGHT_PUNCT = {",", ".", "!", "?", ";", ":", ")", "]", "}"}
    _EN_LEFT_PUNCT = {"(", "[", "{"}

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
            self._logger.exception("输出层获取 speaker_store_service 失败")
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

    def process(self, data: L7Input) -> L7Output:
        """执行输出层分发。"""
        sentence_segments = list(data.sentence_segments or [])
        self._apply_language_punctuation_standardization(
            sentence_segments=sentence_segments,
            language=data.language,
        )
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
                "输出层分发失败: chunk_index={} sentences={}",
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
                    "输出层 speaker_store 写入失败: chunk_index={} sentences={}",
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
            "输出层完成: chunk_index={} sentences={} errors={}",
            data.chunk_index,
            len(sentence_segments),
            len(output_errors),
        )
        return L7Output(
            output_payload=payload,
            output_traces=output_traces,
        )

    def _apply_language_punctuation_standardization(
        self,
        *,
        sentence_segments: Sequence[Any],
        language: str,
    ) -> None:
        """
        在最终出稿前按语言标准化标点样式。

        约束:
        - 仅处理定稿（草稿跳过）。
        - 在最终标点决策之后执行，避免影响上游评分/裁决。
        """
        for sentence in sentence_segments:
            if bool(getattr(sentence, "is_draft", False)):
                continue
            source_language = str(getattr(sentence, "language", "") or language or "auto")
            normalized_text = self._standardize_text_by_language(
                text=str(getattr(sentence, "text", "") or ""),
                language=source_language,
            )
            if normalized_text:
                setattr(sentence, "text", normalized_text)

            text_clean = str(getattr(sentence, "text_clean", "") or "")
            if text_clean:
                normalized_clean = self._standardize_text_by_language(
                    text=text_clean,
                    language=source_language,
                )
                if normalized_clean:
                    setattr(sentence, "text_clean", normalized_clean)

    def _standardize_text_by_language(self, *, text: str, language: str) -> str:
        if not text:
            return text
        lang = self._resolve_language_tag(language=language, text=text)
        if lang == "en":
            return self._standardize_english_punctuation(text)
        if lang in {"zh", "ja"}:
            return self._standardize_cjk_punctuation(text)
        return text

    @staticmethod
    def _resolve_language_tag(*, language: str, text: str) -> str:
        tag = str(language or "auto").lower()
        if tag.startswith("en"):
            return "en"
        if tag.startswith("zh") or tag.startswith("yue"):
            return "zh"
        if tag.startswith("ja") or tag.startswith("jp"):
            return "ja"

        has_cjk = bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", text))
        has_latin = bool(re.search(r"[A-Za-z]", text))
        if has_cjk and not has_latin:
            return "zh"
        if has_latin and not has_cjk:
            return "en"
        return "auto"

    def _standardize_english_punctuation(self, text: str) -> str:
        normalized = self._to_halfwidth_punctuation(text)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = re.sub(r"\s+([,.;:!?\)\]\}])", r"\1", normalized)
        normalized = re.sub(r"([\(\[\{])\s+", r"\1", normalized)

        chars = list(normalized)
        out: list[str] = []
        length = len(chars)
        for index, char in enumerate(chars):
            out.append(char)
            if index >= length - 1:
                continue
            next_char = chars[index + 1]
            if next_char.isspace():
                continue
            if char in {",", ";", ":", "!", "?"} and next_char not in self._EN_RIGHT_PUNCT:
                out.append(" ")
            elif (
                char == "."
                and self._should_insert_space_after_dot(
                    prev_char=chars[index - 1] if index > 0 else "",
                    next_char=next_char,
                )
            ):
                out.append(" ")

        normalized = "".join(out)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = re.sub(r"\s+([,.;:!?\)\]\}])", r"\1", normalized)
        normalized = re.sub(r"([\(\[\{])\s+", r"\1", normalized)
        return normalized

    @staticmethod
    def _should_insert_space_after_dot(*, prev_char: str, next_char: str) -> bool:
        if not prev_char or not next_char:
            return False
        if prev_char.isdigit() and next_char.isdigit():
            return False
        if prev_char.isalpha() and next_char.isalpha():
            # 缩写/省略场景保守处理，不强插空格。
            if prev_char.isupper() and next_char.isupper():
                return False
            return prev_char.islower() and next_char.isupper()
        return next_char.isalnum()

    def _standardize_cjk_punctuation(self, text: str) -> str:
        normalized = self._to_fullwidth_punctuation(text)
        normalized = re.sub(r"\s*([，。！？；：、）】》」』｝])\s*", r"\1", normalized)
        normalized = re.sub(r"\s*([（【《「『｛])\s*", r"\1", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _to_halfwidth_punctuation(self, text: str) -> str:
        normalized = text
        for source, target in self._FULL_TO_HALF_PUNCT.items():
            normalized = normalized.replace(source, target)
        return normalized

    def _to_fullwidth_punctuation(self, text: str) -> str:
        normalized = text
        for source, target in self._HALF_TO_FULL_PUNCT.items():
            normalized = normalized.replace(source, target)
        return normalized

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


# 兼容旧命名（用于过渡期）。
OutputProcessor = OutputLayerProcessor

__all__ = ["OutputLayerProcessor", "OutputProcessor"]
