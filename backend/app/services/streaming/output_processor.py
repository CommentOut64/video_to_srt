"""
L7 输出层处理器（OutputProcessor）。
V3.2.0+dev.20260210.03
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, Optional

from app.core.logging import resolve_loguru_logger
from app.services.alignment.types import L7Input, L7Output
from app.services.homophone.runtime import get_homophone_service, index_chunk_async
from app.services.homophone.service import SentenceRecord


class OutputProcessor:
    """L7 处理器：仅负责格式化与分发，不改文本逻辑。"""

    def __init__(self, *, subtitle_manager: Any, logger: Optional[Any] = None) -> None:
        self._subtitle_manager = subtitle_manager
        self._logger = resolve_loguru_logger(
            logger,
            __name__,
            layer="L7",
            processor_name="output_processor",
        )

    def process(self, data: L7Input) -> L7Output:
        """执行 L7 输出分发。"""
        sentence_segments = list(data.sentence_segments or [])
        sentences_for_manager = copy.deepcopy(sentence_segments)
        output_errors = []
        new_indices = []

        self._apply_global_term_replacements(sentences_for_manager)

        try:
            new_indices = self._subtitle_manager.replace_chunk(data.chunk_index, sentences_for_manager)
        except Exception:
            output_errors.append("E_L7_OUTPUT_CHANNEL_FAIL")
            self._logger.exception(
                "L7 输出失败: chunk_index={} sentences={}",
                data.chunk_index,
                len(sentence_segments),
            )

        if not output_errors:
            self._try_schedule_homophone_index(
                chunk_index=data.chunk_index,
                sentence_segments=sentences_for_manager,
                new_indices=new_indices,
            )

        payload: Dict[str, Any] = {
            "chunk_index": int(data.chunk_index),
            "sentence_count": int(len(sentence_segments)),
            "transport_meta": {
                "channels": ["streaming_subtitle.replace_chunk"],
            },
            "injection_report": dict(data.injection_report or {}),
            "segmentation_report": dict(data.segmentation_report or {}),
            "errors": output_errors,
        }
        self._logger.info(
            "L7 输出完成: chunk_index={} sentences={} errors={}",
            data.chunk_index,
            len(sentence_segments),
            len(output_errors),
        )
        return L7Output(output_payload=payload)

    def _try_schedule_homophone_index(
        self,
        *,
        chunk_index: int,
        sentence_segments: list,
        new_indices: list,
    ) -> None:
        try:
            job_id = str(getattr(self._subtitle_manager, "job_id", "") or "")
            if not job_id:
                return
            sentence_records = self._build_sentence_records(sentence_segments, new_indices)
            if not sentence_records:
                return
            homophone_service = get_homophone_service()
            status = homophone_service.get_index_status(job_id)
            revision = int(status.revision) if status else 1
            language_hint = self._detect_language([item.text for item in sentence_records])
            index_chunk_async(
                job_id=job_id,
                revision=revision,
                chunk_index=int(chunk_index),
                language_hint=language_hint,
                sentences=sentence_records,
            )
        except Exception:
            self._logger.exception(
                "同音侧车触发失败: chunk_index={} sentence_count={}",
                chunk_index,
                len(sentence_segments),
            )

    def _build_sentence_records(
        self,
        sentence_segments: list,
        new_indices: list,
    ) -> list[SentenceRecord]:
        records: list[SentenceRecord] = []
        if new_indices and len(new_indices) >= len(sentence_segments):
            for position, sentence in enumerate(sentence_segments):
                text = str(getattr(sentence, "text_clean", None) or getattr(sentence, "text", "") or "")
                records.append(
                    SentenceRecord(
                        index=int(new_indices[position]),
                        text=text,
                    )
                )
            return records

        for position, sentence in enumerate(sentence_segments):
            text = str(getattr(sentence, "text_clean", None) or getattr(sentence, "text", "") or "")
            records.append(
                SentenceRecord(
                    index=position,
                    text=text,
                )
            )
        return records

    def _apply_global_term_replacements(self, sentence_segments: list) -> None:
        if not sentence_segments:
            return
        try:
            homophone_service = get_homophone_service()
            language_hint = self._detect_language(
                [str(getattr(item, "text_clean", None) or getattr(item, "text", "") or "") for item in sentence_segments]
            )
            for sentence in sentence_segments:
                original_text = str(getattr(sentence, "text_clean", None) or getattr(sentence, "text", "") or "")
                replaced_text = homophone_service.apply_global_terms(
                    text=original_text,
                    language=language_hint,
                    is_modified=bool(getattr(sentence, "is_modified", False)),
                )
                if replaced_text == original_text:
                    continue
                sentence.text = replaced_text
                sentence.text_clean = replaced_text
        except Exception:
            self._logger.exception("全局术语自动替换失败，已降级为原文输出")

    @staticmethod
    def _detect_language(texts: list[str]) -> str:
        text = "\n".join(texts)
        if re.search(r"[ぁ-んァ-ン]", text):
            return "ja"
        if re.search(r"[\u4e00-\u9fff]", text):
            return "zh"
        if re.search(r"[A-Za-z]", text):
            return "en"
        return "zh"

