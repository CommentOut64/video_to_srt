"""
L7 输出层处理器（OutputProcessor）。
V3.2.0+dev.20260207.02
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from app.core.logging import resolve_loguru_logger
from app.services.alignment.types import L7Input, L7Output


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

        try:
            self._subtitle_manager.replace_chunk(data.chunk_index, sentences_for_manager)
        except Exception:
            output_errors.append("E_L7_OUTPUT_CHANNEL_FAIL")
            self._logger.exception(
                "L7 输出失败: chunk_index={} sentences={}",
                data.chunk_index,
                len(sentence_segments),
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

