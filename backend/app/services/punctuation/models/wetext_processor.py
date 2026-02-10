"""
WeTextProcessing 处理器（轻量兜底实现）。
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CharMapping:
    """字符映射信息。"""

    original_range: Tuple[int, int]
    normalized_range: Tuple[int, int]
    mapping_type: str


@dataclass
class TextNormalizationResult:
    """文本标准化结果。"""

    original_text: str
    normalized_text: str
    char_mapping: List[CharMapping]


class WeTextProcessor:
    """中文 ITN 处理器（当前为透传实现）。"""

    def __init__(self) -> None:
        # V3.2.0+dev.20260129.02: 预留线程池避免 GIL 阻塞
        self._thread_pool = ThreadPoolExecutor(max_workers=2)

    async def process(self, text: str) -> TextNormalizationResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._thread_pool, self._sync_process, text)

    def _sync_process(self, text: str) -> TextNormalizationResult:
        """同步处理入口（当前仅做透传与身份映射）。"""
        char_mapping = [
            CharMapping(
                original_range=(idx, idx + 1),
                normalized_range=(idx, idx + 1),
                mapping_type="identity",
            )
            for idx in range(len(text))
        ]
        return TextNormalizationResult(
            original_text=text,
            normalized_text=text,
            char_mapping=char_mapping,
        )
