"""
WeTextProcessing 处理器（基于 wetext 的 WFST 文本标准化）。
V3.1.2+dev.20260128.02: 增强 CharMapping 支持 identity/expand/collapse 类型。
"""

from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Literal, Tuple

from wetext import Normalizer


@dataclass
class CharMapping:
    """字符映射，用于 ITN/TN 后的时间戳对齐。

    mapping_type:
        - identity: 字符不变，1:1 映射
        - expand: 单字符扩展为多字符（如"①"→"1"）
        - collapse: 多字符折叠为少字符（如"一百二十三"→"123"）
    """

    original_range: Tuple[int, int]
    normalized_range: Tuple[int, int]
    mapping_type: Literal["identity", "expand", "collapse"] = "identity"


@dataclass
class TextNormalizationResult:
    """文本标准化结果。"""

    original_text: str
    normalized_text: str
    char_mapping: List[CharMapping]


class WeTextProcessor:
    """基于 wetext 的 WFST 处理器，支持 ITN/TN 文本标准化。

    用于中文逆文本标准化（ITN），如"一百二十三"→"123"。
    通过 CharMapping 机制保持时间戳对齐。
    """

    def __init__(
        self,
        lang: str = "zh",
        operator: Literal["itn", "tn"] = "itn",
        max_workers: int = 2,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._lang = lang
        self._operator = operator
        self._normalizer = Normalizer(lang=lang, operator=operator)
        self.logger.info("WeTextProcessor 初始化完成: lang=%s, operator=%s", lang, operator)

    async def process(self, text: str) -> TextNormalizationResult:
        """异步执行文本标准化，避免阻塞事件循环。"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._thread_pool, self.process_sync, text)

    def process_sync(self, text: str) -> TextNormalizationResult:
        """同步执行文本标准化（供外部直接调用或线程池使用）。"""
        return self._sync_process(text)

    def _sync_process(self, text: str) -> TextNormalizationResult:
        if not text:
            return TextNormalizationResult(original_text="", normalized_text="", char_mapping=[])

        try:
            # 使用 wetext 进行文本标准化
            normalized_text = self._normalizer.normalize(text)
        except Exception as exc:
            self.logger.warning("wetext 标准化失败，返回原文: %s", exc)
            normalized_text = text

        # 构建字符映射（简化版：基于长度差异的启发式映射）
        char_mapping = self._build_char_mapping(text, normalized_text)

        return TextNormalizationResult(
            original_text=text,
            normalized_text=normalized_text,
            char_mapping=char_mapping,
        )

    def _build_char_mapping(
        self, original: str, normalized: str
    ) -> List[CharMapping]:
        """构建原文到标准化文本的字符映射。

        使用 SequenceMatcher 进行差异对齐，识别：
        - identity: 相同字符
        - collapse: 多字符折叠（如"一百二十三"→"123"）
        - expand: 单字符扩展（如"①"→"1"）
        """
        mappings: List[CharMapping] = []

        # 文本相同，返回身份映射
        if original == normalized:
            for i in range(len(original)):
                mappings.append(
                    CharMapping(
                        original_range=(i, i + 1),
                        normalized_range=(i, i + 1),
                        mapping_type="identity",
                    )
                )
            return mappings

        # 使用 SequenceMatcher 进行差异对齐
        matcher = SequenceMatcher(None, original, normalized)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                # 相同部分，逐字符映射
                for offset in range(i2 - i1):
                    mappings.append(
                        CharMapping(
                            original_range=(i1 + offset, i1 + offset + 1),
                            normalized_range=(j1 + offset, j1 + offset + 1),
                            mapping_type="identity",
                        )
                    )
            elif tag == "replace":
                # 替换：根据长度判断 collapse 或 expand
                orig_len = i2 - i1
                norm_len = j2 - j1
                if orig_len > norm_len:
                    mapping_type = "collapse"
                elif orig_len < norm_len:
                    mapping_type = "expand"
                else:
                    mapping_type = "identity"
                mappings.append(
                    CharMapping(
                        original_range=(i1, i2),
                        normalized_range=(j1, j2),
                        mapping_type=mapping_type,
                    )
                )
            elif tag == "delete":
                # 删除：原文有，标准化后无
                mappings.append(
                    CharMapping(
                        original_range=(i1, i2),
                        normalized_range=(j1, j1),
                        mapping_type="collapse",
                    )
                )
            elif tag == "insert":
                # 插入：原文无，标准化后有
                mappings.append(
                    CharMapping(
                        original_range=(i1, i1),
                        normalized_range=(j1, j2),
                        mapping_type="expand",
                    )
                )

        return mappings

    def unload(self) -> None:
        """释放线程池资源。"""
        try:
            self._thread_pool.shutdown(wait=False, cancel_futures=True)
        except Exception as exc:  # pragma: no cover
            self.logger.debug("WeTextProcessor 线程池关闭失败: %s", exc)
