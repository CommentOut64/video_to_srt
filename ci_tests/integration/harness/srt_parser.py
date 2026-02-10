# -*- coding: utf-8 -*-
"""
SRT 解析器。

底层复用 backend.app.utils.text_utils 的解析和格式化函数，
对外提供类型安全的 dataclass 接口。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from app.utils.text_utils import (
    format_srt_timestamp,
    parse_srt_content,
    segments_to_srt,
)


@dataclass
class SRTEntry:
    """SRT 字幕条目。"""

    index: int
    start: float  # 秒
    end: float  # 秒
    text: str

    @property
    def duration(self) -> float:
        """条目时长（秒）。"""
        return self.end - self.start

    @property
    def start_formatted(self) -> str:
        return format_srt_timestamp(self.start)

    @property
    def end_formatted(self) -> str:
        return format_srt_timestamp(self.end)


class SRTParser:
    """SRT 文件解析器，底层委托给 text_utils。"""

    @staticmethod
    def parse_file(path: Path | str) -> List[SRTEntry]:
        """解析 SRT 文件为 SRTEntry 列表。"""
        content = Path(path).read_text(encoding="utf-8")
        return SRTParser.parse_content(content)

    @staticmethod
    def parse_content(content: str) -> List[SRTEntry]:
        """解析 SRT 字符串为 SRTEntry 列表。"""
        raw_segments = parse_srt_content(content)
        return [
            SRTEntry(
                index=seg["index"],
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
            )
            for seg in raw_segments
        ]

    @staticmethod
    def to_srt_string(entries: List[SRTEntry]) -> str:
        """将 SRTEntry 列表转为 SRT 格式字符串。"""
        segments = [
            {"index": e.index, "start": e.start, "end": e.end, "text": e.text}
            for e in entries
        ]
        return segments_to_srt(segments)
