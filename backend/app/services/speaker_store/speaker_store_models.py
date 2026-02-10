"""
Speaker Store 领域数据模型。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeakerProfileView:
    """前端可读的说话人资料视图。"""

    speaker_id: str
    display_name: str
    color_key: str
    status: str
    sample_count: int
    subtitle_count: int


@dataclass(frozen=True)
class SubtitleSpeakerTrace:
    """字幕到说话人绑定追溯项。"""

    sentence_index: int
    text: str
    start: float
    end: float
    speaker_id: str
    turn_id: str | None
    speaker_color_key: str
    binding_source: str

