"""
Bridge 层 TurnGroup 契约模型。

Phase 0 仅定义输入输出契约，不绑定调度策略实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TurnGroup:
    """慢流 Whisper 的最小处理单元。"""

    group_id: str
    target_turn_ids: list[str] = field(default_factory=list)
    context_turn_ids: list[str] = field(default_factory=list)
    speaker_id: str = ""
    audio_segments: list[tuple[float, float]] = field(default_factory=list)
    prompt_text: str = ""
    flush_reason: str = ""
    language: str = "auto"
    source_chunks: list[str] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        """TurnGroup 覆盖的总时长。"""
        if not self.audio_segments:
            return 0.0
        start = min(segment[0] for segment in self.audio_segments)
        end = max(segment[1] for segment in self.audio_segments)
        return max(0.0, float(end) - float(start))
