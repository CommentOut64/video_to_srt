"""
Runtime checkpoint 数据模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class UnitJournalEntry:
    """单元状态日志项。"""

    stage: str
    unit_id: str
    status: str
    payload_json: str | None = None


@dataclass(frozen=True)
class RuntimeCheckpointSnapshot:
    """运行时状态快照（Phase 1 仅包含最小提交信息）。"""

    last_unit_commits: dict[str, str] = field(default_factory=dict)
    # Phase 2: 仅保存转录摘要提示，不复制 checkpoint 全量数据
    transcription_hint: dict[str, Any] | None = None
    # V3.2.4+dev.20260228.01: 运行态字幕真源快照（草稿/定稿统一基线）
    subtitle_runtime: dict[str, Any] | None = None

