"""
Runtime checkpoint 数据模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field


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

