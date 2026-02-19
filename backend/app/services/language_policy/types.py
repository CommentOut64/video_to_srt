"""
语言策略快照契约定义。
V3.2.0+dev.20260219.07
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Tuple


@dataclass(frozen=True)
class LanguagePolicySnapshot:
    """
    语言策略快照（只读）。

    Why:
    - 由准备层编译一次后下传四层，避免各层重复读取配置。
    - 保留 policy_version，保障离线回放与问题复盘可复现。
    """

    policy_version: str
    language_tag: str
    sentence_end_chars: FrozenSet[str] = field(default_factory=frozenset)
    continuation_words: FrozenSet[str] = field(default_factory=frozenset)
    incomplete_endings: FrozenSet[str] = field(default_factory=frozenset)
    semantic_anchor_words: FrozenSet[str] = field(default_factory=frozenset)
    thresholds: Dict[str, float] = field(default_factory=dict)
    carry_rules: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
