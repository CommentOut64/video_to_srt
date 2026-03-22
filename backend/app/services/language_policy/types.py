"""
语言策略快照契约定义。
V3.2.0+dev.20260219.07
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Tuple


WINDOW_KIND_SINGLE_LANGUAGE = "single_language"
WINDOW_KIND_DOMINANT_WITH_ISLANDS = "dominant_with_islands"
WINDOW_KIND_TRUE_MIXED = "true_mixed"


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


@dataclass(frozen=True)
class LanguageWindowClassification:
    """窗口语言分类结果（Phase 3：single/dominant_with_islands/true_mixed）。"""

    dominant_language: str
    window_kind: str
    foreign_run_ratio: float
    can_enter_main_chain: bool
