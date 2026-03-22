"""
语言标签解析器（阶段A基础版）。
V3.2.0+dev.20260219.07
"""

from __future__ import annotations

from typing import Mapping, Optional

from app.services.language_policy.registry import get_language_policy_registry
from app.services.language_policy.types import (
    LanguageWindowClassification,
    WINDOW_KIND_DOMINANT_WITH_ISLANDS,
    WINDOW_KIND_SINGLE_LANGUAGE,
    WINDOW_KIND_TRUE_MIXED,
)


def resolve_language_tag(
    *,
    language_hint: Optional[str],
    fallback: str = "zh",
    preserve_mixed: bool = False,
) -> str:
    """
    解析语言标签并应用别名归一。

    Why:
    - 先在准备层统一 language_tag，再进入策略编译，避免后续多层分叉判断。
    """
    raw_hint = str(language_hint or "").strip().lower()
    if preserve_mixed and raw_hint == "mixed":
        return "mixed"

    registry = get_language_policy_registry()
    normalized = registry.normalize_tag(raw_hint)
    if normalized:
        return normalized

    fallback_tag = str(fallback or "zh").strip().lower() or "zh"
    if preserve_mixed and fallback_tag == "mixed":
        return "mixed"
    normalized_fallback = registry.normalize_tag(fallback_tag)
    return normalized_fallback or "zh"


def classify_window_language_distribution(
    *,
    language_char_counts: Mapping[str, int],
    dominant_language_min_ratio: float,
    foreign_run_max_ratio: float,
    foreign_run_char_max: int,
) -> LanguageWindowClassification:
    """
    窗口语言分类（Phase 3）。

    返回：
    - single_language
    - dominant_with_islands
    - true_mixed
    """
    sanitized_counts: dict[str, int] = {}
    for language, count in dict(language_char_counts or {}).items():
        normalized_language = str(language or "").strip().lower()
        if not normalized_language:
            continue
        try:
            normalized_count = int(count)
        except (TypeError, ValueError):
            continue
        if normalized_count > 0:
            sanitized_counts[normalized_language] = normalized_count

    if not sanitized_counts:
        return LanguageWindowClassification(
            dominant_language="mixed",
            window_kind=WINDOW_KIND_TRUE_MIXED,
            foreign_run_ratio=1.0,
            can_enter_main_chain=False,
        )

    total_chars = float(sum(sanitized_counts.values()))
    dominant_language, dominant_chars = max(
        sanitized_counts.items(),
        key=lambda item: item[1],
    )
    dominant_ratio = float(dominant_chars) / total_chars if total_chars > 0 else 0.0
    foreign_chars = max(int(total_chars - float(dominant_chars)), 0)
    foreign_ratio = float(foreign_chars) / total_chars if total_chars > 0 else 0.0

    if foreign_chars <= 0:
        return LanguageWindowClassification(
            dominant_language=dominant_language,
            window_kind=WINDOW_KIND_SINGLE_LANGUAGE,
            foreign_run_ratio=0.0,
            can_enter_main_chain=True,
        )

    can_be_islands = (
        dominant_ratio >= float(dominant_language_min_ratio)
        and foreign_ratio <= float(foreign_run_max_ratio)
        and foreign_chars <= int(foreign_run_char_max)
    )
    if can_be_islands:
        return LanguageWindowClassification(
            dominant_language=dominant_language,
            window_kind=WINDOW_KIND_DOMINANT_WITH_ISLANDS,
            foreign_run_ratio=foreign_ratio,
            can_enter_main_chain=True,
        )

    return LanguageWindowClassification(
        dominant_language="mixed",
        window_kind=WINDOW_KIND_TRUE_MIXED,
        foreign_run_ratio=foreign_ratio,
        can_enter_main_chain=False,
    )
