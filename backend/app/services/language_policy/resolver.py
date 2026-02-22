"""
语言标签解析器（阶段A基础版）。
V3.2.0+dev.20260219.07
"""

from __future__ import annotations

from typing import Optional

from app.services.language_policy.registry import get_language_policy_registry


def resolve_language_tag(
    *,
    language_hint: Optional[str],
    fallback: str = "zh",
) -> str:
    """
    解析语言标签并应用别名归一。

    Why:
    - 先在准备层统一 language_tag，再进入策略编译，避免后续多层分叉判断。
    """
    registry = get_language_policy_registry()
    normalized = registry.normalize_tag(str(language_hint or "").strip().lower())
    if normalized:
        return normalized
    return str(fallback or "zh").strip().lower() or "zh"
