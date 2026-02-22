"""
语言策略快照编译器。
V3.2.0+dev.20260219.07
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from app.core.logging import resolve_loguru_logger
from app.services.language_policy.registry import get_language_policy_registry
from app.services.language_policy.resolver import resolve_language_tag
from app.services.language_policy.types import LanguagePolicySnapshot
from app.services.model_runtime_config_service import get_model_runtime_config_service


_LOGGER = resolve_loguru_logger(
    None,
    __name__,
    layer="语言适配层",
    processor_name="language_policy_compiler",
)


def _merge_runtime_overrides(explicit_overrides: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    合并运行时覆盖参数。

    优先级：显式入参 > runtime.override.language_policy > runtime.effective.language_policy。
    """
    runtime_bundle = get_model_runtime_config_service().get_effective_runtime_global()
    effective = runtime_bundle.get("effective", {})
    override = runtime_bundle.get("override", {})

    merged: Dict[str, Any] = {}
    if isinstance(effective, dict):
        effective_group = effective.get("language_policy", {})
        if isinstance(effective_group, dict):
            merged.update(effective_group)
    if isinstance(override, dict):
        override_group = override.get("language_policy", {})
        if isinstance(override_group, dict):
            merged.update(override_group)
    if isinstance(explicit_overrides, Mapping):
        merged.update(dict(explicit_overrides))
    return merged


def build_language_policy_snapshot(
    *,
    language_hint: Optional[str],
    runtime_overrides: Optional[Mapping[str, Any]] = None,
) -> LanguagePolicySnapshot:
    """
    生成语言策略快照（准备层单次调用）。
    """
    merged_overrides = _merge_runtime_overrides(runtime_overrides)
    fallback_language = str(merged_overrides.get("default_language", "zh") or "zh")
    language_tag = resolve_language_tag(
        language_hint=language_hint,
        fallback=fallback_language,
    )
    registry = get_language_policy_registry()
    provider = registry.resolve(language_tag)
    snapshot = provider.build_snapshot(
        language_hint=language_tag,
        runtime_overrides=merged_overrides,
    )
    _LOGGER.debug(
        "语言策略快照编译完成: language_tag={} policy_version={}",
        snapshot.language_tag,
        snapshot.policy_version,
    )
    return snapshot
