"""
语言策略快照编译器。
V3.2.0+dev.20260219.07
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from app.core.logging import resolve_loguru_logger
from app.services.language_policy.registry import get_language_policy_registry
from app.services.language_policy.resolver import (
    classify_window_language_distribution,
    resolve_language_tag,
)
from app.services.language_policy.types import (
    LanguageWindowClassification,
    LanguagePolicySnapshot,
    WINDOW_KIND_TRUE_MIXED,
)
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
    feature_scope: str = "legacy",
) -> LanguagePolicySnapshot:
    """
    生成语言策略快照（准备层单次调用）。
    """
    normalized_scope = str(feature_scope or "legacy").strip().lower() or "legacy"
    is_timeanchored_scope = normalized_scope == "timeanchored_alignment"
    merged_overrides = _merge_runtime_overrides(runtime_overrides)
    fallback_language = str(merged_overrides.get("default_language", "zh") or "zh")
    language_tag = resolve_language_tag(
        language_hint=language_hint,
        fallback=fallback_language,
        preserve_mixed=is_timeanchored_scope,
    )
    registry = get_language_policy_registry()
    provider = registry.resolve(
        language_tag,
        preserve_unknown=is_timeanchored_scope,
    )
    snapshot = provider.build_snapshot(
        language_hint=language_tag,
        runtime_overrides=merged_overrides,
    )
    if is_timeanchored_scope:
        decorated_snapshot = _decorate_timeanchored_snapshot(snapshot)
        _LOGGER.debug(
            "语言策略快照编译完成: language_tag={} policy_version={} scope={} window_kind={}",
            decorated_snapshot.language_tag,
            decorated_snapshot.policy_version,
            normalized_scope,
            decorated_snapshot.metadata.get("window_kind", ""),
        )
        return decorated_snapshot
    _LOGGER.debug(
        "语言策略快照编译完成: language_tag={} policy_version={} scope={}",
        snapshot.language_tag,
        snapshot.policy_version,
        normalized_scope,
    )
    return snapshot


def _decorate_timeanchored_snapshot(snapshot: LanguagePolicySnapshot) -> LanguagePolicySnapshot:
    thresholds = dict(snapshot.thresholds or {})
    dominant_language_min_ratio = _safe_float(thresholds.get("dominant_language_min_ratio"), 0.55)
    foreign_run_max_ratio = _safe_float(thresholds.get("foreign_run_max_ratio"), 0.50)
    foreign_run_char_max = _safe_int(thresholds.get("foreign_run_char_max"), 24)

    language_tag = str(snapshot.language_tag or "").strip().lower()
    if language_tag == "mixed":
        classification = LanguageWindowClassification(
            dominant_language="mixed",
            window_kind=WINDOW_KIND_TRUE_MIXED,
            foreign_run_ratio=1.0,
            can_enter_main_chain=False,
        )
    else:
        char_counts = {language_tag: 1} if language_tag else {}
        classification = classify_window_language_distribution(
            language_char_counts=char_counts,
            dominant_language_min_ratio=dominant_language_min_ratio,
            foreign_run_max_ratio=foreign_run_max_ratio,
            foreign_run_char_max=foreign_run_char_max,
        )
    main_languages = {"zh", "ja", "en"}
    main_chain_eligible = (
        classification.can_enter_main_chain
        and classification.window_kind != WINDOW_KIND_TRUE_MIXED
        and classification.dominant_language in main_languages
    )

    metadata = dict(snapshot.metadata or {})
    metadata.update(
        {
            "feature_scope": "timeanchored_alignment",
            "dominant_language": classification.dominant_language,
            "window_kind": classification.window_kind,
            "foreign_run_ratio": classification.foreign_run_ratio,
            "timeanchored_main_chain_eligible": bool(main_chain_eligible),
        }
    )

    return LanguagePolicySnapshot(
        policy_version=snapshot.policy_version,
        language_tag=snapshot.language_tag,
        sentence_end_chars=snapshot.sentence_end_chars,
        continuation_words=snapshot.continuation_words,
        incomplete_endings=snapshot.incomplete_endings,
        semantic_anchor_words=snapshot.semantic_anchor_words,
        thresholds=thresholds,
        carry_rules=dict(snapshot.carry_rules or {}),
        metadata=metadata,
    )


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
