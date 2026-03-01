"""
语言策略注册表。
V3.2.0+dev.20260219.08
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple

from app.core.logging import resolve_loguru_logger
from app.services.language_policy.loader import (
    get_language_policy_config,
    load_language_lexicon,
)
from app.services.language_policy.types import LanguagePolicySnapshot


_LOGGER = resolve_loguru_logger(
    None,
    __name__,
    layer="语言适配层",
    processor_name="language_policy_registry",
)


class LanguagePolicyProvider(Protocol):
    """策略提供者协议。"""

    def build_snapshot(
        self,
        *,
        language_hint: str,
        runtime_overrides: Mapping[str, Any],
    ) -> LanguagePolicySnapshot:
        ...


@dataclass
class StaticLanguagePolicyProvider:
    """基于 YAML + TXT 静态资产的 provider。"""

    language_tag: str

    def build_snapshot(
        self,
        *,
        language_hint: str,
        runtime_overrides: Mapping[str, Any],
    ) -> LanguagePolicySnapshot:
        config = get_language_policy_config()
        language_cfg = (
            config.get("languages", {}).get(self.language_tag, {})
            if isinstance(config, dict)
            else {}
        )
        lexicon_cfg = language_cfg.get("lexicon", {}) if isinstance(language_cfg, dict) else {}

        continuation_words = load_language_lexicon(
            language_tag=self.language_tag,
            filename=str(lexicon_cfg.get("continuation_words", "continuation_words.txt")),
        )
        incomplete_endings = load_language_lexicon(
            language_tag=self.language_tag,
            filename=str(lexicon_cfg.get("incomplete_endings", "incomplete_endings.txt")),
        )
        semantic_anchor_words = load_language_lexicon(
            language_tag=self.language_tag,
            filename=str(lexicon_cfg.get("semantic_anchor_words", "semantic_anchor_words.txt")),
        )

        sentence_end_chars_raw = language_cfg.get("sentence_end_chars", "")
        if isinstance(sentence_end_chars_raw, str):
            sentence_end_chars = frozenset(ch for ch in sentence_end_chars_raw if ch.strip())
        elif isinstance(sentence_end_chars_raw, (list, tuple, set)):
            sentence_end_chars = frozenset(str(item) for item in sentence_end_chars_raw if str(item))
        else:
            sentence_end_chars = frozenset()

        thresholds: Dict[str, float] = {}
        raw_thresholds = language_cfg.get("thresholds", {})
        if isinstance(raw_thresholds, dict):
            for key, value in raw_thresholds.items():
                try:
                    thresholds[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue

        carry_rules: Dict[str, Tuple[float, float]] = {}
        raw_carry_rules = language_cfg.get("carry_rules", {})
        if isinstance(raw_carry_rules, dict):
            for key, value in raw_carry_rules.items():
                if isinstance(value, (list, tuple)) and len(value) >= 2:
                    try:
                        carry_rules[str(key)] = (float(value[0]), float(value[1]))
                    except (TypeError, ValueError):
                        continue

        runtime_thresholds = runtime_overrides.get("thresholds")
        if isinstance(runtime_thresholds, dict):
            for key, value in runtime_thresholds.items():
                try:
                    thresholds[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue

        runtime_carry_rules = runtime_overrides.get("carry_rules")
        if isinstance(runtime_carry_rules, dict):
            for key, value in runtime_carry_rules.items():
                if isinstance(value, (list, tuple)) and len(value) >= 2:
                    try:
                        carry_rules[str(key)] = (float(value[0]), float(value[1]))
                    except (TypeError, ValueError):
                        continue

        cross_chunk_config: Dict[str, Any] = {}
        raw_cross_chunk = language_cfg.get("cross_chunk", {})
        if isinstance(raw_cross_chunk, dict):
            cross_chunk_config.update(raw_cross_chunk)
        runtime_cross_chunk = runtime_overrides.get("cross_chunk")
        if isinstance(runtime_cross_chunk, dict):
            cross_chunk_config.update(runtime_cross_chunk)

        policy_version = str(
            runtime_overrides.get("policy_version")
            or config.get("policy_version")
            or "unknown"
        )
        return LanguagePolicySnapshot(
            policy_version=policy_version,
            language_tag=self.language_tag,
            sentence_end_chars=sentence_end_chars,
            continuation_words=continuation_words,
            incomplete_endings=incomplete_endings,
            semantic_anchor_words=semantic_anchor_words,
            thresholds=thresholds,
            carry_rules=carry_rules,
            metadata={
                "provider": "static_language_policy_provider",
                "language_hint": str(language_hint or ""),
                "cross_chunk": cross_chunk_config,
            },
        )


class LanguagePolicyRegistry:
    """语言策略注册表。"""

    def __init__(
        self,
        *,
        alias_map: Optional[Mapping[str, str]] = None,
        default_language: str = "zh",
    ) -> None:
        self._providers: Dict[str, LanguagePolicyProvider] = {}
        self._alias_map: Dict[str, str] = {
            str(key).strip().lower(): str(value).strip().lower()
            for key, value in dict(alias_map or {}).items()
            if str(key).strip() and str(value).strip()
        }
        self._default_language = str(default_language or "zh").strip().lower()

    def register(self, language_tag: str, provider: LanguagePolicyProvider) -> None:
        tag = self.normalize_tag(language_tag)
        self._providers[tag] = provider

    def normalize_tag(self, language_tag: str) -> str:
        raw = str(language_tag or "").strip().lower()
        if not raw:
            return self._default_language
        return self._alias_map.get(raw, raw)

    def resolve(self, language_tag: str) -> LanguagePolicyProvider:
        normalized = self.normalize_tag(language_tag)
        provider = self._providers.get(normalized)
        if provider is not None:
            return provider
        default_provider = self._providers.get(self._default_language)
        if default_provider is not None:
            return default_provider
        if self._providers:
            return next(iter(self._providers.values()))
        raise RuntimeError("LanguagePolicyRegistry 未注册任何 provider")


_REGISTRY: Optional[LanguagePolicyRegistry] = None


def build_default_language_policy_registry() -> LanguagePolicyRegistry:
    config = get_language_policy_config()
    default_language = str(config.get("default_language", "zh")).strip().lower()
    aliases = config.get("language_aliases", {})
    registry = LanguagePolicyRegistry(
        alias_map=aliases if isinstance(aliases, dict) else {},
        default_language=default_language,
    )

    languages = config.get("languages", {})
    if isinstance(languages, dict):
        for language_tag in languages.keys():
            normalized = registry.normalize_tag(str(language_tag))
            registry.register(
                normalized,
                StaticLanguagePolicyProvider(language_tag=normalized),
            )

    # 兜底确保默认语言可用。
    if default_language not in registry._providers:
        registry.register(default_language, StaticLanguagePolicyProvider(language_tag=default_language))
    _LOGGER.info("语言策略注册完成: languages={}", sorted(registry._providers.keys()))
    return registry


def get_language_policy_registry() -> LanguagePolicyRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = build_default_language_policy_registry()
    return _REGISTRY


def reset_language_policy_registry() -> None:
    global _REGISTRY
    _REGISTRY = None
