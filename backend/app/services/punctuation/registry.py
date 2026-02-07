"""
标点策略注册表。
V3.2.0+dev.20260129.02
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.services.punctuation.base import PunctuationStrategy


def normalize_language(language: Optional[str]) -> str:
    """规范化语言标签（仅保留主语言）。"""
    if not language:
        return ""
    value = str(language).strip().lower()
    if not value:
        return ""
    value = value.replace("_", "-")
    return value.split("-", 1)[0]


class PunctuationRegistry:
    """管理语言到标点策略的映射。"""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._strategies: Dict[str, PunctuationStrategy] = {}
        self._fallback: Optional[PunctuationStrategy] = None
        self._logger = logger or logging.getLogger(__name__)

    def register(self, strategy: PunctuationStrategy) -> None:
        """注册策略。"""
        for language in strategy.supported_languages:
            normalized = normalize_language(language)
            if normalized in {"*", "default"}:
                self._fallback = strategy
                continue
            self._strategies[normalized] = strategy
        self._logger.debug(
            "已注册标点策略: %s -> %s",
            strategy.supported_languages,
            strategy.model_id,
        )

    def get(self, language: str) -> Optional[PunctuationStrategy]:
        """获取匹配语言的策略，找不到则返回兜底策略。"""
        normalized = normalize_language(language)
        if not normalized:
            return self._fallback
        strategy = self._strategies.get(normalized)
        return strategy or self._fallback

    def list_languages(self) -> List[str]:
        """列出已注册语言。"""
        return sorted(self._strategies.keys())

    def set_fallback(self, strategy: PunctuationStrategy) -> None:
        """设置兜底策略。"""
        self._fallback = strategy


_registry: Optional[PunctuationRegistry] = None


def get_punctuation_registry() -> PunctuationRegistry:
    """获取注册表单例。"""
    global _registry
    if _registry is None:
        _registry = PunctuationRegistry()
    return _registry
