"""
标点策略注册表，负责语言路由。
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from app.services.punctuation.base import PunctuationStrategy


def normalize_language(language: Optional[str]) -> str:
    """规范化语言标签，统一成基础语言码。"""
    if not language:
        return "auto"
    language = language.strip().lower()
    if "-" in language:
        return language.split("-", 1)[0]
    if "_" in language:
        return language.split("_", 1)[0]
    return language


class PunctuationRegistry:
    """语言到策略的映射表。"""

    def __init__(self) -> None:
        self._strategies: Dict[str, PunctuationStrategy] = {}
        self._fallback: Optional[PunctuationStrategy] = None
        self._lock = threading.RLock()

    def register(self, strategy: PunctuationStrategy) -> None:
        """注册策略。"""
        with self._lock:
            for language in strategy.supported_languages:
                self._strategies[normalize_language(language)] = strategy

    def set_fallback(self, strategy: PunctuationStrategy) -> None:
        """设置兜底策略。"""
        with self._lock:
            self._fallback = strategy

    def get(self, language: Optional[str]) -> Optional[PunctuationStrategy]:
        """按语言获取策略，失败返回兜底策略。"""
        normalized = normalize_language(language)
        with self._lock:
            return self._strategies.get(normalized) or self._fallback

    def list_languages(self) -> List[str]:
        """列出已注册语言。"""
        with self._lock:
            return sorted(self._strategies.keys())

    def clear(self) -> None:
        """清空注册表（仅用于测试）。"""
        with self._lock:
            self._strategies.clear()
            self._fallback = None
