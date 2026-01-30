"""
标点策略集合。
"""
from app.services.punctuation.registry import PunctuationRegistry
from app.services.punctuation.strategies.fallback import (
    FallbackPunctuationStrategy,
    RuleBasedPunctuationStrategy,
)
from app.services.punctuation.strategies.chinese import ChinesePunctuationStrategy
from app.services.punctuation.strategies.english import EnglishPunctuationStrategy
from app.services.punctuation.strategies.japanese import JapanesePunctuationStrategy


def create_default_registry() -> PunctuationRegistry:
    """创建默认策略注册表（策略模式统一入口）。"""
    registry = PunctuationRegistry()
    registry.register(ChinesePunctuationStrategy())
    registry.register(EnglishPunctuationStrategy())
    registry.register(JapanesePunctuationStrategy())
    registry.set_fallback(FallbackPunctuationStrategy())
    return registry

__all__ = [
    "FallbackPunctuationStrategy",
    "RuleBasedPunctuationStrategy",
    "ChinesePunctuationStrategy",
    "EnglishPunctuationStrategy",
    "JapanesePunctuationStrategy",
    "create_default_registry",
]
