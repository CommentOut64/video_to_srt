"""标点策略集合与注册入口。"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.services.punctuation.registry import PunctuationRegistry
from app.services.punctuation.strategies.chinese import ChinesePunctuationStrategy
from app.services.punctuation.strategies.english import EnglishPunctuationStrategy
from app.services.punctuation.strategies.japanese import JapanesePunctuationStrategy
from app.services.punctuation.strategies.fallback import FallbackPunctuationStrategy


def create_default_registry(
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> PunctuationRegistry:
    """创建默认标点策略注册表。"""
    cfg = config or {}
    log = logger or logging.getLogger(__name__)

    chinese_cfg = cfg.get("chinese", {})
    english_cfg = cfg.get("english", {})
    japanese_cfg = cfg.get("japanese", {})

    registry = PunctuationRegistry()
    registry.register(
        ChinesePunctuationStrategy(
            model_id=chinese_cfg.get("model_id", "punct-ct-transformer-zh"),
            itn_first=bool(chinese_cfg.get("itn_first", True)),
            fallback_on_oov=bool(chinese_cfg.get("fallback_on_oov", True)),
            logger=log,
        )
    )
    registry.register(
        EnglishPunctuationStrategy(
            model_id=english_cfg.get("model_id", "punct-edge-punct-en"),
            overlap_words=int(english_cfg.get("overlap_words", 10)),
            logger=log,
        )
    )
    registry.register(
        JapanesePunctuationStrategy(
            model_id=japanese_cfg.get("model_id", "punct-char-bert-ja"),
            logger=log,
        )
    )
    registry.set_fallback(FallbackPunctuationStrategy())
    return registry
