"""
标点恢复统一服务入口。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml

from app.core.config import config
from app.services.model_runtime_config_service import get_model_runtime_config_service
from app.services.punctuation.base import (
    PunctuationError,
    PunctuationResult,
    WordTimestampLike,
)
from app.services.punctuation.registry import normalize_language
from app.services.punctuation.strategies import create_default_registry


class PunctuationService:
    """标点恢复服务（语言路由 + 降级兜底）。"""

    def __init__(
        self,
        registry=None,
        config_path: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._runtime_service = get_model_runtime_config_service()
        self._config = self._load_yaml_config(config_path)
        self.registry = registry or create_default_registry(self._config, logger=self._logger)

    async def restore(
        self,
        text: str,
        language: Optional[str] = None,
        word_timestamps: Optional[Sequence[WordTimestampLike]] = None,
        context: Optional[str] = None,
        fallback_text: Optional[str] = None,
    ) -> PunctuationResult:
        runtime = self._get_runtime_config()
        if not runtime.get("enable_punctuation", True):
            return self._fallback_result(text, fallback_text, model_id="disabled", runtime=runtime)

        effective_language = normalize_language(
            language or runtime.get("default_language") or self._config.get("default_language")
        )

        strategy = self.registry.get(effective_language)
        if not strategy or strategy.model_id in {"asr_fallback", "fallback"}:
            return self._fallback_result(text, fallback_text, model_id="asr_fallback", runtime=runtime)

        try:
            return await strategy.restore(text, word_timestamps=word_timestamps, context=context)
        except PunctuationError as exc:
            self._logger.warning("标点恢复失败，降级为原文: %s", exc)
            return self._fallback_result(text, fallback_text, model_id="asr_fallback", runtime=runtime)

    def _fallback_result(
        self,
        text: str,
        fallback_text: Optional[str],
        model_id: str,
        runtime: Dict[str, Any],
    ) -> PunctuationResult:
        resolved_text = self._resolve_fallback_text(text, fallback_text, runtime)
        return PunctuationResult(text=resolved_text or "", model_id=model_id)

    def _resolve_fallback_text(
        self,
        text: str,
        fallback_text: Optional[str],
        runtime: Dict[str, Any],
    ) -> str:
        primary = text or ""
        secondary = fallback_text or ""
        if not secondary:
            return primary
        priority = str(
            runtime.get("fallback_priority") or self._config.get("fallback_priority", "fast")
        ).lower()
        if priority not in {"fast", "slow"}:
            self._logger.warning("未知 fallback_priority=%s，回退 fast", priority)
            priority = "fast"
        if priority == "slow":
            return secondary or primary
        return primary or secondary

    def _get_runtime_config(self) -> Dict[str, Any]:
        runtime = self._runtime_service.get_effective_runtime_global()
        effective = runtime.get("effective", {})
        return dict(effective.get("punctuation", {}))

    def _load_yaml_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        path = config_path or (Path(config.BASE_DIR) / "backend" / "app" / "config" / "punctuation.yaml")
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data
        except Exception as exc:
            self._logger.warning("加载标点配置失败，使用默认配置: %s", exc)
            return {}
