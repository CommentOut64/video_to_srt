"""
标点服务入口。
V3.2.0+dev.20260130.06
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Sequence

from app.services.model_runtime_config_service import get_model_runtime_config_service
from app.services.punctuation.base import (
    PunctuationResult,
    PunctuationStrategy,
    WordTimestampLike,
)
from app.services.punctuation.config import get_punctuation_config
from app.services.punctuation.registry import PunctuationRegistry, get_punctuation_registry
from app.services.punctuation.strategies.chinese import ChinesePunctuationStrategy
from app.services.punctuation.strategies.english import EnglishPunctuationStrategy
from app.services.punctuation.strategies.fallback import FallbackPunctuationStrategy
from app.services.punctuation.strategies.japanese import JapanesePunctuationStrategy


class PunctuationService:
    """统一标点恢复服务。"""

    def __init__(
        self,
        registry: Optional[PunctuationRegistry] = None,
        *,
        is_enable_punctuation: Optional[bool] = None,
        default_language: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._registry = registry or get_punctuation_registry()
        self._is_custom_registry = registry is not None
        self._logger = logger or logging.getLogger(__name__)
        self._runtime_service = get_model_runtime_config_service()
        self._static_config = get_punctuation_config()
        self._force_enable_punctuation = is_enable_punctuation
        self._force_default_language = default_language
        self._is_punctuation_enabled = True
        self._default_language = "zh"
        self._fallback_priority = "fast"
        self._logged_runtime = False
        self._logged_strategies: set[str] = set()
        self._logged_first_success_languages: set[str] = set()
        if not self._is_custom_registry:
            self._ensure_default_strategies()

    def _ensure_default_strategies(self) -> None:
        """注册默认策略（幂等）。"""
        self._registry.register(ChinesePunctuationStrategy())
        self._registry.register(EnglishPunctuationStrategy())
        self._registry.register(JapanesePunctuationStrategy())
        self._registry.register(FallbackPunctuationStrategy())

    def _refresh_runtime_config(self) -> None:
        runtime = self._runtime_service.get_effective_runtime_global()
        punct_config = runtime.get("effective", {}).get("punctuation", {})
        self._is_punctuation_enabled = (
            self._force_enable_punctuation
            if self._force_enable_punctuation is not None
            else bool(punct_config.get("enable_punctuation", True))
        )
        self._default_language = (
            self._force_default_language
            or punct_config.get("default_language")
            or self._static_config.get("default_language", "zh")
        )
        self._fallback_priority = (
            punct_config.get("fallback_priority")
            or self._static_config.get("fallback_priority", "fast")
        )

    def get_strategy(self, language: str) -> Optional[PunctuationStrategy]:
        """返回匹配语言的策略。"""
        language = language or self._default_language
        return self._registry.get(language)

    async def restore(
        self,
        text: str,
        language: Optional[str] = None,
        *,
        word_timestamps: Optional[Sequence[WordTimestampLike]] = None,
        context: Optional[str] = None,
        fallback_text: Optional[str] = None,
    ) -> PunctuationResult:
        """恢复标点并返回结果。"""
        self._refresh_runtime_config()
        if not self._logged_runtime:
            self._logger.info(
                "标点服务状态: enabled=%s, default_language=%s, fallback_priority=%s",
                self._is_punctuation_enabled,
                self._default_language,
                self._fallback_priority,
            )
            self._logged_runtime = True
        if not text:
            return self._build_empty_result("", model_id="empty", processing_time_ms=0.0)

        if not self._is_punctuation_enabled:
            return self._build_empty_result(text, model_id="disabled", processing_time_ms=0.0)

        resolved_language = language or self._default_language
        strategy = self.get_strategy(resolved_language)
        if strategy is None:
            self._logger.warning("未找到标点策略，使用兜底逻辑")
            return self._build_fallback_result(text, fallback_text, 0.0)
        if resolved_language not in self._logged_strategies:
            self._logger.info(
                "标点策略启用: language=%s, model_id=%s",
                resolved_language,
                strategy.model_id,
            )
            self._logged_strategies.add(resolved_language)

        if strategy.model_id == "asr_fallback":
            if self._fallback_priority == "slow" and fallback_text:
                return await strategy.restore(
                    text=fallback_text,
                    word_timestamps=None,
                    context=context,
                )
            return await strategy.restore(
                text=text,
                word_timestamps=word_timestamps,
                context=context,
            )

        start = time.perf_counter()
        try:
            result = await strategy.restore(
                text=text,
                word_timestamps=word_timestamps,
                context=context,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._logger.warning("标点恢复失败，回退兜底: %s", exc)
            return self._build_fallback_result(text, fallback_text, elapsed_ms)

        if (
            resolved_language not in self._logged_first_success_languages
            and result.model_id not in {"asr_fallback", "disabled", "empty"}
        ):
            self._logger.info(
                "标点恢复完成(首次): language=%s, model_id=%s, ms=%.2f, text_len=%d, positions=%d",
                resolved_language,
                result.model_id,
                float(result.processing_time_ms or 0.0),
                len(text),
                len(result.punctuation_positions),
            )
            self._logged_first_success_languages.add(resolved_language)

        return result

    async def restore_with_strategy(
        self,
        strategy: PunctuationStrategy,
        text: str,
        *,
        word_timestamps: Optional[Sequence[WordTimestampLike]] = None,
        context: Optional[str] = None,
    ) -> PunctuationResult:
        """使用指定策略恢复标点。"""
        return await strategy.restore(text, word_timestamps=word_timestamps, context=context)

    @staticmethod
    def _build_empty_result(text: str, *, model_id: str, processing_time_ms: float) -> PunctuationResult:
        return PunctuationResult(
            text=text,
            split_points=[],
            punctuation_positions=[],
            confidence=1.0,
            model_id=model_id,
            processing_time_ms=processing_time_ms,
        )

    def _build_fallback_result(
        self,
        text: str,
        fallback_text: Optional[str],
        processing_time_ms: float,
    ) -> PunctuationResult:
        chosen = text
        if self._fallback_priority == "slow" and fallback_text:
            chosen = fallback_text
        return self._build_empty_result(
            chosen,
            model_id="asr_fallback",
            processing_time_ms=processing_time_ms,
        )


_punctuation_service: Optional[PunctuationService] = None


def get_punctuation_service() -> PunctuationService:
    """获取 PunctuationService 单例。"""
    global _punctuation_service
    if _punctuation_service is None:
        _punctuation_service = PunctuationService()
    return _punctuation_service
