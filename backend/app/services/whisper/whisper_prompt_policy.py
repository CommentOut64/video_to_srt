"""
Whisper 提示词注入策略服务。
V3.2.0+dev.20260219.03

设计模式：Strategy Pattern。
原因：将提示词构造、长度预算、上下文重置逻辑集中管理，避免业务链路散落拼接逻辑。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from app.services.model_runtime_config_service import get_model_runtime_config_service

_MODEL_TAG_PATTERN = re.compile(r"<\|[^|]+?\|>")
_TOKEN_MARKER_PATTERN = re.compile(r"▁+")
_SPACE_PATTERN = re.compile(r"\s+")
_LATIN_WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]{1,31}")
_CJK_WORD_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,6}")

_ZH_STOPWORDS = {
    "这个",
    "那个",
    "什么",
    "怎么",
    "因为",
    "所以",
    "然后",
    "我们",
    "你们",
    "他们",
}
_EN_STOPWORDS = {
    "the",
    "and",
    "but",
    "this",
    "that",
    "there",
    "here",
    "when",
    "what",
    "who",
    "where",
    "why",
    "how",
    "then",
    "now",
    "with",
    "from",
}


@dataclass(frozen=True)
class WhisperPromptConfig:
    """Whisper 提示词策略配置。"""

    is_enabled: bool = False
    max_prompt_chars: int = 180
    max_keyword_count: int = 16
    max_context_chars: int = 80
    max_history_chars: int = 160
    reset_pause_sec: float = 1.8
    reset_no_speech_prob: float = 0.65
    reset_low_confidence: float = 0.35


class WhisperPromptPolicy:
    """Whisper 提示词策略执行器。"""

    def __init__(
        self,
        *,
        logger: Optional[logging.Logger] = None,
        user_glossary: Optional[Sequence[str]] = None,
        runtime_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._user_glossary: List[str] = [
            str(item).strip() for item in (user_glossary or []) if str(item).strip()
        ]
        self._runtime_override = dict(runtime_override or {})

    def build_prompt(
        self,
        *,
        previous_text: Optional[str],
        semantic_text: Optional[str],
        pause_gap_sec: Optional[float] = None,
    ) -> Optional[str]:
        """按策略构建 Whisper initial_prompt。"""
        config = self.load_config()
        if not config.is_enabled:
            return None

        normalized_previous = self._normalize_text(previous_text)
        normalized_semantic = self._normalize_text(semantic_text)
        if pause_gap_sec is not None and pause_gap_sec >= config.reset_pause_sec:
            normalized_previous = ""

        keywords = self._collect_keywords(
            semantic_text=normalized_semantic,
            previous_text=normalized_previous,
            max_keyword_count=config.max_keyword_count,
        )
        tail_source = normalized_semantic or normalized_previous
        tail_text = self._clip_tail(tail_source, config.max_context_chars)

        segments: List[str] = []
        keywords_text = ""
        if keywords:
            keywords_text = " ".join(keywords).strip()
            if keywords_text:
                segments.append(keywords_text)
        if tail_text:
            segments.append(tail_text)

        prompt = self._normalize_text(" ".join(segments))
        if not prompt:
            return None
        if len(prompt) > config.max_prompt_chars:
            if keywords_text and len(keywords_text) < config.max_prompt_chars:
                remain = config.max_prompt_chars - len(keywords_text)
                compact_tail = self._clip_tail(tail_text, max(0, remain - 1))
                prompt = self._normalize_text(f"{keywords_text} {compact_tail}")
            else:
                prompt = prompt[: config.max_prompt_chars].rstrip()
        return prompt or None

    def compact_external_context(self, raw_context: Optional[str]) -> str:
        """压缩外部上下文（旧补刀链路使用）。"""
        config = self.load_config()
        if not config.is_enabled:
            return ""
        normalized = self._normalize_text(raw_context)
        if not normalized:
            return ""
        keywords = self._collect_keywords(
            semantic_text=normalized,
            previous_text="",
            max_keyword_count=max(1, config.max_keyword_count // 2),
        )
        segments: List[str] = []
        if keywords:
            segments.append(" ".join(keywords))
        segments.append(self._clip_tail(normalized, config.max_context_chars))
        compacted = self._normalize_text(" ".join(segments))
        if len(compacted) > config.max_prompt_chars:
            compacted = compacted[: config.max_prompt_chars].rstrip()
        return compacted

    def update_history(
        self,
        *,
        previous_text: Optional[str],
        decoded_text: Optional[str],
        confidence: Optional[float] = None,
        avg_no_speech_prob: Optional[float] = None,
        is_hallucination: bool = False,
    ) -> str:
        """更新用于下一轮提示词的历史文本。"""
        config = self.load_config()
        if not config.is_enabled:
            return ""
        if self.should_reset(
            is_hallucination=is_hallucination,
            confidence=confidence,
            avg_no_speech_prob=avg_no_speech_prob,
        ):
            return ""

        normalized_decoded = self._normalize_text(decoded_text)
        if normalized_decoded:
            return normalized_decoded[-config.max_history_chars :]

        normalized_previous = self._normalize_text(previous_text)
        if normalized_previous:
            return normalized_previous[-config.max_history_chars :]
        return ""

    def restore_history(self, previous_text: Optional[str]) -> str:
        """恢复历史上下文（断点恢复时使用）。"""
        config = self.load_config()
        normalized = self._normalize_text(previous_text)
        if not normalized or not config.is_enabled:
            return ""
        return normalized[-config.max_history_chars :]

    def should_reset(
        self,
        *,
        is_hallucination: bool,
        confidence: Optional[float] = None,
        avg_no_speech_prob: Optional[float] = None,
    ) -> bool:
        """判断是否应重置上下文。"""
        config = self.load_config()
        if is_hallucination:
            return True
        if confidence is not None and confidence <= config.reset_low_confidence:
            return True
        if avg_no_speech_prob is not None and avg_no_speech_prob >= config.reset_no_speech_prob:
            return True
        return False

    def describe_config(self) -> Dict[str, Any]:
        """输出当前生效配置，便于诊断。"""
        config = self.load_config()
        return {
            "enabled": config.is_enabled,
            "max_prompt_chars": config.max_prompt_chars,
            "max_keyword_count": config.max_keyword_count,
            "max_context_chars": config.max_context_chars,
            "max_history_chars": config.max_history_chars,
            "reset_pause_sec": config.reset_pause_sec,
            "reset_no_speech_prob": config.reset_no_speech_prob,
            "reset_low_confidence": config.reset_low_confidence,
        }

    def load_config(self) -> WhisperPromptConfig:
        """读取运行参数并映射为策略配置。"""
        runtime_data = get_model_runtime_config_service().get_effective_runtime_global()
        runtime_effective = runtime_data.get("effective", {})
        group_raw = dict(runtime_effective.get("whisper_prompt", {}) or {})
        if self._runtime_override:
            group_raw.update(self._runtime_override)

        return WhisperPromptConfig(
            is_enabled=self._to_bool(group_raw.get("enabled"), default=False),
            max_prompt_chars=self._to_int(group_raw.get("max_prompt_chars"), default=180, minimum=32),
            max_keyword_count=self._to_int(group_raw.get("max_keyword_count"), default=16, minimum=1),
            max_context_chars=self._to_int(group_raw.get("max_context_chars"), default=80, minimum=16),
            max_history_chars=self._to_int(group_raw.get("max_history_chars"), default=160, minimum=32),
            reset_pause_sec=self._to_float(group_raw.get("reset_pause_sec"), default=1.8, minimum=0.0),
            reset_no_speech_prob=self._to_float(
                group_raw.get("reset_no_speech_prob"),
                default=0.65,
                minimum=0.0,
            ),
            reset_low_confidence=self._to_float(
                group_raw.get("reset_low_confidence"),
                default=0.35,
                minimum=0.0,
            ),
        )

    def _collect_keywords(
        self,
        *,
        semantic_text: str,
        previous_text: str,
        max_keyword_count: int,
    ) -> List[str]:
        keywords: List[str] = []
        seen = set()

        def add_keyword(item: str) -> None:
            normalized = item.strip()
            if not normalized:
                return
            key = normalized.lower()
            if key in seen:
                return
            seen.add(key)
            keywords.append(normalized)

        for word in self._user_glossary:
            add_keyword(word)
            if len(keywords) >= max_keyword_count:
                return keywords

        for source in (semantic_text, previous_text):
            for word in self._extract_keywords(source):
                add_keyword(word)
                if len(keywords) >= max_keyword_count:
                    return keywords

        return keywords

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        if not text:
            return []

        words: List[str] = []
        for candidate in _LATIN_WORD_PATTERN.findall(text):
            lowered = candidate.lower()
            if lowered in _EN_STOPWORDS:
                continue
            words.append(candidate)
        for candidate in _CJK_WORD_PATTERN.findall(text):
            if candidate in _ZH_STOPWORDS:
                continue
            words.append(candidate)
        return words

    @staticmethod
    def _normalize_text(text: Optional[str]) -> str:
        if not text:
            return ""
        normalized = str(text)
        normalized = _MODEL_TAG_PATTERN.sub(" ", normalized)
        normalized = _TOKEN_MARKER_PATTERN.sub(" ", normalized)
        normalized = _SPACE_PATTERN.sub(" ", normalized)
        return normalized.strip()

    @staticmethod
    def _clip_tail(text: str, max_chars: int) -> str:
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        return text[-max_chars :].lstrip()

    @staticmethod
    def _to_bool(value: Any, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return default

    @staticmethod
    def _to_int(value: Any, *, default: int, minimum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, parsed)

    @staticmethod
    def _to_float(value: Any, *, default: float, minimum: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, parsed)


def compact_whisper_context_for_patch(
    raw_context: Optional[str],
    *,
    logger: Optional[logging.Logger] = None,
) -> str:
    """旧补刀链路统一上下文压缩入口。"""
    policy = WhisperPromptPolicy(logger=logger)
    return policy.compact_external_context(raw_context)
