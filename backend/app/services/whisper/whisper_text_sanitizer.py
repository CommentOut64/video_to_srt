"""
Whisper 文本清洗器（G-1 最小清洗 + G-2 完整清洗）。
V3.2.0+dev.20260202.07
"""
from __future__ import annotations

import logging
import re
from typing import Optional, Dict, Any, List

from app.services.text_normalizer import TextNormalizer
from app.services.model_runtime_config_service import get_model_runtime_config_service


class WhisperTextSanitizer:
    """Whisper 文本清洗器（策略模式：分层清洗，避免误删正文）。"""

    _REPEAT_CHAR_PATTERN = re.compile(r"(.)\1{9,}")
    _HALLUCINATION_PATTERNS = [
        re.compile(r"^字幕.*?制作", re.IGNORECASE),
        re.compile(r"^翻译[:：]", re.IGNORECASE),
        re.compile(r"^感谢.*?观看", re.IGNORECASE),
        re.compile(r"字幕由.*?提供$", re.IGNORECASE),
    ]

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def sanitize_minimal(self, text: str, prompt: Optional[str] = None) -> str:
        """最小预清洗：移除 Prompt 回显 / 极端重复 / 固定幻觉前缀。"""
        if not text:
            return ""
        raw_text = text
        text = text.strip()

        text = self._remove_prompt_echo(text, prompt)
        text = self._remove_hallucination_markers(text)
        text = self._remove_extreme_repetition(text)

        if raw_text != text:
            self._logger.debug("Whisper 最小清洗完成: raw_len=%d -> clean_len=%d", len(raw_text), len(text))
        return text.strip()

    def sanitize_full(self, text: str, prompt: Optional[str] = None) -> str:
        """完整清洗（G-2）：最小预清洗 + 动态词表 + 重复/幻觉检测。"""
        if not text:
            return ""

        raw_text = text
        text = text.strip()
        runtime = self._load_runtime_config()
        if not runtime.get("enabled", True):
            return text

        text = self._remove_prompt_echo(text, prompt)
        text = self._apply_runtime_patterns(text, runtime.get("patterns") or [])
        text = self._remove_hallucination_markers(text)
        text = self._remove_extreme_repetition(text)

        text = TextNormalizer.clean_whisper_output(text)
        min_len = int(runtime.get("min_text_length", 2) or 2)
        if len(text.strip()) < min_len:
            text = ""

        if raw_text != text:
            self._logger.debug("Whisper 完整清洗完成: raw_len=%d -> clean_len=%d", len(raw_text), len(text))
        return text.strip()

    @staticmethod
    def _remove_prompt_echo(text: str, prompt: Optional[str]) -> str:
        if not text or not prompt:
            return text
        prompt_clean = prompt.strip()
        if prompt_clean and text.startswith(prompt_clean):
            return text[len(prompt_clean):].lstrip()
        return text

    def _remove_extreme_repetition(self, text: str) -> str:
        if not text:
            return text
        if self._REPEAT_CHAR_PATTERN.search(text):
            text = self._REPEAT_CHAR_PATTERN.sub(lambda m: m.group(1) * 2, text)
        if TextNormalizer.REPEATED_PATTERN.search(text):
            truncated = TextNormalizer.truncate_repeated_text(text)
            if truncated != text:
                return truncated
        return text

    def _remove_hallucination_markers(self, text: str) -> str:
        if not text:
            return text
        for pattern in self._HALLUCINATION_PATTERNS:
            text = pattern.sub("", text)
        return text

    def _load_runtime_config(self) -> Dict[str, Any]:
        runtime_data = get_model_runtime_config_service().get_effective_runtime_global()
        return dict(runtime_data.get("effective", {}).get("whisper_sanitize", {}) or {})

    def _apply_runtime_patterns(self, text: str, patterns: List[Dict[str, Any]]) -> str:
        if not text or not patterns:
            return text

        for rule in patterns:
            rule_type = str(rule.get("type", "phrase")).lower()
            value = str(rule.get("value", "") or "")
            position = str(rule.get("position", "any")).lower()
            if not value:
                continue

            try:
                if rule_type == "regex":
                    flags = 0
                    flag_str = str(rule.get("flags", "") or "")
                    if "i" in flag_str.lower():
                        flags |= re.IGNORECASE
                    pattern = re.compile(value, flags)
                    text = self._apply_regex_rule(text, pattern, position)
                else:
                    text = self._apply_phrase_rule(text, value, position)
            except re.error as exc:
                self._logger.warning("Whisper 清洗规则无效: %s (%s)", value, exc)
        return text

    @staticmethod
    def _apply_phrase_rule(text: str, value: str, position: str) -> str:
        if position == "prefix":
            if text.startswith(value):
                return text[len(value):].lstrip()
            return text
        if position == "suffix":
            if text.endswith(value):
                return text[:-len(value)].rstrip()
            return text
        return text.replace(value, "")

    @staticmethod
    def _apply_regex_rule(text: str, pattern: re.Pattern, position: str) -> str:
        if position == "prefix":
            return pattern.sub("", text, count=1).lstrip()
        if position == "suffix":
            suffix_pattern = re.compile(f"(?:{pattern.pattern})\\s*$", pattern.flags)
            return suffix_pattern.sub("", text).rstrip()
        return pattern.sub("", text)
