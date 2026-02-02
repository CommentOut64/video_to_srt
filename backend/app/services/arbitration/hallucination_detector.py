"""
Whisper 幻觉检测器。
V3.2.0+dev.20260202.08
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.services.text_normalizer import TextNormalizer


class HallucinationDetector:
    """幻觉检测器（策略模式：将多道检测规则拆分为可组合步骤，便于扩展）。"""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def is_hallucination(self, result: Dict[str, Any], prompt: Optional[str]) -> bool:
        """判断 Whisper 输出是否为幻觉。"""
        text = str(result.get("text", "") or "")
        if not text.strip():
            self._logger.warning("检测到空输出幻觉: Whisper 返回空文本")
            return True

        if self._has_excessive_underscores(text):
            return True

        if self._has_prompt_echo(text, prompt):
            return True

        if self._has_low_confidence(result, text):
            return True

        if TextNormalizer.is_whisper_hallucination(text):
            self._logger.warning("检测到 Whisper 规则幻觉: text='%s...'", text[:50])
            return True

        return False

    def _has_excessive_underscores(self, text: str) -> bool:
        underscore_ratio = text.count("_") / max(len(text), 1)
        if underscore_ratio > 0.3:
            self._logger.warning(
                "检测到下划线幻觉: 下划线占比 %.1f%%, text='%s...'",
                underscore_ratio * 100,
                text[:50],
            )
            return True
        return False

    def _has_prompt_echo(self, text: str, prompt: Optional[str]) -> bool:
        if not prompt:
            return False
        prompt_words = set(str(prompt).split())
        text_words = set(text.split())
        if not prompt_words:
            return False
        overlap_ratio = len(prompt_words & text_words) / len(prompt_words)
        if overlap_ratio > 0.8 and abs(len(text) - len(prompt)) < len(prompt) * 0.3:
            self._logger.warning(
                "检测到提示词重复: 与 prompt 重叠度 %.1f%%, prompt='%s...', text='%s...'",
                overlap_ratio * 100,
                str(prompt)[:30],
                text[:30],
            )
            return True
        return False

    def _has_low_confidence(self, result: Dict[str, Any], text: str) -> bool:
        raw_result = result.get("raw_result", {}) if isinstance(result, dict) else {}
        segments = raw_result.get("segments", []) if isinstance(raw_result, dict) else []
        if not segments:
            return False

        avg_logprob = sum(s.get("avg_logprob", -0.5) for s in segments) / len(segments)
        avg_no_speech = sum(s.get("no_speech_prob", 0.0) for s in segments) / len(segments)
        if avg_logprob < -1.0:
            self._logger.warning(
                "检测到低置信度幻觉: avg_logprob=%.2f < -1.0, text='%s...'",
                avg_logprob,
                text[:50],
            )
            return True
        if avg_no_speech > 0.6 and text:
            self._logger.warning(
                "检测到静音段误识别: no_speech_prob=%.2f > 0.6, text='%s...'",
                avg_no_speech,
                text[:50],
            )
            return True
        return False
