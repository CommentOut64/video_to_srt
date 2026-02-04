"""
Whisper 幻觉检测器。
V3.2.0+dev.20260204.03
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from app.core.logging import resolve_loguru_logger
from app.services.text_normalizer import TextNormalizer


class HallucinationDetector:
    """幻觉检测器（策略模式：将多道检测规则拆分为可组合步骤，便于扩展）。"""

    _PROMPT_ECHO_MAX_LEN = 120

    def __init__(self, logger: Optional[Any] = None) -> None:
        self._logger = resolve_loguru_logger(
            logger,
            __name__,
            layer="L2",
            processor_name="hallucination_detector",
        )

    def is_hallucination(self, result: Dict[str, Any], prompt: Optional[str]) -> bool:
        """判断 Whisper 输出是否为幻觉。"""
        text = str(result.get("text", "") or "")
        raw_text = str(result.get("text_raw", "") or "")
        candidate_text = text.strip()
        if not candidate_text and raw_text.strip():
            candidate_text = raw_text.strip()
            self._logger.warning("检测到清洗后空文本，改用 raw_text 继续检测: raw_len={}", len(raw_text))
        if not candidate_text:
            self._logger.warning("检测到空输出幻觉: Whisper 返回空文本")
            return True

        if self._has_excessive_underscores(candidate_text):
            return True

        if self._has_prompt_echo(candidate_text, prompt):
            return True

        if self._has_low_confidence(result, candidate_text):
            return True

        if TextNormalizer.is_whisper_hallucination(candidate_text):
            self._logger.warning("检测到 Whisper 规则幻觉: text='%s...'", candidate_text[:50])
            return True

        return False

    def _has_excessive_underscores(self, text: str) -> bool:
        underscore_ratio = text.count("_") / max(len(text), 1)
        if underscore_ratio > 0.3:
            self._logger.warning(
                "检测到下划线幻觉: 下划线占比 {:.1f}%, text='{}...'",
                underscore_ratio * 100,
                text[:50],
            )
            return True
        return False

    def _has_prompt_echo(self, text: str, prompt: Optional[str]) -> bool:
        if not prompt:
            return False
        prompt_clean = str(prompt).strip()
        if not prompt_clean:
            return False
        if len(prompt_clean) > self._PROMPT_ECHO_MAX_LEN:
            # Prompt 过长时不判定为回显，避免误杀正文
            return False
        prompt_words = set(prompt_clean.split())
        text_words = set(text.split())
        if not prompt_words:
            return False
        overlap_ratio = len(prompt_words & text_words) / len(prompt_words)
        if overlap_ratio > 0.8 and abs(len(text) - len(prompt_clean)) < len(prompt_clean) * 0.3:
            self._logger.warning(
                "检测到提示词重复: 与 prompt 重叠度 {:.1f}%, prompt='{}...', text='{}...'",
                overlap_ratio * 100,
                prompt_clean[:30],
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
                "检测到低置信度幻觉: avg_logprob={:.2f} < -1.0, text='{}...'",
                avg_logprob,
                text[:50],
            )
            return True
        if avg_no_speech > 0.6 and text:
            self._logger.warning(
                "检测到静音段误识别: no_speech_prob={:.2f} > 0.6, text='{}...'",
                avg_no_speech,
                text[:50],
            )
            return True
        return False
