"""
ASR 置信度规范化工具。
V3.2.0+dev.20260119.01
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class ConfidenceNormalizer:
    """置信度规范化器。"""

    @staticmethod
    def normalize_sensevoice(logits: np.ndarray) -> float:
        """SenseVoice: CTC softmax 均值。"""
        logits_array = np.asarray(logits)
        if logits_array.size == 0:
            return 0.0
        from scipy.special import softmax

        probs = softmax(logits_array, axis=-1)
        confidence = float(np.mean(np.max(probs, axis=-1)))
        return float(np.clip(confidence, 0.0, 1.0))

    @staticmethod
    def normalize_whisper(avg_logprob: float, no_speech_prob: float) -> float:
        """Whisper: logprob 组合。"""
        confidence = float(1.0 + avg_logprob)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        no_speech = float(np.clip(no_speech_prob, 0.0, 1.0))
        return float(np.clip(confidence * (1.0 - no_speech), 0.0, 1.0))

    @staticmethod
    def normalize_api(raw_confidence: float) -> float:
        """API: 通常已经是 0-1 范围。"""
        return float(np.clip(raw_confidence, 0.0, 1.0))

    @staticmethod
    def normalize_generic(raw_score: float, score_range: Tuple[float, float]) -> float:
        """通用规范化：将任意范围映射到 [0, 1]。"""
        min_score, max_score = score_range
        if max_score <= min_score:
            return 0.0
        normalized = (raw_score - min_score) / (max_score - min_score)
        return float(np.clip(normalized, 0.0, 1.0))
