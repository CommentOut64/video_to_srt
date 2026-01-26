"""
ASR 抽象接口能力与时间戳精度枚举。
V3.2.0+dev.20260119.01
"""

from __future__ import annotations

from enum import Enum


class TimestampPrecision(str, Enum):
    """时间戳精度等级。"""

    FRAME = "frame"    # 帧级（如 CTC）
    WORD = "word"      # 词级
    SEGMENT = "segment"  # 段级
    NONE = "none"      # 无时间戳


class ASRCapability(str, Enum):
    """ASR 引擎能力标记。"""

    WORD_TIMESTAMPS = "word_timestamps"
    CONFIDENCE_SCORES = "confidence_scores"
    LANGUAGE_DETECTION = "language_detection"
    EMOTION_DETECTION = "emotion_detection"
    EVENT_TAGGING = "event_tagging"
    STREAMING = "streaming"
    MULTI_SPEAKER = "multi_speaker"
