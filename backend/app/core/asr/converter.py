"""
时间戳精度转换工具。
V3.2.0+dev.20260119.01
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from app.core.asr.models import Segment, WordTimestamp


class TimestampConverter:
    """时间戳精度转换器。"""

    @staticmethod
    def frame_to_word(
        frame_timestamps: List[WordTimestamp],
        language: str = "zh",
    ) -> List[WordTimestamp]:
        """帧级 → 词级（根据语言做合并）。"""
        if language in {"zh", "yue", "ja", "ko"}:
            return TimestampConverter._merge_by_count(frame_timestamps, count=3)
        return TimestampConverter._merge_by_space(frame_timestamps)

    @staticmethod
    def word_to_segment(
        word_timestamps: List[WordTimestamp],
        max_duration: float = 5.0,
    ) -> List[Segment]:
        """词级 → 段级（按时长或标点切分）。"""
        if not word_timestamps:
            return []

        segments: List[Segment] = []
        current_words: List[WordTimestamp] = []
        current_start: Optional[float] = None

        for word_ts in word_timestamps:
            if not current_words:
                current_start = word_ts.start
            current_words.append(word_ts)

            is_over_duration = word_ts.end - current_start > max_duration
            is_sentence_end = word_ts.word.endswith(("。", ".", "!", "?"))

            if is_over_duration or is_sentence_end:
                segments.append(
                    Segment(
                        start=current_start,
                        end=word_ts.end,
                        text="".join(w.word for w in current_words),
                        confidence=TimestampConverter._mean_confidence(current_words),
                        words=current_words,
                    )
                )
                current_words = []
                current_start = None

        if current_words:
            segments.append(
                Segment(
                start=current_start or current_words[0].start,
                end=current_words[-1].end,
                text="".join(w.word for w in current_words),
                confidence=TimestampConverter._mean_confidence(current_words),
                words=current_words,
            )
            )

        return segments

    @staticmethod
    def fallback_to_pseudo(text: str, start: float, end: float) -> List[WordTimestamp]:
        """降级：文本 → 伪对齐（当引擎不支持字级时间戳）。"""
        if not text:
            return []
        words = list(text)
        step = (end - start) / len(words) if len(words) > 0 else 0.0

        return [
            WordTimestamp(
                word=word,
                start=start + i * step,
                end=start + (i + 1) * step,
                confidence=None,
                is_pseudo=True,
            )
            for i, word in enumerate(words)
        ]

    @staticmethod
    def _merge_by_count(
        timestamps: List[WordTimestamp],
        count: int,
    ) -> List[WordTimestamp]:
        """按字数合并。"""
        if count <= 0:
            return timestamps

        merged: List[WordTimestamp] = []
        for i in range(0, len(timestamps), count):
            chunk = timestamps[i:i + count]
            if not chunk:
                continue
            merged.append(
                WordTimestamp(
                    word="".join(w.word for w in chunk),
                    start=chunk[0].start,
                    end=chunk[-1].end,
                    confidence=TimestampConverter._mean_confidence(chunk),
                    is_pseudo=any(w.is_pseudo for w in chunk),
                )
            )
        return merged

    @staticmethod
    def _merge_by_space(timestamps: List[WordTimestamp]) -> List[WordTimestamp]:
        """按空白字符合并。"""
        merged: List[WordTimestamp] = []
        current: List[WordTimestamp] = []

        for ts in timestamps:
            if ts.word.isspace():
                if current:
                    merged.append(
                        WordTimestamp(
                            word="".join(w.word for w in current),
                            start=current[0].start,
                            end=current[-1].end,
                            confidence=TimestampConverter._mean_confidence(current),
                            is_pseudo=any(w.is_pseudo for w in current),
                        )
                    )
                    current = []
                continue
            current.append(ts)

        if current:
            merged.append(
                WordTimestamp(
                    word="".join(w.word for w in current),
                    start=current[0].start,
                    end=current[-1].end,
                    confidence=TimestampConverter._mean_confidence(current),
                    is_pseudo=any(w.is_pseudo for w in current),
                )
            )

        return merged

    @staticmethod
    def _mean_confidence(words: List[WordTimestamp]) -> float:
        """兼容空值的置信度平均值。"""
        values = [w.confidence for w in words if w.confidence is not None]
        if not values:
            return 0.0
        return float(np.mean(values))
