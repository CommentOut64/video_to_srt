# -*- coding: utf-8 -*-
"""
音频测试数据工厂。

提供便捷方法构造测试用 AudioChunk 和音频数组。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from app.services.audio.chunk_engine import AudioChunk


class AudioFactory:
    """音频测试数据工厂。"""

    @staticmethod
    def create_silent_chunks(
        count: int = 3,
        duration: float = 2.0,
        sample_rate: int = 16000,
        language: str = "zh",
    ) -> List[AudioChunk]:
        """创建指定数量的静音 AudioChunk。

        Args:
            count: chunk 数量
            duration: 每个 chunk 的时长（秒）
            sample_rate: 采样率
            language: 语言代码

        Returns:
            AudioChunk 列表
        """
        chunks: List[AudioChunk] = []
        samples_per_chunk = int(duration * sample_rate)

        for i in range(count):
            start = i * duration
            end = start + duration
            audio = np.zeros(samples_per_chunk, dtype=np.float32)
            chunks.append(
                AudioChunk(
                    index=i,
                    start=start,
                    end=end,
                    audio=audio,
                    sample_rate=sample_rate,
                    language=language,
                )
            )

        return chunks

    @staticmethod
    def create_full_audio_from_chunks(
        chunks: List[AudioChunk],
    ) -> Tuple[np.ndarray, int]:
        """从 chunk 列表拼接完整音频数组。

        Returns:
            (full_audio_array, sample_rate)
        """
        if not chunks:
            return np.zeros(0, dtype=np.float32), 16000

        sample_rate = chunks[0].sample_rate
        total_duration = max(c.end for c in chunks)
        total_samples = int(total_duration * sample_rate)
        full_audio = np.zeros(total_samples, dtype=np.float32)

        for chunk in chunks:
            start_sample = int(chunk.start * sample_rate)
            end_sample = start_sample + len(chunk.audio)
            end_sample = min(end_sample, total_samples)
            length = end_sample - start_sample
            full_audio[start_sample:end_sample] = chunk.audio[:length]

        return full_audio, sample_rate

    @staticmethod
    def create_noise_chunks(
        count: int = 3,
        duration: float = 2.0,
        sample_rate: int = 16000,
        language: str = "zh",
        noise_amplitude: float = 0.1,
    ) -> List[AudioChunk]:
        """创建包含随机噪声的 AudioChunk（模拟非静音场景）。"""
        chunks: List[AudioChunk] = []
        samples_per_chunk = int(duration * sample_rate)
        rng = np.random.default_rng(42)

        for i in range(count):
            start = i * duration
            end = start + duration
            audio = rng.normal(0, noise_amplitude, samples_per_chunk).astype(np.float32)
            chunks.append(
                AudioChunk(
                    index=i,
                    start=start,
                    end=end,
                    audio=audio,
                    sample_rate=sample_rate,
                    language=language,
                )
            )

        return chunks
