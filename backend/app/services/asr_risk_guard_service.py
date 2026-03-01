"""
ASR 风险前置守卫服务。

职责：
- 在音频预检前，基于 chunk 内多点采样做 ASR 风险评估；
- 命中风险时上游可直接强制全局分离并旁路 DNSMOS 预检。
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import librosa
import numpy as np

from app.services.audio.chunk_engine import AudioChunk


@dataclass(frozen=True)
class ASRRiskGuardConfig:
    """ASR 风险守卫参数。"""

    sample_window_sec: float = 1.2
    sample_positions: Tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
    min_chunk_duration_sec: float = 1.5
    min_window_duration_sec: float = 0.8
    min_window_rms: float = 0.01
    flatness_n_fft: int = 2048
    flatness_hop_length: int = 512

    chunk_flatness_mean_min: float = 0.33
    chunk_flatness_max_min: float = 0.65
    chunk_high_flatness_ratio_min: float = 0.40
    chunk_min_valid_windows: int = 4

    video_min_risk_chunks: int = 2
    video_min_risk_ratio: float = 0.05


@dataclass
class ASRRiskChunkMetric:
    """单个 chunk 的 ASR 风险指标。"""

    chunk_index: int
    start_time: float
    end_time: float
    valid_window_count: int
    mean_flatness: float
    max_flatness: float
    high_flatness_ratio: float
    mean_rms: float
    median_crest_db: float
    is_chunk_risky: bool


@dataclass
class ASRRiskGuardResult:
    """视频级 ASR 风险判定结果。"""

    is_risk_detected: bool
    reason: str
    analyzed_chunk_count: int
    risk_chunk_count: int
    risk_chunk_ratio: float
    risk_chunk_indices: List[int]
    max_chunk_flatness_mean: float
    chunk_metrics: List[ASRRiskChunkMetric]


class ASRRiskGuardService:
    """
    ASR 风险守卫服务。

    设计说明：
    - 采用“模板方法”组织流程：采样 -> 指标提取 -> chunk 判定 -> 视频级约束判定；
    - 原因是把“采样策略”和“阈值策略”稳定拆分，后续只需改参数即可迭代。
    """

    def __init__(
        self,
        config: ASRRiskGuardConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or ASRRiskGuardConfig()
        self.logger = logger or logging.getLogger(__name__)

    def evaluate_chunks(self, chunks: Sequence[AudioChunk]) -> ASRRiskGuardResult:
        """执行 ASR 风险评估。"""
        if not chunks:
            return ASRRiskGuardResult(
                is_risk_detected=False,
                reason="无 chunk 输入，跳过 ASR 风险检测",
                analyzed_chunk_count=0,
                risk_chunk_count=0,
                risk_chunk_ratio=0.0,
                risk_chunk_indices=[],
                max_chunk_flatness_mean=0.0,
                chunk_metrics=[],
            )

        metrics: List[ASRRiskChunkMetric] = []
        for chunk in chunks:
            chunk_metric = self._analyze_chunk(chunk)
            if chunk_metric is not None:
                metrics.append(chunk_metric)

        analyzed_chunk_count = len(metrics)
        if analyzed_chunk_count == 0:
            return ASRRiskGuardResult(
                is_risk_detected=False,
                reason="无有效 chunk（时长或能量不足），跳过 ASR 风险检测",
                analyzed_chunk_count=0,
                risk_chunk_count=0,
                risk_chunk_ratio=0.0,
                risk_chunk_indices=[],
                max_chunk_flatness_mean=0.0,
                chunk_metrics=[],
            )

        risk_chunk_indices = [item.chunk_index for item in metrics if item.is_chunk_risky]
        risk_chunk_count = len(risk_chunk_indices)
        risk_chunk_ratio = risk_chunk_count / max(1, analyzed_chunk_count)

        required_risk_chunks = max(
            self.config.video_min_risk_chunks,
            int(math.ceil(analyzed_chunk_count * self.config.video_min_risk_ratio)),
        )
        is_risk_detected = (
            risk_chunk_count >= required_risk_chunks
            and risk_chunk_ratio >= self.config.video_min_risk_ratio
        )
        max_chunk_flatness_mean = max(item.mean_flatness for item in metrics)

        if is_risk_detected:
            reason = (
                "ASR 风险命中：chunk 内多点采样平坦度异常，"
                f"risk={risk_chunk_count}/{analyzed_chunk_count} ({risk_chunk_ratio:.2%})"
            )
        else:
            reason = (
                "ASR 风险未命中："
                f"risk={risk_chunk_count}/{analyzed_chunk_count} ({risk_chunk_ratio:.2%})"
            )

        return ASRRiskGuardResult(
            is_risk_detected=is_risk_detected,
            reason=reason,
            analyzed_chunk_count=analyzed_chunk_count,
            risk_chunk_count=risk_chunk_count,
            risk_chunk_ratio=risk_chunk_ratio,
            risk_chunk_indices=risk_chunk_indices,
            max_chunk_flatness_mean=max_chunk_flatness_mean,
            chunk_metrics=metrics,
        )

    def save_report(self, job_dir: Path, result: ASRRiskGuardResult) -> None:
        """保存 ASR 风险检测报告，便于追踪判定原因。"""
        report_path = Path(job_dir) / "asr_risk_guard.json"
        payload: Dict[str, object] = {
            "timestamp": datetime.now().isoformat(),
            "is_risk_detected": result.is_risk_detected,
            "reason": result.reason,
            "summary": {
                "analyzed_chunk_count": result.analyzed_chunk_count,
                "risk_chunk_count": result.risk_chunk_count,
                "risk_chunk_ratio": round(result.risk_chunk_ratio, 6),
                "risk_chunk_indices": result.risk_chunk_indices,
                "max_chunk_flatness_mean": round(result.max_chunk_flatness_mean, 6),
            },
            "config": asdict(self.config),
            "chunks": [asdict(item) for item in result.chunk_metrics],
        }
        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

    def _analyze_chunk(self, chunk: AudioChunk) -> ASRRiskChunkMetric | None:
        if chunk.audio is None:
            return None
        sample_rate = int(chunk.sample_rate or 16000)
        audio = np.asarray(chunk.audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1).astype(np.float32)
        duration_sec = float(len(audio) / max(1, sample_rate))
        if duration_sec < self.config.min_chunk_duration_sec:
            return None

        window_metrics = self._analyze_windows(audio=audio, sample_rate=sample_rate)
        if not window_metrics:
            return None

        flatness_values = [item["flatness"] for item in window_metrics]
        rms_values = [item["rms"] for item in window_metrics]
        crest_values = [item["crest_db"] for item in window_metrics]
        mean_flatness = float(np.mean(flatness_values))
        max_flatness = float(np.max(flatness_values))
        high_flatness_ratio = float(
            np.mean([value >= self.config.chunk_flatness_mean_min for value in flatness_values])
        )
        valid_window_count = len(window_metrics)

        is_chunk_risky = (
            valid_window_count >= self.config.chunk_min_valid_windows
            and mean_flatness >= self.config.chunk_flatness_mean_min
            and max_flatness >= self.config.chunk_flatness_max_min
            and high_flatness_ratio >= self.config.chunk_high_flatness_ratio_min
        )

        return ASRRiskChunkMetric(
            chunk_index=int(chunk.index),
            start_time=float(chunk.start),
            end_time=float(chunk.end),
            valid_window_count=valid_window_count,
            mean_flatness=mean_flatness,
            max_flatness=max_flatness,
            high_flatness_ratio=high_flatness_ratio,
            mean_rms=float(np.mean(rms_values)),
            median_crest_db=float(np.median(crest_values)),
            is_chunk_risky=is_chunk_risky,
        )

    def _analyze_windows(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, float]]:
        window_size = int(self.config.sample_window_sec * sample_rate)
        min_window_size = int(self.config.min_window_duration_sec * sample_rate)
        if window_size <= 0 or len(audio) < min_window_size:
            return []

        results: List[Dict[str, float]] = []
        for ratio in self.config.sample_positions:
            center_index = int(ratio * len(audio))
            start_index = max(0, center_index - window_size // 2)
            end_index = min(len(audio), start_index + window_size)
            start_index = max(0, end_index - window_size)
            window = audio[start_index:end_index]
            if len(window) < min_window_size:
                continue

            rms = float(np.sqrt(np.mean(window.astype(np.float64) ** 2)))
            if rms < self.config.min_window_rms:
                continue

            peak = float(np.max(np.abs(window)))
            crest_db = float(20.0 * np.log10(max(peak, 1e-9) / max(rms, 1e-9)))
            flatness = self._compute_flatness(window)
            results.append(
                {
                    "flatness": flatness,
                    "rms": rms,
                    "crest_db": crest_db,
                }
            )
        return results

    def _compute_flatness(self, audio: np.ndarray) -> float:
        try:
            stft = np.abs(
                librosa.stft(
                    audio,
                    n_fft=self.config.flatness_n_fft,
                    hop_length=self.config.flatness_hop_length,
                )
            )
            flatness = librosa.feature.spectral_flatness(S=stft)[0]
            if flatness.size == 0:
                return 0.0
            return float(np.mean(flatness))
        except ValueError as exc:
            self.logger.debug("ASR 风险平坦度计算失败（值错误）: %s", exc)
            return 0.0
        except RuntimeError as exc:
            self.logger.debug("ASR 风险平坦度计算失败（运行时错误）: %s", exc)
            return 0.0
