"""
Pyannote segmentation 服务（Phase 2）。

设计模式：Adapter Pattern。
原因：对业务层隐藏 pyannote 推理细节，统一模型管理与
推理输入输出，避免在编排层直接耦合第三方 API。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from app.services.model_manager_v2 import get_model_manager_v2
from app.services.pyannote_compat import load_pyannote_model


@dataclass(frozen=True)
class SegmentationFrame:
    """单帧 segmentation 输出。"""

    time: float
    score: float


@dataclass(frozen=True)
class SegmentationResult:
    """segmentation 推理结果。"""

    boundaries: list[float]
    frames: list[SegmentationFrame]


@dataclass(frozen=True)
class PyannoteSegmentationConfig:
    """pyannote segmentation 配置。"""

    model_id: str = "pyannote-segmentation-3-0"
    boundary_threshold: float = 0.55
    min_boundary_interval_sec: float = 0.20
    prefer_device: str = "auto"


def map_frame_times_to_word_boundaries(
    *,
    frame_times: Sequence[float],
    word_boundaries: Sequence[float],
    tolerance_sec: float = 0.22,
    is_enable_mapping: bool = False,
) -> List[Dict[str, Any]]:
    """
    将 pyannote 帧时间映射到词边界（阶段3契约）。

    Why:
    - 统一 `raw_time/snapped_time/delta_ms/mapping_reason/mapping_quality` 字段口径，
      供 FactBuilder 在不改主链裁决的前提下产出可观测事实。
    """
    normalized_frames = sorted({float(item) for item in frame_times or []})
    normalized_boundaries = sorted({float(item) for item in word_boundaries or []})
    mappings: List[Dict[str, Any]] = []
    if not normalized_frames:
        return mappings

    if not is_enable_mapping:
        for raw_time in normalized_frames:
            mappings.append(
                {
                    "raw_time": raw_time,
                    "snapped_time": raw_time,
                    "delta_ms": 0.0,
                    "mapping_reason": "time_mapping_disabled",
                    "mapping_quality": "disabled",
                }
            )
        return mappings

    if not normalized_boundaries:
        for raw_time in normalized_frames:
            mappings.append(
                {
                    "raw_time": raw_time,
                    "snapped_time": raw_time,
                    "delta_ms": 0.0,
                    "mapping_reason": "no_word_boundaries",
                    "mapping_quality": "deferred",
                }
            )
        return mappings

    tolerance = max(0.0, float(tolerance_sec))
    for raw_time in normalized_frames:
        snapped_time = min(normalized_boundaries, key=lambda item: abs(item - raw_time))
        delta_ms = (float(snapped_time) - float(raw_time)) * 1000.0
        if abs(float(snapped_time) - float(raw_time)) <= tolerance:
            mappings.append(
                {
                    "raw_time": float(raw_time),
                    "snapped_time": float(snapped_time),
                    "delta_ms": float(delta_ms),
                    "mapping_reason": "word_boundary_mapper",
                    "mapping_quality": "snapped",
                }
            )
        else:
            mappings.append(
                {
                    "raw_time": float(raw_time),
                    "snapped_time": float(raw_time),
                    "delta_ms": 0.0,
                    "mapping_reason": "no_anchor_within_tolerance",
                    "mapping_quality": "deferred",
                }
            )
    return mappings


class PyannoteSegmentationService:
    """使用 `pyannote.audio` 模型进行说话人变化边界检测。"""

    def __init__(
        self,
        config: Optional[PyannoteSegmentationConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or PyannoteSegmentationConfig()
        self.logger = logger or logging.getLogger(__name__)

    def run(
        self,
        *,
        audio: np.ndarray,
        sample_rate: int,
    ) -> SegmentationResult:
        """执行 segmentation 并返回边界点。"""
        if audio.size == 0:
            return SegmentationResult(boundaries=[], frames=[])

        model = self._acquire_model()
        if model is None:
            return SegmentationResult(boundaries=[], frames=[])

        try:
            import torch
            from pyannote.audio import Inference
        except ImportError as exc:
            raise RuntimeError("未安装 pyannote.audio，无法执行 segmentation") from exc

        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        # 设计说明（Strategy Pattern）：
        # segmentation-3.0 属于 permutation-invariant 任务，默认推理会返回
        # 原始 chunk 级三维张量 (chunk, frame, class)，其 sliding_window 表示
        # chunk 时间轴而非 frame 时间轴。
        # 这里先把 class 维度压缩为单分数轨（与说话人身份无关），再让
        # pyannote 执行重叠聚合，得到稳定的一维 frame 时间轴。
        inference = Inference(
            model,
            pre_aggregation_hook=self._collapse_permutation_invariant_scores,
        )
        sliding_scores = inference({"waveform": waveform, "sample_rate": sample_rate})

        data = np.asarray(sliding_scores.data, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.ndim != 2:
            raise RuntimeError(f"segmentation 输出维度不支持: {data.shape}")

        frame_scores = np.max(data, axis=1)
        frame_step = float(sliding_scores.sliding_window.step)
        frame_start = float(sliding_scores.sliding_window.start)
        audio_duration_sec = float(audio.shape[0]) / float(sample_rate)

        boundaries: list[float] = []
        frames: list[SegmentationFrame] = []
        last_boundary = -1e9
        threshold = float(self.config.boundary_threshold)

        for idx, score in enumerate(frame_scores.tolist()):
            time_sec = frame_start + idx * frame_step
            if time_sec > audio_duration_sec:
                time_sec = audio_duration_sec
            value = float(score)
            frames.append(SegmentationFrame(time=time_sec, score=value))
            if value < threshold:
                continue
            if time_sec - last_boundary < self.config.min_boundary_interval_sec:
                continue
            boundaries.append(time_sec)
            last_boundary = time_sec

        return SegmentationResult(boundaries=boundaries, frames=frames)

    @staticmethod
    def _collapse_permutation_invariant_scores(scores: np.ndarray) -> np.ndarray:
        """将 permutation-invariant 多类输出压缩为单分数轨后再聚合。"""
        if scores.ndim == 3:
            return np.max(scores, axis=2, keepdims=True)
        if scores.ndim == 2:
            return scores[..., np.newaxis]
        raise RuntimeError(f"不支持的 segmentation 原始维度: {scores.shape}")

    def _acquire_model(self):
        """通过 ModelManagerV2 获取模型目录，并用 pyannote 官方方式加载。"""
        manager = get_model_manager_v2()
        spec = manager.registry.get(self.config.model_id)
        effective_model = manager.runtime_config_service.get_effective_model(spec).get("effective", {})
        device = str(effective_model.get("device") or self.config.prefer_device)

        local_path = Path(manager.ensure_available(self.config.model_id))
        if not local_path.exists():
            raise FileNotFoundError(f"segmentation 模型目录不存在: {local_path}")

        model = load_pyannote_model(
            checkpoint=str(local_path),
            logger=self.logger,
        )

        try:
            if device.startswith("cuda"):
                import torch

                if torch.cuda.is_available():
                    model = model.to(torch.device(device))
        except Exception as exc:
            self.logger.warning("segmentation 模型切换设备失败，回退默认设备: %s", exc)

        return model
