"""
预处理产物契约模型。

用于承载 Phase 0 的结构化预处理输出，逐步替代裸 List[AudioChunk]。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.services.audio.chunk_engine import AudioChunk


@dataclass
class PreprocessArtifacts:
    """预处理域的统一产物对象。"""

    chunks: list[AudioChunk]
    vad_intervals: list[tuple[float, float]] = field(default_factory=list)
    full_audio: np.ndarray | None = None
    sample_rate: int = 16000
    language_map: dict[int, dict[str, Any]] = field(default_factory=dict)

