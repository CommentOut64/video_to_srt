"""ChunkProjector：将终稿句子投影到 chunk 归属。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

from app.models.sensevoice_models import SentenceSegment


@dataclass(frozen=True)
class ChunkWindow:
    """chunk 时间窗。"""

    chunk_ref: int | str
    start: float
    end: float

    def __post_init__(self) -> None:
        if float(self.end) < float(self.start):
            raise ValueError(f"ChunkWindow 非法：end({self.end}) < start({self.start})")


@dataclass(frozen=True)
class ChunkProjection:
    """单 chunk 投影结果。"""

    chunk_window: ChunkWindow
    sentence_segments: tuple[SentenceSegment, ...]


class ChunkProjector:
    """chunk 归属投影器。"""

    def project(
        self,
        *,
        sentence_segments: Sequence[SentenceSegment],
        chunk_windows: Sequence[ChunkWindow],
    ) -> tuple[ChunkProjection, ...]:
        if not chunk_windows:
            return tuple()

        buckets: list[list[SentenceSegment]] = [[] for _ in chunk_windows]
        for sentence in sentence_segments:
            target_index = self._resolve_target_window(sentence=sentence, chunk_windows=chunk_windows)
            cloned = deepcopy(sentence)
            cloned.chunk_uid = str(chunk_windows[target_index].chunk_ref)
            buckets[target_index].append(cloned)

        projections: list[ChunkProjection] = []
        for index, window in enumerate(chunk_windows):
            projections.append(
                ChunkProjection(
                    chunk_window=window,
                    sentence_segments=tuple(buckets[index]),
                )
            )
        return tuple(projections)

    @staticmethod
    def _resolve_target_window(
        *,
        sentence: SentenceSegment,
        chunk_windows: Sequence[ChunkWindow],
    ) -> int:
        best_index = 0
        best_overlap = -1.0
        sent_start = float(sentence.start)
        sent_end = float(sentence.end)
        for index, window in enumerate(chunk_windows):
            overlap = max(
                0.0,
                min(sent_end, float(window.end)) - max(sent_start, float(window.start)),
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index
        if best_overlap > 0.0:
            return best_index

        # 若无重叠，按句子中心点就近投影到最近 chunk。
        center = (sent_start + sent_end) / 2.0
        distances = [
            abs(center - ((float(window.start) + float(window.end)) / 2.0))
            for window in chunk_windows
        ]
        return min(range(len(distances)), key=lambda idx: distances[idx])


__all__ = ["ChunkProjection", "ChunkProjector", "ChunkWindow"]
