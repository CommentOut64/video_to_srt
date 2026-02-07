"""
对齐质量统计计算器。
V3.2.0+dev.20260202.07
"""
# V3.2.0+dev.20260205.09: 对齐统计支持词级置信度空值。
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from app.models.confidence_models import AlignedWord, AlignmentStatus
from app.services.alignment.gap_resolver import GapResolution


@dataclass
class QualityStats:
    """对齐质量统计指标。"""

    coverage: float
    gap_ratio: float
    alignment_score: float
    gap_positions: List[int] = field(default_factory=list)
    gap_resolution: Optional[GapResolution] = None


class QualityStatsCalculator:
    """质量统计计算器。"""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def compute(
        self,
        aligned_words: List[AlignedWord],
        total_tokens: Optional[int] = None,
        gap_positions: Optional[List[int]] = None,
        gap_resolution: Optional[GapResolution] = None,
    ) -> QualityStats:
        if not aligned_words:
            return QualityStats(
                coverage=0.0,
                gap_ratio=0.0,
                alignment_score=0.0,
                gap_positions=[],
                gap_resolution=gap_resolution,
            )

        total_tokens = total_tokens if total_tokens is not None else len(aligned_words)
        gap_count = sum(1 for w in aligned_words if w.is_pseudo or w.alignment_status == AlignmentStatus.PSEUDO)
        aligned_count = max(total_tokens - gap_count, 0)

        coverage = aligned_count / max(total_tokens, 1)
        gap_ratio = gap_count / max(len(aligned_words), 1)
        alignment_score = self._compute_alignment_score(aligned_words)

        if gap_positions is None:
            gap_positions = self._find_gap_clusters(aligned_words)

        return QualityStats(
            coverage=coverage,
            gap_ratio=gap_ratio,
            alignment_score=alignment_score,
            gap_positions=gap_positions,
            gap_resolution=gap_resolution,
        )

    def _compute_alignment_score(self, aligned_words: List[AlignedWord]) -> float:
        matched_count = sum(1 for w in aligned_words if w.alignment_status == AlignmentStatus.MATCHED)
        match_ratio = matched_count / max(len(aligned_words), 1)
        confidences = [w.final_confidence for w in aligned_words if w.final_confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return 0.6 * match_ratio + 0.4 * avg_confidence

    @staticmethod
    def _find_gap_clusters(aligned_words: List[AlignedWord]) -> List[int]:
        clusters: List[int] = []
        in_gap = False
        for idx, word in enumerate(aligned_words):
            is_gap = word.is_pseudo or word.alignment_status == AlignmentStatus.PSEUDO
            if is_gap and not in_gap:
                clusters.append(idx)
                in_gap = True
            elif not is_gap:
                in_gap = False
        return clusters
