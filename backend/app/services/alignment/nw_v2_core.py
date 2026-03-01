"""
Needleman-Wunsch V2 统一内核。
V3.2.0+dev.20260215.11
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, Sequence, Tuple

import numpy as np


class AlignmentPriorProvider(Protocol):
    """对齐先验提供器（M3 语义先验预留）。"""

    def prior_matrix(
        self,
        seq1: Sequence[str],
        seq2: Sequence[str],
    ) -> List[List[float]]:
        """返回 seq1 x seq2 的先验矩阵。"""


class ZeroPriorProvider:
    """默认先验实现：零矩阵。"""

    def prior_matrix(
        self,
        seq1: Sequence[str],
        seq2: Sequence[str],
    ) -> List[List[float]]:
        return [[0.0 for _ in seq2] for _ in seq1]


@dataclass
class NeedlemanWunschScoreConfig:
    """NW V2 打分配置。"""

    base_match_score: float = 2.0
    base_mismatch_penalty: float = -1.0
    base_gap_penalty: float = -2.0
    confidence_alpha: float = 0.6
    prior_beta: float = 0.4
    is_enable_weighted_scoring: bool = False


class NeedlemanWunschV2Core:
    """
    Needleman-Wunsch 统一内核（Strategy Pattern）。

    Why:
    - 对齐服务与缓冲池共用同一 DP 内核，避免双实现漂移。
    - 通过开关兼容固定分值（V1）与加权分值（V2）。
    """

    def __init__(self, config: Optional[NeedlemanWunschScoreConfig] = None) -> None:
        self.config = config or NeedlemanWunschScoreConfig()

    def align(
        self,
        seq1: Sequence[str],
        seq2: Sequence[str],
        *,
        seq1_confidences: Optional[Sequence[Optional[float]]] = None,
        seq2_confidences: Optional[Sequence[Optional[float]]] = None,
        prior_provider: Optional[AlignmentPriorProvider] = None,
        match_fn: Optional[Callable[[str, str], bool]] = None,
    ) -> List[Tuple[Optional[int], Optional[int]]]:
        m, n = len(seq1), len(seq2)
        if m == 0 and n == 0:
            return []

        score = np.zeros((m + 1, n + 1), dtype=np.float64)
        traceback = np.zeros((m + 1, n + 1), dtype=np.int8)

        conf1 = self._normalize_confidences(seq1_confidences, m)
        conf2 = self._normalize_confidences(seq2_confidences, n)
        prior = self._build_prior_matrix(
            seq1=seq1,
            seq2=seq2,
            prior_provider=prior_provider,
        )
        matcher = match_fn or self._default_match_fn

        for i in range(1, m + 1):
            score[i][0] = score[i - 1][0] + self._delete_penalty(conf1[i - 1])
            traceback[i][0] = 1

        for j in range(1, n + 1):
            score[0][j] = score[0][j - 1] + self._insert_penalty(conf2[j - 1])
            traceback[0][j] = 2

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                is_match = bool(matcher(str(seq1[i - 1]), str(seq2[j - 1])))
                conf_pair = (conf1[i - 1] + conf2[j - 1]) / 2.0
                prior_pair = prior[i - 1][j - 1]

                match = score[i - 1][j - 1] + self._match_or_mismatch_score(
                    is_match=is_match,
                    conf_pair=conf_pair,
                    prior_pair=prior_pair,
                )
                delete = score[i - 1][j] + self._delete_penalty(conf1[i - 1])
                insert = score[i][j - 1] + self._insert_penalty(conf2[j - 1])

                max_score = max(match, delete, insert)
                score[i][j] = max_score

                # tie-break 与历史实现保持一致：优先对角，再上，再左。
                if max_score == match:
                    traceback[i][j] = 0
                elif max_score == delete:
                    traceback[i][j] = 1
                else:
                    traceback[i][j] = 2

        alignment_path: List[Tuple[Optional[int], Optional[int]]] = []
        i, j = m, n
        while i > 0 or j > 0:
            if i == 0:
                alignment_path.append((None, j - 1))
                j -= 1
            elif j == 0:
                alignment_path.append((i - 1, None))
                i -= 1
            else:
                direction = traceback[i][j]
                if direction == 0:
                    alignment_path.append((i - 1, j - 1))
                    i -= 1
                    j -= 1
                elif direction == 1:
                    alignment_path.append((i - 1, None))
                    i -= 1
                else:
                    alignment_path.append((None, j - 1))
                    j -= 1

        alignment_path.reverse()
        return alignment_path

    def _match_or_mismatch_score(
        self,
        *,
        is_match: bool,
        conf_pair: float,
        prior_pair: float,
    ) -> float:
        if not self.config.is_enable_weighted_scoring:
            return (
                float(self.config.base_match_score)
                if is_match
                else float(self.config.base_mismatch_penalty)
            )
        if is_match:
            return float(
                self.config.base_match_score
                + self.config.confidence_alpha * conf_pair
                + self.config.prior_beta * prior_pair
            )
        return float(
            self.config.base_mismatch_penalty
            * (0.5 + conf_pair)
        )

    def _delete_penalty(self, seq1_confidence: float) -> float:
        if not self.config.is_enable_weighted_scoring:
            return float(self.config.base_gap_penalty)
        return float(self.config.base_gap_penalty * (0.5 + seq1_confidence))

    def _insert_penalty(self, seq2_confidence: float) -> float:
        if not self.config.is_enable_weighted_scoring:
            return float(self.config.base_gap_penalty)
        return float(self.config.base_gap_penalty * (0.5 + seq2_confidence))

    @staticmethod
    def _normalize_confidences(
        confidences: Optional[Sequence[Optional[float]]],
        size: int,
    ) -> List[float]:
        if not confidences:
            return [0.0 for _ in range(size)]
        normalized = [0.0 for _ in range(size)]
        for index in range(size):
            value: Optional[float]
            if index < len(confidences):
                value = confidences[index]
            else:
                value = None
            if value is None:
                normalized[index] = 0.0
            else:
                normalized[index] = max(0.0, min(1.0, float(value)))
        return normalized

    @staticmethod
    def _build_prior_matrix(
        *,
        seq1: Sequence[str],
        seq2: Sequence[str],
        prior_provider: Optional[AlignmentPriorProvider],
    ) -> List[List[float]]:
        provider = prior_provider or ZeroPriorProvider()
        try:
            matrix = provider.prior_matrix(seq1, seq2)
        except Exception:
            matrix = ZeroPriorProvider().prior_matrix(seq1, seq2)

        if len(matrix) != len(seq1):
            return ZeroPriorProvider().prior_matrix(seq1, seq2)
        for row in matrix:
            if len(row) != len(seq2):
                return ZeroPriorProvider().prior_matrix(seq1, seq2)

        normalized: List[List[float]] = []
        for row in matrix:
            normalized.append([max(0.0, min(1.0, float(item))) for item in row])
        return normalized

    @staticmethod
    def _default_match_fn(left: str, right: str) -> bool:
        return left == right


__all__ = [
    "AlignmentPriorProvider",
    "NeedlemanWunschScoreConfig",
    "NeedlemanWunschV2Core",
    "ZeroPriorProvider",
]
