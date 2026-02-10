# -*- coding: utf-8 -*-
"""
断句准确率评估器。

以 SRT 条目 start 为断句边界，计算 precision/recall/F1。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

from ci_tests.integration.harness import SRTEntry


@dataclass
class SegmentationEvalResult:
    """断句评估结果。"""

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    over_segmentation: int = 0  # 过度分句数
    under_segmentation: int = 0  # 不足分句数
    matched_boundaries: int = 0
    ref_boundary_count: int = 0
    hyp_boundary_count: int = 0


class SegmentationEvaluator:
    """断句准确率评估器。"""

    @staticmethod
    def evaluate(
        ref_entries: List[SRTEntry],
        hyp_entries: List[SRTEntry],
        boundary_tolerance_ms: float = 500.0,
    ) -> SegmentationEvalResult:
        """评估断句准确率。

        以每个条目的 start 时间作为断句边界。
        """
        if not ref_entries:
            return SegmentationEvalResult()

        # 提取边界（跳过第一个，因为第一个 start 不是断句点）
        ref_boundaries = [e.start for e in ref_entries[1:]]
        hyp_boundaries = [e.start for e in hyp_entries[1:]]

        result = SegmentationEvalResult(
            ref_boundary_count=len(ref_boundaries),
            hyp_boundary_count=len(hyp_boundaries),
        )

        if not ref_boundaries and not hyp_boundaries:
            result.precision = 1.0
            result.recall = 1.0
            result.f1 = 1.0
            return result

        # 匹配边界
        tolerance_s = boundary_tolerance_ms / 1000.0
        matched_ref: Set[int] = set()
        matched_hyp: Set[int] = set()

        for i, ref_b in enumerate(ref_boundaries):
            for j, hyp_b in enumerate(hyp_boundaries):
                if j in matched_hyp:
                    continue
                if abs(ref_b - hyp_b) <= tolerance_s:
                    matched_ref.add(i)
                    matched_hyp.add(j)
                    break

        result.matched_boundaries = len(matched_ref)

        # Precision: 假设边界中有多少是正确的
        if hyp_boundaries:
            result.precision = len(matched_hyp) / len(hyp_boundaries)

        # Recall: 参考边界中有多少被找到
        if ref_boundaries:
            result.recall = len(matched_ref) / len(ref_boundaries)

        # F1
        if result.precision + result.recall > 0:
            result.f1 = (
                2 * result.precision * result.recall
                / (result.precision + result.recall)
            )

        # 过度/不足分句
        result.over_segmentation = len(hyp_boundaries) - len(matched_hyp)
        result.under_segmentation = len(ref_boundaries) - len(matched_ref)

        return result
