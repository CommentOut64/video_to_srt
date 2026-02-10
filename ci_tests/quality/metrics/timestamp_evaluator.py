# -*- coding: utf-8 -*-
"""
时间戳偏移评估器。

贪心最近匹配对齐，计算 avg/max/p95 偏移。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from ci_tests.integration.harness import SRTEntry


@dataclass
class TimestampEvalResult:
    """时间戳评估结果。"""

    avg_offset_ms: float = 0.0
    max_offset_ms: float = 0.0
    p95_offset_ms: float = 0.0
    within_tolerance_ratio: float = 1.0  # 容差内比例
    matched_count: int = 0
    total_ref_count: int = 0
    all_offsets_ms: List[float] = field(default_factory=list)


class TimestampEvaluator:
    """时间戳偏移评估器。"""

    @staticmethod
    def evaluate(
        ref_entries: List[SRTEntry],
        hyp_entries: List[SRTEntry],
        tolerance_ms: float = 200.0,
    ) -> TimestampEvalResult:
        """评估时间戳偏移。

        使用贪心最近匹配对齐参考和假设条目。
        """
        if not ref_entries:
            return TimestampEvalResult()

        result = TimestampEvalResult(total_ref_count=len(ref_entries))
        all_offsets: List[float] = []

        # 贪心匹配：对每个参考条目找最近的假设条目
        used_hyp = set()
        for ref in ref_entries:
            best_hyp_idx = -1
            best_offset = float("inf")

            for i, hyp in enumerate(hyp_entries):
                if i in used_hyp:
                    continue
                # 计算 start 时间偏移
                offset = abs(ref.start - hyp.start) * 1000
                if offset < best_offset:
                    best_offset = offset
                    best_hyp_idx = i

            if best_hyp_idx >= 0:
                used_hyp.add(best_hyp_idx)
                hyp = hyp_entries[best_hyp_idx]

                # 记录 start 和 end 偏移
                start_offset = abs(ref.start - hyp.start) * 1000
                end_offset = abs(ref.end - hyp.end) * 1000
                all_offsets.extend([start_offset, end_offset])
                result.matched_count += 1

        result.all_offsets_ms = all_offsets

        if all_offsets:
            result.avg_offset_ms = sum(all_offsets) / len(all_offsets)
            result.max_offset_ms = max(all_offsets)

            # P95
            sorted_offsets = sorted(all_offsets)
            p95_idx = int(len(sorted_offsets) * 0.95)
            p95_idx = min(p95_idx, len(sorted_offsets) - 1)
            result.p95_offset_ms = sorted_offsets[p95_idx]

            # 容差内比例
            within = sum(1 for o in all_offsets if o <= tolerance_ms)
            result.within_tolerance_ratio = within / len(all_offsets)

        return result
