# -*- coding: utf-8 -*-
"""
SRT 比较器。

比较两份 SRT 的文本差异和时间戳偏移，生成结构化的比较结果。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .srt_parser import SRTEntry


@dataclass
class TimestampDiff:
    """单条时间戳差异。"""

    entry_index: int
    field: str  # "start" 或 "end"
    reference_value: float
    candidate_value: float
    offset_ms: float  # 毫秒


@dataclass
class TextDiff:
    """单条文本差异。"""

    entry_index: int
    reference_text: str
    candidate_text: str


@dataclass
class ComparisonResult:
    """SRT 比较结果。"""

    is_identical: bool = True
    text_diffs: List[TextDiff] = field(default_factory=list)
    timestamp_diffs: List[TimestampDiff] = field(default_factory=list)
    segment_count_match: bool = True
    reference_count: int = 0
    candidate_count: int = 0
    max_timestamp_offset_ms: float = 0.0
    avg_timestamp_offset_ms: float = 0.0


class SRTComparator:
    """比较两组 SRTEntry，输出结构化的差异报告。"""

    @staticmethod
    def compare(
        reference: List[SRTEntry],
        candidate: List[SRTEntry],
        tolerance_ms: float = 100.0,
    ) -> ComparisonResult:
        """
        比较参考 SRT 和候选 SRT。

        Args:
            reference: 参考条目列表
            candidate: 候选条目列表
            tolerance_ms: 时间戳容差（毫秒）

        Returns:
            ComparisonResult
        """
        result = ComparisonResult(
            reference_count=len(reference),
            candidate_count=len(candidate),
        )

        if len(reference) != len(candidate):
            result.segment_count_match = False
            result.is_identical = False

        # 逐条比较（取最小长度）
        compare_count = min(len(reference), len(candidate))
        all_offsets: List[float] = []

        for i in range(compare_count):
            ref = reference[i]
            cand = candidate[i]

            # 文本比较
            if ref.text.strip() != cand.text.strip():
                result.text_diffs.append(
                    TextDiff(
                        entry_index=i + 1,
                        reference_text=ref.text,
                        candidate_text=cand.text,
                    )
                )
                result.is_identical = False

            # 时间戳比较
            for ts_field in ("start", "end"):
                ref_val = getattr(ref, ts_field)
                cand_val = getattr(cand, ts_field)
                offset_ms = abs(ref_val - cand_val) * 1000

                all_offsets.append(offset_ms)

                if offset_ms > tolerance_ms:
                    result.timestamp_diffs.append(
                        TimestampDiff(
                            entry_index=i + 1,
                            field=ts_field,
                            reference_value=ref_val,
                            candidate_value=cand_val,
                            offset_ms=offset_ms,
                        )
                    )
                    result.is_identical = False

        if all_offsets:
            result.max_timestamp_offset_ms = max(all_offsets)
            result.avg_timestamp_offset_ms = sum(all_offsets) / len(all_offsets)

        return result

    @staticmethod
    def format_report(result: ComparisonResult) -> str:
        """将比较结果格式化为人类可读报告。"""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  SRT 比较报告")
        lines.append("=" * 60)

        lines.append(f"条目数量: 参考={result.reference_count}, 候选={result.candidate_count}")
        lines.append(f"条目数量匹配: {'是' if result.segment_count_match else '否'}")
        lines.append(f"完全一致: {'是' if result.is_identical else '否'}")
        lines.append(f"时间戳最大偏移: {result.max_timestamp_offset_ms:.1f}ms")
        lines.append(f"时间戳平均偏移: {result.avg_timestamp_offset_ms:.1f}ms")

        if result.text_diffs:
            lines.append("")
            lines.append(f"[文本差异] 共 {len(result.text_diffs)} 处")
            for diff in result.text_diffs[:10]:  # 最多显示 10 条
                lines.append(f"  #{diff.entry_index}:")
                lines.append(f"    参考: {diff.reference_text}")
                lines.append(f"    候选: {diff.candidate_text}")

        if result.timestamp_diffs:
            lines.append("")
            lines.append(f"[时间戳差异] 共 {len(result.timestamp_diffs)} 处 (超出容差)")
            for diff in result.timestamp_diffs[:10]:
                lines.append(
                    f"  #{diff.entry_index} {diff.field}: "
                    f"参考={diff.reference_value:.3f}s, "
                    f"候选={diff.candidate_value:.3f}s, "
                    f"偏移={diff.offset_ms:.1f}ms"
                )

        lines.append("=" * 60)
        return "\n".join(lines)
