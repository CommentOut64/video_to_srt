# -*- coding: utf-8 -*-
"""
质量评估报告生成器。

生成 JSON + TXT 双格式报告。
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .evaluator import QualityEvalResult


class QualityReportGenerator:
    """质量评估报告生成器。"""

    @staticmethod
    def to_dict(result: QualityEvalResult) -> dict:
        """将评估结果转为 JSON 可序列化字典。"""
        report: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "overall_score": round(result.overall_score, 2),
            "grade": result.grade,
            "entry_counts": {
                "reference": result.reference_entry_count,
                "hypothesis": result.hypothesis_entry_count,
            },
        }

        # CER/WER
        if result.cer_result:
            report["cer"] = {
                "value": round(result.cer_result.cer * 100, 2),
                "edit_distance": result.cer_result.edit_distance,
                "insertions": result.cer_result.insertions,
                "deletions": result.cer_result.deletions,
                "substitutions": result.cer_result.substitutions,
            }

        if result.wer_result:
            report["wer"] = {
                "value": round(result.wer_result.wer * 100, 2),
                "edit_distance": result.wer_result.edit_distance,
            }

        # 时间戳
        if result.timestamp_result:
            report["timestamp"] = {
                "avg_offset_ms": round(result.timestamp_result.avg_offset_ms, 1),
                "max_offset_ms": round(result.timestamp_result.max_offset_ms, 1),
                "p95_offset_ms": round(result.timestamp_result.p95_offset_ms, 1),
                "within_tolerance_ratio": round(result.timestamp_result.within_tolerance_ratio, 3),
            }

        # 断句
        if result.segmentation_result:
            report["segmentation"] = {
                "precision": round(result.segmentation_result.precision, 3),
                "recall": round(result.segmentation_result.recall, 3),
                "f1": round(result.segmentation_result.f1, 3),
                "over_segmentation": result.segmentation_result.over_segmentation,
                "under_segmentation": result.segmentation_result.under_segmentation,
            }

        return report

    @staticmethod
    def format_text(result: QualityEvalResult) -> str:
        """格式化为人类可读文本报告。"""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  AnchorFlux 质量评估报告")
        lines.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)
        lines.append("")

        # 综合评分
        lines.append(f"[综合评分] 总分: {result.overall_score:.1f}/100 (等级: {result.grade})")
        lines.append("")

        # CER/WER
        if result.cer_result:
            lines.append(f"[字符错误率] CER: {result.cer_result.cer * 100:.1f}%")
        if result.wer_result:
            lines.append(f"[词错误率] WER: {result.wer_result.wer * 100:.1f}%")

        # 时间戳
        if result.timestamp_result:
            ts = result.timestamp_result
            lines.append("")
            lines.append("[时间戳偏移]")
            lines.append(f"  平均: {ts.avg_offset_ms:.1f}ms")
            lines.append(f"  最大: {ts.max_offset_ms:.1f}ms")
            lines.append(f"  P95: {ts.p95_offset_ms:.1f}ms")
            lines.append(f"  容差内比例: {ts.within_tolerance_ratio * 100:.1f}%")

        # 断句
        if result.segmentation_result:
            seg = result.segmentation_result
            lines.append("")
            lines.append("[断句准确率]")
            lines.append(f"  F1: {seg.f1 * 100:.1f}%")
            lines.append(f"  精确率: {seg.precision * 100:.1f}%")
            lines.append(f"  召回率: {seg.recall * 100:.1f}%")
            lines.append(f"  过度分句: {seg.over_segmentation}")
            lines.append(f"  不足分句: {seg.under_segmentation}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    @staticmethod
    def save(result: QualityEvalResult, output_dir: Path | str) -> Path:
        """保存报告到文件（JSON + TXT）。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON
        json_path = output_dir / f"quality_report_{timestamp}.json"
        report_dict = QualityReportGenerator.to_dict(result)
        json_path.write_text(
            json.dumps(report_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # TXT
        txt_path = output_dir / f"quality_report_{timestamp}.txt"
        txt_path.write_text(
            QualityReportGenerator.format_text(result),
            encoding="utf-8",
        )

        return json_path
