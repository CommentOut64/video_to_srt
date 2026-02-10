# -*- coding: utf-8 -*-
"""
集成测试报告构建器。

生成 JSON + TXT 双格式的测试报告。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StageReport:
    """单阶段报告。"""

    stage_name: str
    status: str  # "passed" / "failed" / "skipped"
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class ReportBuilder:
    """集成测试报告构建器。"""

    @staticmethod
    def build_integration_report(
        result: Any,
        config: Any,
        stages: Optional[List[StageReport]] = None,
    ) -> dict:
        """构建 JSON 格式的集成测试报告。

        Args:
            result: PipelineTestResult 或类似对象
            config: PipelineTestConfig 或类似对象
            stages: 阶段报告列表

        Returns:
            JSON 可序列化的字典
        """
        report: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "success": getattr(result, "success", False),
            "config": {
                "transcription_profile": getattr(config, "transcription_profile", "unknown"),
                "use_real_engines": getattr(config, "use_real_engines", False),
                "use_real_preprocessing": getattr(config, "use_real_preprocessing", False),
                "chunk_count": getattr(config, "chunk_count", 0),
            },
            "summary": {
                "total_contexts": len(getattr(result, "contexts", [])),
                "total_sentences": len(getattr(result, "sentences", [])),
                "srt_length": len(getattr(result, "srt_content", "")),
                "has_error": getattr(result, "error", None) is not None,
            },
            "stage_timings": getattr(result, "stage_timings", {}),
        }

        if stages:
            report["stages"] = [asdict(s) for s in stages]

        if getattr(result, "error", None):
            report["error"] = str(result.error)

        return report

    @staticmethod
    def format_text_report(report: dict) -> str:
        """将 JSON 报告格式化为人类可读文本。"""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  AnchorFlux 集成测试报告")
        lines.append(f"  生成时间: {report.get('generated_at', 'N/A')}")
        lines.append("=" * 60)
        lines.append("")

        # 结果概要
        success = report.get("success", False)
        lines.append(f"[结果] {'通过' if success else '失败'}")

        summary = report.get("summary", {})
        lines.append(f"[上下文数量] {summary.get('total_contexts', 0)}")
        lines.append(f"[句子数量] {summary.get('total_sentences', 0)}")
        lines.append(f"[SRT 长度] {summary.get('srt_length', 0)} 字符")

        # 配置
        config = report.get("config", {})
        lines.append("")
        lines.append("[配置]")
        lines.append(f"  转录模式: {config.get('transcription_profile', 'N/A')}")
        lines.append(f"  真实引擎: {config.get('use_real_engines', False)}")
        lines.append(f"  真实预处理: {config.get('use_real_preprocessing', False)}")
        lines.append(f"  Chunk 数量: {config.get('chunk_count', 0)}")

        # 阶段耗时
        timings = report.get("stage_timings", {})
        if timings:
            lines.append("")
            lines.append("[阶段耗时]")
            for stage, ms in timings.items():
                lines.append(f"  {stage}: {ms:.1f}ms")

        # 阶段详情
        stages = report.get("stages", [])
        if stages:
            lines.append("")
            lines.append("[阶段详情]")
            for stage in stages:
                status_icon = "通过" if stage["status"] == "passed" else "失败"
                lines.append(f"  {stage['stage_name']}: {status_icon} ({stage['duration_ms']:.1f}ms)")
                if stage.get("errors"):
                    for err in stage["errors"]:
                        lines.append(f"    错误: {err}")

        # 错误信息
        if report.get("error"):
            lines.append("")
            lines.append(f"[错误] {report['error']}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    @staticmethod
    def save_report(report: dict, output_dir: Path | str) -> Path:
        """保存报告到文件（JSON + TXT 双格式）。

        Returns:
            JSON 文件路径
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON 格式
        json_path = output_dir / f"integration_report_{timestamp}.json"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        # TXT 格式
        txt_path = output_dir / f"integration_report_{timestamp}.txt"
        txt_path.write_text(ReportBuilder.format_text_report(report), encoding="utf-8")

        return json_path
