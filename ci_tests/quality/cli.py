# -*- coding: utf-8 -*-
"""
质量评估 CLI 入口。

使用方法：
    # 仅比较两个 SRT
    python -m tests.quality.cli --reference input/standard.srt --hypothesis output/generated.srt

    # 指定输出目录
    python -m tests.quality.cli --reference input/standard.srt --hypothesis output/generated.srt --output-dir results/
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# 确保 backend 目录在 Python 路径中
_project_root = Path(__file__).resolve().parent.parent.parent
_backend_path = _project_root / "backend"
if str(_backend_path) not in sys.path:
    sys.path.insert(0, str(_backend_path))

import click

from .evaluator import QualityEvalConfig, QualityEvaluator
from .report_generator import QualityReportGenerator


@click.command()
@click.option(
    "--reference",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="参考 SRT 文件路径",
)
@click.option(
    "--hypothesis",
    "-h",
    "hypothesis_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="待评估 SRT 文件路径",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("quality_reports"),
    help="报告输出目录",
)
@click.option(
    "--language",
    "-l",
    default="zh",
    help="语言代码 (zh/en/ja)",
)
@click.option(
    "--timestamp-tolerance",
    type=float,
    default=200.0,
    help="时间戳容差 (毫秒)",
)
@click.option(
    "--boundary-tolerance",
    type=float,
    default=500.0,
    help="断句边界容差 (毫秒)",
)
def main(
    reference: Path,
    hypothesis_path: Path,
    output_dir: Path,
    language: str,
    timestamp_tolerance: float,
    boundary_tolerance: float,
) -> None:
    """AnchorFlux 质量评估工具。"""
    click.echo("=" * 50)
    click.echo("  AnchorFlux 质量评估")
    click.echo("=" * 50)

    config = QualityEvalConfig(
        reference_srt_path=reference,
        hypothesis_srt_path=hypothesis_path,
        language=language,
        timestamp_tolerance_ms=timestamp_tolerance,
        boundary_tolerance_ms=boundary_tolerance,
    )

    click.echo(f"参考文件: {reference}")
    click.echo(f"待评估文件: {hypothesis_path}")
    click.echo("")

    evaluator = QualityEvaluator(config)
    result = evaluator.evaluate()

    # 输出结果
    click.echo(QualityReportGenerator.format_text(result))

    # 保存报告
    json_path = QualityReportGenerator.save(result, output_dir)
    click.echo(f"报告已保存: {json_path}")


if __name__ == "__main__":
    main()
