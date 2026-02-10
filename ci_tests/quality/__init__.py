# -*- coding: utf-8 -*-
"""
质量评估体系。

提供转录质量评估能力：
- CER/WER 计算
- 时间戳偏移评估
- 断句准确率评估
- 综合评分与报告生成
"""
import sys
from pathlib import Path

# 确保 backend 目录在 Python 路径中
_project_root = Path(__file__).resolve().parent.parent.parent
_backend_path = _project_root / "backend"
if str(_backend_path) not in sys.path:
    sys.path.insert(0, str(_backend_path))

from .evaluator import QualityEvaluator, QualityEvalConfig, QualityEvalResult
from .report_generator import QualityReportGenerator

__all__ = [
    "QualityEvaluator",
    "QualityEvalConfig",
    "QualityEvalResult",
    "QualityReportGenerator",
]
