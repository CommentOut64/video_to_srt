# -*- coding: utf-8 -*-
"""
质量评估 pytest 入口。

使用方法：
    pytest tests/quality/test_quality_evaluation.py -v \
        --reference=input/standard.srt --hypothesis=output/generated.srt
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ci_tests.integration.harness import SRTEntry, SRTParser
from ci_tests.quality.evaluator import QualityEvalConfig, QualityEvaluator
from ci_tests.quality.metrics import CERCalculator, WERCalculator


class TestMetricsUnit:
    """指标计算单元测试。"""

    def test_cer_identical(self):
        """完全相同文本 CER 应为 0。"""
        result = CERCalculator.calculate("你好世界", "你好世界")
        assert result.cer == 0.0
        assert result.edit_distance == 0

    def test_cer_different(self):
        """不同文本应有正确的 CER。"""
        result = CERCalculator.calculate("你好世界", "你好中国")
        assert result.cer > 0
        assert result.substitutions == 2

    def test_wer_chinese(self):
        """中文 WER 按字计算。"""
        result = WERCalculator.calculate("你好世界", "你好中国", "zh")
        assert result.wer > 0


@pytest.mark.slow
@pytest.mark.quality
class TestQualityEvaluation:
    """质量评估集成测试。"""

    def test_evaluate_with_fixtures(
        self,
        reference_srt_path: Path | None,
        hypothesis_srt_path: Path | None,
    ):
        """使用 fixture 进行评估。"""
        if reference_srt_path is None:
            pytest.skip("需要 --reference 参数")
        if hypothesis_srt_path is None:
            pytest.skip("需要 --hypothesis 参数")

        config = QualityEvalConfig(
            reference_srt_path=reference_srt_path,
            hypothesis_srt_path=hypothesis_srt_path,
        )
        evaluator = QualityEvaluator(config)
        result = evaluator.evaluate()

        # 断言质量阈值
        assert result.cer_result is not None
        assert result.cer_result.cer < 0.15, f"CER {result.cer_result.cer:.2%} 超过 15%"

        assert result.timestamp_result is not None
        assert result.timestamp_result.within_tolerance_ratio > 0.85

        assert result.segmentation_result is not None
        assert result.segmentation_result.f1 > 0.70

        assert result.grade in ("A", "B", "C")
