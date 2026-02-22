# -*- coding: utf-8 -*-
"""
质量评估引擎。

综合 CER/WER/时间戳/断句评估，生成综合评分。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ci_tests.integration.harness.srt_parser import SRTEntry, SRTParser

from .metrics import (
    CERCalculator,
    CERResult,
    SegmentationEvaluator,
    SegmentationEvalResult,
    TimestampEvaluator,
    TimestampEvalResult,
    WERCalculator,
    WERResult,
)


@dataclass
class QualityEvalConfig:
    """质量评估配置。"""

    video_path: Optional[Path] = None
    reference_srt_path: Optional[Path] = None
    hypothesis_srt_path: Optional[Path] = None
    transcription_profile: str = "sv_whisper_dual"
    preprocessing_preset: str = "default"
    language: str = "zh"
    timestamp_tolerance_ms: float = 200.0
    boundary_tolerance_ms: float = 500.0


@dataclass
class QualityEvalResult:
    """质量评估结果。"""

    # 综合评分
    overall_score: float = 0.0
    grade: str = "F"  # A/B/C/D/F

    # 子指标
    cer_result: Optional[CERResult] = None
    wer_result: Optional[WERResult] = None
    timestamp_result: Optional[TimestampEvalResult] = None
    segmentation_result: Optional[SegmentationEvalResult] = None

    # 元信息
    reference_entry_count: int = 0
    hypothesis_entry_count: int = 0
    config: Optional[QualityEvalConfig] = None


class QualityEvaluator:
    """质量评估器。"""

    # 综合评分权重
    WEIGHT_TEXT = 0.50  # CER/WER 权重
    WEIGHT_TIMESTAMP = 0.30  # 时间戳权重
    WEIGHT_SEGMENTATION = 0.20  # 断句权重

    # 等级阈值
    GRADE_THRESHOLDS = {
        "A": 90,
        "B": 75,
        "C": 60,
        "D": 40,
    }

    def __init__(self, config: QualityEvalConfig) -> None:
        self.config = config

    def evaluate(
        self,
        reference_entries: Optional[List[SRTEntry]] = None,
        hypothesis_entries: Optional[List[SRTEntry]] = None,
    ) -> QualityEvalResult:
        """执行质量评估。"""
        # 加载 SRT
        if reference_entries is None and self.config.reference_srt_path:
            reference_entries = SRTParser.parse_file(self.config.reference_srt_path)

        if hypothesis_entries is None and self.config.hypothesis_srt_path:
            hypothesis_entries = SRTParser.parse_file(self.config.hypothesis_srt_path)

        if not reference_entries:
            raise ValueError("缺少参考 SRT")
        if not hypothesis_entries:
            raise ValueError("缺少待评估 SRT")

        result = QualityEvalResult(
            reference_entry_count=len(reference_entries),
            hypothesis_entry_count=len(hypothesis_entries),
            config=self.config,
        )

        # 计算各项指标
        result.cer_result = self._calc_cer(reference_entries, hypothesis_entries)
        result.wer_result = self._calc_wer(reference_entries, hypothesis_entries)
        result.timestamp_result = self._calc_timestamp(reference_entries, hypothesis_entries)
        result.segmentation_result = self._calc_segmentation(reference_entries, hypothesis_entries)

        # 综合评分
        result.overall_score = self._calc_overall_score(result)
        result.grade = self._calc_grade(result.overall_score)

        return result

    def _calc_cer(self, ref: List[SRTEntry], hyp: List[SRTEntry]) -> CERResult:
        """计算整体 CER。"""
        ref_text = "".join(e.text for e in ref)
        hyp_text = "".join(e.text for e in hyp)
        return CERCalculator.calculate(ref_text, hyp_text)

    def _calc_wer(self, ref: List[SRTEntry], hyp: List[SRTEntry]) -> WERResult:
        """计算整体 WER。"""
        ref_text = "".join(e.text for e in ref)
        hyp_text = "".join(e.text for e in hyp)
        return WERCalculator.calculate(ref_text, hyp_text, self.config.language)

    def _calc_timestamp(self, ref: List[SRTEntry], hyp: List[SRTEntry]) -> TimestampEvalResult:
        """计算时间戳偏移。"""
        return TimestampEvaluator.evaluate(ref, hyp, self.config.timestamp_tolerance_ms)

    def _calc_segmentation(self, ref: List[SRTEntry], hyp: List[SRTEntry]) -> SegmentationEvalResult:
        """计算断句准确率。"""
        return SegmentationEvaluator.evaluate(ref, hyp, self.config.boundary_tolerance_ms)

    def _calc_overall_score(self, result: QualityEvalResult) -> float:
        """计算综合评分 (0-100)。"""
        scores = []

        # 文本准确率得分 (CER 越低越好)
        if result.cer_result:
            cer = result.cer_result.cer
            text_score = max(0, (1 - cer)) * 100
            scores.append((text_score, self.WEIGHT_TEXT))

        # 时间戳得分
        if result.timestamp_result:
            ts_score = result.timestamp_result.within_tolerance_ratio * 100
            scores.append((ts_score, self.WEIGHT_TIMESTAMP))

        # 断句得分
        if result.segmentation_result:
            seg_score = result.segmentation_result.f1 * 100
            scores.append((seg_score, self.WEIGHT_SEGMENTATION))

        if not scores:
            return 0.0

        total_weight = sum(w for _, w in scores)
        weighted_sum = sum(s * w for s, w in scores)
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _calc_grade(self, score: float) -> str:
        """根据分数计算等级。"""
        for grade, threshold in self.GRADE_THRESHOLDS.items():
            if score >= threshold:
                return grade
        return "F"
