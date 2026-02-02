"""
仲裁器（多阶段渐进决策）。
V3.2.0+dev.20260202.08
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from app.services.arbitration.hallucination_detector import HallucinationDetector


@dataclass
class ArbitrationResult:
    """仲裁结果（包含决策上下文）。"""

    text_source: str
    punct_source: str
    reason: str
    coverage: float
    sv_score: float
    wh_score: float
    gap_positions: List[int] = field(default_factory=list)


class Arbiter:
    """仲裁器（策略模式：按阶段执行规则，便于扩展/替换）。"""

    def __init__(
        self,
        *,
        logger: Optional[logging.Logger] = None,
        hallucination_detector: Optional[HallucinationDetector] = None,
        coverage_high: float = 0.85,
        coverage_mid: float = 0.5,
        len_ratio_min: float = 0.65,
        len_ratio_max: float = 3.0,
        quality_margin: float = 0.9,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._hallucination_detector = hallucination_detector or HallucinationDetector(
            logger=self._logger
        )
        self._coverage_high = coverage_high
        self._coverage_mid = coverage_mid
        self._len_ratio_min = len_ratio_min
        self._len_ratio_max = len_ratio_max
        self._quality_margin = quality_margin

    def arbitrate(
        self,
        sv_result: Dict[str, Any],
        whisper_result: Dict[str, Any],
        *,
        prompt: Optional[str] = None,
    ) -> ArbitrationResult:
        """执行多阶段渐进仲裁。"""
        sv_text = str(sv_result.get("text_clean") or sv_result.get("text") or "")
        wh_text = str(whisper_result.get("text_clean") or whisper_result.get("text") or "")

        if not wh_text.strip():
            return self._build_result(
                text_source="sv",
                punct_source="sv",
                reason="empty_whisper",
                sv_result=sv_result,
                whisper_result=whisper_result,
                coverage=0.0,
            )

        if self._hallucination_detector.is_hallucination(whisper_result, prompt):
            return self._build_result(
                text_source="sv",
                punct_source="sv",
                reason="hallucination",
                sv_result=sv_result,
                whisper_result=whisper_result,
                coverage=0.0,
            )

        if not sv_text.strip():
            return self._build_result(
                text_source="whisper",
                punct_source="merged",
                reason="sv_empty",
                sv_result=sv_result,
                whisper_result=whisper_result,
                coverage=1.0,
            )

        if self._length_mismatch(sv_text, wh_text):
            return self._build_result(
                text_source="sv",
                punct_source="sv",
                reason="length_mismatch",
                sv_result=sv_result,
                whisper_result=whisper_result,
                coverage=0.0,
            )

        coverage = self._compute_coverage(sv_text, wh_text)
        if coverage < self._coverage_mid:
            return self._build_result(
                text_source="sv",
                punct_source="sv",
                reason="low_coverage",
                sv_result=sv_result,
                whisper_result=whisper_result,
                coverage=coverage,
            )

        sv_score = self._score_sv(sv_result, sv_text)
        wh_score = self._score_wh(whisper_result, wh_text)
        if wh_score >= sv_score * self._quality_margin:
            punct_source = "merged" if coverage >= self._coverage_high else "sv"
            return ArbitrationResult(
                text_source="whisper",
                punct_source=punct_source,
                reason="accepted",
                coverage=coverage,
                sv_score=sv_score,
                wh_score=wh_score,
                gap_positions=[],
            )

        return ArbitrationResult(
            text_source="sv",
            punct_source="sv",
            reason="low_quality_wh",
            coverage=coverage,
            sv_score=sv_score,
            wh_score=wh_score,
            gap_positions=[],
        )

    def _build_result(
        self,
        *,
        text_source: str,
        punct_source: str,
        reason: str,
        sv_result: Dict[str, Any],
        whisper_result: Dict[str, Any],
        coverage: float,
    ) -> ArbitrationResult:
        return ArbitrationResult(
            text_source=text_source,
            punct_source=punct_source,
            reason=reason,
            coverage=coverage,
            sv_score=self._score_sv(sv_result, str(sv_result.get("text_clean") or "")),
            wh_score=self._score_wh(whisper_result, str(whisper_result.get("text_clean") or "")),
            gap_positions=[],
        )

    def _length_mismatch(self, sv_text: str, wh_text: str) -> bool:
        len_sv = max(len(sv_text), 1)
        len_wh = max(len(wh_text), 1)
        ratio = len_wh / len_sv
        if ratio < self._len_ratio_min:
            return True
        if ratio > self._len_ratio_max:
            return True
        return False

    @staticmethod
    def _compute_coverage(sv_text: str, wh_text: str) -> float:
        """使用序列相似度估算覆盖率。"""
        if not sv_text or not wh_text:
            return 0.0
        normalized_sv = " ".join(sv_text.split())
        normalized_wh = " ".join(wh_text.split())
        return SequenceMatcher(None, normalized_sv, normalized_wh).ratio()

    @staticmethod
    def _score_sv(result: Dict[str, Any], text: str) -> float:
        base = float(result.get("confidence", 0.5) or 0.5)
        if not text.strip():
            return base * 0.5
        return base

    @staticmethod
    def _score_wh(result: Dict[str, Any], text: str) -> float:
        base = float(result.get("confidence", 0.5) or 0.5)
        if not text.strip():
            return base * 0.5
        return base
