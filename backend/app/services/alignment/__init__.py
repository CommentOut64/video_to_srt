"""
对齐服务模块

Phase 3 实现 - 2025-12-10

提供双流对齐算法和关键词提取功能。
"""

from .alignment_processor import AlignmentProcessor
from .alignment_service import AlignmentService
from .default_aligner import DefaultAligner, AlignmentLevel
from .gap_resolver import GapResolver, GapResolution, GapResolutionResult
from .keyword_extractor import KeywordExtractor
from .quality_stats import QualityStatsCalculator, QualityStats

__all__ = [
    "AlignmentService",
    "AlignmentProcessor",
    "DefaultAligner",
    "AlignmentLevel",
    "GapResolver",
    "GapResolution",
    "GapResolutionResult",
    "KeywordExtractor",
    "QualityStatsCalculator",
    "QualityStats",
]
