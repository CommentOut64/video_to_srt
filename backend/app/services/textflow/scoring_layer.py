"""
评分层统一入口（真实实现 + 合证适配）。
V3.2.0+dev.20260215.24
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.logging import resolve_loguru_logger
from app.services.alignment.types import AnnotatedWord, ScoringLayerInput, ScoringLayerOutput
from app.services.punctuation.semantic_injector import SemanticInjector
from app.services.segmentation.soft_cut.evidence_fusion import (
    EvidenceFusion as ScoringEvidenceFusion,
    EvidenceFusionConfig as ScoringEvidenceFusionConfig,
)


class ScoringSemanticInjectionProcessor:
    """评分层处理器：仅负责标点注入与覆盖率门控。"""

    def __init__(
        self,
        *,
        logger: Optional[Any] = None,
        min_mapping_coverage: float = 0.6,
    ) -> None:
        self._logger = resolve_loguru_logger(
            logger,
            __name__,
            layer="评分层",
            processor_name="semantic_injection_processor",
        )
        self._injector = SemanticInjector(
            logger=self._logger,
            min_mapping_coverage=min_mapping_coverage,
        )

    def process(self, data: ScoringLayerInput) -> ScoringLayerOutput:
        """执行评分层注入主路径。"""
        aligned_words = data.alignment_result.aligned_words or []
        clean_text_ref = str(data.punct_track.clean_text_ref or "")
        positions = list(data.punct_track.positions or [])

        if not aligned_words:
            return ScoringLayerOutput(
                annotated_words=[],
                injection_report={
                    "mapping_coverage": 0.0,
                    "mismatch_count": 0.0,
                    "error_code": "E_SCORING_EMPTY_ALIGNMENT",
                    "blocked": 1.0,
                },
            )

        if not clean_text_ref or not positions:
            base_words = self._build_base_annotated_words(
                aligned_words,
                speaker_id=data.speaker_id,
                turn_id=data.turn_id,
            )
            return ScoringLayerOutput(
                annotated_words=base_words,
                injection_report={
                    "mapping_coverage": 1.0,
                    "mismatch_count": 0.0,
                    "error_code": "",
                    "blocked": 0.0,
                },
            )

        injection = self._injector.inject(
            aligned_words,
            clean_text_ref,
            positions,
            language=data.language,
        )
        mismatch_count = float(len(injection.unmatched_positions or []))
        error_code = "E_SCORING_INJECTION_BLOCKED" if injection.is_mapping_blocked else ""
        report: Dict[str, Any] = {
            "mapping_coverage": float(injection.mapping_coverage),
            "mismatch_count": mismatch_count,
            "error_code": error_code,
            "blocked": 1.0 if injection.is_mapping_blocked else 0.0,
        }

        annotated_words: List[AnnotatedWord]
        if injection.is_mapping_blocked:
            annotated_words = self._build_base_annotated_words(
                aligned_words,
                speaker_id=data.speaker_id,
                turn_id=data.turn_id,
            )
        else:
            annotated_words = [
                AnnotatedWord(
                    word=item.word,
                    start=item.start,
                    end=item.end,
                    trailing_punct=item.trailing_punct,
                    confidence=source.final_confidence,
                    confidence_source=source.confidence_source,
                    speaker_id=data.speaker_id,
                    turn_id=data.turn_id,
                    track_id="main",
                )
                for item, source in zip(injection.annotated_words, aligned_words)
            ]

        self._logger.info(
            "评分层注入完成: words={} blocked={} coverage={:.2f} mismatch={}",
            len(annotated_words),
            int(report["blocked"]),
            report["mapping_coverage"],
            int(report["mismatch_count"]),
        )
        return ScoringLayerOutput(annotated_words=annotated_words, injection_report=report)

    @staticmethod
    def _build_base_annotated_words(
        aligned_words: List[Any],
        *,
        speaker_id: Optional[str],
        turn_id: Optional[str],
    ) -> List[AnnotatedWord]:
        return [
            AnnotatedWord(
                word=word.word,
                start=word.start,
                end=word.end,
                trailing_punct="",
                confidence=word.final_confidence,
                confidence_source=word.confidence_source,
                speaker_id=speaker_id,
                turn_id=turn_id,
                track_id="main",
            )
            for word in aligned_words
        ]


# 兼容旧命名（用于过渡期）。
SemanticInjectionProcessor = ScoringSemanticInjectionProcessor

__all__ = [
    "ScoringSemanticInjectionProcessor",
    "ScoringEvidenceFusion",
    "ScoringEvidenceFusionConfig",
    "SemanticInjectionProcessor",
]

