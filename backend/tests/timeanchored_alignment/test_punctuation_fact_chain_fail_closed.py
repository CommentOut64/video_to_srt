from __future__ import annotations

from app.pipelines.dual_pipeline.services.alignment_stage_service import AlignmentStageService
from app.services.alignment.types import PunctTrack
from app.services.punctuation.base import PuncPosition


def test_fail_closed_when_punct_track_exists_but_fact_chain_is_empty() -> None:
    metrics = AlignmentStageService._build_punctuation_chain_health_metrics(
        chosen_source="slow",
        punct_track=PunctTrack(
            clean_text_ref="你好世界",
            positions=[PuncPosition(char_index=3, punctuation="。", confidence=0.99)],
            source="fast",
        ),
        preparation_punctuation_count=0,
        punctuation_fact_count=0,
    )

    assert metrics["punctuation_chain_broken_flag"] == 1
    assert metrics["punctuation_chain_broken_reason"] == "punct_track_not_projected_to_preparation"
