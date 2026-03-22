from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.pipelines.dual_pipeline.services.alignment_stage_service import AlignmentStageService
from app.services.alignment.types import CharMapping, L2Input, QualitySignals, TextTrack
from app.services.alignment.types import L2Output, TextTrackBundle
from app.services.arbitration.arbiter import ArbitrationResult, TextArbiterProcessor


def _build_track(*, text: str, source: str) -> TextTrack:
    mapping = [CharMapping(raw_idx=idx, clean_idx=idx) for idx, _ in enumerate(text)]
    return TextTrack(
        raw_text=text,
        text_itn_raw=text,
        text_clean=text,
        char_mapping=mapping,
        raw_to_clean=list(range(len(text))),
        clean_to_raw=list(range(len(text))),
        language="zh",
        source=source,
    )


def test_auto_mode_keeps_hallucination_gate_behavior() -> None:
    processor = TextArbiterProcessor(config={"hallucination_block": True})
    result = processor.process(
        L2Input(
            sv_track=_build_track(text="快流文本", source="sv"),
            whisper_track=_build_track(text="慢流文本", source="whisper"),
            quality_signals=QualitySignals(
                is_hallucination=True,
                confidence_fast=0.8,
                confidence_slow=0.9,
            ),
            edge_selection_mode="auto",
        )
    )

    assert result.arbitration_result.chosen_source == "fast"
    assert result.arbitration_result.reason == "hallucination"
    assert result.arbitration_result.is_edge_selection_bypassed is False
    assert result.arbitration_result.forced_source is None
    assert result.chosen_text_track is not None
    assert result.chosen_text_track.text_clean == "快流文本"


def test_force_slow_bypasses_gate() -> None:
    processor = TextArbiterProcessor(config={"hallucination_block": True})
    result = processor.process(
        L2Input(
            sv_track=_build_track(text="快流文本", source="sv"),
            whisper_track=_build_track(text="慢流文本", source="whisper"),
            quality_signals=QualitySignals(
                is_hallucination=True,
                confidence_fast=0.8,
                confidence_slow=0.9,
            ),
            edge_selection_mode="force_slow",
        )
    )

    assert result.arbitration_result.chosen_source == "slow"
    assert result.arbitration_result.reason == "forced_slow"
    assert result.arbitration_result.is_edge_selection_bypassed is True
    assert result.arbitration_result.forced_source == "slow"
    assert result.arbitration_result.error_code is None
    assert result.chosen_text_track is not None
    assert result.chosen_text_track.text_clean == "慢流文本"


def test_force_fast_bypasses_gate() -> None:
    processor = TextArbiterProcessor(config={"hallucination_block": True})
    result = processor.process(
        L2Input(
            sv_track=_build_track(text="快流文本", source="sv"),
            whisper_track=_build_track(text="慢流文本", source="whisper"),
            quality_signals=QualitySignals(
                is_hallucination=False,
                confidence_fast=0.2,
                confidence_slow=0.95,
            ),
            edge_selection_mode="force_fast",
        )
    )

    assert result.arbitration_result.chosen_source == "fast"
    assert result.arbitration_result.reason == "forced_fast"
    assert result.arbitration_result.is_edge_selection_bypassed is True
    assert result.arbitration_result.forced_source == "fast"
    assert result.arbitration_result.error_code is None
    assert result.chosen_text_track is not None
    assert result.chosen_text_track.text_clean == "快流文本"


def test_force_mode_missing_source_returns_error_without_fallback() -> None:
    processor = TextArbiterProcessor()
    result = processor.process(
        L2Input(
            sv_track=_build_track(text="快流文本", source="sv"),
            whisper_track=None,
            quality_signals=QualitySignals(
                is_hallucination=False,
                confidence_fast=0.8,
                confidence_slow=0.0,
            ),
            edge_selection_mode="force_slow",
        )
    )

    assert result.arbitration_result.chosen_source == "slow"
    assert result.arbitration_result.reason == "forced_source_missing"
    assert result.arbitration_result.is_edge_selection_bypassed is True
    assert result.arbitration_result.forced_source == "slow"
    assert result.arbitration_result.error_code == "E_L2_ARBITRATION_FORCED_SOURCE_MISSING"
    assert result.chosen_text_track is None


def test_alignment_stage_raises_when_forced_source_missing() -> None:
    tracks = TextTrackBundle(
        sv_track=_build_track(text="快流文本", source="sv"),
        whisper_track=None,
        chosen_track=None,
    )
    arbitration_result = ArbitrationResult(
        chosen_source="slow",
        reason="forced_source_missing",
        sv_score=0.8,
        wh_score=0.0,
        coverage=0.0,
        error_code="E_L2_ARBITRATION_FORCED_SOURCE_MISSING",
        is_edge_selection_bypassed=True,
        forced_source="slow",
        edge_selection_mode="force_slow",
    )

    with pytest.raises(ValueError, match="强制选边源缺失"):
        AlignmentStageService._apply_arbitration_track_selection(
            host=SimpleNamespace(_clone_text_track=lambda track, source: track),
            chunk_index=5,
            tracks=tracks,
            arbitration_output=L2Output(
                chosen_text_track=None,
                arbitration_result=arbitration_result,
            ),
        )
