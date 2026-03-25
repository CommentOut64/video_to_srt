from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from app.services.textflow.contracts import (
    CanonicalTextStream,
    ConsumedBoundaryPunct,
    CoreToken,
    PunctuationFact,
    RenderPolicy,
    SegmentPlan,
    SegmentationOptions,
)


def test_core_token_is_frozen() -> None:
    token = CoreToken(
        token_id="tk-1",
        index=0,
        text_core="stream",
        normalized_text="stream",
        start=0.0,
        end=0.2,
        confidence=0.95,
        source="slow",
    )
    with pytest.raises(FrozenInstanceError):
        token.text_core = "changed"


def test_core_token_rejects_edge_punctuation() -> None:
    with pytest.raises(ValueError, match="text_core"):
        CoreToken(
            token_id="tk-1",
            index=0,
            text_core="stream.",
            normalized_text="stream",
            start=0.0,
            end=0.2,
            source="slow",
        )


@pytest.mark.parametrize("text_source", ["fast", "slow", "aligned"])
def test_canonical_text_stream_accepts_explicit_text_source(text_source: str) -> None:
    stream = CanonicalTextStream(
        stream_id="stream-1",
        chunk_ref="chunk-1",
        language="zh",
        text_source=text_source,
    )
    assert stream.text_source == text_source


def test_canonical_text_stream_rejects_unknown_text_source() -> None:
    with pytest.raises(ValueError, match="text_source"):
        CanonicalTextStream(
            stream_id="stream-1",
            chunk_ref="chunk-1",
            language="zh",
            text_source="legacy",
        )


def test_punctuation_fact_rejects_unknown_attach_mode() -> None:
    with pytest.raises(ValueError, match="attach_mode"):
        PunctuationFact(
            fact_id="pf-1",
            left_token_index=0,
            right_token_index=1,
            attach_mode="inside",
            raw_text="，",
            normalized_text="，",
            punct_class="weak",
            source="slow",
        )


def test_segmentation_options_validate_token_and_duration_range() -> None:
    with pytest.raises(ValueError, match="min_tokens"):
        SegmentationOptions(min_tokens=5, max_tokens=3)
    with pytest.raises(ValueError, match="min_duration"):
        SegmentationOptions(min_duration=5.0, max_duration=3.0)


def test_segment_plan_accepts_consumed_boundary_punct() -> None:
    consumed = ConsumedBoundaryPunct(
        fact_id="pf-1",
        raw_text="?",
        normalized_text="?",
        punct_class="sentence_end",
        source="slow",
        render_hint="keep",
    )
    plan = SegmentPlan(
        segment_id="seg-1",
        token_start=0,
        token_end=3,
        start=0.0,
        end=1.2,
        boundary_reason="punctuation",
        boundary_score=0.9,
        hard_boundary=False,
        consumed_boundary_punct=consumed,
    )
    assert plan.consumed_boundary_punct is consumed


def test_render_policy_defaults_follow_design() -> None:
    policy = RenderPolicy()
    assert policy.show_inner_punctuation is True
    assert policy.show_terminal_period is False
    assert policy.show_terminal_non_period_punct is True
