from __future__ import annotations

from app.services.textflow.contracts import (
    CanonicalTextStream,
    ConsumedBoundaryPunct,
    CoreToken,
    PunctuationFact,
    RenderPolicy,
    SegmentPlan,
    SegmentationResult,
)
from app.services.textflow.render_core import RenderCore


def _build_stream() -> CanonicalTextStream:
    return CanonicalTextStream(
        stream_id="single-source",
        chunk_ref="chunk-77",
        language="en",
        text_source="slow",
        tokens=(
            CoreToken(
                token_id="tk-0",
                index=0,
                text_core="Hello",
                normalized_text="hello",
                start=0.0,
                end=0.5,
                source="slow",
            ),
            CoreToken(
                token_id="tk-1",
                index=1,
                text_core="world",
                normalized_text="world",
                start=0.5,
                end=1.0,
                source="slow",
            ),
        ),
        punctuation_facts=(
            PunctuationFact(
                fact_id="pf-comma",
                left_token_index=0,
                right_token_index=1,
                attach_mode="between",
                raw_text=",",
                normalized_text=",",
                punct_class="weak",
                source="slow",
            ),
        ),
    )


def test_render_core_owns_inner_and_terminal_punctuation_decisions() -> None:
    core = RenderCore()
    segmentation = SegmentationResult(
        segments=(
            SegmentPlan(
                segment_id="seg-77",
                token_start=0,
                token_end=1,
                start=0.0,
                end=1.0,
                consumed_boundary_punct=ConsumedBoundaryPunct(
                    fact_id="pf-question",
                    raw_text="?",
                    normalized_text="?",
                    punct_class="sentence_end",
                    source="slow",
                    render_hint="keep",
                ),
                trace={"legacy_text": "Hello world??"},
            ),
        ),
    )

    shown = core.render(
        canonical_stream=_build_stream(),
        segmentation_result=segmentation,
        policy=RenderPolicy(show_inner_punctuation=True, show_terminal_non_period_punct=True),
    )
    hidden = core.render(
        canonical_stream=_build_stream(),
        segmentation_result=segmentation,
        policy=RenderPolicy(show_inner_punctuation=False, show_terminal_non_period_punct=False),
    )

    assert shown.subtitles[0].text_display == "Hello, world?"
    assert hidden.subtitles[0].text_display == "Hello world"
    assert shown.subtitles[0].trace["legacy_text"] == "Hello world??"
