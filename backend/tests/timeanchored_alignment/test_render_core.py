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


def _build_stream_en() -> CanonicalTextStream:
    return CanonicalTextStream(
        stream_id="s-en",
        chunk_ref="chunk-en",
        language="en",
        text_source="slow",
        tokens=(
            CoreToken(
                token_id="t0",
                index=0,
                text_core="Hello",
                normalized_text="hello",
                start=0.0,
                end=0.5,
                source="slow",
            ),
            CoreToken(
                token_id="t1",
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


def _build_stream_zh() -> CanonicalTextStream:
    return CanonicalTextStream(
        stream_id="s-zh",
        chunk_ref="chunk-zh",
        language="zh",
        text_source="slow",
        tokens=(
            CoreToken(
                token_id="t0",
                index=0,
                text_core="你好",
                normalized_text="你好",
                start=0.0,
                end=0.5,
                source="slow",
            ),
            CoreToken(
                token_id="t1",
                index=1,
                text_core="世界",
                normalized_text="世界",
                start=0.5,
                end=1.0,
                source="slow",
            ),
            CoreToken(
                token_id="t2",
                index=2,
                text_core="1.4",
                normalized_text="1.4",
                start=1.0,
                end=1.4,
                source="slow",
            ),
            CoreToken(
                token_id="t3",
                index=3,
                text_core="克",
                normalized_text="克",
                start=1.4,
                end=1.7,
                source="slow",
            ),
        ),
        punctuation_facts=(
            PunctuationFact(
                fact_id="pf-comma-zh",
                left_token_index=1,
                right_token_index=2,
                attach_mode="between",
                raw_text=",",
                normalized_text=",",
                punct_class="weak",
                source="slow",
            ),
        ),
    )


def test_render_core_renders_inner_punctuation_and_terminal_non_period() -> None:
    core = RenderCore()
    stream = _build_stream_en()
    segmentation = SegmentationResult(
        segments=(
            SegmentPlan(
                segment_id="seg-1",
                token_start=0,
                token_end=1,
                start=0.0,
                end=1.0,
                boundary_reason="punctuation",
                boundary_score=0.9,
                consumed_boundary_punct=ConsumedBoundaryPunct(
                    fact_id="pf-question",
                    raw_text="?",
                    normalized_text="?",
                    punct_class="sentence_end",
                    source="slow",
                    render_hint="keep",
                ),
            ),
        )
    )

    result = core.render(
        canonical_stream=stream,
        segmentation_result=segmentation,
        policy=RenderPolicy(),
    )

    assert len(result.subtitles) == 1
    assert result.subtitles[0].text_core_joined == "Hello world"
    assert result.subtitles[0].text_display == "Hello, world?"
    assert result.subtitles[0].terminal_punct == "?"


def test_render_core_hides_terminal_period_by_default() -> None:
    core = RenderCore()
    stream = _build_stream_zh()
    segmentation = SegmentationResult(
        segments=(
            SegmentPlan(
                segment_id="seg-2",
                token_start=0,
                token_end=3,
                start=0.0,
                end=1.7,
                boundary_reason="punctuation",
                boundary_score=0.9,
                consumed_boundary_punct=ConsumedBoundaryPunct(
                    fact_id="pf-period",
                    raw_text=".",
                    normalized_text=".",
                    punct_class="sentence_end",
                    source="slow",
                    render_hint="drop_period_default",
                ),
            ),
        )
    )

    result = core.render(
        canonical_stream=stream,
        segmentation_result=segmentation,
        policy=RenderPolicy(),
    )

    assert result.subtitles[0].text_display == "你好世界，1.4克"
    assert result.subtitles[0].terminal_punct is None


def test_render_core_supports_show_terminal_period() -> None:
    core = RenderCore()
    stream = _build_stream_zh()
    segmentation = SegmentationResult(
        segments=(
            SegmentPlan(
                segment_id="seg-3",
                token_start=0,
                token_end=3,
                start=0.0,
                end=1.7,
                boundary_reason="punctuation",
                boundary_score=0.9,
                consumed_boundary_punct=ConsumedBoundaryPunct(
                    fact_id="pf-period",
                    raw_text=".",
                    normalized_text=".",
                    punct_class="sentence_end",
                    source="slow",
                    render_hint="show_period",
                ),
            ),
        )
    )

    result = core.render(
        canonical_stream=stream,
        segmentation_result=segmentation,
        policy=RenderPolicy(show_terminal_period=True),
    )

    assert result.subtitles[0].text_display == "你好世界，1.4克。"
    assert result.subtitles[0].terminal_punct == "。"


def test_render_core_can_hide_inner_punctuation() -> None:
    core = RenderCore()
    stream = _build_stream_en()
    segmentation = SegmentationResult(
        segments=(
            SegmentPlan(
                segment_id="seg-4",
                token_start=0,
                token_end=1,
                start=0.0,
                end=1.0,
                boundary_reason="pause",
                boundary_score=0.6,
            ),
        )
    )

    result = core.render(
        canonical_stream=stream,
        segmentation_result=segmentation,
        policy=RenderPolicy(show_inner_punctuation=False),
    )

    assert result.subtitles[0].text_display == "Hello world"
