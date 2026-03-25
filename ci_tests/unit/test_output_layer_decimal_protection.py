"""RenderCore 小数点保护测试。"""

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


def _build_decimal_stream() -> CanonicalTextStream:
    return CanonicalTextStream(
        stream_id="decimal-stream",
        chunk_ref="chunk-79",
        language="zh",
        text_source="slow",
        tokens=(
            CoreToken(
                token_id="tk-0",
                index=0,
                text_core="低一瓶中的氰化钠总含量为",
                normalized_text="低一瓶中的氰化钠总含量为",
                start=0.0,
                end=0.8,
                source="slow",
            ),
            CoreToken(
                token_id="tk-1",
                index=1,
                text_core="1.4",
                normalized_text="1.4",
                start=0.8,
                end=1.0,
                source="slow",
            ),
            CoreToken(
                token_id="tk-2",
                index=2,
                text_core="到",
                normalized_text="到",
                start=1.0,
                end=1.1,
                source="slow",
            ),
            CoreToken(
                token_id="tk-3",
                index=3,
                text_core="1.6",
                normalized_text="1.6",
                start=1.1,
                end=1.3,
                source="slow",
            ),
            CoreToken(
                token_id="tk-4",
                index=4,
                text_core="克",
                normalized_text="克",
                start=1.3,
                end=1.4,
                source="slow",
            ),
        ),
        punctuation_facts=(
            PunctuationFact(
                fact_id="pf-period",
                left_token_index=4,
                right_token_index=None,
                attach_mode="trailing",
                raw_text=".",
                normalized_text=".",
                punct_class="sentence_end",
                source="slow",
            ),
        ),
    )


def test_render_core_keeps_decimal_text_when_hiding_terminal_period() -> None:
    core = RenderCore()
    result = core.render(
        canonical_stream=_build_decimal_stream(),
        segmentation_result=SegmentationResult(
            segments=(
                SegmentPlan(
                    segment_id="seg-79",
                    token_start=0,
                    token_end=4,
                    start=0.0,
                    end=1.4,
                    consumed_boundary_punct=ConsumedBoundaryPunct(
                        fact_id="pf-period",
                        raw_text=".",
                        normalized_text=".",
                        punct_class="sentence_end",
                        source="slow",
                        render_hint="drop_period_default",
                    ),
                ),
            ),
        ),
    )

    assert result.subtitles[0].text_display == "低一瓶中的氰化钠总含量为1.4到1.6克"
    assert result.subtitles[0].terminal_punct is None


def test_render_core_keeps_non_decimal_terminal_period_when_policy_enables_it() -> None:
    core = RenderCore()
    result = core.render(
        canonical_stream=CanonicalTextStream(
            stream_id="plain-stream",
            chunk_ref="chunk-plain",
            language="zh",
            text_source="slow",
            tokens=(
                CoreToken(
                    token_id="tk-plain-0",
                    index=0,
                    text_core="这是测试",
                    normalized_text="这是测试",
                    start=0.0,
                    end=0.8,
                    source="slow",
                ),
            ),
        ),
        segmentation_result=SegmentationResult(
            segments=(
                SegmentPlan(
                    segment_id="seg-plain",
                    token_start=0,
                    token_end=0,
                    start=0.0,
                    end=0.8,
                    consumed_boundary_punct=ConsumedBoundaryPunct(
                        fact_id="pf-plain-period",
                        raw_text=".",
                        normalized_text=".",
                        punct_class="sentence_end",
                        source="slow",
                        render_hint="keep",
                    ),
                ),
            ),
        ),
        policy=RenderPolicy(show_terminal_period=True),
    )

    assert result.subtitles[0].text_display == "这是测试。"
    assert result.subtitles[0].terminal_punct == "。"
