from __future__ import annotations

from app.services.textflow.contracts import RenderResult, RenderedSubtitle
from app.services.textflow.subtitle_delivery import SubtitleDelivery


def test_subtitle_delivery_builds_unified_batch_from_render_result() -> None:
    delivery = SubtitleDelivery()
    batch = delivery.build_batch(
        render_result=RenderResult(
            subtitles=(
                RenderedSubtitle(
                    segment_id="seg-1",
                    start=1.0,
                    end=2.4,
                    text_display="你好，世界？",
                    text_core_joined="你好世界",
                    terminal_punct="？",
                    speaker_id="spk-1",
                    turn_id="turn-1",
                    text_source="slow",
                    trace={"split_reason": "punctuation"},
                ),
            ),
            render_report={"segment_count": 1},
            output_trace=({"segment_id": "seg-1", "text_display": "你好，世界？"},),
        ),
        chunk_id="chunk-7",
        chunk_index=7,
        ingress_context={"unit_kind": "chunk", "unit_id": "chunk-7"},
    )

    assert batch.chunk_id == "chunk-7"
    assert batch.chunk_index == 7
    assert batch.items[0].segment_id == "seg-1"
    assert batch.items[0].text == "你好，世界？"
    assert batch.items[0].status == "final"
    assert batch.items[0].source == "slow"
    assert batch.items[0].speaker_id == "spk-1"
    assert batch.render_report["segment_count"] == 1
    assert batch.diagnostics["ingress_context"]["unit_kind"] == "chunk"
    assert batch.diagnostics["render_output_trace"][0]["segment_id"] == "seg-1"


def test_subtitle_delivery_keeps_future_ready_ingress_fields_in_diagnostics_only() -> None:
    delivery = SubtitleDelivery()
    batch = delivery.build_batch(
        render_result=RenderResult(
            subtitles=(
                RenderedSubtitle(
                    segment_id="seg-2",
                    start=3.0,
                    end=4.0,
                    text_display="Hello world",
                    text_core_joined="Hello world",
                    text_source="fast",
                ),
            ),
        ),
        chunk_id="chunk-unknown",
        ingress_context={
            "unit_kind": "slow_window",
            "unit_id": "window-2",
            "slow_window_id": "window-2",
            "turn_group_id": "turn-group-9",
            "window_coverage": 0.6,
            "source_chunk_ids": ["chunk-2", "chunk-3"],
            "projection_chunk_ids": ["chunk-3"],
        },
    )

    assert batch.items[0].chunk_id == "chunk-unknown"
    assert batch.items[0].text == "Hello world"
    assert batch.diagnostics["ingress_context"]["slow_window_id"] == "window-2"
    assert batch.diagnostics["ingress_context"]["window_coverage"] == 0.6
    assert batch.diagnostics["ingress_context"]["source_chunk_ids"] == ["chunk-2", "chunk-3"]
    assert batch.diagnostics["ingress_context"]["projection_chunk_ids"] == ["chunk-3"]
