from __future__ import annotations

from app.services.textflow.contracts import SegmentationIngressContext


def test_ingress_context_supports_current_chunk_shape() -> None:
    context = SegmentationIngressContext(
        unit_kind="chunk",
        unit_id="chunk-17",
        chunk_id="chunk-17",
        chunk_index=17,
    )

    assert context.unit_kind == "chunk"
    assert context.unit_id == "chunk-17"
    assert context.chunk_id == "chunk-17"
    assert context.chunk_index == 17
    assert context.slow_window_id is None
    assert context.turn_group_id is None
    assert context.window_coverage is None
    assert context.source_chunk_ids == ()


def test_ingress_context_supports_future_window_shape_with_optional_fields() -> None:
    context = SegmentationIngressContext(
        unit_kind="slow_window",
        unit_id="window-3",
        slow_window_id="window-3",
        turn_group_id="turn-group-8",
        source_chunk_ids=("chunk-1", "chunk-2"),
    )

    assert context.unit_kind == "slow_window"
    assert context.unit_id == "window-3"
    assert context.slow_window_id == "window-3"
    assert context.turn_group_id == "turn-group-8"
    assert context.source_chunk_ids == ("chunk-1", "chunk-2")
    assert context.chunk_id is None
    assert context.chunk_index is None
    assert context.window_coverage is None
