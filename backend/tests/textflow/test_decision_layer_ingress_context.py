from __future__ import annotations

from app.models.sensevoice_models import WordTimestamp
from app.services.alignment.types import DecisionLayerInput
from app.services.punctuation.final_splitter import FinalSplitter
from app.services.textflow.contracts import SegmentationIngressContext
from app.services.textflow.decision_layer import SegmentationProcessor


def test_decision_layer_preserves_future_ready_ingress_context_in_canonical_stream() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    ingress_context = SegmentationIngressContext(
        unit_kind="slow_window",
        unit_id="window-7",
        slow_window_id="window-7",
        turn_group_id="turn-group-4",
        window_coverage=0.8,
        source_chunk_ids=("chunk-1", "chunk-2"),
        projection_chunk_ids=("chunk-2",),
        metadata={"trace_scope": "chunk6"},
    )
    decision_input = DecisionLayerInput(
        annotated_words=[],
        vad_intervals=[],
        ingress_context=ingress_context,
    )

    stream = processor._build_canonical_stream_from_words(
        words=[
            WordTimestamp(word="Hello", start=0.0, end=0.4),
            WordTimestamp(word="world", start=0.5, end=0.9),
        ],
        data=decision_input,
        stream_id="main",
        chunk_index=3,
    )

    assert stream.metadata["ingress_context"]["unit_kind"] == "slow_window"
    assert stream.metadata["ingress_context"]["slow_window_id"] == "window-7"
    assert stream.metadata["ingress_context"]["turn_group_id"] == "turn-group-4"
    assert stream.metadata["ingress_context"]["window_coverage"] == 0.8
    assert stream.metadata["ingress_context"]["source_chunk_ids"] == ["chunk-1", "chunk-2"]
    assert stream.metadata["ingress_context"]["projection_chunk_ids"] == ["chunk-2"]
    assert stream.diagnostics.ingress_context["metadata"]["trace_scope"] == "chunk6"
