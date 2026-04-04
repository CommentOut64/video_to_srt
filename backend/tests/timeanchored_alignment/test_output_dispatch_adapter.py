from __future__ import annotations

import pytest

from app.models.sensevoice_models import SentenceSegment
from app.services.alignment.types import OutputLayerInput
from app.services.textflow.contracts import ChunkSentenceIndex, SentenceRecord, SubtitleBatch, SubtitleItem
from app.services.textflow.output_dispatch_adapter import OutputDispatchAdapter, OutputLayerProcessor


class _DummySubtitleManager:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.calls: list[tuple[str, int]] = []

    def replace_chunk_batch(self, subtitle_batch: SubtitleBatch) -> list[int]:
        self.calls.append((subtitle_batch.chunk_id, len(subtitle_batch.items)))
        if self.should_fail:
            raise RuntimeError("replace_chunk failed")
        return [10 + idx for idx in range(len(subtitle_batch.items))]


class _DummySpeakerStore:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def upsert_subtitle_speaker_links(self, payload) -> None:
        self.items.extend(list(payload or []))


def test_output_dispatch_adapter_dispatch_success() -> None:
    subtitle_manager = _DummySubtitleManager()
    speaker_store = _DummySpeakerStore()
    adapter = OutputDispatchAdapter(
        subtitle_manager=subtitle_manager,
        speaker_store_service_getter=lambda: speaker_store,
    )
    subtitle_batch = SubtitleBatch(
        chunk_id="chunk-7",
        chunk_index=7,
        items=(
            SubtitleItem(
                segment_id="seg-1",
                chunk_id="chunk-7",
                text="Hello, world",
                start=0.0,
                end=1.2,
                source="aligned",
                speaker_id="spk-1",
                turn_id="turn-1",
                trace={
                    "split_reason": "punctuation",
                    "window_id": "w1",
                    "mapped_cut_time": 1.2,
                    "mapping_quality": "boundary",
                    "mapping_reason": "render_core",
                },
            ),
        ),
        diagnostics={
            "output_trace": [
                {
                    "sentence_index": 0,
                    "split_reason": "punctuation",
                    "split_risk": "",
                    "window_id": "w1",
                    "mapped_cut_time": 1.2,
                    "mapping_quality": "boundary",
                    "mapping_reason": "render_core",
                    "sentence_start": 0.0,
                    "sentence_end": 1.2,
                }
            ]
        },
    )

    payload = adapter.dispatch(
        sentence_records=[
            SentenceRecord(
                sentence_id="seg-1",
                text="Hello, world",
                start=0.0,
                end=1.2,
                source_chunk_ids=("chunk-7",),
                overlap_chunk_ids=("chunk-7",),
                replace_scope_chunk_ids=("chunk-7",),
                route="timeanchored",
                trace={
                    "split_reason": "punctuation",
                    "window_id": "w1",
                    "mapped_cut_time": 1.2,
                    "mapping_quality": "boundary",
                    "mapping_reason": "render_core",
                },
                metadata={"speaker_id": "spk-1", "turn_id": "turn-1", "source": "aligned"},
            ),
        ],
        chunk_sentence_indices=[ChunkSentenceIndex(chunk_id="chunk-7", sentence_ids=("seg-1",))],
        subtitle_batch=subtitle_batch,
        injection_report={"mapping_coverage": 1.0},
        segmentation_report={"route": "timeanchored"},
        unknown_sentence_filtered_count=0,
    )

    assert subtitle_manager.calls == [("chunk-7", 1)]
    assert payload["chunk_index"] == 7
    assert payload["chunk_uid"] == "chunk-7"
    assert payload["sentence_count"] == 1
    assert payload["sentence_segments"][0]["text"] == "Hello, world"
    assert payload["sentence_segments"][0]["source_chunk_ids"] == ["chunk-7"]
    assert payload["output_trace"][0]["split_reason"] == "punctuation"
    assert payload["transport_meta"]["subtitle_channel"]["status"] == "ok"
    assert payload["transport_meta"]["speaker_store_channel"]["status"] == "ok"
    assert payload["transport_meta"]["speaker_store_channel"]["upsert_count"] == 1
    assert payload["errors"] == []
    assert payload["segmentation_report"]["unknown_sentence_filtered_count"] == 0
    assert len(speaker_store.items) == 1


def test_output_dispatch_adapter_marks_output_channel_failure() -> None:
    subtitle_manager = _DummySubtitleManager(should_fail=True)
    adapter = OutputDispatchAdapter(subtitle_manager=subtitle_manager)
    subtitle_batch = SubtitleBatch(
        chunk_id="chunk-1",
        chunk_index=1,
        items=(
            SubtitleItem(
                segment_id="seg-x",
                chunk_id="chunk-1",
                text="x",
                start=0.0,
                end=0.1,
            ),
        ),
    )

    payload = adapter.dispatch(
        sentence_records=[
            SentenceRecord(
                sentence_id="seg-x",
                text="x",
                start=0.0,
                end=0.1,
                source_chunk_ids=("chunk-1",),
                replace_scope_chunk_ids=("chunk-1",),
            ),
        ],
        chunk_sentence_indices=[ChunkSentenceIndex(chunk_id="chunk-1", sentence_ids=("seg-x",))],
        subtitle_batch=subtitle_batch,
        injection_report={},
        segmentation_report={},
        unknown_sentence_filtered_count=0,
    )

    assert payload["transport_meta"]["subtitle_channel"]["status"] == "failed"
    assert payload["transport_meta"]["speaker_store_channel"]["status"] == "skipped_upstream_failed"
    assert "E_OUTPUT_CHANNEL_FAIL" in payload["errors"]


def test_output_layer_processor_uses_sentence_records_as_internal_truth() -> None:
    subtitle_manager = _DummySubtitleManager()
    processor = OutputLayerProcessor(subtitle_manager=subtitle_manager)

    payload = processor.process(
        OutputLayerInput(
            chunk_index="chunk-9",
            sentence_segments=[
                SentenceSegment(
                    text="旧句段文本",
                    text_clean="旧句段文本",
                    start=0.0,
                    end=0.5,
                )
            ],
            sentence_records=[
                SentenceRecord(
                    sentence_id="seg-9-0",
                    text="统一 DTO 文本",
                    start=1.0,
                    end=2.0,
                    source_chunk_ids=("chunk-9",),
                    overlap_chunk_ids=("chunk-9",),
                    replace_scope_chunk_ids=("chunk-9",),
                    route="timeanchored",
                    trace={
                        "split_reason": "render_core",
                        "window_id": "",
                        "mapped_cut_time": 2.0,
                        "mapping_quality": "boundary",
                        "mapping_reason": "render_core",
                    },
                    metadata={"source": "render_core"},
                )
            ],
            chunk_sentence_indices=[
                ChunkSentenceIndex(chunk_id="chunk-9", sentence_ids=("seg-9-0",))
            ],
            subtitle_batch=SubtitleBatch(
                chunk_id="chunk-9",
                chunk_index=9,
                items=(
                    SubtitleItem(
                        segment_id="seg-9-0",
                        chunk_id="chunk-9",
                        text="统一 DTO 文本",
                        start=1.0,
                        end=2.0,
                        source="render_core",
                    ),
                ),
                diagnostics={
                    "output_trace": [
                        {
                            "sentence_index": 0,
                            "split_reason": "render_core",
                            "split_risk": "",
                            "window_id": "",
                            "mapped_cut_time": 2.0,
                            "mapping_quality": "boundary",
                            "mapping_reason": "render_core",
                            "sentence_start": 1.0,
                            "sentence_end": 2.0,
                        }
                    ]
                },
            ),
            language="zh",
        )
    ).output_payload

    assert subtitle_manager.calls == [("chunk-9", 1)]
    assert payload["sentence_segments"][0]["text"] == "统一 DTO 文本"
    assert payload["sentence_segments"][0]["source_chunk_ids"] == ["chunk-9"]
    assert payload["output_trace"][0]["split_reason"] == "render_core"


def test_output_layer_processor_rejects_missing_sentence_records() -> None:
    subtitle_manager = _DummySubtitleManager()
    processor = OutputLayerProcessor(subtitle_manager=subtitle_manager)

    with pytest.raises(ValueError, match="sentence_records"):
        processor.process(
            OutputLayerInput(
                chunk_index="chunk-3",
                sentence_segments=[
                    SentenceSegment(
                        text="旧句段文本",
                        text_clean="旧句段文本",
                        start=0.0,
                        end=0.5,
                    )
                ],
                language="zh",
            )
        )
