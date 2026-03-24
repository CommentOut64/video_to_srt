from __future__ import annotations

from app.services.punctuation.final_splitter import FinalSplitter
from app.services.textflow.decision_layer import SegmentationProcessor
from app.services.textflow.contracts import RenderResult, RenderedSubtitle


def test_decision_layer_render_bridge_maps_rendered_subtitle_to_sentence_segment_and_trace() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    render_result = RenderResult(
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
                trace={
                    "split_reason": "punctuation",
                    "split_risk": "",
                    "window_id": "w1",
                    "mapped_cut_time": 2.4,
                    "mapping_quality": "boundary",
                    "mapping_reason": "render_core",
                },
            ),
        )
    )

    sentences, traces = processor._build_legacy_sentences_from_render_result(
        render_result=render_result
    )

    assert len(sentences) == 1
    assert sentences[0].text == "你好，世界？"
    assert sentences[0].text_clean == "你好，世界？"
    assert sentences[0].speaker_id == "spk-1"
    assert sentences[0].turn_id == "turn-1"
    assert sentences[0].split_reason == "punctuation"
    assert sentences[0].window_id == "w1"
    assert sentences[0].mapped_cut_time == 2.4
    assert sentences[0].mapping_quality == "boundary"
    assert sentences[0].mapping_reason == "render_core"
    assert sentences[0].segment_id == "seg-1"
    assert sentences[0].sentence_uid == "seg-1"

    assert len(traces) == 1
    assert traces[0].sentence_index == 0
    assert traces[0].split_reason == "punctuation"
    assert traces[0].window_id == "w1"
    assert traces[0].mapped_cut_time == 2.4
