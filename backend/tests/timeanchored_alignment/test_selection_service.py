from __future__ import annotations

from types import SimpleNamespace

from app.services.alignment.types import CharMapping, L2Output, QualitySignals, TextTrack, TextTrackBundle
from app.services.arbitration.arbiter import ArbitrationResult
from app.services.timeanchored_alignment.selection.service import TextSelectionService


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


def test_selection_service_builds_truth_decision_and_report() -> None:
    service = TextSelectionService(logger=None)
    chosen_track = _build_track(text="快流文本", source="chosen")
    tracks = TextTrackBundle(
        sv_track=_build_track(text="快流文本", source="sv"),
        whisper_track=_build_track(text="慢流文本", source="whisper"),
    )
    ctx = SimpleNamespace(
        job_id="job-selection-service",
        chunk_index=2,
        edge_selection_mode="auto",
        ready_slow_window=None,
        audio_chunk=SimpleNamespace(chunk_id="chunk-2"),
    )
    selection_inputs = service.build_selection_inputs(
        ctx=ctx,
        tracks=tracks,
        quality_signals=QualitySignals(
            confidence_fast=0.91,
            confidence_slow=0.33,
            length_ratio=1.0,
            is_hallucination=True,
        ),
        sv_result={"text_itn_raw": "快流文本"},
        whisper_result={},
    )

    outcome = service.select(
        ctx=ctx,
        selection_inputs=selection_inputs,
        arbitration_processor=lambda _l2_input: L2Output(
            chosen_text_track=chosen_track,
            arbitration_result=ArbitrationResult(
                chosen_source="fast",
                reason="hallucination",
                sv_score=0.91,
                wh_score=0.33,
                coverage=0.88,
            ),
        ),
        clone_text_track=lambda track, source: _build_track(text=track.text_clean, source=source),
    )

    assert outcome.selected_text_truth.text == "快流文本"
    assert outcome.selected_text_truth.text_source == "fast"
    assert outcome.selection_decision.chosen_source == "fast"
    assert outcome.selection_decision.reason_code == "hallucination"
    assert outcome.selection_report.primary_reason_code == "hallucination"
    assert outcome.selection_report.metrics["confidence_fast"] == 0.91
    assert outcome.selection_inputs is selection_inputs


def test_selection_service_maps_mixed_source_to_mixed_decision() -> None:
    service = TextSelectionService(logger=None)
    chosen_track = _build_track(text="混合文本", source="chosen")
    tracks = TextTrackBundle(
        sv_track=_build_track(text="快流文本", source="sv"),
        whisper_track=_build_track(text="慢流文本", source="whisper"),
    )
    ctx = SimpleNamespace(
        job_id="job-selection-service-mixed",
        chunk_index=8,
        edge_selection_mode="auto",
        ready_slow_window=None,
        audio_chunk=SimpleNamespace(chunk_id="chunk-8"),
    )
    selection_inputs = service.build_selection_inputs(
        ctx=ctx,
        tracks=tracks,
        quality_signals=QualitySignals(
            confidence_fast=0.72,
            confidence_slow=0.73,
            length_ratio=1.0,
        ),
        sv_result={"text_itn_raw": "快流文本"},
        whisper_result={"text_raw": "慢流文本"},
    )

    outcome = service.select(
        ctx=ctx,
        selection_inputs=selection_inputs,
        arbitration_processor=lambda _l2_input: L2Output(
            chosen_text_track=chosen_track,
            arbitration_result=ArbitrationResult(
                chosen_source="mixed",
                reason="mixed_dominant_plus_island",
                sv_score=0.72,
                wh_score=0.73,
                coverage=0.95,
            ),
        ),
        clone_text_track=lambda track, source: _build_track(text=track.text_clean, source=source),
    )

    assert outcome.selected_text_truth.text == "混合文本"
    assert outcome.selected_text_truth.text_source == "mixed"
    assert outcome.selection_decision.decision == "mixed"
    assert outcome.selection_decision.chosen_source == "mixed"
    assert outcome.selection_report.chosen_source == "mixed"
    assert outcome.selection_report.primary_reason_code == "mixed_dominant_plus_island"


def test_selection_service_apply_runtime_selection_writes_whisper_view_once() -> None:
    service = TextSelectionService(logger=None)
    chosen_track = _build_track(text="快流文本", source="chosen")
    outcome = SimpleNamespace(
        chosen_track=chosen_track,
        chosen_text_clean="快流文本",
        selection_decision=SimpleNamespace(chosen_source="fast"),
    )
    tracks = TextTrackBundle(
        sv_track=_build_track(text="快流文本", source="sv"),
        whisper_track=_build_track(text="慢流文本", source="whisper"),
    )
    whisper_result = {"text": "慢流文本", "text_clean": "慢流文本", "text_itn_raw": "慢流文本"}
    sv_result = {"text_itn_raw": "快流文本"}

    chosen_text = service.apply_runtime_selection(
        tracks=tracks,
        whisper_result=whisper_result,
        sv_result=sv_result,
        selection_outcome=outcome,
    )

    assert chosen_text == "快流文本"
    assert tracks.chosen_track is chosen_track
    assert whisper_result["text"] == "快流文本"
    assert whisper_result["text_clean"] == "快流文本"
    assert whisper_result["text_itn_raw"] == "快流文本"
