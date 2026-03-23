from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from app.models.sensevoice_models import SentenceSegment
from app.pipelines.dual_pipeline.services.diagnostic_trace_service import DiagnosticTraceService
from app.services.alignment.types import (
    AlignedFacts,
    AlignmentResult,
    FusedEvidence,
    L2Output,
    PunctTrack,
    TextTrack,
    TextTrackBundle,
)
from app.services.arbitration.arbiter import ArbitrationResult


def _track(text: str, source: str) -> TextTrack:
    return TextTrack(
        raw_text=text,
        text_itn_raw=text,
        text_clean=text,
        char_mapping=[],
        raw_to_clean=[],
        clean_to_raw=[],
        language="zh",
        source=source,
    )


def test_emit_layer_trace_full_includes_timeanchored_raw_mount_trace() -> None:
    service = DiagnosticTraceService(logger=Mock())
    captured: dict[str, object] = {}
    ctx = SimpleNamespace(
        job_id="job-diag",
        chunk_index=7,
        job_dir=None,
        arbitration_result=ArbitrationResult(
            chosen_source="slow",
            reason="test",
            sv_score=0.8,
            wh_score=0.9,
            coverage=1.0,
        ),
        hetero_alignment_report={
            "alignment_report": {
                "name": "alignment",
                "metrics": {
                    "raw_mount_trace": {
                        "mapping_pairs": [
                            {
                                "text_index": 0,
                                "time_index": 0,
                                "text": "你",
                                "time_text": "你",
                            }
                        ],
                        "time_units": [{"index": 0, "text": "你"}],
                        "text_units": [{"index": 0, "text": "你"}],
                    }
                },
            }
        },
    )

    service.emit_layer_trace_full(
        ctx,
        tracks=TextTrackBundle(
            sv_track=_track("你", "sv"),
            whisper_track=_track("你", "whisper"),
            chosen_track=_track("你", "chosen"),
        ),
        sv_result={"text_clean": "你", "words": []},
        whisper_result={"text_clean": "你", "raw_result": {"segments": []}},
        arbitration_output=L2Output(
            chosen_text_track=_track("你", "chosen"),
            arbitration_result=ctx.arbitration_result,
        ),
        punct_track=PunctTrack(clean_text_ref="你", positions=[], source="none"),
        alignment_result=AlignmentResult(
            aligned_words=[],
            alignment_score=1.0,
            gap_ratio=0.0,
            gap_positions=[],
            coverage=1.0,
        ),
        aligned_facts=AlignedFacts(),
        fused_evidence=FusedEvidence(),
        words_for_split=[],
        injection_stats={},
        split_stats={},
        final_sentences=[
            SentenceSegment(
                text="你",
                text_clean="你",
                start=0.0,
                end=0.1,
            )
        ],
        output_traces=[],
        append_debug_layer_trace_line=lambda _job_dir, payload, logger=None: captured.update(
            payload=payload
        ),
    )

    payload = captured["payload"]
    assert payload["timeanchored_alignment"]["raw_mount_trace"]["mapping_pairs"][0]["text"] == "你"
