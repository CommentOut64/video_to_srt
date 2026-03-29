from __future__ import annotations

from types import SimpleNamespace

from app.pipelines.dual_pipeline.services.alignment_stage_service import AlignmentStageService
from app.services.timeanchored_alignment.slow_window.contracts import (
    WindowChunkBinding,
    WindowCoverage,
)


def test_job_p20260328_202014_safe_window_fallback_keeps_monotonic_window_span() -> None:
    service = AlignmentStageService(
        host=SimpleNamespace(
            logger=SimpleNamespace(),
            _edge_selection_mode="force_slow",
            _postprocess_trace_enabled=False,
            _postprocess_trace_level="summary",
            _anchor_mount_graph="off",
        )
    )
    stage_result = SimpleNamespace(
        decision_ingress=SimpleNamespace(
            window_id="window-202014",
            anchored_token_units=tuple(),
            coverage=WindowCoverage(
                core_segments=((202.014, 205.000),),
                left_guard_sec=0.0,
                right_guard_sec=0.0,
                chunk_bindings=(
                    WindowChunkBinding(
                        chunk_id="chunk-202",
                        chunk_index=202,
                        chunk_start=202.014,
                        chunk_end=203.500,
                        overlap_ratio=1.0,
                        role="owner",
                        is_owner=True,
                    ),
                    WindowChunkBinding(
                        chunk_id="chunk-203",
                        chunk_index=203,
                        chunk_start=203.500,
                        chunk_end=205.000,
                        overlap_ratio=1.0,
                        role="core",
                        is_owner=False,
                    ),
                ),
            ),
            timeline_validity="fatal",
        )
    )
    ctx = SimpleNamespace(
        audio_chunk=SimpleNamespace(start=202.000, end=205.100),
    )

    start, end = service._resolve_safe_window_fallback_time_span(
        stage_result=stage_result,
        ctx=ctx,
    )

    assert start == 202.014
    assert end == 205.000
    assert end > start
