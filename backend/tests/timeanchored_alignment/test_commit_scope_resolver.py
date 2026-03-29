from __future__ import annotations

from app.pipelines.dual_pipeline.services.commit_scope_resolver import (
    CommitScopeResolver,
)
from app.services.textflow.contracts import SubtitleBatch, SubtitleItem
from app.services.timeanchored_alignment.slow_window.contracts import (
    WindowChunkBinding,
    WindowCoverage,
)


def test_commit_scope_resolver_builds_window_scope_from_coverage_and_owner_batch() -> None:
    resolver = CommitScopeResolver()
    coverage = WindowCoverage(
        core_segments=((10.0, 12.0),),
        left_guard_sec=0.2,
        right_guard_sec=0.3,
        chunk_bindings=(
            WindowChunkBinding(
                chunk_id="chunk-10",
                chunk_index=10,
                chunk_start=10.0,
                chunk_end=11.0,
                overlap_ratio=0.7,
                role="owner",
                is_owner=True,
            ),
            WindowChunkBinding(
                chunk_id="chunk-11",
                chunk_index=11,
                chunk_start=10.9,
                chunk_end=12.0,
                overlap_ratio=0.6,
                role="core",
                is_owner=False,
            ),
        ),
    )
    owner_batch = SubtitleBatch(
        chunk_id="chunk-10",
        chunk_index=10,
        items=(
            SubtitleItem(
                segment_id="timeanchored:window-10:seg:0",
                chunk_id="chunk-10",
                start=10.1,
                end=11.8,
                text="测试输出",
                source="render_core",
            ),
        ),
    )

    scope = resolver.build(
        window_id="window-10",
        source_chunk_ids=("chunk-10", "chunk-11"),
        source_chunk_indices=(10, 11),
        coverage=coverage,
        owner_batch=owner_batch,
        timeline_validity="repairable",
    )

    assert scope.window_id == "window-10"
    assert scope.source_chunk_ids == ("chunk-10", "chunk-11")
    assert scope.coverage_segments == ((10.0, 12.0),)
    assert scope.affected_segment_ids == ("timeanchored:window-10:seg:0",)
    assert scope.timeline_validity == "repairable"
    assert scope.generation_id.startswith("window-10:")
