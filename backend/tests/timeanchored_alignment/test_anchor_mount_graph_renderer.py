from __future__ import annotations

from types import SimpleNamespace

from app.pipelines.dual_pipeline.services.anchor_mount_graph_renderer import (
    AnchorMountGraphRenderer,
)


def test_anchor_mount_graph_renderer_builds_summary_and_svg() -> None:
    renderer = AnchorMountGraphRenderer()
    hooks = (
        SimpleNamespace(hook_id="h1", hook_text="你好"),
        SimpleNamespace(hook_id="h2", hook_text="世界，测试长文本换行能力"),
    )
    items = (
        SimpleNamespace(
            unit_id="token-1",
            unit_index=0,
            display_text="你好",
            mount_status="anchored",
            anchor_kind="lexical",
            source_hook_ids=("h1",),
            source_chunk_ids=("chunk-1",),
            source_chunk_indices=(1,),
            match_confidence=1.0,
            alignment_block_id="block-a",
        ),
        SimpleNamespace(
            unit_id="token-2",
            unit_index=1,
            display_text="世界",
            mount_status="inferred",
            anchor_kind="soft_pronunciation",
            source_hook_ids=("h2",),
            source_chunk_ids=("chunk-1", "chunk-2"),
            source_chunk_indices=(1, 2),
            match_confidence=0.6,
            alignment_block_id=None,
        ),
        SimpleNamespace(
            unit_id="token-3",
            unit_index=2,
            display_text="!",
            mount_status="unresolved",
            anchor_kind="none",
            source_hook_ids=(),
            source_chunk_ids=("chunk-2",),
            source_chunk_indices=(2,),
            match_confidence=0.0,
            alignment_block_id=None,
        ),
    )
    envelopes = (
        SimpleNamespace(
            provisional_start=0.0,
            provisional_end=0.4,
            left_bound=0.0,
            right_bound=0.4,
            envelope_kind="anchored",
            source_chunk_ids=("chunk-1",),
            source_chunk_indices=(1,),
        ),
        SimpleNamespace(
            provisional_start=0.46,
            provisional_end=0.9,
            left_bound=0.44,
            right_bound=0.92,
            envelope_kind="inferred",
            source_chunk_ids=("chunk-1", "chunk-2"),
            source_chunk_indices=(1, 2),
        ),
        SimpleNamespace(
            provisional_start=0.92,
            provisional_end=1.0,
            left_bound=0.92,
            right_bound=1.0,
            envelope_kind="unresolved",
            source_chunk_ids=("chunk-2",),
            source_chunk_indices=(2,),
        ),
    )
    boundary_evidences = (
        SimpleNamespace(
            split_idx=0,
            event_time=0.43,
            left_end=0.4,
            right_start=0.46,
            reason="gap_pause",
            score=0.72,
            hard_flag=False,
            metadata={"blocked_by_lock": False},
        ),
        SimpleNamespace(
            split_idx=1,
            event_time=0.91,
            left_end=0.9,
            right_start=0.92,
            reason="anchor_block_close",
            score=0.52,
            hard_flag=False,
            metadata={"left_block_id": "block-a", "right_block_id": "block-b"},
        ),
    )
    graph = renderer.build_graph_payload(
        window_id="w1",
        items=items,
        envelopes=envelopes,
        boundary_evidences=boundary_evidences,
        hooks=hooks,
        metrics={"coverage_ratio": 0.66},
    )
    assert graph["summary"]["success"] == 1
    assert graph["summary"]["fallback"] == 1
    assert graph["summary"]["unresolved"] == 1
    assert graph["mounts"][0] == {
        "unit_id": "token-1",
        "unit_index": 0,
        "text": "你好",
        "mount_status": "anchored",
        "classification": "success",
        "anchor_kind": "lexical",
        "hook_ids": ["h1"],
        "hooks": [{"hook_id": "h1", "text": "你好"}],
        "source_chunk_ids": ["chunk-1"],
        "source_chunk_indices": [1],
        "alignment_block_id": "block-a",
        "match_confidence": 1.0,
        "envelope": {
            "kind": "anchored",
            "provisional_start": 0.0,
            "provisional_end": 0.4,
            "left_bound": 0.0,
            "right_bound": 0.4,
            "source_chunk_ids": ["chunk-1"],
            "source_chunk_indices": [1],
        },
        "boundary_after": [
            {
                "split_idx": 0,
                "event_time": 0.43,
                "left_end": 0.4,
                "right_start": 0.46,
                "reason": "gap_pause",
                "score": 0.72,
                "hard_flag": False,
                "metadata": {"blocked_by_lock": False},
            }
        ],
    }

    svg = renderer.render_svg(graph)
    assert "<svg" in svg
    assert "anchored" in svg
    assert "inferred" in svg
    assert "unresolved" in svg
    assert "unit[0] id: token-1 text: 你好" in svg
    assert "time:" in svg
    assert "0.000 -&gt; 0.400" in svg or "0.000 -> 0.400" in svg
    assert "chunk-1" in svg
    assert "block-a" in svg
    assert "anchor_kind: lexical" in svg
    assert "boundary_after[0] gap_pause score=0.720" in svg
    assert "hook[h1] text: 你好" in svg
    assert "hook[h2] text: 世界" in svg
