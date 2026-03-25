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
            slot_id="s1",
            slot_index=0,
            display_text="你好",
            mount_status="anchored",
            source_hook_ids=("h1",),
            match_confidence=1.0,
        ),
        SimpleNamespace(
            slot_id="s2",
            slot_index=1,
            display_text="世界",
            mount_status="inferred",
            source_hook_ids=("h2",),
            match_confidence=0.6,
        ),
        SimpleNamespace(
            slot_id="s3",
            slot_index=2,
            display_text="!",
            mount_status="unresolved",
            source_hook_ids=(),
            match_confidence=0.0,
        ),
    )
    graph = renderer.build_graph_payload(
        window_id="w1",
        items=items,
        hooks=hooks,
        metrics={"coverage_ratio": 0.66},
    )
    assert graph["summary"]["success"] == 1
    assert graph["summary"]["fallback"] == 1
    assert graph["summary"]["unresolved"] == 1

    svg = renderer.render_svg(graph)
    assert "<svg" in svg
    assert "anchored" in svg
    assert "inferred" in svg
    assert "unresolved" in svg
    assert "hook[h1] text: 你好" in svg
    assert "hook[h2] text: 世界" in svg
