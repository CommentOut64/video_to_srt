from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from app.pipelines.dual_pipeline.services.alignment_stage_service import AlignmentStageService
from app.services.timeanchored_alignment.window_recovery_planner import WindowRecoveryPlanner


def _load_anchor_mount_graph() -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    graph_path = (
        repo_root
        / "jobs"
        / "p-20260329-201905-tr-test-en-1-pxac"
        / "debug"
        / "postprocess"
        / "chunk_0011"
        / "21_anchor_mount.graph.json"
    )
    return json.loads(graph_path.read_text(encoding="utf-8"))


def _parse_char_span(unit_id: str) -> tuple[int, int]:
    span_text = str(unit_id).split(":", 1)[1]
    start_text, end_text = span_text.split("-", 1)
    return int(start_text), int(end_text)


def test_job_p20260329_201905_typical_fatal_window_now_prefers_partial_commit() -> None:
    graph = _load_anchor_mount_graph()
    planner = WindowRecoveryPlanner()
    service = AlignmentStageService(
        host=SimpleNamespace(
            logger=Mock(),
            _edge_selection_mode="force_slow",
            _postprocess_trace_enabled=False,
            _postprocess_trace_level="summary",
            _anchor_mount_graph="off",
        )
    )
    stage_result = SimpleNamespace(
        decision_ingress=SimpleNamespace(
            anchored_token_units=tuple(
                SimpleNamespace(
                    unit_id=str(item["unit_id"]),
                    token_index=int(item["unit_index"]),
                    char_start=_parse_char_span(str(item["unit_id"]))[0],
                    char_end=_parse_char_span(str(item["unit_id"]))[1],
                    start=float(item["envelope"]["provisional_start"]),
                    end=float(item["envelope"]["provisional_end"]),
                    mount_status=str(item["mount_status"]),
                )
                for item in graph["mounts"]
            ),
            source_chunk_ids=("chunk-11", "chunk-12"),
            timeline_validity=str(graph["metrics"]["timeline_validity"]),
        ),
        anchor_mount_result=SimpleNamespace(
            should_fallback=True,
            timeline_validity=str(graph["metrics"]["timeline_validity"]),
        ),
    )
    stage_result.window_recovery_plan = planner.build(stage_result=stage_result)

    text_route, edge_route, final_route, error_code = service._resolve_anchor_mount_routes(
        ctx=SimpleNamespace(edge_selection_mode="force_slow"),
        stage_result=stage_result,
    )

    assert stage_result.window_recovery_plan.has_recoverable_spans is True
    assert service._should_use_safe_window_fallback(stage_result=stage_result) is False
    assert text_route == "slow"
    assert edge_route == "partial_commit"
    assert final_route == "partial_commit"
    assert error_code is None
