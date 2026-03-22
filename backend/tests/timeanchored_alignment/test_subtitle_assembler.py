from __future__ import annotations

from app.services.timeanchored_alignment.contracts import (
    AlignmentItem,
    AlignmentMetrics,
    FinalAlignmentResult,
)
from app.services.timeanchored_alignment.edge_selector import FailedSpan
from app.services.timeanchored_alignment.subtitle_assembler import SubtitleAssembler


def _item(
    text: str,
    *,
    start: float,
    end: float,
    status: str = "direct",
    source: str = "text_aligner",
) -> AlignmentItem:
    return AlignmentItem(
        text=text,
        start=start,
        end=end,
        status=status,
        source=source,
        confidence=0.9,
        reason="test",
    )


def _result(route: str, items: tuple[AlignmentItem, ...]) -> FinalAlignmentResult:
    failed = sum(1 for row in items if row.status == "failed")
    return FinalAlignmentResult(
        items=items,
        route=route,
        metrics=AlignmentMetrics(
            coverage=float(len(items) - failed) / float(max(len(items), 1)),
            duration_ratio=1.0,
            failed_count=failed,
            route_confidence=0.8,
        ),
    )


def test_assemble_mixed_route_splices_failed_span_with_fallback() -> None:
    assembler = SubtitleAssembler()
    base = _result(
        "mixed",
        (
            _item("你", start=0.0, end=0.1, status="direct"),
            _item("坏", start=0.1, end=0.2, status="failed"),
            _item("好", start=0.2, end=0.3, status="direct"),
        ),
    )
    fallback = _result(
        "fast",
        (
            _item("你", start=0.0, end=0.1, status="direct", source="edge_selector_fast"),
            _item("啊", start=0.1, end=0.2, status="direct", source="edge_selector_fast"),
            _item("好", start=0.2, end=0.3, status="direct", source="edge_selector_fast"),
        ),
    )

    stream = assembler.assemble(
        base_result=base,
        fallback_result=fallback,
        failed_spans=(FailedSpan(start=1, end=1),),
    )

    assert [item.text for item in stream] == ["你", "啊", "好"]
    assert stream[1].source == "edge_selector_fast"
    assert stream[1].status == "estimated"


def test_assemble_enforces_monotonic_time() -> None:
    assembler = SubtitleAssembler()
    base = _result(
        "text",
        (
            _item("A", start=0.5, end=0.7),
            _item("B", start=0.4, end=0.6),
            _item("C", start=0.6, end=0.8),
        ),
    )

    stream = assembler.assemble(base_result=base)

    assert stream[0].start <= stream[1].start <= stream[2].start
    assert stream[0].end <= stream[1].end <= stream[2].end
