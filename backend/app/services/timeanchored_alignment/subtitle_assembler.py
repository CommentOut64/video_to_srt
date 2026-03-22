"""SubtitleAssembler：将各层结果组装为有序终稿流。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.services.timeanchored_alignment.contracts import AlignmentItem, FinalAlignmentResult
from app.services.timeanchored_alignment.edge_selector import FailedSpan


FinalAlignedStream = tuple[AlignmentItem, ...]


@dataclass(frozen=True)
class SubtitleAssembler:
    """终稿流组装器。"""

    def assemble(
        self,
        *,
        base_result: FinalAlignmentResult,
        fallback_result: FinalAlignmentResult | None = None,
        failed_spans: Sequence[FailedSpan] = (),
    ) -> FinalAlignedStream:
        """输出单调有序的终稿 AlignmentItem 流。"""
        stream = list(base_result.items)
        if base_result.route == "mixed" and fallback_result is not None and failed_spans:
            stream = self._splice_failed_spans(
                stream=stream,
                fallback=fallback_result.items,
                failed_spans=failed_spans,
            )
        elif not stream and fallback_result is not None:
            stream = list(fallback_result.items)
        return self._ensure_monotonic(tuple(stream))

    @staticmethod
    def _splice_failed_spans(
        *,
        stream: list[AlignmentItem],
        fallback: Sequence[AlignmentItem],
        failed_spans: Sequence[FailedSpan],
    ) -> list[AlignmentItem]:
        if not stream or not fallback:
            return stream
        max_index = len(stream) - 1
        for span in failed_spans:
            start = max(0, int(span.start))
            end = min(max_index, int(span.end))
            for idx in range(start, end + 1):
                if idx >= len(fallback):
                    continue
                src = fallback[idx]
                stream[idx] = AlignmentItem(
                    text=src.text,
                    start=src.start,
                    end=src.end,
                    status="estimated",
                    source=src.source,
                    confidence=src.confidence,
                    reason="mixed_span_fallback",
                )
        return stream

    @staticmethod
    def _ensure_monotonic(stream: FinalAlignedStream) -> FinalAlignedStream:
        if not stream:
            return tuple()
        cursor = float(stream[0].start)
        fixed: list[AlignmentItem] = []
        for item in stream:
            start = max(float(item.start), cursor)
            end = max(float(item.end), start)
            cursor = end
            reason = item.reason
            if start != float(item.start) or end != float(item.end):
                reason = f"{reason}|monotonic_adjusted" if reason else "monotonic_adjusted"
            fixed.append(
                AlignmentItem(
                    text=item.text,
                    start=start,
                    end=end,
                    status=item.status,
                    source=item.source,
                    confidence=item.confidence,
                    reason=reason,
                )
            )
        return tuple(fixed)


__all__ = ["FinalAlignedStream", "SubtitleAssembler"]
