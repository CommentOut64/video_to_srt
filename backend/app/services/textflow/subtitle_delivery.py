"""SubtitleDelivery：统一南向字幕 DTO 投递层。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.services.textflow.contracts import (
    RenderResult,
    SubtitleBatch,
    SubtitleItem,
)


class SubtitleDelivery:
    """将 RenderResult 收口为统一字幕批次 DTO。"""

    def build_batch(
        self,
        *,
        render_result: RenderResult,
        chunk_id: str,
        chunk_index: Optional[int] = None,
        ingress_context: Optional[Dict[str, Any]] = None,
        source: str = "render_core",
    ) -> SubtitleBatch:
        items = tuple(
            SubtitleItem(
                segment_id=item.segment_id,
                chunk_id=chunk_id,
                start=float(item.start),
                end=float(item.end),
                text=str(item.text_display),
                status="final",
                source=str(item.text_source or "unknown"),
                speaker_id=item.speaker_id,
                turn_id=item.turn_id,
                trace=dict(item.trace or {}),
            )
            for item in render_result.subtitles
        )
        diagnostics = {
            "source": str(source or "render_core"),
            "render_output_trace": [dict(entry) for entry in render_result.output_trace],
        }
        if ingress_context:
            diagnostics["ingress_context"] = dict(ingress_context)
        return SubtitleBatch(
            chunk_id=str(chunk_id),
            chunk_index=chunk_index,
            items=items,
            render_report=dict(render_result.render_report or {}),
            diagnostics=diagnostics,
        )
