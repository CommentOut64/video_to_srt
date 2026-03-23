"""统一切分核心入口（Phase 3）。"""

from __future__ import annotations

from typing import Callable, Optional

from app.services.alignment.types import DecisionLayerInput, DecisionLayerOutput


class SegmentationCore:
    """统一切分层核心入口。

    Phase 3 目标是先收口入口 ownership：由 `SegmentationCore` 统一承接最终切分调用，
    具体策略仍复用现有裁决层实现，避免一次性迁移引入行为回归。
    """

    def __init__(
        self,
        *,
        process_impl: Callable[..., DecisionLayerOutput],
    ) -> None:
        self._process_impl = process_impl

    def process(
        self,
        data: DecisionLayerInput,
        *,
        stream_id: str = "main",
        chunk_index: Optional[int] = None,
        is_last_chunk: bool = False,
    ) -> DecisionLayerOutput:
        return self._process_impl(
            data=data,
            stream_id=stream_id,
            chunk_index=chunk_index,
            is_last_chunk=is_last_chunk,
        )

