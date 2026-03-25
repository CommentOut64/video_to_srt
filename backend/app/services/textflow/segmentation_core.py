"""统一切分核心入口（Phase 3）。"""

from __future__ import annotations

from typing import Any, Optional

from app.services.alignment.types import DecisionLayerInput, DecisionLayerOutput


class SegmentationCore:
    """统一切分层核心入口。

    由 `SegmentationCore` 统一承接最终切分调用与主流程 ownership。
    具体切分细节仍复用 DecisionLayer 上的规则与后处理 helper，但不再通过
    `process_impl=self._run_legacy_segmentation` 这种 legacy 委托壳接入。
    """

    def __init__(
        self,
        *,
        processor: Any,
    ) -> None:
        self._processor = processor

    def process(
        self,
        data: DecisionLayerInput,
        *,
        stream_id: str = "main",
        chunk_index: Optional[int] = None,
        is_last_chunk: bool = False,
    ) -> DecisionLayerOutput:
        return self._processor._run_segmentation_core(
            data=data,
            stream_id=stream_id,
            chunk_index=chunk_index,
            is_last_chunk=is_last_chunk,
        )
