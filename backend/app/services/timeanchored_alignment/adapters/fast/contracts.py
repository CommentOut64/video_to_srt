"""快流时间基适配器协议。"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence

from app.services.timeanchored_alignment.contracts import TimeBasePackage


class FastTimeAdapter(Protocol):
    """任意快流观测适配器都应满足的最小协议。"""

    @property
    def can_decode_ctc(self) -> bool:
        """是否能直接消费完整 CTC logits。"""

    def build_time_base(
        self,
        *,
        ctc_logits: Any,
        language: str,
        frame_stride: float = 0.06,
        encoder_out_lens: Optional[int] = None,
        compact_acoustic_trace: Optional[dict[str, Any]] = None,
        raw_tokens: Optional[Sequence[dict[str, Any]]] = None,
    ) -> TimeBasePackage:
        """把快流观测转换成统一 TimeBasePackage。"""
