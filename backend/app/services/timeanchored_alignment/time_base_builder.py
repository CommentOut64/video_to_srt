"""TimeBaseBuilder：从 FastWorker 上下文构建时间基底。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from app.services.timeanchored_alignment.adapters.fast import FastTimeAdapter, SenseVoiceTimeAdapter
from app.services.timeanchored_alignment.contracts import TimeBasePackage


class TimeBaseBuilder:
    """协调适配器并管理 ctc 大对象生命周期。"""

    def __init__(self, adapter: Optional[FastTimeAdapter] = None) -> None:
        self._adapter = adapter or SenseVoiceTimeAdapter(vocab=None, blank_id=0)

    def build(self, ctx: Any) -> Optional[TimeBasePackage]:
        sv_result = getattr(ctx, "sv_result", None) or {}
        if not isinstance(sv_result, dict) or not sv_result:
            return None

        language = str(sv_result.get("language") or "auto")
        frame_stride = float(sv_result.get("ctc_frame_stride", 0.06) or 0.06)
        ctc_logits = sv_result.get("ctc_logits")
        compact_trace = sv_result.get("ctc_compact_trace") or {}
        encoder_out_lens = sv_result.get("encoder_out_lens")
        raw_tokens = sv_result.get("raw_tokens") or compact_trace.get("raw_tokens") or []

        if ctc_logits is None and not raw_tokens:
            return None

        logits_for_build = ctc_logits
        if ctc_logits is not None and not self._adapter.can_decode_ctc:
            # 未提供 vocab 时，回退到紧凑轨迹路径，避免硬依赖完整矩阵。
            logits_for_build = None

        try:
            package = self._adapter.build_time_base(
                ctc_logits=logits_for_build,
                compact_acoustic_trace=compact_trace,
                raw_tokens=raw_tokens,
                language=language,
                frame_stride=frame_stride,
                encoder_out_lens=encoder_out_lens,
            )
        finally:
            # 构建后立即释放完整矩阵，防止在上下文中长期滞留。
            sv_result.pop("ctc_logits", None)

        metadata = dict(getattr(package, "metadata", {}) or {})
        if encoder_out_lens is not None:
            metadata["encoder_out_lens"] = int(encoder_out_lens)
        blank_track = sv_result.get("blank_track")
        if blank_track is not None:
            metadata["blank_track"] = [float(item) for item in list(blank_track)]
        sparse_logits = sv_result.get("sparse_logits")
        if sparse_logits is not None:
            metadata["sparse_logits"] = list(sparse_logits)
        if metadata != dict(getattr(package, "metadata", {}) or {}):
            package = replace(package, metadata=metadata)

        return package
