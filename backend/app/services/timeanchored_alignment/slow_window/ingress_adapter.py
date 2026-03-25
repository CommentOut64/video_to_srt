"""SlowWindowBuilder 入口适配。"""

from __future__ import annotations

import time
from typing import Iterable, Optional

from app.services.punctuation.semantic_buffer import SemanticChunk
from app.services.timeanchored_alignment.slow_window.contracts import SlowWindowIngressUnit


class SlowWindowIngressAdapter:
    """将旧 SemanticChunk 收口成强类型 ingress。"""

    def adapt(
        self,
        chunk: SemanticChunk,
        *,
        speaker_id: Optional[str],
        turn_id: Optional[str],
        source_chunk_indices: Iterable[int],
        arrived_at: Optional[float] = None,
    ) -> SlowWindowIngressUnit:
        if chunk is None:
            raise ValueError("SlowWindowIngressAdapter 输入 chunk 不能为空")

        source_chunk_ids = tuple(str(item) for item in (getattr(chunk, "source_chunks", None) or ()) if str(item))
        resolved_indices = tuple(int(item) for item in source_chunk_indices)
        if not resolved_indices:
            raise ValueError("SlowWindowIngressAdapter 需要显式 source_chunk_indices")
        audio_range = getattr(chunk, "audio_range", (0.0, 0.0))
        return SlowWindowIngressUnit(
            semantic_chunk_id=str(getattr(chunk, "chunk_id", "") or ""),
            text=str(getattr(chunk, "text", "") or ""),
            sentences=tuple(getattr(chunk, "sentences", ()) or ()),
            punctuation_decision=getattr(chunk, "punctuation_decision", None),
            audio_range=(float(audio_range[0]), float(audio_range[1])),
            language=str(getattr(chunk, "language", "") or ""),
            source_chunk_ids=source_chunk_ids,
            source_chunk_indices=resolved_indices,
            speaker_id=str(speaker_id or getattr(chunk, "speaker_id", None) or "unknown"),
            turn_id=str(turn_id) if turn_id else None,
            word_timestamps=tuple(),
            arrived_at=float(arrived_at if arrived_at is not None else time.time()),
        )
