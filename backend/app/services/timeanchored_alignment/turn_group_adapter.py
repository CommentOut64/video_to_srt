"""SlowInferenceWindowEnvelope 到 TurnGroupEnvelope 的薄适配器。"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from app.services.bridge.turn_group_builder import TurnGroupEnvelope
from app.services.bridge.turn_group_models import TurnGroup
from app.services.timeanchored_alignment.contracts import SlowInferenceWindowEnvelope


class TurnGroupAdapter:
    """将新组窗契约转换为现有慢流消费契约。"""

    def to_turn_group_envelope(self, envelope: SlowInferenceWindowEnvelope) -> TurnGroupEnvelope:
        if envelope is None or envelope.window is None:
            raise ValueError("TurnGroupAdapter 输入窗口不能为空")

        window = envelope.window
        metadata = dict(window.metadata or {})
        speaker_id = str(metadata.get("speaker_id") or "unknown")
        flush_reason = str(metadata.get("flush_reason") or "window_flush")
        primary_language = str(metadata.get("primary_language") or window.language or "auto")
        audio_segments = self._resolve_audio_segments(window.start, window.end, metadata)
        prompt_text = self._build_prompt_from_hints(window.hints)

        group_metadata: Dict[str, Any] = {
            "window_id": window.window_id,
            "primary_language": primary_language,
            "is_mixed_window": bool(metadata.get("is_mixed_window", False)),
            "route": str(metadata.get("route") or "main_chain"),
            "window_mode": str(metadata.get("window_mode") or ""),
        }
        group_metadata.update({key: value for key, value in metadata.items() if key not in group_metadata})

        group = TurnGroup(
            group_id=window.window_id,
            target_turn_ids=[str(item) for item in metadata.get("turn_ids", ())],
            context_turn_ids=[str(item) for item in metadata.get("context_turn_ids", ())],
            speaker_id=speaker_id,
            audio_segments=audio_segments,
            prompt_text=prompt_text,
            flush_reason=flush_reason,
            language=primary_language,
            source_chunks=[str(item) for item in envelope.source_contexts],
            metadata=group_metadata,
        )
        return TurnGroupEnvelope(
            group=group,
            sentences=[],
            punctuation_decision=None,
        )

    @staticmethod
    def _build_prompt_from_hints(hints: Tuple[str, ...]) -> str:
        normalized = [str(item).strip() for item in hints if str(item).strip()]
        return " ".join(normalized).strip()

    @staticmethod
    def _resolve_audio_segments(start: float, end: float, metadata: Dict[str, Any]) -> list[tuple[float, float]]:
        resolved = []
        raw_segments = metadata.get("audio_segments", ())
        for item in raw_segments:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            seg_start = float(item[0])
            seg_end = float(item[1])
            if seg_end <= seg_start:
                continue
            resolved.append((seg_start, seg_end))
        if resolved:
            return resolved
        start_sec = float(start)
        end_sec = float(end)
        if end_sec <= start_sec:
            end_sec = start_sec + 1e-3
        return [(start_sec, end_sec)]
