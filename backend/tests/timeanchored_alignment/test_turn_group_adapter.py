from __future__ import annotations

from app.services.timeanchored_alignment.contracts import (
    SlowInferenceWindow,
    SlowInferenceWindowEnvelope,
)
from app.services.timeanchored_alignment.turn_group_adapter import TurnGroupAdapter


def _build_window_envelope(*, is_mixed: bool = False) -> SlowInferenceWindowEnvelope:
    window = SlowInferenceWindow(
        window_id="sw-000123",
        start=10.0,
        end=16.0,
        language="mixed" if is_mixed else "zh",
        chunk_indices=(8, 9),
        hints=("OpenAI", "Whisper", "ctx:zh:测试上下文"),
        metadata={
            "flush_reason": "speaker_change",
            "speaker_id": "spk-9",
            "turn_ids": ("turn-1", "turn-2"),
            "context_turn_ids": ("turn-0",),
            "audio_segments": ((10.0, 12.5), (12.8, 16.0)),
            "primary_language": "mixed" if is_mixed else "zh",
            "is_mixed_window": is_mixed,
            "route": "fallback" if is_mixed else "main_chain",
        },
    )
    return SlowInferenceWindowEnvelope(
        window=window,
        aggregated_time_base=None,
        source_contexts=("chunk-8", "chunk-9"),
    )


def test_turn_group_adapter_converts_window_envelope() -> None:
    adapter = TurnGroupAdapter()
    envelope = _build_window_envelope(is_mixed=False)

    output = adapter.to_turn_group_envelope(envelope)

    assert output.group.group_id == "sw-000123"
    assert output.group.language == "zh"
    assert output.group.source_chunks == ["chunk-8", "chunk-9"]


def test_turn_group_adapter_maps_chunk_speaker_flush_and_prompt() -> None:
    adapter = TurnGroupAdapter()
    envelope = _build_window_envelope(is_mixed=False)

    output = adapter.to_turn_group_envelope(envelope)

    assert output.group.speaker_id == "spk-9"
    assert output.group.flush_reason == "speaker_change"
    assert output.group.target_turn_ids == ["turn-1", "turn-2"]
    assert output.group.context_turn_ids == ["turn-0"]
    assert output.group.audio_segments == [(10.0, 12.5), (12.8, 16.0)]
    assert "OpenAI" in output.group.prompt_text
    assert "Whisper" in output.group.prompt_text


def test_turn_group_adapter_marks_mixed_window_for_fallback() -> None:
    adapter = TurnGroupAdapter()
    envelope = _build_window_envelope(is_mixed=True)

    output = adapter.to_turn_group_envelope(envelope)

    assert output.group.metadata["is_mixed_window"] is True
    assert output.group.metadata["route"] == "fallback"
    assert output.group.metadata["window_id"] == "sw-000123"
