from __future__ import annotations

from app.models.sensevoice_models import SentenceSegment
from app.services.punctuation.semantic_buffer import SemanticChunk
from app.services.timeanchored_alignment.hint_builder import HintBuilder


def _build_chunk(text: str, *, start: float, end: float, language: str = "en") -> SemanticChunk:
    return SemanticChunk(
        chunk_id=f"chunk-{int(start * 10)}",
        text=text,
        sentences=[SentenceSegment(text=text, text_clean=text, start=start, end=end)],
        punctuation_result=None,
        punctuation_decision=None,
        pending_tail="",
        audio_range=(start, end),
        language=language,
        source_chunks=[f"chunk-{int(start * 10)}"],
        speaker_id="spk-1",
    )


def test_hint_builder_outputs_terms_and_tail_context_only() -> None:
    builder = HintBuilder(max_hint_items=6)
    long_text = (
        "OpenAI Whisper GPU batching keeps latency stable while domain terms like Kubernetes and PyTorch stay consistent."
    )
    chunks = [
        _build_chunk(long_text, start=0.0, end=2.0, language="en"),
        _build_chunk("Focus on Kubernetes rollout and Whisper prompt hygiene.", start=2.0, end=4.0, language="en"),
    ]

    hints = builder.build_hints(chunks, primary_language="en")
    merged = " ".join(hints)

    assert "OpenAI" in merged
    assert "Kubernetes" in merged
    assert "Whisper" in merged
    assert long_text not in merged


def test_hint_builder_deduplicates_and_respects_limit() -> None:
    builder = HintBuilder(max_hint_items=4)
    chunks = [
        _build_chunk("Whisper Whisper Whisper latency tuning", start=0.0, end=1.0),
        _build_chunk("latency tuning and prompt budget", start=1.0, end=2.0),
    ]

    hints = builder.build_hints(chunks, primary_language="en")

    assert len(hints) <= 4
    assert len(set(hints)) == len(hints)
