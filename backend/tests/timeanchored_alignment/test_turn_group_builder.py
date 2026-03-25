from __future__ import annotations

from app.models.sensevoice_models import SentenceSegment
from app.services.bridge.flush_policy import FlushPolicyConfig
from app.services.bridge.turn_group_builder import TurnGroupBuilder
from app.services.punctuation.semantic_buffer import SemanticChunk


def _build_chunk(
    *,
    chunk_id: str,
    source_chunks: list[str],
    start: float,
    end: float,
    text: str = "测试文本",
) -> SemanticChunk:
    return SemanticChunk(
        chunk_id=chunk_id,
        text=text,
        sentences=[SentenceSegment(text=text, text_clean=text, start=start, end=end)],
        punctuation_result=None,
        punctuation_decision=None,
        pending_tail="",
        audio_range=(start, end),
        language="zh",
        source_chunks=source_chunks,
        speaker_id="spk-1",
    )


def test_turn_group_builder_dedupes_overlapping_source_chunks_in_order() -> None:
    builder = TurnGroupBuilder(
        flush_config=FlushPolicyConfig(
            min_audio_sec=99.0,
            min_token_count=999,
            max_wait_sec=99.0,
            tail_idle_sec=99.0,
            long_pause_cut_sec=99.0,
        )
    )

    assert builder.add_chunk(
        _build_chunk(
            chunk_id="semantic-1",
            source_chunks=["chunk-0", "chunk-1"],
            start=0.0,
            end=1.0,
        ),
        speaker_id="spk-1",
        turn_id="turn-1",
        now=0.0,
    ) == []
    assert builder.add_chunk(
        _build_chunk(
            chunk_id="semantic-2",
            source_chunks=["chunk-1", "chunk-2"],
            start=1.0,
            end=2.0,
        ),
        speaker_id="spk-1",
        turn_id="turn-1",
        now=0.1,
    ) == []

    envelope = builder.flush("eof_flush")

    assert envelope is not None
    assert envelope.group.source_chunks == ["chunk-0", "chunk-1", "chunk-2"]
