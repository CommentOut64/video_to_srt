from __future__ import annotations

from app.models.sensevoice_models import SentenceSegment
from app.services.timeanchored_alignment.chunk_projector import (
    ChunkProjector,
    ChunkWindow,
)


def _segment(text: str, *, start: float, end: float) -> SentenceSegment:
    return SentenceSegment(text=text, text_clean=text, start=start, end=end)


def test_cross_chunk_sentence_assigned_by_max_overlap() -> None:
    projector = ChunkProjector()
    windows = (
        ChunkWindow(chunk_ref=0, start=0.0, end=1.0),
        ChunkWindow(chunk_ref=1, start=1.0, end=2.0),
    )
    segments = (
        _segment("chunk0", start=0.1, end=0.8),
        _segment("cross", start=0.9, end=1.4),
    )

    projected = projector.project(sentence_segments=segments, chunk_windows=windows)

    assert [seg.text for seg in projected[0].sentence_segments] == ["chunk0"]
    assert [seg.text for seg in projected[1].sentence_segments] == ["cross"]


def test_projector_keeps_empty_chunk_for_replace_chunk_cleanup() -> None:
    projector = ChunkProjector()
    windows = (
        ChunkWindow(chunk_ref=0, start=0.0, end=1.0),
        ChunkWindow(chunk_ref=1, start=1.0, end=2.0),
    )
    segments = (_segment("only-first", start=0.1, end=0.7),)

    projected = projector.project(sentence_segments=segments, chunk_windows=windows)

    assert len(projected) == 2
    assert [seg.text for seg in projected[0].sentence_segments] == ["only-first"]
    assert projected[1].sentence_segments == ()
