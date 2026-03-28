from __future__ import annotations

from app.services.timeanchored_alignment.contracts import TokenUnit
from app.services.timeanchored_alignment.preparation.token_provenance_binder import (
    TokenProvenanceBinder,
)
from app.services.timeanchored_alignment.slow_window.contracts import WindowSourceUnit


def test_token_provenance_binder_keeps_token_units_and_provenance() -> None:
    binder = TokenProvenanceBinder()
    text = "No you're fine"
    token_units = (
        TokenUnit(token_text="No", language="en", char_start=0, char_end=2),
        TokenUnit(token_text="you're", language="en", char_start=3, char_end=9),
        TokenUnit(token_text="fine", language="en", char_start=10, char_end=14),
    )
    source_units = (
        WindowSourceUnit(
            unit_id="unit-1",
            semantic_chunk_id="sem-1",
            text="No you're",
            audio_start=0.0,
            audio_end=0.8,
            source_chunk_ids=("chunk-1",),
            source_chunk_indices=(1,),
            speaker_id="speaker-a",
            turn_id="turn-a",
            language="en",
            arrived_at=1.0,
        ),
        WindowSourceUnit(
            unit_id="unit-2",
            semantic_chunk_id="sem-2",
            text="fine",
            audio_start=0.8,
            audio_end=1.2,
            source_chunk_ids=("chunk-2",),
            source_chunk_indices=(2,),
            speaker_id="speaker-b",
            turn_id="turn-b",
            language="en",
            arrived_at=2.0,
        ),
    )

    prepared = binder.bind(
        text=text,
        token_units=token_units,
        source_units=source_units,
    )

    assert [item.token_text for item in prepared] == ["No", "you're", "fine"]
    assert [item.source_chunk_ids for item in prepared] == [("chunk-1",), ("chunk-1",), ("chunk-2",)]
    assert [item.source_unit_ids for item in prepared] == [("unit-1",), ("unit-1",), ("unit-2",)]
