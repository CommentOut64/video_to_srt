from __future__ import annotations

from app.services.alignment.types import CharMapping, TextTrack, TextTrackBundle
from app.services.punctuation.base import PuncPosition
from app.services.textflow.canonical_text_stream_adapter import CanonicalTextStreamAdapter
from app.services.timeanchored_alignment.contracts import BoundaryEvidence, ProtectedSpan


def _track(*, text: str, source: str, punct_positions: list[PuncPosition]) -> TextTrack:
    mapping = [CharMapping(raw_idx=i, clean_idx=i) for i, _ in enumerate(text)]
    return TextTrack(
        raw_text=text,
        text_itn_raw=text,
        text_clean=text,
        char_mapping=mapping,
        raw_to_clean=list(range(len(text))),
        clean_to_raw=list(range(len(text))),
        language="en",
        source=source,
        punct_positions=punct_positions,
    )


def test_adapter_builds_canonical_stream_and_keeps_diagnostics() -> None:
    adapter = CanonicalTextStreamAdapter()
    tracks = TextTrackBundle(
        sv_track=_track(
            text="Hello, world?",
            source="sensevoice",
            punct_positions=[
                PuncPosition(char_index=5, punctuation=",", confidence=0.95),
                PuncPosition(char_index=12, punctuation="?", confidence=0.95),
            ],
        ),
        whisper_track=_track(
            text="Hello, world?",
            source="whisper",
            punct_positions=[
                PuncPosition(char_index=5, punctuation=",", confidence=0.88),
            ],
        ),
        chosen_track=_track(
            text="Hello, world?",
            source="sensevoice",
            punct_positions=[],
        ),
    )
    boundaries = (
        BoundaryEvidence(
            split_idx=1,
            event_time=1.2,
            left_end=1.1,
            right_start=1.2,
            reason="pause",
            score=0.7,
            hard_flag=False,
        ),
    )
    spans = (ProtectedSpan(start=0, end=5, kind="word"),)

    stream = adapter.build(
        stream_id="stream-1",
        chunk_ref="chunk-1",
        tracks=tracks,
        text_source="fast",
        language="en",
        candidate_boundaries=boundaries,
        protected_spans=spans,
        raw_mount_trace={"mapping_pairs": [{"a": 1}]},
    )

    assert [token.text_core for token in stream.tokens] == ["Hello", "world"]
    assert stream.candidate_boundaries == boundaries
    assert stream.protected_spans == spans
    assert stream.diagnostics.raw_mount_trace["mapping_pairs"] == [{"a": 1}]

    # fast 优先：逗号应保留 fast，slow 同位同类逗号应被去重淘汰并写入 dedup_log。
    comma = [fact for fact in stream.punctuation_facts if fact.normalized_text == ","]
    assert len(comma) == 1
    assert comma[0].source == "fast"
    assert any(record.get("reason") == "dedup_lower_priority" for record in stream.diagnostics.dedup_log)

    assert len(stream.diagnostics.raw_fast_punctuation) == 2
    assert len(stream.diagnostics.raw_slow_punctuation) == 1
    assert len(stream.diagnostics.raw_aligned_punctuation) == 0


def test_adapter_respects_text_source_priority_for_slow() -> None:
    adapter = CanonicalTextStreamAdapter()
    tracks = TextTrackBundle(
        sv_track=_track(
            text="Hi, all!",
            source="sensevoice",
            punct_positions=[PuncPosition(char_index=2, punctuation=",", confidence=0.8)],
        ),
        whisper_track=_track(
            text="Hi, all!",
            source="whisper",
            punct_positions=[PuncPosition(char_index=2, punctuation=",", confidence=0.9)],
        ),
        chosen_track=_track(
            text="Hi, all!",
            source="whisper",
            punct_positions=[],
        ),
    )

    stream = adapter.build(
        stream_id="stream-2",
        chunk_ref="chunk-2",
        tracks=tracks,
        text_source="slow",
        language="en",
    )
    comma = [fact for fact in stream.punctuation_facts if fact.normalized_text == ","]
    assert len(comma) == 1
    assert comma[0].source == "slow"
