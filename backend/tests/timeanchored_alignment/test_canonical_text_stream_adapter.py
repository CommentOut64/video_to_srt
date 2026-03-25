from __future__ import annotations

from app.services.alignment.types import AlignedFacts, AnnotatedWord, CharMapping, TextTrack, TextTrackBundle
from app.services.punctuation.base import PuncPosition
from app.services.textflow.canonical_text_stream_adapter import CanonicalTextStreamAdapter
from app.services.textflow.contracts import SegmentationIngressContext
from app.services.timeanchored_alignment.contracts import BoundaryEvidence, ProtectedSpan


def _track(
    *,
    text: str,
    source: str,
    punct_positions: list[PuncPosition],
    clean_to_word: list[int | None] | None = None,
    language: str = "en",
) -> TextTrack:
    mapping = [CharMapping(raw_idx=i, clean_idx=i) for i, _ in enumerate(text)]
    return TextTrack(
        raw_text=text,
        text_itn_raw=text,
        text_clean=text,
        char_mapping=mapping,
        raw_to_clean=list(range(len(text))),
        clean_to_raw=list(range(len(text))),
        language=language,
        source=source,
        clean_to_word=list(clean_to_word or []),
        punct_positions=punct_positions,
        mapping_coverage=1.0 if text else 0.0,
    )


def _build_clean_to_word(text: str, spans: list[tuple[str, int | None]]) -> list[int | None]:
    mapping: list[int | None] = []
    cursor = 0
    for chunk_text, word_index in spans:
        assert text[cursor : cursor + len(chunk_text)] == chunk_text
        mapping.extend([word_index] * len(chunk_text))
        cursor += len(chunk_text)
    assert cursor == len(text)
    return mapping


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
        ingress_context=SegmentationIngressContext(
            unit_kind="chunk",
            unit_id="chunk-1",
            chunk_id="chunk-1",
            chunk_index=1,
        ),
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
    assert stream.diagnostics.ingress_context["unit_kind"] == "chunk"
    assert stream.metadata["ingress_context"]["chunk_id"] == "chunk-1"

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
        ingress_context=SegmentationIngressContext(
            unit_kind="slow_window",
            unit_id="window-2",
            slow_window_id="window-2",
            turn_group_id="turn-group-2",
            window_coverage=0.5,
            source_chunk_ids=("chunk-2",),
            projection_chunk_ids=("chunk-2",),
        ),
        tracks=tracks,
        text_source="slow",
        language="en",
    )
    comma = [fact for fact in stream.punctuation_facts if fact.normalized_text == ","]
    assert len(comma) == 1
    assert comma[0].source == "slow"
    assert stream.chunk_ref == "chunk-2"
    assert stream.diagnostics.ingress_context["slow_window_id"] == "window-2"
    assert stream.diagnostics.ingress_context["turn_group_id"] == "turn-group-2"
    assert stream.diagnostics.ingress_context["window_coverage"] == 0.5
    assert stream.metadata["ingress_context"]["source_chunk_ids"] == ["chunk-2"]
    assert stream.metadata["ingress_context"]["projection_chunk_ids"] == ["chunk-2"]


def test_adapter_uses_clean_to_word_to_keep_chinese_word_level_tokens() -> None:
    adapter = CanonicalTextStreamAdapter()
    text = "幸运的他躲过了一劫，警方一共发现了4瓶毒可乐"
    words = [
        AnnotatedWord(word="幸运的他", start=0.0, end=0.4, confidence=0.98),
        AnnotatedWord(word="躲过了", start=0.4, end=0.8, confidence=0.97),
        AnnotatedWord(word="一劫", start=0.8, end=1.1, confidence=0.96),
        AnnotatedWord(word="警方", start=1.2, end=1.5, confidence=0.95),
        AnnotatedWord(word="一共", start=1.5, end=1.8, confidence=0.94),
        AnnotatedWord(word="发现了", start=1.8, end=2.2, confidence=0.93),
        AnnotatedWord(word="4", start=2.2, end=2.3, confidence=0.92),
        AnnotatedWord(word="瓶", start=2.3, end=2.4, confidence=0.91),
        AnnotatedWord(word="毒可乐", start=2.4, end=2.9, confidence=0.9),
    ]
    clean_to_word = _build_clean_to_word(
        text,
        [
            ("幸运的他", 0),
            ("躲过了", 1),
            ("一劫", 2),
            ("，", None),
            ("警方", 3),
            ("一共", 4),
            ("发现了", 5),
            ("4", 6),
            ("瓶", 7),
            ("毒可乐", 8),
        ],
    )

    stream = adapter.build(
        stream_id="stream-zh",
        chunk_ref="chunk-zh",
        tracks=TextTrackBundle(
            chosen_track=_track(
                text=text,
                source="aligned",
                punct_positions=[],
                clean_to_word=clean_to_word,
                language="zh",
            )
        ),
        text_source="aligned",
        language="zh",
        aligned_facts=AlignedFacts(annotated_words=words),
    )

    assert [token.text_core for token in stream.tokens] == [word.word for word in words]
    assert [token.index for token in stream.tokens] == list(range(len(words)))
    assert stream.tokens[0].start == 0.0
    assert stream.tokens[-1].end == 2.9
    assert len(stream.punctuation_facts) == 1
    assert stream.punctuation_facts[0].left_token_index == 2
    assert stream.punctuation_facts[0].right_token_index == 3


def test_adapter_word_aligned_tokens_strip_trailing_punctuation_from_word_mapped_span() -> None:
    adapter = CanonicalTextStreamAdapter()
    text = "大家好,，欢迎"
    words = [
        AnnotatedWord(word="大家好,", start=0.0, end=0.5, confidence=0.98),
        AnnotatedWord(word="欢迎", start=0.5, end=1.0, confidence=0.97),
    ]
    clean_to_word = _build_clean_to_word(
        text,
        [
            ("大家好,", 0),
            ("，", None),
            ("欢迎", 1),
        ],
    )

    stream = adapter.build(
        stream_id="stream-punct-tail",
        chunk_ref="chunk-punct-tail",
        tracks=TextTrackBundle(
            chosen_track=_track(
                text=text,
                source="aligned",
                punct_positions=[],
                clean_to_word=clean_to_word,
                language="zh",
            )
        ),
        text_source="aligned",
        language="zh",
        aligned_facts=AlignedFacts(annotated_words=words),
    )

    assert [token.text_core for token in stream.tokens] == ["大家好", "欢迎"]


def test_adapter_maps_clean_text_punctuation_positions_to_current_token_trailing_anchor() -> None:
    adapter = CanonicalTextStreamAdapter()

    stream = adapter.build(
        stream_id="stream-clean-punct",
        chunk_ref="chunk-clean-punct",
        tracks=TextTrackBundle(
            chosen_track=_track(
                text="你好世界",
                source="aligned",
                punct_positions=[
                    PuncPosition(char_index=1, punctuation="，", confidence=0.99),
                    PuncPosition(char_index=3, punctuation="。", confidence=0.99),
                ],
                clean_to_word=[0, 1, 2, 3],
                language="zh",
            )
        ),
        text_source="aligned",
        language="zh",
        aligned_facts=AlignedFacts(
            annotated_words=[
                AnnotatedWord(word="你", start=0.0, end=0.1, confidence=0.9),
                AnnotatedWord(word="好", start=0.1, end=0.2, confidence=0.9),
                AnnotatedWord(word="世", start=0.2, end=0.3, confidence=0.9),
                AnnotatedWord(word="界", start=0.3, end=0.4, confidence=0.9),
            ]
        ),
    )

    facts = list(stream.punctuation_facts)
    assert len(facts) == 2
    assert (facts[0].left_token_index, facts[0].right_token_index, facts[0].attach_mode) == (1, None, "trailing")
    assert (facts[1].left_token_index, facts[1].right_token_index, facts[1].attach_mode) == (3, None, "trailing")
