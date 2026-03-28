from __future__ import annotations

from app.services.alignment.types import AnnotatedWord, DecisionLayerInput
from app.services.punctuation.final_splitter import FinalSplitter
from app.services.textflow.contracts import SegmentationIngressContext
from app.services.textflow.decision_layer import SegmentationProcessor


def _build_timeanchored_decision_input(
    *,
    unit_id: str,
    chunk_id: str,
    chunk_index: int,
    words: list[tuple[str, float, float]],
) -> DecisionLayerInput:
    ingress_context = SegmentationIngressContext(
        unit_kind="slow_window",
        unit_id=unit_id,
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        slow_window_id=unit_id,
        source_chunk_ids=(chunk_id,),
        projection_chunk_ids=(),
    )
    return DecisionLayerInput(
        annotated_words=[
            AnnotatedWord(
                word=word,
                start=start,
                end=end,
                confidence=0.9,
                confidence_source="aligned",
            )
            for word, start, end in words
        ],
        vad_intervals=[],
        ingress_context=ingress_context,
    )


def test_timeanchored_ingress_does_not_fragment_english_multiword_clause() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    decision_input = _build_timeanchored_decision_input(
        unit_id="window-qvvs-fragment",
        chunk_id="chunk-qvvs-fragment",
        chunk_index=1,
        words=[
            ("killinging", 18.20, 18.68),
            ("five", 18.74, 19.52),
            ("strangers", 19.94, 20.24),
            ("is", 20.30, 20.44),
            ("probably", 20.46, 20.78),
            ("more", 20.82, 21.04),
            ("beneficial", 21.08, 21.62),
            ("for", 21.70, 21.82),
            ("society,", 21.84, 22.30),
            ("so", 22.36, 22.50),
            ("I'll", 22.52, 22.72),
            ("do", 22.74, 22.88),
            ("that", 22.90, 23.30),
        ],
    )

    output = processor.process(
        decision_input,
        stream_id="timeanchored:window-qvvs-fragment",
        chunk_index=1,
        is_last_chunk=True,
    )

    assert [segment.text for segment in output.sentence_segments] == [
        "killinging five strangers is probably more beneficial for society, so I'll do that"
    ]


def test_timeanchored_ingress_prefers_prefix_split_over_clause_break_before_preposition() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    decision_input = _build_timeanchored_decision_input(
        unit_id="window-v356-prefix",
        chunk_id="chunk-v356-prefix",
        chunk_index=1,
        words=[
            ("Oh", 16.04, 16.10),
            ("God,", 16.52, 16.94),
            ("killinging", 18.20, 18.68),
            ("five", 18.74, 19.52),
            ("strangers", 19.94, 20.24),
            ("is", 20.30, 20.44),
            ("probably", 20.46, 20.78),
            ("more", 20.82, 21.04),
            ("beneficial", 21.08, 21.62),
            ("for", 21.70, 21.82),
            ("society,", 21.84, 22.30),
            ("so", 22.36, 22.50),
            ("I'll", 22.52, 22.72),
            ("do", 22.74, 22.88),
            ("that", 22.90, 23.30),
        ],
    )

    output = processor.process(
        decision_input,
        stream_id="timeanchored:window-v356-prefix",
        chunk_index=1,
        is_last_chunk=True,
    )

    assert [segment.text for segment in output.sentence_segments] == [
        "Oh God",
        "killinging five strangers is probably more beneficial for society, so I'll do that",
    ]


def test_timeanchored_ingress_allows_small_overflow_to_avoid_breaking_preposition_object() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    decision_input = _build_timeanchored_decision_input(
        unit_id="window-v356-overflow",
        chunk_id="chunk-v356-overflow",
        chunk_index=3,
        words=[
            ("Bennet,", 30.84, 31.32),
            ("okay,", 31.38, 31.74),
            ("sure", 31.82, 32.12),
            ("a", 32.18, 32.22),
            ("trolley", 32.24, 32.70),
            ("is", 32.72, 32.86),
            ("heading", 32.88, 33.22),
            ("towards", 33.24, 33.66),
            ("five", 33.96, 34.32),
            ("people,", 34.36, 34.86),
            ("you", 34.92, 35.10),
            ("can", 35.14, 35.32),
            ("pull", 35.36, 35.68),
            ("the", 35.72, 35.84),
            ("lever", 35.88, 36.22),
            ("to", 36.24, 36.34),
            ("diver", 36.38, 36.62),
            ("the", 36.66, 36.76),
            ("track", 36.80, 36.90),
            ("One", 37.325, 37.56),
            ("person", 37.58, 37.96),
            ("said,", 37.98, 38.34),
            ("at", 38.40, 38.52),
            ("least", 38.54, 38.86),
            ("that's", 38.88, 39.22),
            ("what", 39.24, 39.46),
            ("you", 39.48, 39.62),
            ("think", 39.64, 39.90),
            ("is", 39.92, 40.06),
            ("happening,", 40.08, 40.52),
        ],
    )

    output = processor.process(
        decision_input,
        stream_id="timeanchored:window-v356-overflow",
        chunk_index=3,
        is_last_chunk=True,
    )

    assert [segment.text for segment in output.sentence_segments] == [
        "Bennet, okay, sure a trolley is heading towards five people, you can pull the lever to diver the track",
        "One person said, at least that's what you think is happening,",
    ]
