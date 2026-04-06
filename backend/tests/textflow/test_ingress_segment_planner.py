from __future__ import annotations

from collections import Counter

from app.services.alignment.types import AnnotatedWord, DecisionLayerInput
from app.services.punctuation.final_splitter import FinalSplitter
from app.services.textflow.contracts import SegmentationIngressContext
from app.services.textflow.decision_layer import SegmentationProcessor
from app.services.textflow.ingress_segment_planner import IngressSegmentPlanner, _BoundaryFeature
from app.services.timeanchored_alignment.contracts import BoundaryEvidence


def _build_timeanchored_decision_input(
    *,
    unit_id: str,
    chunk_id: str,
    chunk_index: int,
    words: list[tuple[str, float, float]],
    canonical_candidate_boundaries: tuple[BoundaryEvidence, ...] = (),
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
        canonical_candidate_boundaries=canonical_candidate_boundaries,
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
        "One person said, at least that's what you think is happening",
    ]


def test_timeanchored_ingress_does_not_split_pause_only_english_clause_tail() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    decision_input = _build_timeanchored_decision_input(
        unit_id="window-tr-tail-hold",
        chunk_id="chunk-tr-tail-hold",
        chunk_index=0,
        words=[
            ("There's", 6.856, 7.096),
            ("plenty", 7.216, 7.276),
            ("of", 7.516, 7.576),
            ("time", 7.696, 7.756),
            ("left", 7.996, 8.056),
            ("to", 8.176, 8.236),
            ("go", 8.356, 8.416),
            ("and", 8.536, 8.596),
            ("hunt", 8.716, 8.776),
            ("evil", 9.016, 9.076),
            ("down", 9.316, 9.976),
        ],
    )

    output = processor.process(
        decision_input,
        stream_id="timeanchored:window-tr-tail-hold",
        chunk_index=0,
        is_last_chunk=True,
    )

    assert [segment.text for segment in output.sentence_segments] == [
        "There's plenty of time left to go and hunt evil down"
    ]


def test_timeanchored_ingress_does_not_split_before_right_idea_clause() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    decision_input = _build_timeanchored_decision_input(
        unit_id="window-tr-right-idea",
        chunk_id="chunk-tr-right-idea",
        chunk_index=2,
        words=[
            ("Probably", 21.348, 21.708),
            ("Monica,", 21.888, 22.428),
            ("she", 22.548, 22.608),
            ("has", 22.848, 22.908),
            ("the", 23.088, 23.148),
            ("right", 23.268, 23.328),
            ("idea", 23.448, 23.508),
            ("in", 23.928, 23.988),
            ("a", 24.108, 24.168),
            ("lot", 24.228, 24.288),
            ("of", 24.408, 24.468),
            ("ways", 24.588, 25.428),
        ],
        canonical_candidate_boundaries=(
            BoundaryEvidence(
                split_idx=4,
                event_time=23.208,
                left_end=23.148,
                right_start=23.268,
                reason="gap_pause",
                score=0.24,
                hard_flag=False,
                metadata={"gap_sec": 0.12},
            ),
            BoundaryEvidence(
                split_idx=5,
                event_time=23.388,
                left_end=23.328,
                right_start=23.448,
                reason="gap_pause",
                score=0.24,
                hard_flag=False,
                metadata={"gap_sec": 0.12},
            ),
        ),
    )

    output = processor.process(
        decision_input,
        stream_id="timeanchored:window-tr-right-idea",
        chunk_index=2,
        is_last_chunk=True,
    )

    assert [segment.text for segment in output.sentence_segments] == [
        "Probably Monica, she has the right idea in a lot of ways"
    ]


def test_timeanchored_ingress_must_split_defers_pause_only_clause_break_until_late_boundary() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    decision_input = _build_timeanchored_decision_input(
        unit_id="window-tr-must-split-late-hold",
        chunk_id="chunk-tr-must-split-late-hold",
        chunk_index=4,
        words=[
            ("A", 7.468, 7.528),
            ("trolley", 7.768, 8.068),
            ("is", 8.128, 8.188),
            ("heading", 8.248, 8.308),
            ("towards", 8.488, 8.548),
            ("your", 8.788, 8.848),
            ("best", 8.968, 9.028),
            ("friend,", 9.268, 9.568),
            ("you", 9.688, 9.748),
            ("can", 9.868, 9.928),
            ("pull", 10.048, 10.108),
            ("the", 10.228, 10.288),
            ("lever", 10.408, 10.468),
            ("to", 10.708, 10.768),
            ("divert", 10.828, 11.068),
            ("it", 11.068, 11.128),
            ("to", 11.188, 11.248),
            ("the", 11.308, 11.368),
            ("other", 11.428, 11.488),
            ("track,", 11.668, 11.908),
            ("killing", 12.028, 12.088),
            ("five", 12.328, 12.388),
            ("strangers", 12.688, 13.048),
            ("instead,", 13.228, 13.588),
            ("What", 13.768, 13.828),
            ("do", 14.008, 14.068),
            ("you", 14.128, 14.188),
            ("do", 14.308, 14.968),
        ],
    )

    output = processor._run_segmentation_core(
        decision_input,
        stream_id="timeanchored:window-tr-must-split-late-hold",
        chunk_index=4,
        is_last_chunk=True,
    )

    assert output.segmentation_result is not None
    assert output.segmentation_result.segments[0].token_end > 6


def test_timeanchored_ingress_must_split_prefers_late_safe_boundary_over_early_pause_peak() -> None:
    planner = IngressSegmentPlanner()
    words = [
        AnnotatedWord(word="Oh", start=16.04, end=16.10, confidence=0.9, confidence_source="aligned"),
        AnnotatedWord(word="God,", start=16.52, end=16.94, confidence=0.9, confidence_source="aligned"),
        AnnotatedWord(word="killinging", start=18.20, end=18.50, confidence=0.9, confidence_source="aligned"),
        AnnotatedWord(word="five", start=18.68, end=18.74, confidence=0.9, confidence_source="aligned"),
        AnnotatedWord(word="strangers", start=19.04, end=19.40, confidence=0.9, confidence_source="aligned"),
        AnnotatedWord(word="is", start=19.52, end=19.58, confidence=0.9, confidence_source="aligned"),
        AnnotatedWord(word="probably", start=19.76, end=19.82, confidence=0.9, confidence_source="aligned"),
        AnnotatedWord(word="more", start=20.24, end=20.30, confidence=0.9, confidence_source="aligned"),
        AnnotatedWord(word="beneficial", start=20.48, end=20.54, confidence=0.9, confidence_source="aligned"),
        AnnotatedWord(word="for", start=21.08, end=21.14, confidence=0.9, confidence_source="aligned"),
        AnnotatedWord(word="society,", start=21.32, end=21.92, confidence=0.9, confidence_source="aligned"),
        AnnotatedWord(word="so", start=21.98, end=22.04, confidence=0.9, confidence_source="aligned"),
        AnnotatedWord(word="I'll", start=22.16, end=22.40, confidence=0.9, confidence_source="aligned"),
        AnnotatedWord(word="do", start=22.46, end=22.60, confidence=0.9, confidence_source="aligned"),
        AnnotatedWord(word="that", start=22.66, end=23.30, confidence=0.9, confidence_source="aligned"),
    ]
    words_for_split = SegmentationProcessor._build_words_for_split(words)
    feature_map = {
        6: _BoundaryFeature(
            split_idx=6,
            event_time=20.03,
            left_end=19.82,
            right_start=20.24,
            reasons=("gap_pause", "gap_pause"),
            scores=(0.7, 0.7),
        ),
        10: _BoundaryFeature(
            split_idx=10,
            event_time=21.95,
            left_end=21.92,
            right_start=21.98,
            reasons=("punctuation_soft",),
            scores=(0.6,),
        ),
        11: _BoundaryFeature(
            split_idx=11,
            event_time=22.10,
            left_end=22.04,
            right_start=22.16,
            reasons=("gap_pause", "blank_valley"),
            scores=(0.5, 0.4),
        ),
    }

    candidate, _, _, _ = planner._select_best_segment_end(
        processor=object(),
        words_for_split=words_for_split,
        segment_start_idx=0,
        feature_map=feature_map,
        accept_threshold=planner._DEFAULT_ACCEPT_THRESHOLD,
        max_segment_sec=planner._DEFAULT_MAX_SEGMENT_SEC,
        max_overflow_sec=planner._DEFAULT_MAX_OVERFLOW_SEC,
        language_is_cjk=False,
        rejection_stats=Counter(),
    )

    assert candidate is not None
    assert candidate.split_idx == 10


def test_timeanchored_ingress_does_not_cut_on_speaker_change_without_real_break() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    decision_input = _build_timeanchored_decision_input(
        unit_id="window-tr-weak-speaker-change",
        chunk_id="chunk-tr-weak-speaker-change",
        chunk_index=5,
        words=[
            ("Tell", 0.00, 0.20),
            ("me", 0.22, 0.34),
            ("your", 0.36, 0.54),
            ("answer", 0.56, 0.88),
            ("right", 0.90, 1.12),
            ("now", 1.14, 1.34),
            ("please", 1.36, 1.62),
        ],
        canonical_candidate_boundaries=(
            BoundaryEvidence(
                split_idx=2,
                event_time=0.55,
                left_end=0.54,
                right_start=0.56,
                reason="speaker_change",
                score=0.95,
                hard_flag=True,
                metadata={},
            ),
        ),
    )

    output = processor.process(
        decision_input,
        stream_id="timeanchored:window-tr-weak-speaker-change",
        chunk_index=5,
        is_last_chunk=True,
    )

    assert [segment.text for segment in output.sentence_segments] == [
        "Tell me your answer right now please"
    ]


def test_timeanchored_ingress_must_split_does_not_let_synthetic_pause_peak_beat_late_safe_boundary() -> None:
    planner = IngressSegmentPlanner()
    words = [
        AnnotatedWord(word="Tell", start=0.00, end=0.18, confidence=0.9, confidence_source="aligned"),
        AnnotatedWord(word="me", start=0.20, end=0.34, confidence=0.9, confidence_source="aligned"),
        AnnotatedWord(word="What's", start=0.36, end=0.46, confidence=0.9, confidence_source="aligned"),
        AnnotatedWord(word="your", start=0.90, end=1.00, confidence=0.9, confidence_source="aligned"),
        AnnotatedWord(word="answer", start=1.02, end=1.42, confidence=0.9, confidence_source="aligned"),
        AnnotatedWord(word="right", start=1.46, end=1.62, confidence=0.9, confidence_source="aligned"),
        AnnotatedWord(word="now,", start=1.66, end=2.08, confidence=0.9, confidence_source="aligned"),
        AnnotatedWord(word="please", start=2.12, end=2.78, confidence=0.9, confidence_source="aligned"),
    ]
    words_for_split = SegmentationProcessor._build_words_for_split(words)
    feature_map = {
        2: _BoundaryFeature(
            split_idx=2,
            event_time=0.68,
            left_end=0.46,
            right_start=0.90,
            reasons=("gap_pause", "speaker_change"),
            scores=(0.92, 0.91),
            metadata=(
                {
                    "left_mount_status": "synthetic",
                    "right_mount_status": "local_interpolation",
                },
            ),
        ),
        6: _BoundaryFeature(
            split_idx=6,
            event_time=2.08,
            left_end=2.08,
            right_start=2.12,
            reasons=("punctuation_soft",),
            scores=(0.65,),
            metadata=({},),
        ),
    }

    candidate, _, _, _ = planner._select_best_segment_end(
        processor=object(),
        words_for_split=words_for_split,
        segment_start_idx=0,
        feature_map=feature_map,
        accept_threshold=planner._DEFAULT_ACCEPT_THRESHOLD,
        max_segment_sec=2.4,
        max_overflow_sec=0.45,
        language_is_cjk=False,
        rejection_stats=Counter(),
    )

    assert candidate is not None
    assert candidate.split_idx == 6
