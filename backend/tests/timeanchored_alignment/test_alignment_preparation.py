from __future__ import annotations

from dataclasses import replace

from app.services.timeanchored_alignment.contracts import (
    TimeBasePackage,
    TimeBaseQuality,
    TimeBaseUnit,
)
from app.services.timeanchored_alignment.preparation.assembler import (
    AlignmentPreparationAssembler,
)
from app.services.alignment.types import PunctTrack
from app.services.punctuation.base import PuncPosition
from app.services.timeanchored_alignment.preparation.display_projection_builder import (
    DisplayProjectionBuilder,
)
from app.services.timeanchored_alignment.preparation.safe_pre_normalizer import (
    SafePreNormalizer,
)
from app.services.timeanchored_alignment.preparation.structure_protector import (
    StructureProtector,
)
from app.services.timeanchored_alignment.slow_window.contracts import (
    DialogueShapeSnapshot,
    PromptSeed,
    ReadySlowWindow,
    WindowBatchHint,
    WindowChunkBinding,
    WindowCoverage,
    WindowLanguageProfile,
    WindowSourceUnit,
)
from app.services.timeanchored_alignment.window_time_base_assembler import (
    WindowTimeBasePackage,
)


def _build_ready_window(*, text_a: str = "你好", text_b: str = "世界") -> ReadySlowWindow:
    return ReadySlowWindow(
        window_id="window-001",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        window_mode="steady",
        flush_reason="test_flush",
        audio_segments=((0.0, 1.0), (1.0, 2.0)),
        coverage=WindowCoverage(
            core_segments=((0.0, 2.0),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-1",
                    chunk_index=1,
                    chunk_start=0.0,
                    chunk_end=1.0,
                    overlap_ratio=1.0,
                    role="owner",
                    is_owner=True,
                ),
                WindowChunkBinding(
                    chunk_id="chunk-2",
                    chunk_index=2,
                    chunk_start=1.0,
                    chunk_end=2.0,
                    overlap_ratio=1.0,
                    role="core",
                    is_owner=False,
                ),
            ),
        ),
        source_semantic_chunk_ids=("sem-1", "sem-2"),
        source_chunk_ids=("chunk-1", "chunk-2"),
        source_chunk_indices=(1, 2),
        source_units=(
            WindowSourceUnit(
                unit_id="unit-1",
                semantic_chunk_id="sem-1",
                text=text_a,
                audio_start=0.0,
                audio_end=1.0,
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
                speaker_id="speaker-a",
                turn_id="turn-a",
                language="zh",
                arrived_at=1.0,
            ),
            WindowSourceUnit(
                unit_id="unit-2",
                semantic_chunk_id="sem-2",
                text=text_b,
                audio_start=1.0,
                audio_end=2.0,
                source_chunk_ids=("chunk-2",),
                source_chunk_indices=(2,),
                speaker_id="speaker-b",
                turn_id="turn-b",
                language="zh",
                arrived_at=2.0,
            ),
        ),
        dialogue_shape=DialogueShapeSnapshot(
            shape="two_party_stable",
            speaker_count=2,
            dominant_speaker_id="speaker-a",
            dominant_speaker_ratio=0.5,
            speaker_switch_count=1,
            speaker_switch_density=0.5,
            turn_count=2,
            avg_turn_duration_sec=1.0,
        ),
        language_profile=WindowLanguageProfile(
            primary_language="zh",
            language_mix_state="single_language",
            decision_domains=("timeanchored_alignment",),
            should_bypass_whisper=False,
        ),
        prompt_seed=PromptSeed(text="你好世界"),
        batch_hint=WindowBatchHint(
            duration_bucket="short",
            token_estimate=4,
            acoustic_density_hint="medium",
            queue_priority=1,
        ),
        created_at=3.0,
    )


def _build_english_ready_window(
    *,
    text_a: str = "Wouldn't it make sense.",
    text_b: str = "It's not like I even wanted to graduate anyways.",
) -> ReadySlowWindow:
    return ReadySlowWindow(
        window_id="window-en-001",
        owner_chunk_id="chunk-en-1",
        owner_chunk_index=1,
        window_mode="steady",
        flush_reason="test_flush",
        audio_segments=((0.0, 1.0), (1.0, 2.0)),
        coverage=WindowCoverage(
            core_segments=((0.0, 2.0),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-en-1",
                    chunk_index=1,
                    chunk_start=0.0,
                    chunk_end=1.0,
                    overlap_ratio=1.0,
                    role="owner",
                    is_owner=True,
                ),
                WindowChunkBinding(
                    chunk_id="chunk-en-2",
                    chunk_index=2,
                    chunk_start=1.0,
                    chunk_end=2.0,
                    overlap_ratio=1.0,
                    role="core",
                    is_owner=False,
                ),
            ),
        ),
        source_semantic_chunk_ids=("sem-en-1", "sem-en-2"),
        source_chunk_ids=("chunk-en-1", "chunk-en-2"),
        source_chunk_indices=(1, 2),
        source_units=(
            WindowSourceUnit(
                unit_id="unit-en-1",
                semantic_chunk_id="sem-en-1",
                text=text_a,
                audio_start=0.0,
                audio_end=1.0,
                source_chunk_ids=("chunk-en-1",),
                source_chunk_indices=(1,),
                speaker_id="speaker-en",
                turn_id="turn-en-1",
                language="en",
                arrived_at=1.0,
            ),
            WindowSourceUnit(
                unit_id="unit-en-2",
                semantic_chunk_id="sem-en-2",
                text=text_b,
                audio_start=1.0,
                audio_end=2.0,
                source_chunk_ids=("chunk-en-2",),
                source_chunk_indices=(2,),
                speaker_id="speaker-en",
                turn_id="turn-en-2",
                language="en",
                arrived_at=2.0,
            ),
        ),
        dialogue_shape=DialogueShapeSnapshot(
            shape="single_speaker",
            speaker_count=1,
            dominant_speaker_id="speaker-en",
            dominant_speaker_ratio=1.0,
            speaker_switch_count=0,
            speaker_switch_density=0.0,
            turn_count=2,
            avg_turn_duration_sec=1.0,
        ),
        language_profile=WindowLanguageProfile(
            primary_language="en",
            language_mix_state="single_language",
            decision_domains=("timeanchored_alignment",),
            should_bypass_whisper=False,
        ),
        prompt_seed=PromptSeed(text=f"{text_a} {text_b}".strip()),
        batch_hint=WindowBatchHint(
            duration_bucket="short",
            token_estimate=16,
            acoustic_density_hint="medium",
            queue_priority=1,
        ),
        created_at=3.0,
    )


def _build_window_time_base() -> WindowTimeBasePackage:
    units = (
        TimeBaseUnit(text="你", start=0.0, end=0.1, confidence=0.9, token_type="word"),
        TimeBaseUnit(text="好", start=0.1, end=0.2, confidence=0.9, token_type="word"),
        TimeBaseUnit(text="世", start=1.0, end=1.1, confidence=0.95, token_type="word"),
        TimeBaseUnit(text="界", start=1.1, end=1.2, confidence=0.95, token_type="word"),
    )
    ready_window = _build_ready_window()
    return WindowTimeBasePackage(
        window_id=ready_window.window_id,
        language="zh",
        raw_units=units,
        word_units=units,
        quality=TimeBaseQuality(
            blank_ratio=0.1,
            avg_max_prob=0.9,
            low_prob_ratio=0.05,
        ),
        source_chunk_ids=ready_window.source_chunk_ids,
        source_chunk_indices=ready_window.source_chunk_indices,
        chunk_bindings=ready_window.coverage.chunk_bindings,
    )


def _build_whisper_result(text: str) -> dict:
    return {
        "text": text,
        "text_clean": text,
        "text_itn_raw": text,
        "confidence": 0.9,
        "language": "zh",
        "raw_result": {"segments": [{"avg_logprob": -0.1}]},
    }


def _build_whisper_result_with_word_timestamps() -> dict:
    return {
        "text": "补刀A补刀B",
        "text_clean": "补刀A补刀B",
        "text_itn_raw": "补刀A补刀B",
        "confidence": 0.9,
        "language": "zh",
        "raw_result": {
            "segments": [
                {
                    "start": 0.0,
                    "end": 0.5,
                    "text": "补刀A",
                    "avg_logprob": -0.1,
                    "words": [
                        {
                            "word": "补刀A",
                            "start": 0.0,
                            "end": 0.5,
                            "probability": 0.9,
                        }
                    ],
                },
                {
                    "start": 0.5,
                    "end": 1.0,
                    "text": "补刀B",
                    "avg_logprob": -0.1,
                    "words": [
                        {
                            "word": "补刀B",
                            "start": 0.5,
                            "end": 1.0,
                            "probability": 0.9,
                        }
                    ],
                },
            ]
        },
    }


class _RecordingSafePreNormalizer(SafePreNormalizer):
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def normalize(self, *, text: str, language: str):
        self._calls.append("safe")
        return super().normalize(text=text, language=language)


class _RecordingStructureProtector(StructureProtector):
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def protect(self, *, text: str):
        self._calls.append("protect")
        return super().protect(text=text)


class _RecordingDisplayProjectionBuilder(DisplayProjectionBuilder):
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def build(self, *, source_text: str, protected_units, source_language: str):
        self._calls.append("display")
        return super().build(
            source_text=source_text,
            protected_units=protected_units,
            source_language=source_language,
        )


def test_alignment_preparation_runs_safe_structure_display_in_order() -> None:
    calls: list[str] = []
    assembler = AlignmentPreparationAssembler(
        safe_pre_normalizer=_RecordingSafePreNormalizer(calls),
        structure_protector=_RecordingStructureProtector(calls),
        display_projection_builder=_RecordingDisplayProjectionBuilder(calls),
    )

    assembler.prepare(
        ready_window=_build_ready_window(),
        window_time_base=_build_window_time_base(),
        whisper_result=_build_whisper_result("你好，世界！"),
        default_language="zh",
    )

    assert calls == ["safe", "protect", "display"]


def test_alignment_preparation_slots_carry_speaker_turn_and_source_chunk_fields() -> None:
    package = AlignmentPreparationAssembler().prepare(
        ready_window=_build_ready_window(),
        window_time_base=_build_window_time_base(),
        whisper_result=_build_whisper_result("你好世界"),
        default_language="zh",
    )

    assert package.slow_text.slots
    assert package.slow_text.slots[0].speaker_id == "speaker-a"
    assert package.slow_text.slots[0].turn_id == "turn-a"
    assert package.slow_text.slots[0].source_chunk_ids == ("chunk-1",)
    assert package.slow_text.slots[0].source_chunk_indices == (1,)
    assert package.slow_text.slots[-1].speaker_id == "speaker-b"
    assert package.slow_text.slots[-1].turn_id == "turn-b"
    assert package.slow_text.slots[-1].source_chunk_ids == ("chunk-2",)
    assert package.slow_text.slots[-1].source_chunk_indices == (2,)


def test_alignment_preparation_hooks_only_come_from_window_time_base_package() -> None:
    window_time_base = _build_window_time_base()
    package = AlignmentPreparationAssembler().prepare(
        ready_window=_build_ready_window(text_a="完全", text_b="不同"),
        window_time_base=window_time_base,
        whisper_result=_build_whisper_result("完全不同的慢流文本"),
        default_language="zh",
    )

    assert tuple(hook.hook_text for hook in package.fast_hooks) == tuple(
        unit.text for unit in window_time_base.word_units
    )


def test_alignment_preparation_display_text_stays_lexical_and_punctuation_only_becomes_evidence() -> None:
    package = AlignmentPreparationAssembler().prepare(
        ready_window=_build_ready_window(),
        window_time_base=_build_window_time_base(),
        whisper_result=_build_whisper_result("你好，世界！"),
        default_language="zh",
    )

    assert package.slow_text.window_text.text == "你好世界"
    assert package.slow_text.window_text.display_text == "你好世界"
    assert all(mark not in package.slow_text.window_text.display_text for mark in ("，", "！"))
    assert [item.mark for item in package.slow_text.punctuation_evidences] == ["，", "！"]
    assert package.slow_text.punctuation_evidences[0].source_char_index == 1
    assert package.slow_text.punctuation_evidences[1].source_char_index == 3


def test_alignment_preparation_preserves_slow_timestamps_for_edge_selector_fallback() -> None:
    package = AlignmentPreparationAssembler().prepare(
        ready_window=_build_ready_window(text_a="补刀A", text_b="补刀B"),
        window_time_base=_build_window_time_base(),
        whisper_result=_build_whisper_result_with_word_timestamps(),
        default_language="zh",
    )

    assert package.compat.text_truth.units
    assert any(unit.start is not None and unit.end is not None for unit in package.compat.text_truth.units)


def test_alignment_preparation_keeps_slot_boundaries_stable_when_projection_preserves_contractions() -> None:
    whisper_text = "Wouldn't it make sense. It's not like I even wanted to graduate anyways."
    package = AlignmentPreparationAssembler().prepare(
        ready_window=_build_english_ready_window(),
        window_time_base=_build_window_time_base(),
        whisper_result=_build_whisper_result(whisper_text),
        default_language="en",
    )

    assert package.slow_text.window_text.text == (
        "Wouldn't it make sense It's not like I even wanted to graduate anyways"
    )
    assert tuple(slot.text for slot in package.slow_text.slots) == (
        "Wouldn't it make sense",
        "It's not like I even wanted to graduate anyways",
    )


def test_alignment_preparation_splits_slots_inside_same_source_unit_when_punctuation_exists() -> None:
    text_a = (
        "Then tomorrow we can celebrate her birthday, and maybe even get her a lava lamp. "
        "Who is my favorite DDLC character?"
    )
    text_b = "Probably Monica she has the right idea in a lot of ways,"
    whisper_text = f"{text_a} {text_b}".strip()
    package = AlignmentPreparationAssembler().prepare(
        ready_window=_build_english_ready_window(text_a=text_a, text_b=text_b),
        window_time_base=_build_window_time_base(),
        whisper_result=_build_whisper_result(whisper_text),
        default_language="en",
    )

    assert len(package.slow_text.slots) > 2
    assert sum(
        1 for slot in package.slow_text.slots if slot.source_chunk_indices == (1,)
    ) >= 2


def test_alignment_preparation_refines_undersegmented_slots_with_pronunciation_hints() -> None:
    text_a = (
        "Then tomorrow we can celebrate her birthday and maybe even get her a laval lamp "
        "Who is my favorite DDLC character?"
    )
    text_b = "Probably Monica, she has the right idea in a lot of ways."
    whisper_text = f"{text_a} {text_b}".strip()
    package = AlignmentPreparationAssembler().prepare(
        ready_window=_build_english_ready_window(text_a=text_a, text_b=text_b),
        window_time_base=_build_window_time_base(),
        whisper_result=_build_whisper_result(whisper_text),
        default_language="en",
    )

    assert len(package.slow_text.slots) > 4


def test_assembler_accepts_external_punct_track_and_merges_evidence() -> None:
    punct_track = PunctTrack(
        clean_text_ref="你好世界",
        positions=[PuncPosition(char_index=3, punctuation="。", confidence=0.99)],
        source="fast",
    )
    package = AlignmentPreparationAssembler().prepare(
        ready_window=_build_ready_window(),
        window_time_base=_build_window_time_base(),
        whisper_result=_build_whisper_result("你好世界"),
        default_language="zh",
        external_punct_track=punct_track,
    )

    assert len(package.slow_text.punctuation_evidences) > 0
    assert any(item.mark == "。" for item in package.slow_text.punctuation_evidences)
