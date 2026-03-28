from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.punctuation_fact_mapper import (
    PunctuationFactMapper,
)
from app.services.timeanchored_alignment.anchor_mount.service import AnchorMountAlignmentService
from app.services.timeanchored_alignment.preparation.assembler import (
    AlignmentPreparationAssembler,
)
from app.services.timeanchored_alignment.preparation.contracts import (
    PreparedTokenUnit,
    PunctuationEvidence,
    SlowWindowTextPackage,
)
from app.services.timeanchored_alignment.contracts import (
    TimeBaseQuality,
    TimeBaseUnit,
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
        window_id="anchor-window-001",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        window_mode="steady",
        flush_reason="test",
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
        created_at=2.0,
    )


def _build_window_time_base() -> WindowTimeBasePackage:
    units = (
        TimeBaseUnit(text="你", start=0.0, end=0.1, confidence=0.95, token_type="word"),
        TimeBaseUnit(text="好", start=0.1, end=0.2, confidence=0.95, token_type="word"),
        TimeBaseUnit(text="世", start=1.0, end=1.1, confidence=0.95, token_type="word"),
        TimeBaseUnit(text="界", start=1.1, end=1.2, confidence=0.95, token_type="word"),
    )
    ready_window = _build_ready_window()
    return WindowTimeBasePackage(
        window_id=ready_window.window_id,
        language="zh",
        raw_units=units,
        word_units=units,
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.9, low_prob_ratio=0.05),
        source_chunk_ids=ready_window.source_chunk_ids,
        source_chunk_indices=ready_window.source_chunk_indices,
        chunk_bindings=ready_window.coverage.chunk_bindings,
    )


def _build_preparation_package(*, whisper_text: str):
    return AlignmentPreparationAssembler().prepare(
        ready_window=_build_ready_window(),
        window_time_base=_build_window_time_base(),
        whisper_result={
            "text": whisper_text,
            "text_clean": whisper_text,
            "text_itn_raw": whisper_text,
            "confidence": 0.9,
            "language": "zh",
            "raw_result": {"segments": [{"avg_logprob": -0.1}]},
        },
        default_language="zh",
    )


def test_anchor_mount_maps_punctuation_into_token_space_and_generates_pair_states() -> None:
    service = AnchorMountAlignmentService()
    preparation = _build_preparation_package(whisper_text="你好，“世界”！")

    result = service.align(preparation=preparation, language="zh")

    facts = result.anchor_mount_result.punctuation_facts
    assert [fact.normalized_text for fact in facts] == ["，", "“", "”", "！"]
    assert facts[1].right_token_index == 2
    assert facts[2].left_token_index == 3
    assert facts[3].left_token_index == 3
    assert result.anchor_mount_result.punctuation_pair_states[0].state == "closed"


def test_anchor_mount_only_forwards_alignment_specific_boundary_evidences() -> None:
    service = AnchorMountAlignmentService()
    preparation = _build_preparation_package(whisper_text="你好，世界！")

    result = service.align(preparation=preparation, language="zh")

    assert result.anchor_mount_result.boundary_evidences
    reasons = {evidence.reason for evidence in result.anchor_mount_result.boundary_evidences}
    assert reasons <= {"lexical_boundary", "anchor_block_close"}
    assert "punctuation_sentence_end" not in reasons
    assert result.decision_ingress.anchored_token_units
    assert all(
        token.start is not None and token.end is not None
        for token in result.decision_ingress.anchored_token_units
    )
    assert [
        (token.start, token.end)
        for token in result.decision_ingress.anchored_token_units
    ] == [
        (envelope.provisional_start, envelope.provisional_end)
        for envelope in result.anchor_mount_result.envelopes
    ]


def test_punctuation_fact_mapper_fail_closed_records_unmapped_diagnostics() -> None:
    mapper = PunctuationFactMapper()

    facts, pair_states, diagnostics = mapper.map(
        window_text=SlowWindowTextPackage(
            text="你好",
            display_text="你好",
            source_language="zh",
        ),
        token_units=(
            PreparedTokenUnit(
                unit_id="unit-0",
                token_text="你",
                normalized_text="你",
                char_start=0,
                char_end=1,
                speaker_id="speaker-a",
                turn_id="turn-a",
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
            ),
            PreparedTokenUnit(
                unit_id="unit-1",
                token_text="好",
                normalized_text="好",
                char_start=1,
                char_end=2,
                speaker_id="speaker-a",
                turn_id="turn-a",
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
            ),
        ),
        punctuation_evidences=(
            PunctuationEvidence(mark="！", source_char_index=9, attach_side="after"),
        ),
    )

    assert facts == ()
    assert pair_states == ()
    assert len(diagnostics.unmapped) == 1
    assert diagnostics.unmapped[0]["reason"] == "char_index_out_of_window_text"
