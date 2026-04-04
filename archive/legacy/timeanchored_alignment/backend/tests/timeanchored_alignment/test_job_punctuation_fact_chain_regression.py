from __future__ import annotations

from app.pipelines.dual_pipeline.services.alignment_stage_service import AlignmentStageService
from app.services.alignment.types import PunctTrack
from app.services.punctuation.base import PuncPosition
from app.services.timeanchored_alignment.anchor_mount.service import AnchorMountAlignmentService
from app.services.timeanchored_alignment.contracts import TimeBaseQuality, TimeBaseUnit
from app.services.timeanchored_alignment.preparation.assembler import AlignmentPreparationAssembler
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
from app.services.timeanchored_alignment.window_time_base_assembler import WindowTimeBasePackage


def _build_ready_window() -> ReadySlowWindow:
    return ReadySlowWindow(
        window_id="window-e2e-punct-chain-001",
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
                text="你好世界",
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
                text="今天继续测试",
                audio_start=1.0,
                audio_end=2.0,
                source_chunk_ids=("chunk-2",),
                source_chunk_indices=(2,),
                speaker_id="speaker-a",
                turn_id="turn-b",
                language="zh",
                arrived_at=2.0,
            ),
        ),
        dialogue_shape=DialogueShapeSnapshot(
            shape="single_speaker",
            speaker_count=1,
            dominant_speaker_id="speaker-a",
            dominant_speaker_ratio=1.0,
            speaker_switch_count=0,
            speaker_switch_density=0.0,
            turn_count=2,
            avg_turn_duration_sec=1.0,
        ),
        language_profile=WindowLanguageProfile(
            primary_language="zh",
            language_mix_state="single_language",
            decision_domains=("timeanchored_alignment",),
            should_bypass_whisper=False,
        ),
        prompt_seed=PromptSeed(text="你好世界今天继续测试"),
        batch_hint=WindowBatchHint(
            duration_bucket="short",
            token_estimate=10,
            acoustic_density_hint="medium",
            queue_priority=1,
        ),
        created_at=3.0,
    )


def _build_window_time_base() -> WindowTimeBasePackage:
    ready_window = _build_ready_window()
    units = (
        TimeBaseUnit(text="你", start=0.0, end=0.1, confidence=0.95, token_type="word"),
        TimeBaseUnit(text="好", start=0.1, end=0.2, confidence=0.95, token_type="word"),
        TimeBaseUnit(text="世", start=0.2, end=0.3, confidence=0.95, token_type="word"),
        TimeBaseUnit(text="界", start=0.3, end=0.4, confidence=0.95, token_type="word"),
        TimeBaseUnit(text="今", start=1.0, end=1.1, confidence=0.95, token_type="word"),
        TimeBaseUnit(text="天", start=1.1, end=1.2, confidence=0.95, token_type="word"),
    )
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


def _build_whisper_result(text: str) -> dict:
    return {
        "text": text,
        "text_clean": text,
        "text_itn_raw": text,
        "confidence": 0.9,
        "language": "zh",
        "raw_result": {"segments": [{"avg_logprob": -0.1}]},
    }


def test_job_has_punctuation_points_then_preparation_should_not_be_zero() -> None:
    punct_track = PunctTrack(
        clean_text_ref="你好世界今天继续测试",
        positions=[
            PuncPosition(char_index=3, punctuation="。", confidence=0.99),
            PuncPosition(char_index=9, punctuation="！", confidence=0.99),
        ],
        source="fast",
    )
    preparation = AlignmentPreparationAssembler().prepare(
        ready_window=_build_ready_window(),
        window_time_base=_build_window_time_base(),
        whisper_result=_build_whisper_result("你好世界今天继续测试"),
        default_language="zh",
        external_punct_track=punct_track,
    )
    stage_result = AnchorMountAlignmentService().align(preparation=preparation, language="zh")

    metrics = AlignmentStageService._build_punctuation_chain_health_metrics(
        chosen_source="slow",
        punct_track=punct_track,
        preparation_punctuation_count=len(preparation.slow_text.punctuation_evidences),
        punctuation_fact_count=len(stage_result.decision_ingress.punctuation_facts),
    )
    boundary_reasons = {
        str(evidence.reason)
        for evidence in stage_result.decision_ingress.boundary_evidences
    }

    assert len(punct_track.positions) > 0
    assert metrics["preparation_punctuation_count"] > 0
    assert metrics["punctuation_fact_count"] > 0
    assert metrics["punctuation_chain_broken_flag"] == 0
    assert boundary_reasons <= {"lexical_boundary", "anchor_block_close"}
    assert "punctuation_sentence_end" not in boundary_reasons
