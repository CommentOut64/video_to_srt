from __future__ import annotations

from app.services.textflow.decision_ingress_adapter import DecisionIngressAdapter
from app.services.timeanchored_alignment.anchor_mount.contracts import (
    BoundaryHint,
    CrossChunkLock,
    DecisionIngressPackage,
    DecisionToken,
    PunctuationFact,
    PunctuationPairState,
)
from app.services.timeanchored_alignment.slow_window.contracts import (
    WindowChunkBinding,
    WindowCoverage,
)


def _build_package() -> DecisionIngressPackage:
    return DecisionIngressPackage(
        window_id="window-11",
        owner_chunk_id="chunk-11",
        owner_chunk_index=11,
        source_chunk_ids=("chunk-11", "chunk-12"),
        source_chunk_indices=(11, 12),
        language="zh",
        policy_snapshot=None,
        tokens=(
            DecisionToken(
                token_id="token-1",
                slot_index=0,
                text_core="hello",
                display_text="hello",
                normalized_text="hello",
                start=0.0,
                end=0.5,
                left_bound=0.0,
                right_bound=0.55,
                speaker_id="spk-1",
                turn_id="turn-1",
                source_chunk_ids=("chunk-11",),
                source_chunk_indices=(11,),
                source_hook_ids=("hook-1",),
                metadata={"slot_id": "slot-1", "match_confidence": 0.91},
            ),
            DecisionToken(
                token_id="token-2",
                slot_index=1,
                text_core="world",
                display_text="world",
                normalized_text="world",
                start=0.55,
                end=1.0,
                left_bound=0.5,
                right_bound=1.0,
                speaker_id="spk-1",
                turn_id="turn-1",
                source_chunk_ids=("chunk-12",),
                source_chunk_indices=(12,),
                source_hook_ids=("hook-2",),
                metadata={"slot_id": "slot-2", "match_confidence": 0.93},
            ),
        ),
        punctuation_facts=(
            PunctuationFact(
                fact_id="fact-1",
                left_slot_index=1,
                right_slot_index=None,
                attach_mode="trailing",
                normalized_text=".",
                punct_class="sentence_end",
                confidence=0.87,
                source="aligned",
                group_id="grp-1",
                boundary_weight=0.8,
                render_default=True,
                metadata={},
            ),
        ),
        punctuation_pair_states=(
            PunctuationPairState(
                group_id="grp-quote",
                pair_kind="quote",
                open_fact_id="open-1",
                close_fact_id=None,
                state="open",
                metadata={},
            ),
        ),
        boundary_hints=(
            BoundaryHint(
                split_after_slot_id="slot-1",
                decision_time=0.52,
                score=0.9,
                reason="pause_long",
                hard_flag=False,
                blocked_by_lock=False,
            ),
        ),
        cross_chunk_locks=(
            CrossChunkLock(
                lock_id="lock-1",
                slot_ids=("slot-1", "slot-2"),
                hook_ids=("hook-1",),
                reason="quote_scope",
                source_chunk_ids=("chunk-11", "chunk-12"),
                source_chunk_indices=(11, 12),
            ),
        ),
        coverage=WindowCoverage(
            core_segments=((0.0, 1.0),),
            left_guard_sec=0.1,
            right_guard_sec=0.1,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-11",
                    chunk_index=11,
                    chunk_start=0.0,
                    chunk_end=0.7,
                    overlap_ratio=0.7,
                    role="owner",
                    is_owner=True,
                ),
                WindowChunkBinding(
                    chunk_id="chunk-12",
                    chunk_index=12,
                    chunk_start=0.55,
                    chunk_end=1.2,
                    overlap_ratio=0.45,
                    role="core",
                    is_owner=False,
                ),
            ),
        ),
        quality_metrics={"alignment_score": 0.92, "soft_anchor_ratio": 0.88},
        should_fallback=True,
    )


def test_decision_ingress_adapter_keeps_compat_fields_but_cleans_projection_boundary() -> None:
    result = DecisionIngressAdapter().build(package=_build_package())

    ingress_context = result.ingress_context
    assert isinstance(ingress_context.source_chunk_ids, tuple)
    assert ingress_context.source_chunk_ids == ("chunk-11", "chunk-12")
    assert ingress_context.projection_chunk_ids == ()

    metadata = ingress_context.metadata
    assert metadata["source_chunk_indices"] == [11, 12]
    assert metadata["quality_metrics"] == {"alignment_score": 0.92, "soft_anchor_ratio": 0.88}
    assert metadata["should_fallback"] is True
    assert metadata["cross_chunk_locks"][0]["lock_id"] == "lock-1"
    assert metadata["decision_ingress_version"] == "window_first_compat_v1"
    assert metadata["fallback_projection_mode"] == "layer_internal_compat"

    decision_input = result.decision_input
    assert decision_input.allow_fast_draft_fallback is True
    assert decision_input.fallback_clean_text_ref
    assert len(decision_input.fallback_punctuation_positions) == 1

    assert result.compat_report["fallback_projection_mode"] == "layer_internal_compat"
