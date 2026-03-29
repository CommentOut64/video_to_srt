"""Decision ingress 收口。"""

from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorMountInputView,
    AnchoredTokenUnit,
    AnchorMountResult,
    DecisionIngressPackage,
)


class DecisionIngressAssembler:
    """把 AnchorMount 真相收口成 DecisionIngressPackage。"""

    def build(
        self,
        *,
        input_view: AnchorMountInputView,
        result: AnchorMountResult,
    ) -> DecisionIngressPackage:
        anchored_token_units = tuple(
            AnchoredTokenUnit(
                unit_id=item.unit_id,
                token_text=item.token_text,
                normalized_text=item.normalized_text,
                start=envelope.provisional_start,
                end=envelope.provisional_end,
                left_bound=envelope.left_bound,
                right_bound=envelope.right_bound,
                speaker_id=item.speaker_id,
                turn_id=item.turn_id,
                mount_status=item.mount_status,
                anchor_kind=item.anchor_kind,
                source_chunk_ids=item.source_chunk_ids,
                source_chunk_indices=item.source_chunk_indices,
                source_hook_ids=item.source_hook_ids,
                match_confidence=item.match_confidence,
                cross_chunk_lock_ids=item.cross_chunk_lock_ids,
            )
            for item, envelope in zip(result.items, result.envelopes)
        )
        return DecisionIngressPackage(
            window_id=input_view.window_id,
            owner_chunk_id=input_view.owner_chunk_id,
            owner_chunk_index=input_view.owner_chunk_index,
            source_chunk_ids=input_view.source_chunk_ids,
            source_chunk_indices=input_view.source_chunk_indices,
            language=input_view.language,
            policy_snapshot=input_view.policy_snapshot,
            anchored_token_units=anchored_token_units,
            punctuation_facts=result.punctuation_facts,
            punctuation_pair_states=result.punctuation_pair_states,
            boundary_evidences=result.boundary_evidences,
            cross_chunk_locks=result.cross_chunk_locks,
            coverage=input_view.coverage,
            quality_metrics=dict(result.metrics),
            timeline_validity=result.timeline_validity,
            should_fallback=result.should_fallback,
        )
