"""对齐层专属边界证据组装。"""

from __future__ import annotations

from app.services.timeanchored_alignment.contracts import BoundaryEvidence
from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorMountItem,
    CrossChunkLock,
    PunctuationFact,
    TemporalEnvelope,
)


class BoundaryEvidenceBuilder:
    """只生成对齐层专属的 lock-aware 边界证据。"""

    def build(
        self,
        *,
        items: tuple[AnchorMountItem, ...],
        envelopes: tuple[TemporalEnvelope, ...],
        punctuation_facts: tuple[PunctuationFact, ...],
        cross_chunk_locks: tuple[CrossChunkLock, ...],
    ) -> tuple[BoundaryEvidence, ...]:
        evidences: list[BoundaryEvidence] = []
        lock_pairs = {
            tuple(lock.unit_ids)
            for lock in cross_chunk_locks
            if len(lock.unit_ids) >= 2
        }

        for index in range(len(envelopes) - 1):
            left = envelopes[index]
            right = envelopes[index + 1]
            unit_pair = (items[index].unit_id, items[index + 1].unit_id)
            event_time = (left.provisional_end + right.provisional_start) / 2.0
            base_metadata = {
                "blocked_by_lock": unit_pair in lock_pairs,
                "punctuation_fact_count": len(punctuation_facts),
                "left_block_id": items[index].alignment_block_id,
                "right_block_id": items[index + 1].alignment_block_id,
            }
            if self._is_disjoint_chunk_provenance(
                left_chunk_ids=items[index].source_chunk_ids,
                right_chunk_ids=items[index + 1].source_chunk_ids,
            ):
                evidences.append(
                    BoundaryEvidence(
                        split_idx=index,
                        event_time=event_time,
                        left_end=float(left.provisional_end),
                        right_start=float(right.provisional_start),
                        score=0.58,
                        reason="lexical_boundary",
                        hard_flag=False,
                        metadata=base_metadata,
                    )
                )
            if (
                items[index].alignment_block_id
                and items[index + 1].alignment_block_id
                and items[index].alignment_block_id != items[index + 1].alignment_block_id
            ):
                evidences.append(
                    BoundaryEvidence(
                        split_idx=index,
                        event_time=event_time,
                        left_end=float(left.provisional_end),
                        right_start=float(right.provisional_start),
                        score=0.52,
                        reason="anchor_block_close",
                        hard_flag=False,
                        metadata=base_metadata,
                    )
                )

        dedup: dict[tuple[int, str], BoundaryEvidence] = {}
        for evidence in evidences:
            key = (int(evidence.split_idx), str(evidence.reason))
            current = dedup.get(key)
            if current is None or evidence.score > current.score:
                dedup[key] = evidence
        return tuple(sorted(dedup.values(), key=lambda item: (item.event_time, item.reason)))

    @staticmethod
    def _is_disjoint_chunk_provenance(
        *,
        left_chunk_ids: tuple[str, ...],
        right_chunk_ids: tuple[str, ...],
    ) -> bool:
        left = {str(item) for item in left_chunk_ids if str(item)}
        right = {str(item) for item in right_chunk_ids if str(item)}
        if not left or not right:
            return left != right
        return left.isdisjoint(right)
