"""边界提示组装。"""

from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorMountItem,
    BoundaryHint,
    CrossChunkLock,
    PunctuationFact,
    TemporalEnvelope,
)


class BoundaryHintAssembler:
    """从标点与停顿事实生成 lock-aware 边界提示。"""

    def build(
        self,
        *,
        items: tuple[AnchorMountItem, ...],
        envelopes: tuple[TemporalEnvelope, ...],
        punctuation_facts: tuple[PunctuationFact, ...],
        cross_chunk_locks: tuple[CrossChunkLock, ...],
    ) -> tuple[BoundaryHint, ...]:
        hints: list[BoundaryHint] = []
        lock_pairs = {
            tuple(lock.slot_ids)
            for lock in cross_chunk_locks
            if len(lock.slot_ids) >= 2
        }

        for fact in punctuation_facts:
            if fact.left_slot_index is None:
                continue
            slot_id = items[fact.left_slot_index].slot_id
            envelope = envelopes[fact.left_slot_index]
            reason = "punctuation_sentence_end" if fact.punct_class == "sentence_end" else "punctuation_soft"
            hints.append(
                BoundaryHint(
                    split_after_slot_id=slot_id,
                    decision_time=float(envelope.provisional_end),
                    score=1.0 if fact.punct_class == "sentence_end" else 0.6,
                    reason=reason,
                    hard_flag=fact.punct_class == "sentence_end",
                    blocked_by_lock=False,
                )
            )

        for index in range(len(envelopes) - 1):
            left = envelopes[index]
            right = envelopes[index + 1]
            gap = max(0.0, float(right.provisional_start) - float(left.provisional_end))
            slot_pair = (items[index].slot_id, items[index + 1].slot_id)
            if gap >= 0.25:
                hints.append(
                    BoundaryHint(
                        split_after_slot_id=items[index].slot_id,
                        decision_time=(left.provisional_end + right.provisional_start) / 2.0,
                        score=min(1.0, 0.5 + gap),
                        reason="gap_pause",
                        hard_flag=gap >= 0.45,
                        blocked_by_lock=slot_pair in lock_pairs,
                    )
                )
            elif gap >= 0.12:
                hints.append(
                    BoundaryHint(
                        split_after_slot_id=items[index].slot_id,
                        decision_time=(left.provisional_end + right.provisional_start) / 2.0,
                        score=min(0.8, 0.35 + gap),
                        reason="blank_valley",
                        hard_flag=False,
                        blocked_by_lock=slot_pair in lock_pairs,
                    )
                )
            if (
                items[index].speaker_id
                and items[index + 1].speaker_id
                and items[index].speaker_id != items[index + 1].speaker_id
            ):
                hints.append(
                    BoundaryHint(
                        split_after_slot_id=items[index].slot_id,
                        decision_time=(left.provisional_end + right.provisional_start) / 2.0,
                        score=0.8,
                        reason="speaker_change",
                        hard_flag=True,
                        blocked_by_lock=slot_pair in lock_pairs,
                    )
                )
            if items[index].source_chunk_ids != items[index + 1].source_chunk_ids:
                hints.append(
                    BoundaryHint(
                        split_after_slot_id=items[index].slot_id,
                        decision_time=(left.provisional_end + right.provisional_start) / 2.0,
                        score=0.58,
                        reason="lexical_boundary",
                        hard_flag=False,
                        blocked_by_lock=slot_pair in lock_pairs,
                    )
                )
            if (
                items[index].source_hook_ids
                and items[index + 1].source_hook_ids
                and items[index].source_hook_ids != items[index + 1].source_hook_ids
            ):
                hints.append(
                    BoundaryHint(
                        split_after_slot_id=items[index].slot_id,
                        decision_time=(left.provisional_end + right.provisional_start) / 2.0,
                        score=0.52,
                        reason="anchor_block_close",
                        hard_flag=False,
                        blocked_by_lock=slot_pair in lock_pairs,
                    )
                )

        dedup: dict[tuple[str, str], BoundaryHint] = {}
        for hint in hints:
            key = (hint.split_after_slot_id, hint.reason)
            current = dedup.get(key)
            if current is None or hint.score > current.score:
                dedup[key] = hint
        return tuple(sorted(dedup.values(), key=lambda item: (item.decision_time, item.reason)))
