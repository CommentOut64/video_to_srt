"""跨 chunk 锁构建。"""

from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorMountItem,
    CrossChunkLock,
    TemporalEnvelope,
)


class CrossChunkLockBuilder:
    """构建保护 lock，避免下游在稳定跨 chunk 词组中切断。"""

    def build(
        self,
        *,
        items: tuple[AnchorMountItem, ...],
        envelopes: tuple[TemporalEnvelope, ...],
    ) -> tuple[CrossChunkLock, ...]:
        locks: list[CrossChunkLock] = []
        for index in range(1, len(items)):
            left = items[index - 1]
            right = items[index]
            if left.source_chunk_ids != right.source_chunk_ids and (
                envelopes[index - 1].envelope_kind in {"anchored", "merged"}
                and envelopes[index].envelope_kind in {"anchored", "merged"}
            ):
                locks.append(
                    CrossChunkLock(
                        lock_id=f"lock-{index - 1}-{index}",
                        slot_ids=(left.slot_id, right.slot_id),
                        hook_ids=tuple(left.source_hook_ids + right.source_hook_ids),
                        reason="stable_anchor_group",
                        source_chunk_ids=tuple(dict.fromkeys(left.source_chunk_ids + right.source_chunk_ids)),
                        source_chunk_indices=tuple(
                            dict.fromkeys(left.source_chunk_indices + right.source_chunk_indices)
                        ),
                    )
                )
        return tuple(locks)
