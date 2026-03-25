"""局部扩展块构建。"""

from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorCandidate,
    LocalAlignmentBlock,
)


class LocalExtension:
    """当前版本按 seed 直接形成局部块。"""

    def build(
        self,
        *,
        candidates: tuple[AnchorCandidate, ...],
    ) -> tuple[LocalAlignmentBlock, ...]:
        blocks: list[LocalAlignmentBlock] = []
        for index, candidate in enumerate(candidates):
            blocks.append(
                LocalAlignmentBlock(
                    block_id=f"block-{index}",
                    slot_indices=candidate.slot_indices,
                    hook_indices=candidate.hook_indices,
                    score=candidate.score,
                    block_kind="anchored" if candidate.is_hard else "partial",
                    anchor_kind=candidate.anchor_kind,
                )
            )
        return tuple(blocks)
