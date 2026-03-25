"""局部块串联为单调主链。"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorMountInputView,
    LocalAlignmentBlock,
)


@dataclass(frozen=True)
class ChainSolveResult:
    committed_blocks: tuple[LocalAlignmentBlock, ...]
    slot_to_hook_indices: tuple[tuple[int, ...], ...]
    slot_anchor_kinds: tuple[str, ...]
    unresolved_slot_indices: tuple[int, ...]
    reseed_count: int


class ChainSolver:
    """贪心维持 slot/hook 单调关系。"""

    def solve(
        self,
        *,
        input_view: AnchorMountInputView,
        blocks: tuple[LocalAlignmentBlock, ...],
    ) -> ChainSolveResult:
        committed: list[LocalAlignmentBlock] = []
        reseed_count = 0
        ordered_blocks = sorted(blocks, key=lambda item: (item.slot_indices[0], item.hook_indices[0]))

        for block in ordered_blocks:
            conflict_start = self._find_conflict_start(committed=committed, block=block)
            if conflict_start is not None:
                reseed_count += 1
                suffix = committed[conflict_start:]
                suffix_score = sum(item.score for item in suffix)
                if float(block.score) <= float(suffix_score):
                    continue
                committed = committed[:conflict_start]
            committed.append(block)

        slot_to_hook_indices: list[tuple[int, ...]] = [tuple() for _ in input_view.slots]
        slot_anchor_kinds: list[str] = ["none" for _ in input_view.slots]
        for block in committed:
            for slot_index in block.slot_indices:
                slot_to_hook_indices[slot_index] = tuple(block.hook_indices)
                slot_anchor_kinds[slot_index] = block.anchor_kind

        unresolved = tuple(
            index for index, hook_indices in enumerate(slot_to_hook_indices) if not hook_indices
        )
        return ChainSolveResult(
            committed_blocks=tuple(committed),
            slot_to_hook_indices=tuple(slot_to_hook_indices),
            slot_anchor_kinds=tuple(slot_anchor_kinds),
            unresolved_slot_indices=unresolved,
            reseed_count=reseed_count,
        )

    @staticmethod
    def _find_conflict_start(
        *,
        committed: list[LocalAlignmentBlock],
        block: LocalAlignmentBlock,
    ) -> int | None:
        block_slot_set = set(block.slot_indices)
        current_hook_min = min(block.hook_indices)
        current_hook_max = max(block.hook_indices)
        for index, item in enumerate(committed):
            item_slot_set = set(item.slot_indices)
            if block_slot_set & item_slot_set:
                return index
            item_hook_min = min(item.hook_indices)
            item_hook_max = max(item.hook_indices)
            if current_hook_min <= item_hook_max and current_hook_max >= item_hook_min:
                return index
            if index == len(committed) - 1 and current_hook_min < item_hook_max:
                return index
        return None
