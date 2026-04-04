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
    unit_to_hook_indices: tuple[tuple[int, ...], ...]
    unit_anchor_kinds: tuple[str, ...]
    unresolved_unit_indices: tuple[int, ...]
    reseed_count: int


class ChainSolver:
    """贪心维持 unit/hook 单调关系。"""

    def solve(
        self,
        *,
        input_view: AnchorMountInputView,
        blocks: tuple[LocalAlignmentBlock, ...],
    ) -> ChainSolveResult:
        committed: list[LocalAlignmentBlock] = []
        reseed_count = 0
        ordered_blocks = sorted(blocks, key=lambda item: (item.unit_indices[0], item.hook_indices[0]))

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

        unit_to_hook_indices: list[tuple[int, ...]] = [tuple() for _ in input_view.token_units]
        unit_anchor_kinds: list[str] = ["none" for _ in input_view.token_units]
        for block in committed:
            if len(block.unit_indices) == len(block.hook_indices):
                for unit_index, hook_index in zip(block.unit_indices, block.hook_indices, strict=False):
                    unit_to_hook_indices[unit_index] = (int(hook_index),)
                    unit_anchor_kinds[unit_index] = block.anchor_kind
                continue
            for unit_index in block.unit_indices:
                unit_to_hook_indices[unit_index] = tuple(block.hook_indices)
                unit_anchor_kinds[unit_index] = block.anchor_kind

        unresolved = tuple(
            index for index, hook_indices in enumerate(unit_to_hook_indices) if not hook_indices
        )
        return ChainSolveResult(
            committed_blocks=tuple(committed),
            unit_to_hook_indices=tuple(unit_to_hook_indices),
            unit_anchor_kinds=tuple(unit_anchor_kinds),
            unresolved_unit_indices=unresolved,
            reseed_count=reseed_count,
        )

    @staticmethod
    def _find_conflict_start(
        *,
        committed: list[LocalAlignmentBlock],
        block: LocalAlignmentBlock,
    ) -> int | None:
        block_unit_set = set(block.unit_indices)
        current_hook_min = min(block.hook_indices)
        current_hook_max = max(block.hook_indices)
        for index, item in enumerate(committed):
            item_unit_set = set(item.unit_indices)
            if block_unit_set & item_unit_set:
                return index
            item_hook_min = min(item.hook_indices)
            item_hook_max = max(item.hook_indices)
            if current_hook_min <= item_hook_max and current_hook_max >= item_hook_min:
                return index
            if index == len(committed) - 1 and current_hook_min < item_hook_max:
                return index
        return None
