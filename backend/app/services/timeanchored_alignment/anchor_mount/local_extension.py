"""局部扩展块构建。"""

from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorCandidate,
    AnchorMountInputView,
    LocalAlignmentBlock,
)


_RUN_ELIGIBLE_KINDS = {"exact", "normalized"}
_RUN_BOUNDARY_MARKS = {"。", "！", "？", ".", "!", "?", "，", "、", ",", ";", "；", ":", "："}


class LocalExtension:
    """把可确认的连续 lexical run 收束成真实 block。"""

    def build(
        self,
        *,
        input_view: AnchorMountInputView,
        candidates: tuple[AnchorCandidate, ...],
    ) -> tuple[LocalAlignmentBlock, ...]:
        run_blocks, consumed_pairs = self._build_run_blocks(
            input_view=input_view,
            candidates=candidates,
        )
        residual_blocks = [
            self._candidate_to_block(candidate)
            for candidate in candidates
            if not self._is_consumed_singleton(candidate, consumed_pairs)
        ]
        all_blocks = [*run_blocks, *residual_blocks]
        return tuple(
            LocalAlignmentBlock(
                block_id=f"block-{index}",
                unit_indices=block.unit_indices,
                hook_indices=block.hook_indices,
                score=block.score,
                block_kind=block.block_kind,
                anchor_kind=block.anchor_kind,
            )
            for index, block in enumerate(all_blocks)
        )

    def _build_run_blocks(
        self,
        *,
        input_view: AnchorMountInputView,
        candidates: tuple[AnchorCandidate, ...],
    ) -> tuple[list[LocalAlignmentBlock], set[tuple[int, int]]]:
        boundary_after_indices = self._collect_boundary_after_indices(input_view=input_view)
        grouped: dict[int, list[AnchorCandidate]] = {}
        for candidate in candidates:
            if not self._is_run_eligible(candidate):
                continue
            unit_index = int(candidate.unit_indices[0])
            hook_index = int(candidate.hook_indices[0])
            grouped.setdefault(hook_index - unit_index, []).append(candidate)

        blocks: list[LocalAlignmentBlock] = []
        consumed_pairs: set[tuple[int, int]] = set()
        for group in grouped.values():
            ordered = sorted(group, key=lambda item: (item.unit_indices[0], item.hook_indices[0]))
            current_run: list[AnchorCandidate] = []
            for candidate in ordered:
                if not current_run:
                    current_run = [candidate]
                    continue
                previous = current_run[-1]
                previous_unit = int(previous.unit_indices[0])
                previous_hook = int(previous.hook_indices[0])
                current_unit = int(candidate.unit_indices[0])
                current_hook = int(candidate.hook_indices[0])
                if (
                    current_unit == previous_unit + 1
                    and current_hook == previous_hook + 1
                    and not self._has_structural_boundary(
                        input_view=input_view,
                        left_unit_index=previous_unit,
                        right_unit_index=current_unit,
                        boundary_after_indices=boundary_after_indices,
                    )
                ):
                    current_run.append(candidate)
                    continue
                self._flush_run(current_run=current_run, blocks=blocks, consumed_pairs=consumed_pairs)
                current_run = [candidate]
            self._flush_run(current_run=current_run, blocks=blocks, consumed_pairs=consumed_pairs)
        return blocks, consumed_pairs

    @staticmethod
    def _flush_run(
        *,
        current_run: list[AnchorCandidate],
        blocks: list[LocalAlignmentBlock],
        consumed_pairs: set[tuple[int, int]],
    ) -> None:
        if len(current_run) < 2:
            return
        unit_indices = tuple(int(candidate.unit_indices[0]) for candidate in current_run)
        hook_indices = tuple(int(candidate.hook_indices[0]) for candidate in current_run)
        anchor_kind = "exact" if all(candidate.anchor_kind == "exact" for candidate in current_run) else "normalized"
        score = sum(float(candidate.score) for candidate in current_run) / float(len(current_run))
        blocks.append(
            LocalAlignmentBlock(
                block_id="run-pending",
                unit_indices=unit_indices,
                hook_indices=hook_indices,
                score=score,
                block_kind="anchored",
                anchor_kind=anchor_kind,
            )
        )
        consumed_pairs.update(zip(unit_indices, hook_indices, strict=False))

    @staticmethod
    def _candidate_to_block(candidate: AnchorCandidate) -> LocalAlignmentBlock:
        return LocalAlignmentBlock(
            block_id="candidate-pending",
            unit_indices=candidate.unit_indices,
            hook_indices=candidate.hook_indices,
            score=candidate.score,
            block_kind="anchored" if candidate.is_hard else "partial",
            anchor_kind=candidate.anchor_kind,
        )

    @staticmethod
    def _is_run_eligible(candidate: AnchorCandidate) -> bool:
        return (
            candidate.is_hard
            and candidate.anchor_kind in _RUN_ELIGIBLE_KINDS
            and len(candidate.unit_indices) == 1
            and len(candidate.hook_indices) == 1
        )

    @staticmethod
    def _is_consumed_singleton(candidate: AnchorCandidate, consumed_pairs: set[tuple[int, int]]) -> bool:
        return (
            len(candidate.unit_indices) == 1
            and len(candidate.hook_indices) == 1
            and (int(candidate.unit_indices[0]), int(candidate.hook_indices[0])) in consumed_pairs
        )

    def _collect_boundary_after_indices(self, *, input_view: AnchorMountInputView) -> set[int]:
        token_units = tuple(input_view.token_units or ())
        boundaries: set[int] = set()
        for evidence in input_view.punctuation_evidences:
            mark = str(evidence.mark or "")
            if mark not in _RUN_BOUNDARY_MARKS:
                continue
            char_index = int(evidence.source_char_index)
            boundary_index: int | None = None
            if str(evidence.attach_side or "after") == "after":
                token_index = self._token_for_char(token_units=token_units, char_index=char_index)
                if token_index is not None:
                    boundary_index = token_index
                else:
                    boundary_index = self._previous_token(token_units=token_units, char_index=char_index)
            else:
                next_index = self._next_token(token_units=token_units, char_index=char_index)
                if next_index is not None and next_index > 0:
                    boundary_index = next_index - 1
                else:
                    boundary_index = self._previous_token(token_units=token_units, char_index=char_index)
            if boundary_index is not None and 0 <= boundary_index < len(token_units) - 1:
                boundaries.add(boundary_index)
        return boundaries

    @staticmethod
    def _has_structural_boundary(
        *,
        input_view: AnchorMountInputView,
        left_unit_index: int,
        right_unit_index: int,
        boundary_after_indices: set[int],
    ) -> bool:
        left_unit = input_view.token_units[left_unit_index]
        right_unit = input_view.token_units[right_unit_index]
        return (
            left_unit_index in boundary_after_indices
            or left_unit.turn_id != right_unit.turn_id
            or left_unit.speaker_id != right_unit.speaker_id
        )

    @staticmethod
    def _token_for_char(*, token_units, char_index: int) -> int | None:
        for index, token_unit in enumerate(token_units):
            if token_unit.char_start <= char_index < token_unit.char_end:
                return index
        return None

    @staticmethod
    def _next_token(*, token_units, char_index: int) -> int | None:
        for index, token_unit in enumerate(token_units):
            if char_index < token_unit.char_start:
                return index
        return None

    @staticmethod
    def _previous_token(*, token_units, char_index: int) -> int | None:
        candidate: int | None = None
        for index, token_unit in enumerate(token_units):
            if token_unit.char_end <= char_index:
                candidate = index
                continue
            break
        return candidate
