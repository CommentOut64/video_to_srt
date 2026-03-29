"""在主锚岛之间做局部 gap rescue。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.alignment.nw_v2_core import NeedlemanWunschV2Core
from app.services.timeanchored_alignment.anchor_mount.window_alignment_state import (
    WindowAlignmentState,
)


def _normalize(text: str) -> str:
    return "".join(
        char.lower()
        for char in str(text or "")
        if char.isalnum() or ("\u4e00" <= char <= "\u9fff")
    )


@dataclass(frozen=True)
class AnchorIsland:
    island_id: str
    block_ids: tuple[str, ...]
    unit_indices: tuple[int, ...]
    hook_indices: tuple[int, ...]


@dataclass(frozen=True)
class GapRescueMatch:
    unit_indices: tuple[int, ...]
    hook_indices: tuple[int, ...]
    anchor_kind: str
    score: float


@dataclass(frozen=True)
class AnchorGap:
    gap_id: str
    left_island_id: str | None
    right_island_id: str | None
    unit_indices: tuple[int, ...]
    hook_indices: tuple[int, ...]
    rescue_matches: tuple[GapRescueMatch, ...] = field(default_factory=tuple)


class GapRescueAligner:
    """复用 NW 内核在 anchor island 之间寻找可晋升匹配。"""

    def __init__(
        self,
        *,
        nw_core: NeedlemanWunschV2Core | None = None,
    ) -> None:
        self._nw_core = nw_core or NeedlemanWunschV2Core()

    def run(
        self,
        *,
        state: WindowAlignmentState,
        attempt_rescue: bool = True,
        increment_round: bool = True,
    ) -> WindowAlignmentState:
        islands = self._build_anchor_islands(state=state)
        gaps = self._build_open_gaps(
            state=state,
            islands=islands,
            attempt_rescue=attempt_rescue,
        )
        diagnostics = dict(state.diagnostics)
        diagnostics["anchor_island_count"] = len(islands)
        diagnostics["open_gap_count"] = len(gaps)
        diagnostics["gap_rescue_match_count"] = sum(
            len(gap.rescue_matches) for gap in gaps
        )
        return state.with_updates(
            anchor_islands=islands,
            open_gaps=gaps,
            rescue_round_index=(
                state.rescue_round_index + 1
                if increment_round
                else state.rescue_round_index
            ),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _build_anchor_islands(
        *,
        state: WindowAlignmentState,
    ) -> tuple[AnchorIsland, ...]:
        solve_result = state.main_chain
        if solve_result is None:
            return tuple()
        islands: list[AnchorIsland] = []
        for index, block in enumerate(getattr(solve_result, "committed_blocks", ()) or ()):
            islands.append(
                AnchorIsland(
                    island_id=f"island-{index}",
                    block_ids=(str(block.block_id),),
                    unit_indices=tuple(int(item) for item in block.unit_indices),
                    hook_indices=tuple(int(item) for item in block.hook_indices),
                )
            )
        return tuple(islands)

    def _build_open_gaps(
        self,
        *,
        state: WindowAlignmentState,
        islands: tuple[AnchorIsland, ...],
        attempt_rescue: bool,
    ) -> tuple[AnchorGap, ...]:
        if len(islands) < 2:
            return tuple()
        gaps: list[AnchorGap] = []
        for index, (left, right) in enumerate(zip(islands, islands[1:], strict=False)):
            unit_start = max(left.unit_indices) + 1
            unit_end = min(right.unit_indices)
            hook_start = max(left.hook_indices) + 1
            hook_end = min(right.hook_indices)
            unit_indices = tuple(range(unit_start, unit_end))
            hook_indices = tuple(range(hook_start, hook_end))
            if not unit_indices and not hook_indices:
                continue
            rescue_matches = (
                self._align_gap(
                    state=state,
                    unit_indices=unit_indices,
                    hook_indices=hook_indices,
                )
                if attempt_rescue and unit_indices and hook_indices
                else tuple()
            )
            gaps.append(
                AnchorGap(
                    gap_id=f"gap-{index}",
                    left_island_id=left.island_id,
                    right_island_id=right.island_id,
                    unit_indices=unit_indices,
                    hook_indices=hook_indices,
                    rescue_matches=rescue_matches,
                )
            )
        return tuple(gaps)

    def _align_gap(
        self,
        *,
        state: WindowAlignmentState,
        unit_indices: tuple[int, ...],
        hook_indices: tuple[int, ...],
    ) -> tuple[GapRescueMatch, ...]:
        input_view = state.input_view
        unit_texts = [input_view.token_units[index].token_text for index in unit_indices]
        hook_texts = [input_view.fast_hooks[index].hook_text for index in hook_indices]
        unit_keys = [_normalize(text) for text in unit_texts]
        hook_keys = [_normalize(text) for text in hook_texts]
        path = self._nw_core.align(
            unit_texts,
            hook_texts,
            match_fn=lambda left, right: _normalize(left) == _normalize(right),
        )
        unit_key_counts = {key: unit_keys.count(key) for key in unit_keys if key}
        hook_key_counts = {key: hook_keys.count(key) for key in hook_keys if key}
        matches: list[GapRescueMatch] = []
        for unit_offset, hook_offset in path:
            if unit_offset is None or hook_offset is None:
                continue
            unit_key = unit_keys[unit_offset]
            hook_key = hook_keys[hook_offset]
            if not unit_key or unit_key != hook_key:
                continue
            if unit_key_counts.get(unit_key, 0) != 1 or hook_key_counts.get(hook_key, 0) != 1:
                continue
            unit_text = unit_texts[unit_offset]
            hook_text = hook_texts[hook_offset]
            matches.append(
                GapRescueMatch(
                    unit_indices=(int(unit_indices[unit_offset]),),
                    hook_indices=(int(hook_indices[hook_offset]),),
                    anchor_kind="exact" if unit_text == hook_text else "normalized",
                    score=0.9 if unit_text == hook_text else 0.82,
                )
            )
        return tuple(matches)
