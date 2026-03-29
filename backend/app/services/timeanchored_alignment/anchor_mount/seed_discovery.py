"""Seed-and-chain 的初始锚点发现。"""

from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorCandidate,
    AnchorMountInputView,
)

MAX_TOKEN_MERGE_WIDTH = 2
MAX_HOOK_SEQUENCE_WIDTH = 3


def _normalize(text: str) -> str:
    return "".join(
        char.lower()
        for char in str(text or "")
        if char.isalnum() or ("\u4e00" <= char <= "\u9fff")
    )


class SeedDiscovery:
    """从 token units 与 hooks 中提取 hard/soft seeds。"""

    def discover(self, *, input_view: AnchorMountInputView) -> tuple[AnchorCandidate, ...]:
        hints_by_key = {
            _normalize(hint.token_text): hint
            for hint in input_view.pronunciation_hints
            if _normalize(hint.token_text)
        }
        candidates: list[AnchorCandidate] = []
        for unit_index, token_unit in enumerate(input_view.token_units):
            unit_key = _normalize(token_unit.token_text)
            if not unit_key:
                continue
            matched = False
            for hook_index, hook in enumerate(input_view.fast_hooks):
                hook_key = _normalize(hook.hook_text)
                if token_unit.token_text == hook.hook_text:
                    candidates.append(
                        self._build_candidate(
                            unit_indices=(unit_index,),
                            hook_indices=(hook_index,),
                            anchor_kind="exact",
                            score=1.0,
                            is_hard=True,
                        )
                    )
                    matched = True
                elif unit_key and unit_key == hook_key:
                    candidates.append(
                        self._build_candidate(
                            unit_indices=(unit_index,),
                            hook_indices=(hook_index,),
                            anchor_kind="normalized",
                            score=0.88,
                            is_hard=True,
                        )
                    )
                    matched = True
            if not matched and unit_key in hints_by_key:
                for hook_index, hook in enumerate(input_view.fast_hooks):
                    if unit_key == _normalize(hook.hook_text):
                        candidates.append(
                            self._build_candidate(
                                unit_indices=(unit_index,),
                                hook_indices=(hook_index,),
                                anchor_kind="soft_pronunciation",
                                score=0.65,
                                is_hard=False,
                            )
                        )
            if not matched and len(input_view.token_units) > 1:
                merge_candidate = self._match_bounded_adjacent_merge(
                    input_view=input_view,
                    unit_index=unit_index,
                )
                if merge_candidate is not None:
                    candidates.append(merge_candidate)
            if not matched:
                sequence_candidate = self._match_contiguous_hook_sequence(
                    unit_key=unit_key,
                    hooks=input_view.fast_hooks,
                    unit_index=unit_index,
                )
                if sequence_candidate is not None:
                    candidates.append(sequence_candidate)
        return tuple(candidates)

    @staticmethod
    def _build_candidate(
        *,
        unit_indices: tuple[int, ...],
        hook_indices: tuple[int, ...],
        anchor_kind: str,
        score: float,
        is_hard: bool,
    ) -> AnchorCandidate:
        unit_part = "-".join(str(index) for index in unit_indices)
        hook_part = "-".join(str(index) for index in hook_indices)
        return AnchorCandidate(
            candidate_id=f"{anchor_kind}:{unit_part}:{hook_part}",
            unit_indices=unit_indices,
            hook_indices=hook_indices,
            anchor_kind=anchor_kind,
            score=score,
            is_hard=is_hard,
        )

    def _match_bounded_adjacent_merge(
        self,
        *,
        input_view: AnchorMountInputView,
        unit_index: int,
    ) -> AnchorCandidate | None:
        max_width = min(MAX_TOKEN_MERGE_WIDTH, len(input_view.token_units) - unit_index)
        for width in range(2, max_width + 1):
            unit_span = tuple(range(unit_index, unit_index + width))
            merged_key = _normalize(
                "".join(input_view.token_units[index].token_text for index in unit_span)
            )
            if not merged_key:
                continue
            for hook_index, hook in enumerate(input_view.fast_hooks):
                if merged_key == _normalize(hook.hook_text):
                    return self._build_candidate(
                        unit_indices=unit_span,
                        hook_indices=(hook_index,),
                        anchor_kind="merge",
                        score=0.82,
                        is_hard=True,
                    )
        return None

    @staticmethod
    def _match_contiguous_hook_sequence(
        *,
        unit_key: str,
        hooks,
        unit_index: int,
    ) -> AnchorCandidate | None:
        if not unit_key:
            return None

        normalized_hooks = [
            _normalize(getattr(hook, "hook_text", ""))
            for hook in hooks
        ]
        max_width = SeedDiscovery._sequence_width_limit(unit_key=unit_key)
        for start_index in range(len(normalized_hooks)):
            merged_parts: list[str] = []
            hook_indices: list[int] = []
            for hook_index in range(start_index, min(len(normalized_hooks), start_index + max_width)):
                hook_key = normalized_hooks[hook_index]
                if not hook_key:
                    continue
                merged_parts.append(hook_key)
                hook_indices.append(hook_index)
                merged_key = "".join(merged_parts)
                if merged_key == unit_key:
                    return SeedDiscovery._build_candidate(
                        unit_indices=(unit_index,),
                        hook_indices=tuple(hook_indices),
                        anchor_kind="sequence",
                        score=0.9,
                        is_hard=True,
                    )
                if len(merged_key) > len(unit_key):
                    break
        return None

    @staticmethod
    def _sequence_width_limit(*, unit_key: str) -> int:
        return MAX_HOOK_SEQUENCE_WIDTH
