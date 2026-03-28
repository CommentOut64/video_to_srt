"""Seed-and-chain 的初始锚点发现。"""

from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorCandidate,
    AnchorMountInputView,
)


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
                        AnchorCandidate(
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
                        AnchorCandidate(
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
                            AnchorCandidate(
                                unit_indices=(unit_index,),
                                hook_indices=(hook_index,),
                                anchor_kind="soft_pronunciation",
                                score=0.65,
                                is_hard=False,
                            )
                        )
            if not matched and len(input_view.token_units) > 1:
                for next_unit_index in range(unit_index + 1, len(input_view.token_units)):
                    merged_key = _normalize(
                        token_unit.token_text + input_view.token_units[next_unit_index].token_text
                    )
                    for hook_index, hook in enumerate(input_view.fast_hooks):
                        if merged_key and merged_key == _normalize(hook.hook_text):
                            candidates.append(
                                AnchorCandidate(
                                    unit_indices=(unit_index, next_unit_index),
                                    hook_indices=(hook_index,),
                                    anchor_kind="merge",
                                    score=0.82,
                                    is_hard=True,
                                )
                            )
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
        for start_index in range(len(normalized_hooks)):
            merged_parts: list[str] = []
            hook_indices: list[int] = []
            for hook_index in range(start_index, len(normalized_hooks)):
                hook_key = normalized_hooks[hook_index]
                if not hook_key:
                    continue
                merged_parts.append(hook_key)
                hook_indices.append(hook_index)
                merged_key = "".join(merged_parts)
                if merged_key == unit_key:
                    return AnchorCandidate(
                        unit_indices=(unit_index,),
                        hook_indices=tuple(hook_indices),
                        anchor_kind="sequence",
                        score=0.9,
                        is_hard=True,
                    )
                if len(merged_key) > len(unit_key):
                    break
        return None
