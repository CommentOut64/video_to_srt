"""弱信息 seed 降权。"""

from __future__ import annotations

from dataclasses import replace

from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorCandidate,
    AnchorMountInputView,
)


_LOW_INFO_TOKENS = {"a", "an", "the", "of", "to", "in", "了", "的", "啊", "呢"}


class LowComplexityMask:
    """对低信息 seed 做保守降权。"""

    def apply(
        self,
        *,
        candidates: tuple[AnchorCandidate, ...],
        input_view: AnchorMountInputView,
    ) -> tuple[AnchorCandidate, ...]:
        masked: list[AnchorCandidate] = []
        for candidate in candidates:
            unit_text = "".join(
                str(input_view.token_units[index].token_text or "")
                for index in candidate.unit_indices
            ).strip().lower()
            if unit_text in _LOW_INFO_TOKENS or len(unit_text) <= 1:
                masked.append(
                    replace(
                        candidate,
                        score=max(0.35, float(candidate.score) * 0.6),
                        is_hard=False,
                    )
                )
                continue
            masked.append(candidate)
        return tuple(masked)
