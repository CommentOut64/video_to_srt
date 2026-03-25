"""hook owner 决议。"""

from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorMountInputView,
    HookClaimRecord,
    TemporalEnvelope,
)


class HookClaimResolver:
    """当前窗口内把 hook ownership finalize。"""

    def resolve(
        self,
        *,
        input_view: AnchorMountInputView,
        envelopes: tuple[TemporalEnvelope, ...],
    ) -> tuple[HookClaimRecord, ...]:
        claims: list[HookClaimRecord] = []
        seen: set[str] = set()
        for envelope in envelopes:
            for hook_id in envelope.source_hook_ids:
                if hook_id in seen:
                    continue
                seen.add(hook_id)
                claims.append(
                    HookClaimRecord(
                        hook_id=hook_id,
                        owner_window_id=input_view.window_id,
                        claim_level="hard" if envelope.envelope_kind in {"anchored", "merged"} else "guard_only",
                        claim_reason=envelope.envelope_kind,
                        overlap_ratio=1.0 if envelope.envelope_kind in {"anchored", "merged"} else 0.5,
                        anchor_score=envelope.confidence,
                        finalized=True,
                    )
                )
        return tuple(claims)
