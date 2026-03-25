from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.contracts import AnchorMountInputView
from app.services.timeanchored_alignment.anchor_mount.seed_discovery import SeedDiscovery
from app.services.timeanchored_alignment.preparation.contracts import (
    FastHook,
    SlowSlot,
    SlowWindowTextPackage,
)
from app.services.timeanchored_alignment.slow_window.contracts import (
    WindowChunkBinding,
    WindowCoverage,
)


def test_seed_discovery_matches_single_slot_to_multiple_contiguous_hooks() -> None:
    input_view = AnchorMountInputView(
        window_id="window-seed-001",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        language="en",
        slots=(
            SlowSlot(
                slot_id="slot-0",
                text="hello world",
                char_start=0,
                char_end=11,
                speaker_id=None,
                turn_id=None,
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
            ),
        ),
        window_text=SlowWindowTextPackage(
            text="hello world",
            display_text="hello world",
            source_language="en",
        ),
        punctuation_evidences=tuple(),
        fast_hooks=(
            FastHook(
                hook_text="hello",
                start=0.0,
                end=0.4,
                confidence=0.95,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
                token_type="word",
            ),
            FastHook(
                hook_text="world",
                start=0.4,
                end=0.8,
                confidence=0.95,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
                token_type="word",
            ),
        ),
        pronunciation_hints=tuple(),
        policy_snapshot=None,
        coverage=WindowCoverage(
            core_segments=((0.0, 0.8),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-1",
                    chunk_index=1,
                    chunk_start=0.0,
                    chunk_end=0.8,
                    overlap_ratio=1.0,
                    role="owner",
                    is_owner=True,
                ),
            ),
        ),
    )

    candidates = SeedDiscovery().discover(input_view=input_view)

    assert candidates
    assert candidates[0].slot_indices == (0,)
    assert candidates[0].hook_indices == (0, 1)
