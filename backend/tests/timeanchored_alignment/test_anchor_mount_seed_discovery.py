from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.contracts import AnchorMountInputView
from app.services.timeanchored_alignment.anchor_mount.seed_discovery import SeedDiscovery
from app.services.timeanchored_alignment.preparation.contracts import (
    FastHook,
    PreparedTokenUnit,
    SlowWindowTextPackage,
)
from app.services.timeanchored_alignment.slow_window.contracts import (
    WindowChunkBinding,
    WindowCoverage,
)


def test_seed_discovery_matches_single_token_unit_to_multiple_contiguous_hooks() -> None:
    input_view = AnchorMountInputView(
        window_id="window-seed-001",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        language="en",
        token_units=(
            PreparedTokenUnit(
                unit_id="unit-0",
                token_text="hello world",
                normalized_text="hello world",
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
    assert candidates[0].unit_indices == (0,)
    assert candidates[0].hook_indices == (0, 1)


def test_seed_discovery_rejects_non_adjacent_merge_candidates() -> None:
    input_view = AnchorMountInputView(
        window_id="window-seed-002",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        language="en",
        token_units=(
            PreparedTokenUnit(
                unit_id="unit-0",
                token_text="alpha",
                normalized_text="alpha",
                char_start=0,
                char_end=5,
                speaker_id=None,
                turn_id=None,
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
            ),
            PreparedTokenUnit(
                unit_id="unit-1",
                token_text="noise",
                normalized_text="noise",
                char_start=6,
                char_end=11,
                speaker_id=None,
                turn_id=None,
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
            ),
            PreparedTokenUnit(
                unit_id="unit-2",
                token_text="beta",
                normalized_text="beta",
                char_start=12,
                char_end=16,
                speaker_id=None,
                turn_id=None,
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
            ),
        ),
        window_text=SlowWindowTextPackage(
            text="alpha noise beta",
            display_text="alpha noise beta",
            source_language="en",
        ),
        punctuation_evidences=tuple(),
        fast_hooks=(
            FastHook(
                hook_text="alphabeta",
                start=0.0,
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

    assert not any(candidate.anchor_kind == "merge" for candidate in candidates)


def test_seed_discovery_limits_hook_sequence_width_to_three() -> None:
    input_view = AnchorMountInputView(
        window_id="window-seed-003",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        language="en",
        token_units=(
            PreparedTokenUnit(
                unit_id="unit-0",
                token_text="alphabet",
                normalized_text="alphabet",
                char_start=0,
                char_end=8,
                speaker_id=None,
                turn_id=None,
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
            ),
        ),
        window_text=SlowWindowTextPackage(
            text="alphabet",
            display_text="alphabet",
            source_language="en",
        ),
        punctuation_evidences=tuple(),
        fast_hooks=(
            FastHook(
                hook_text="al",
                start=0.0,
                end=0.1,
                confidence=0.95,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
                token_type="word",
            ),
            FastHook(
                hook_text="ph",
                start=0.1,
                end=0.2,
                confidence=0.95,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
                token_type="word",
            ),
            FastHook(
                hook_text="ab",
                start=0.2,
                end=0.3,
                confidence=0.95,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
                token_type="word",
            ),
            FastHook(
                hook_text="et",
                start=0.3,
                end=0.4,
                confidence=0.95,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
                token_type="word",
            ),
        ),
        pronunciation_hints=tuple(),
        policy_snapshot=None,
        coverage=WindowCoverage(
            core_segments=((0.0, 0.4),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-1",
                    chunk_index=1,
                    chunk_start=0.0,
                    chunk_end=0.4,
                    overlap_ratio=1.0,
                    role="owner",
                    is_owner=True,
                ),
            ),
        ),
    )

    candidates = SeedDiscovery().discover(input_view=input_view)

    assert not any(candidate.anchor_kind == "sequence" for candidate in candidates)


def test_seed_discovery_limits_cjk_hook_sequence_width_to_three() -> None:
    input_view = AnchorMountInputView(
        window_id="window-seed-004",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        language="zh",
        token_units=(
            PreparedTokenUnit(
                unit_id="unit-0",
                token_text="中华人民共和国",
                normalized_text="中华人民共和国",
                char_start=0,
                char_end=7,
                speaker_id=None,
                turn_id=None,
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
            ),
        ),
        window_text=SlowWindowTextPackage(
            text="中华人民共和国",
            display_text="中华人民共和国",
            source_language="zh",
        ),
        punctuation_evidences=tuple(),
        fast_hooks=(
            FastHook(
                hook_text="中华",
                start=0.0,
                end=0.1,
                confidence=0.95,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
                token_type="word",
            ),
            FastHook(
                hook_text="人民",
                start=0.1,
                end=0.2,
                confidence=0.95,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
                token_type="word",
            ),
            FastHook(
                hook_text="共和",
                start=0.2,
                end=0.3,
                confidence=0.95,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
                token_type="word",
            ),
            FastHook(
                hook_text="国",
                start=0.3,
                end=0.4,
                confidence=0.95,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
                token_type="word",
            ),
        ),
        pronunciation_hints=tuple(),
        policy_snapshot=None,
        coverage=WindowCoverage(
            core_segments=((0.0, 0.4),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-1",
                    chunk_index=1,
                    chunk_start=0.0,
                    chunk_end=0.4,
                    overlap_ratio=1.0,
                    role="owner",
                    is_owner=True,
                ),
            ),
        ),
    )

    candidates = SeedDiscovery().discover(input_view=input_view)

    assert not any(candidate.anchor_kind == "sequence" for candidate in candidates)
