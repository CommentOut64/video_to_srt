from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.chain_solver import ChainSolver
from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorMountInputView,
    LocalAlignmentBlock,
)
from app.services.timeanchored_alignment.preparation.contracts import (
    FastHook,
    PreparedTokenUnit,
    SlowWindowTextPackage,
)
from app.services.timeanchored_alignment.slow_window.contracts import (
    WindowChunkBinding,
    WindowCoverage,
)


def _build_input_view() -> AnchorMountInputView:
    token_units = tuple(
        PreparedTokenUnit(
            unit_id=f"unit-{index}",
            token_text=f"tok{index}",
            normalized_text=f"tok{index}",
            char_start=index * 4,
            char_end=index * 4 + 4,
            speaker_id=None,
            turn_id=None,
            source_chunk_ids=("chunk-0",),
            source_chunk_indices=(0,),
        )
        for index in range(8)
    )
    hooks = tuple(
        FastHook(
            hook_text=f"tok{index}",
            start=index * 0.2,
            end=index * 0.2 + 0.18,
            confidence=0.95,
            source_chunk_id="chunk-0",
            source_chunk_index=0,
            token_type="word",
        )
        for index in range(8)
    )
    return AnchorMountInputView(
        window_id="window-benchmark-001",
        owner_chunk_id="chunk-0",
        owner_chunk_index=0,
        source_chunk_ids=("chunk-0",),
        source_chunk_indices=(0,),
        language="en",
        token_units=token_units,
        window_text=SlowWindowTextPackage(
            text=" ".join(f"tok{index}" for index in range(8)),
            display_text=" ".join(f"tok{index}" for index in range(8)),
            source_language="en",
        ),
        punctuation_evidences=tuple(),
        fast_hooks=hooks,
        pronunciation_hints=tuple(),
        policy_snapshot=None,
        coverage=WindowCoverage(
            core_segments=((0.0, 1.6),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-0",
                    chunk_index=0,
                    chunk_start=0.0,
                    chunk_end=1.6,
                    overlap_ratio=1.0,
                    role="owner",
                    is_owner=True,
                ),
            ),
        ),
    )


def test_anchor_mount_performance_benchmark_exposes_sparse_dp_metrics() -> None:
    input_view = _build_input_view()
    blocks = tuple(
        LocalAlignmentBlock(
            block_id=f"block-{index}",
            unit_indices=(index,),
            hook_indices=(index,),
            score=0.8,
            block_kind="anchored",
            anchor_kind="exact",
        )
        for index in range(8)
    )

    result = ChainSolver().solve(input_view=input_view, blocks=blocks)

    assert result.compatibility_edge_count >= 0
    assert result.avg_in_degree >= 0.0
    assert result.max_in_degree >= 0
    assert result.solver_elapsed_ms >= 0.0
