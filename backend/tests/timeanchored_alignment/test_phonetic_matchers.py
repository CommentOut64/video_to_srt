from __future__ import annotations

import pytest

from app.services.timeanchored_alignment.phonetic_matchers import (
    PhoneticAlignmentPair,
    align_monotonic_sequences,
    project_token_phone_keys,
)


def _is_monotonic(indices: list[int | None]) -> bool:
    values = [item for item in indices if item is not None]
    return values == sorted(values)


def test_align_monotonic_sequences_same_length() -> None:
    result = align_monotonic_sequences(
        left_keys=("ni", "hao"),
        right_keys=("ni", "hao"),
    )
    assert result == (
        PhoneticAlignmentPair(left_index=0, right_index=0, is_match=True),
        PhoneticAlignmentPair(left_index=1, right_index=1, is_match=True),
    )


def test_align_monotonic_sequences_single_gap() -> None:
    result = align_monotonic_sequences(
        left_keys=("ni", "hao"),
        right_keys=("ni", "ma", "hao"),
    )
    gap_count = sum(1 for row in result if row.left_index is None or row.right_index is None)
    assert gap_count == 1
    assert _is_monotonic([row.left_index for row in result])
    assert _is_monotonic([row.right_index for row in result])


def test_align_monotonic_sequences_only_outputs_monotonic_indices() -> None:
    result = align_monotonic_sequences(
        left_keys=("a", "b", "a"),
        right_keys=("a", "a", "b"),
    )
    assert _is_monotonic([row.left_index for row in result])
    assert _is_monotonic([row.right_index for row in result])


def test_project_token_phone_keys_monotonic_mapping() -> None:
    result = project_token_phone_keys(
        token_count=2,
        phone_keys=("ow", "pen", "ai"),
        token_to_phone_spans=((0, 0, 1), (1, 2, 2)),
    )
    assert result == ("ow|pen", "ai")


def test_project_token_phone_keys_rejects_non_monotonic_span() -> None:
    with pytest.raises(ValueError):
        project_token_phone_keys(
            token_count=2,
            phone_keys=("ow", "pen", "ai"),
            token_to_phone_spans=((0, 1, 2), (1, 0, 0)),
        )

