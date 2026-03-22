"""Phase5 发音匹配纯函数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from app.services.alignment.nw_v2_core import NeedlemanWunschV2Core


@dataclass(frozen=True)
class PhoneticAlignmentPair:
    left_index: Optional[int]
    right_index: Optional[int]
    is_match: bool


def align_monotonic_sequences(
    *,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
) -> Tuple[PhoneticAlignmentPair, ...]:
    """按单调路径对齐两组发音键。"""
    nw_core = NeedlemanWunschV2Core()
    path = nw_core.align(
        seq1=tuple(str(item or "") for item in left_keys),
        seq2=tuple(str(item or "") for item in right_keys),
        match_fn=lambda left, right: left == right,
    )
    rows = [
        PhoneticAlignmentPair(
            left_index=left_index,
            right_index=right_index,
            is_match=(
                left_index is not None
                and right_index is not None
                and str(left_keys[left_index]) == str(right_keys[right_index])
            ),
        )
        for left_index, right_index in path
    ]
    _validate_monotonic_alignment(tuple(rows))
    return tuple(rows)


def project_token_phone_keys(
    *,
    token_count: int,
    phone_keys: Sequence[str],
    token_to_phone_spans: Sequence[tuple[int, int, int]],
) -> Tuple[str, ...]:
    """
    将 token->phone span 投影为 token 粒度发音键。

    返回长度固定为 token_count 的 tuple，未覆盖 token 返回空串。
    """
    if token_count < 0:
        raise ValueError("token_count 不能为负数")

    projected = ["" for _ in range(token_count)]
    prev_token_index = -1
    prev_phone_end = -1

    for token_index, phone_start, phone_end in token_to_phone_spans:
        if token_index < 0 or token_index >= token_count:
            raise ValueError("token_index 越界")
        if phone_start < 0 or phone_end < 0:
            raise ValueError("phone span 不能为负数")
        if phone_end < phone_start:
            raise ValueError("phone_end 必须 >= phone_start")
        if phone_end >= len(phone_keys):
            raise ValueError("phone span 越界")
        if token_index < prev_token_index:
            raise ValueError("token_to_phone_spans 非单调：token_index 回退")
        if token_index == prev_token_index and phone_start <= prev_phone_end:
            raise ValueError("token_to_phone_spans 非单调：同 token 的 phone span 重叠")
        if token_index > prev_token_index and phone_start <= prev_phone_end:
            raise ValueError("token_to_phone_spans 非单调：跨 token 的 phone span 回退")

        segment = [str(item or "") for item in phone_keys[phone_start : phone_end + 1]]
        if not projected[token_index]:
            projected[token_index] = "|".join(segment)
        else:
            projected[token_index] = f"{projected[token_index]}|{'|'.join(segment)}"

        prev_token_index = token_index
        prev_phone_end = phone_end

    return tuple(projected)


def _validate_monotonic_alignment(rows: Tuple[PhoneticAlignmentPair, ...]) -> None:
    prev_left = -1
    prev_right = -1
    for row in rows:
        if row.left_index is not None:
            if row.left_index < prev_left:
                raise ValueError("对齐结果非单调：left_index 回退")
            prev_left = row.left_index
        if row.right_index is not None:
            if row.right_index < prev_right:
                raise ValueError("对齐结果非单调：right_index 回退")
            prev_right = row.right_index


__all__ = [
    "PhoneticAlignmentPair",
    "align_monotonic_sequences",
    "project_token_phone_keys",
]

