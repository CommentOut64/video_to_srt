"""Window coverage 规划。"""

from __future__ import annotations

from collections import OrderedDict

from app.services.timeanchored_alignment.slow_window.contracts import (
    WindowChunkBinding,
    WindowCoverage,
    WindowSourceUnit,
)


class CoveragePlanner:
    """为 ReadySlowWindow 冻结 chunk 归属与 owner。"""

    def build(self, source_units: tuple[WindowSourceUnit, ...]) -> WindowCoverage:
        if not source_units:
            raise ValueError("CoveragePlanner 需要至少一个 source unit")

        window_start = min(unit.audio_start for unit in source_units)
        window_end = max(unit.audio_end for unit in source_units)
        total_duration = max(window_end - window_start, 1e-6)

        spans: "OrderedDict[tuple[str, int], dict[str, float]]" = OrderedDict()
        for unit in source_units:
            unit_duration = max(unit.audio_end - unit.audio_start, 0.0)
            for chunk_id, chunk_index in zip(unit.source_chunk_ids, unit.source_chunk_indices):
                key = (chunk_id, chunk_index)
                record = spans.setdefault(
                    key,
                    {
                        "chunk_start": unit.audio_start,
                        "chunk_end": unit.audio_end,
                        "overlap": 0.0,
                    },
                )
                record["chunk_start"] = min(record["chunk_start"], unit.audio_start)
                record["chunk_end"] = max(record["chunk_end"], unit.audio_end)
                record["overlap"] += unit_duration

        owner_key = max(spans.items(), key=lambda item: (item[1]["overlap"], -item[0][1]))[0]
        ordered_keys = list(spans.keys())
        owner_position = ordered_keys.index(owner_key)
        left_guard_sec = 0.0
        right_guard_sec = 0.0
        owner_record = spans[owner_key]
        core_start = owner_record["chunk_start"]
        core_end = owner_record["chunk_end"]
        bindings = []
        for (chunk_id, chunk_index), record in spans.items():
            is_owner = (chunk_id, chunk_index) == owner_key
            position = ordered_keys.index((chunk_id, chunk_index))
            role = "core"
            if is_owner:
                role = "owner"
            elif position == 0 and position < owner_position:
                role = "left_guard"
                left_guard_sec = max(left_guard_sec, core_start - record["chunk_start"])
            elif position == len(ordered_keys) - 1 and position > owner_position:
                role = "right_guard"
                right_guard_sec = max(right_guard_sec, record["chunk_end"] - core_end)
            bindings.append(
                WindowChunkBinding(
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    chunk_start=record["chunk_start"],
                    chunk_end=record["chunk_end"],
                    overlap_ratio=min(1.0, record["overlap"] / total_duration),
                    role=role,
                    is_owner=is_owner,
                )
            )

        return WindowCoverage(
            core_segments=((core_start, core_end),),
            left_guard_sec=left_guard_sec,
            right_guard_sec=right_guard_sec,
            chunk_bindings=tuple(bindings),
        )
