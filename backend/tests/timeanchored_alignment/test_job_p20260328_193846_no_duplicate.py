from __future__ import annotations

from app.services.streaming_subtitle import StreamingSubtitleManager
from app.services.textflow.contracts import SubtitleBatch, SubtitleItem


def test_job_p20260328_193846_window_group_replay_does_not_duplicate_latest_scope_output() -> None:
    manager = StreamingSubtitleManager("job-p20260328-193846")

    batch = SubtitleBatch(
        chunk_id="ow-window-193846",
        chunk_index=None,
        items=(
            SubtitleItem(
                segment_id="window-193846:seg:0",
                chunk_id="ow-window-193846",
                start=193.846,
                end=195.200,
                text="同一窗口重复回放也不能累积重复字幕",
                source="render_core",
            ),
        ),
        diagnostics={
            "projection": {
                "projection_mode": "window_group",
                "window_id": "window-193846",
                "replace_scope_chunk_ids": ["193", "194"],
                "generation_id": "window-193846:0000000000000002:latest",
            }
        },
    )
    stale_batch = SubtitleBatch(
        chunk_id="ow-window-193846",
        chunk_index=None,
        items=(
            SubtitleItem(
                segment_id="window-193846:seg:0",
                chunk_id="ow-window-193846",
                start=193.846,
                end=195.200,
                text="旧结果不应覆盖新结果",
                source="render_core",
            ),
        ),
        diagnostics={
            "projection": {
                "projection_mode": "window_group",
                "window_id": "window-193846",
                "replace_scope_chunk_ids": ["193", "194"],
                "generation_id": "window-193846:0000000000000001:stale",
            }
        },
    )

    manager.replace_chunk_batch(batch)
    manager.replace_chunk_batch(batch)
    manager.replace_chunk_batch(stale_batch)

    snapshot = manager.to_checkpoint_data()
    items = [
        item
        for item in snapshot["subtitle_items_snapshot"]
        if item["chunk_id"] == "ow-window-193846"
    ]

    assert len(items) == 1
    assert items[0]["text"] == "同一窗口重复回放也不能累积重复字幕"
