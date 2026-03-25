from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.streaming_subtitle import StreamingSubtitleManager


def test_restore_from_checkpoint_prefers_subtitle_items_snapshot_over_legacy_sentences_snapshot() -> None:
    manager = StreamingSubtitleManager("job-checkpoint-prefer-items")

    restored = manager.restore_from_checkpoint(
        {
            "subtitle_items_snapshot": [
                {
                    "segment_id": "chunk-5-seg-0",
                    "chunk_id": "chunk-5",
                    "start": 1.0,
                    "end": 2.0,
                    "text": "统一 DTO 文本",
                    "status": "final",
                    "source": "render_core",
                    "speaker_id": "spk-1",
                    "turn_id": "turn-1",
                    "trace": {"split_reason": "render_core"},
                    "legacy_index": 7,
                }
            ],
            "sentences_snapshot": [
                {
                    "_index": 7,
                    "text": "旧快照文本",
                    "start": 1.0,
                    "end": 2.0,
                    "source": "sensevoice",
                    "_is_draft": False,
                    "_is_finalized": True,
                }
            ],
            "sentence_count": 8,
            "chunk_sentences_map": {"chunk-5": [7]},
        }
    )

    assert restored is True
    assert manager.sentences[7].text == "统一 DTO 文本"
    assert manager.sentences[7].segment_id == "chunk-5-seg-0"
    assert manager.chunk_sentences[5] == [7]
