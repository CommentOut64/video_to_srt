"""
Speaker Store 服务层。

Phase 0 只提供仓储初始化封装，后续 phase 再扩展读写业务。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from app.services.speaker_store.speaker_store_repository import SpeakerStoreRepository


class SpeakerStoreService:
    """Speaker Store 领域服务。"""

    def __init__(self, job_dir: Path, db_file_name: str = "speaker_store.db") -> None:
        self.job_dir = Path(job_dir)
        self.db_path = self.job_dir / db_file_name
        self.repository = SpeakerStoreRepository(self.db_path)

    def upsert_subtitle_speaker_link(
        self,
        *,
        sentence_index: int,
        turn_id: str | None,
        speaker_id: str,
        start: float,
        end: float,
        text_hash: str | None,
        binding_source: str,
    ) -> None:
        """写入单条字幕-说话人绑定。"""
        self.repository.upsert_subtitle_speaker_link(
            sentence_index=sentence_index,
            turn_id=turn_id,
            speaker_id=speaker_id,
            start=start,
            end=end,
            text_hash=text_hash,
            binding_source=binding_source,
        )

    def upsert_turn_speaker_link(
        self,
        *,
        turn_id: str,
        speaker_id: str,
        block_id: str,
        start: float,
        end: float,
        boundary_confidence: float,
        source: str,
    ) -> None:
        """写入单条 turn-说话人绑定。"""
        self.repository.upsert_turn_speaker_link(
            turn_id=turn_id,
            speaker_id=speaker_id,
            block_id=block_id,
            start=start,
            end=end,
            boundary_confidence=boundary_confidence,
            source=source,
        )

    def upsert_turn_speaker_links(self, items: Sequence[dict[str, object]]) -> None:
        """批量写入 turn-说话人绑定。"""
        self.repository.upsert_turn_speaker_links(items)

    def upsert_subtitle_speaker_links(self, items: Sequence[dict[str, object]]) -> None:
        """批量写入字幕-说话人绑定。"""
        self.repository.upsert_subtitle_speaker_links(items)
