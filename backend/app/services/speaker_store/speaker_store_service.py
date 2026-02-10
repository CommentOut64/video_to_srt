"""
Speaker Store 服务层。

Phase 0 只提供仓储初始化封装，后续 phase 再扩展读写业务。
"""

from __future__ import annotations

from pathlib import Path

from app.services.speaker_store.speaker_store_repository import SpeakerStoreRepository


class SpeakerStoreService:
    """Speaker Store 领域服务。"""

    def __init__(self, job_dir: Path, db_file_name: str = "speaker_store.db") -> None:
        self.job_dir = Path(job_dir)
        self.db_path = self.job_dir / db_file_name
        self.repository = SpeakerStoreRepository(self.db_path)

