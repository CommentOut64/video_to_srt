"""
Speaker Store 服务层。

设计模式：Facade Pattern（门面模式）
原因：为 API/流水线提供稳定调用面，避免上层直接依赖 SQL 细节。
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

    def list_speaker_profiles(self) -> list[dict[str, object]]:
        """查询所有说话人 profile。"""
        return self.repository.list_speaker_profiles()

    def get_speaker_profile(self, *, speaker_id: str) -> dict[str, object] | None:
        """查询单个说话人 profile。"""
        return self.repository.get_speaker_profile(speaker_id=speaker_id)

    def update_speaker_profile(
        self,
        *,
        speaker_id: str,
        display_name: str | None = None,
        color_key: str | None = None,
        is_locked: bool | None = None,
        status: str | None = None,
    ) -> dict[str, object] | None:
        """更新说话人 profile。"""
        return self.repository.update_speaker_profile(
            speaker_id=speaker_id,
            display_name=display_name,
            color_key=color_key,
            is_locked=is_locked,
            status=status,
        )

    def list_subtitle_speaker_links(
        self,
        *,
        speaker_id: str | None = None,
        sentence_indices: Sequence[int] | None = None,
    ) -> list[dict[str, object]]:
        """查询字幕-说话人绑定。"""
        return self.repository.list_subtitle_speaker_links(
            speaker_id=speaker_id,
            sentence_indices=sentence_indices,
        )

    def get_subtitle_speaker_link(self, *, sentence_index: int) -> dict[str, object] | None:
        """查询单条字幕-说话人绑定。"""
        return self.repository.get_subtitle_speaker_link(sentence_index=sentence_index)

    def get_subtitle_speaker_map(
        self,
        *,
        sentence_indices: Sequence[int],
    ) -> dict[int, dict[str, object]]:
        """按句子索引批量查询 speaker 绑定，返回映射。"""
        links = self.list_subtitle_speaker_links(sentence_indices=sentence_indices)
        mapping: dict[int, dict[str, object]] = {}
        for row in links:
            mapping[int(row["sentence_index"])] = row
        return mapping

    def rebind_subtitle_speaker(
        self,
        *,
        sentence_index: int,
        speaker_id: str,
        binding_source: str = "user",
    ) -> dict[str, object] | None:
        """改绑句级 speaker。"""
        return self.repository.rebind_subtitle_speaker(
            sentence_index=sentence_index,
            speaker_id=speaker_id,
            binding_source=binding_source,
        )

    def merge_speakers(
        self,
        *,
        source_speaker_id: str,
        target_speaker_id: str,
    ) -> dict[str, object] | None:
        """合并说话人。"""
        return self.repository.merge_speakers(
            source_speaker_id=source_speaker_id,
            target_speaker_id=target_speaker_id,
        )

    def append_audit_log(
        self,
        *,
        action: str,
        speaker_id: str | None = None,
        sentence_index: int | None = None,
        payload: dict[str, object] | str | None = None,
    ) -> int:
        """写入审计日志。"""
        return self.repository.append_audit_log(
            action=action,
            speaker_id=speaker_id,
            sentence_index=sentence_index,
            payload=payload,
        )
