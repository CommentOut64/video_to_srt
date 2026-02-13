"""
Speaker Store 仓储。

设计模式：Repository Pattern
原因：隔离 SQL 细节，保证 speaker_store.db 的结构真相由单点维护。
"""

from __future__ import annotations

import logging
import time
from typing import Sequence
from pathlib import Path

from app.services.storage import MigrationStep, SQLiteEngine, SQLiteMigrator


class SpeakerStoreRepository:
    """任务级 speaker_store.db 仓储。"""

    DOMAIN = "speaker_store"
    SCHEMA_VERSION = 1

    def __init__(self, db_path: Path, logger: logging.Logger | None = None) -> None:
        self.db_path = Path(db_path)
        self.logger = logger or logging.getLogger(__name__)
        self.engine = SQLiteEngine(self.db_path)
        self.migrator = SQLiteMigrator()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.engine.connect() as conn:
            self.migrator.ensure_migrated(
                conn=conn,
                domain=self.DOMAIN,
                target_version=self.SCHEMA_VERSION,
                steps=[MigrationStep(version=1, handler=self._create_v1_schema)],
            )

    @staticmethod
    def _create_v1_schema(conn) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS speaker_profiles (
                speaker_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                color_key TEXT NOT NULL,
                status TEXT NOT NULL,
                is_locked INTEGER NOT NULL DEFAULT 0,
                sample_count INTEGER NOT NULL DEFAULT 0,
                quality_score REAL NOT NULL DEFAULT 0,
                centroid_blob BLOB,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS turn_speaker_links (
                turn_id TEXT PRIMARY KEY,
                speaker_id TEXT NOT NULL,
                block_id TEXT NOT NULL,
                start REAL NOT NULL,
                end REAL NOT NULL,
                boundary_confidence REAL NOT NULL,
                source TEXT NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (speaker_id) REFERENCES speaker_profiles(speaker_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS subtitle_speaker_links (
                sentence_index INTEGER PRIMARY KEY,
                turn_id TEXT,
                speaker_id TEXT NOT NULL,
                start REAL NOT NULL,
                end REAL NOT NULL,
                text_hash TEXT,
                binding_source TEXT NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (speaker_id) REFERENCES speaker_profiles(speaker_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS speaker_audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                speaker_id TEXT,
                sentence_index INTEGER,
                payload_json TEXT,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_turn_speaker
            ON turn_speaker_links (speaker_id, start)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_subtitle_speaker
            ON subtitle_speaker_links (speaker_id, start)
            """
        )

    def _ensure_speaker_profile_exists(self, *, speaker_id: str) -> None:
        """确保 speaker_profiles 存在对应主键，避免外键写入失败。"""
        now = time.time()
        with self.engine.connect() as conn:
            conn.execute(
                """
                INSERT INTO speaker_profiles (
                    speaker_id,
                    display_name,
                    color_key,
                    status,
                    is_locked,
                    sample_count,
                    quality_score,
                    centroid_blob,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, 0, 0, 0, NULL, ?, ?)
                ON CONFLICT(speaker_id) DO UPDATE SET
                    updated_at = excluded.updated_at
                """,
                (speaker_id, speaker_id, "speaker-default", "candidate", now, now),
            )

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
        """按 sentence_index 幂等写入字幕-说话人绑定。"""
        normalized_speaker_id = str(speaker_id or "unknown")
        self._ensure_speaker_profile_exists(speaker_id=normalized_speaker_id)
        now = time.time()
        with self.engine.connect() as conn:
            conn.execute(
                """
                INSERT INTO subtitle_speaker_links (
                    sentence_index,
                    turn_id,
                    speaker_id,
                    start,
                    end,
                    text_hash,
                    binding_source,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sentence_index) DO UPDATE SET
                    turn_id = excluded.turn_id,
                    speaker_id = excluded.speaker_id,
                    start = excluded.start,
                    end = excluded.end,
                    text_hash = excluded.text_hash,
                    binding_source = excluded.binding_source,
                    updated_at = excluded.updated_at
                """,
                (
                    int(sentence_index),
                    turn_id,
                    normalized_speaker_id,
                    float(start),
                    float(end),
                    text_hash,
                    str(binding_source or "l7"),
                    now,
                ),
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
        """按 turn_id 幂等写入 turn-说话人绑定。"""
        normalized_speaker_id = str(speaker_id or "unknown")
        self._ensure_speaker_profile_exists(speaker_id=normalized_speaker_id)
        now = time.time()
        with self.engine.connect() as conn:
            conn.execute(
                """
                INSERT INTO turn_speaker_links (
                    turn_id,
                    speaker_id,
                    block_id,
                    start,
                    end,
                    boundary_confidence,
                    source,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(turn_id) DO UPDATE SET
                    speaker_id = excluded.speaker_id,
                    block_id = excluded.block_id,
                    start = excluded.start,
                    end = excluded.end,
                    boundary_confidence = excluded.boundary_confidence,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (
                    str(turn_id),
                    normalized_speaker_id,
                    str(block_id),
                    float(start),
                    float(end),
                    float(boundary_confidence),
                    str(source or "unknown"),
                    now,
                ),
            )

    def upsert_turn_speaker_links(
        self,
        items: Sequence[dict[str, object]],
    ) -> None:
        """批量幂等写入 turn-说话人绑定。"""
        for item in items:
            self.upsert_turn_speaker_link(
                turn_id=str(item.get("turn_id") or ""),
                speaker_id=str(item.get("speaker_id") or "unknown"),
                block_id=str(item.get("block_id") or "unknown"),
                start=float(item.get("start") or 0.0),
                end=float(item.get("end") or 0.0),
                boundary_confidence=float(item.get("boundary_confidence") or 0.0),
                source=str(item.get("source") or "unknown"),
            )

    def upsert_subtitle_speaker_links(
        self,
        items: Sequence[dict[str, object]],
    ) -> None:
        """批量幂等写入字幕-说话人绑定。"""
        for item in items:
            self.upsert_subtitle_speaker_link(
                sentence_index=int(item["sentence_index"]),
                turn_id=(
                    None
                    if item.get("turn_id") is None
                    else str(item.get("turn_id"))
                ),
                speaker_id=str(item.get("speaker_id") or "unknown"),
                start=float(item.get("start") or 0.0),
                end=float(item.get("end") or 0.0),
                text_hash=(
                    None
                    if item.get("text_hash") is None
                    else str(item.get("text_hash"))
                ),
                binding_source=str(item.get("binding_source") or "l7"),
            )
