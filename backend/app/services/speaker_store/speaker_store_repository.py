"""
Speaker Store 仓储。

设计模式：Repository Pattern
原因：隔离 SQL 细节，保证 speaker_store.db 的结构真相由单点维护。
"""

from __future__ import annotations

import logging
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

