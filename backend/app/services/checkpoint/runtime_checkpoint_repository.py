"""
Runtime Checkpoint 仓储。

Phase 0 仅完成 runtime_state.db 基础建表，不切换现有恢复逻辑。
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.services.storage import MigrationStep, SQLiteEngine, SQLiteMigrator


class RuntimeCheckpointRepository:
    """运行时状态仓储（runtime_state.db）。"""

    DOMAIN = "runtime_checkpoint"
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
            CREATE TABLE IF NOT EXISTS unit_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stage TEXT NOT NULL,
                unit_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unit_commits (
                stage TEXT PRIMARY KEY,
                last_unit_id TEXT NOT NULL,
                committed_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS control_signals (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_unit_journal_stage_status
            ON unit_journal (stage, status)
            """
        )

