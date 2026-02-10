"""
Runtime Checkpoint 仓储。

Phase 1：补齐 unit journal / checkpoint 镜像读写接口，
为 PauseBarrier 与 runtime_state 优先恢复提供底层能力。
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

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
    def _create_v1_schema(conn: sqlite3.Connection) -> None:
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

    def append_unit_journal(
        self,
        *,
        stage: str,
        unit_id: str,
        status: str,
        payload_json: Optional[str] = None,
    ) -> None:
        """写入单元状态日志。"""
        with self.engine.connect() as conn:
            conn.execute(
                """
                INSERT INTO unit_journal (stage, unit_id, status, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (stage, unit_id, status, payload_json, time.time()),
            )

    def upsert_unit_commit(self, *, stage: str, last_unit_id: str) -> None:
        """更新阶段最近已提交单元。"""
        with self.engine.connect() as conn:
            conn.execute(
                """
                INSERT INTO unit_commits (stage, last_unit_id, committed_at)
                VALUES (?, ?, ?)
                ON CONFLICT(stage) DO UPDATE SET
                    last_unit_id = excluded.last_unit_id,
                    committed_at = excluded.committed_at
                """,
                (stage, last_unit_id, time.time()),
            )

    def get_last_unit_commit(self, stage: str) -> Optional[str]:
        """读取某阶段最后已提交单元。"""
        with self.engine.connect() as conn:
            row = conn.execute(
                "SELECT last_unit_id FROM unit_commits WHERE stage = ?",
                (stage,),
            ).fetchone()
        if not row:
            return None
        return str(row["last_unit_id"])

    def list_unit_commits(self) -> dict[str, str]:
        """读取所有阶段提交点。"""
        with self.engine.connect() as conn:
            rows = conn.execute("SELECT stage, last_unit_id FROM unit_commits").fetchall()
        return {str(row["stage"]): str(row["last_unit_id"]) for row in rows}

    def upsert_control_signal(self, *, key: str, value: Optional[str]) -> None:
        """写入控制信号键值。"""
        with self.engine.connect() as conn:
            conn.execute(
                """
                INSERT INTO control_signals (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, time.time()),
            )

    def get_control_signal(self, key: str) -> Optional[str]:
        """读取控制信号键值。"""
        with self.engine.connect() as conn:
            row = conn.execute(
                "SELECT value FROM control_signals WHERE key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        value = row["value"]
        return None if value is None else str(value)

    def has_any_state(self) -> bool:
        """判断 runtime_state 是否已有可恢复状态。"""
        with self.engine.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    (SELECT COUNT(1) FROM unit_journal) AS journal_count,
                    (SELECT COUNT(1) FROM unit_commits) AS commit_count
                """
            ).fetchone()
        if not row:
            return False
        return bool(row["journal_count"] or row["commit_count"])
