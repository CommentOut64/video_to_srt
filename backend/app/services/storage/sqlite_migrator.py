"""
SQLite 迁移执行器。

设计模式：模板方法模式（Template Method Pattern）
原因：统一版本表维护与迁移生命周期，业务仓储仅提供迁移步骤。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import sqlite3


MigrationHandler = Callable[[sqlite3.Connection], None]


@dataclass(frozen=True)
class MigrationStep:
    """单个迁移步骤。"""

    version: int
    handler: MigrationHandler


class SQLiteMigrator:
    """轻量迁移器。"""

    METADATA_TABLE = "schema_metadata"

    def ensure_migrated(
        self,
        conn: sqlite3.Connection,
        domain: str,
        target_version: int,
        steps: Sequence[MigrationStep],
    ) -> None:
        """执行增量迁移直到目标版本。"""
        self._ensure_metadata_table(conn)
        current_version = self._get_current_version(conn, domain)
        effective_steps = sorted(
            [step for step in steps if current_version < step.version <= target_version],
            key=lambda step: step.version,
        )

        for step in effective_steps:
            step.handler(conn)
            self._set_current_version(conn, domain, step.version)

        if target_version > current_version and not effective_steps:
            self._set_current_version(conn, domain, target_version)

    def _ensure_metadata_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.METADATA_TABLE} (
                domain TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )

    def _get_current_version(self, conn: sqlite3.Connection, domain: str) -> int:
        row = conn.execute(
            f"SELECT version FROM {self.METADATA_TABLE} WHERE domain = ?",
            (domain,),
        ).fetchone()
        if row is None:
            return 0
        return int(row["version"])

    def _set_current_version(self, conn: sqlite3.Connection, domain: str, version: int) -> None:
        conn.execute(
            f"""
            INSERT INTO {self.METADATA_TABLE} (domain, version, updated_at)
            VALUES (?, ?, strftime('%s', 'now'))
            ON CONFLICT(domain) DO UPDATE SET
                version = excluded.version,
                updated_at = excluded.updated_at
            """,
            (domain, int(version)),
        )

