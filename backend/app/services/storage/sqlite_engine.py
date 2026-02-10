"""
SQLite 统一连接工厂。

设计模式：工厂模式（Factory Pattern）
原因：集中管理连接级 PRAGMA，避免各仓储重复配置与行为漂移。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SQLitePragmaConfig:
    """SQLite 基础运行参数。"""

    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    busy_timeout_ms: int = 5000
    foreign_keys: bool = True


class SQLiteEngine:
    """SQLite 连接工厂。"""

    def __init__(self, db_path: Path, pragma: SQLitePragmaConfig | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.pragma = pragma or SQLitePragmaConfig()

    def connect(self) -> sqlite3.Connection:
        """创建并初始化连接。"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        if self.pragma.foreign_keys:
            conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(f"PRAGMA journal_mode = {self.pragma.journal_mode}")
        conn.execute(f"PRAGMA synchronous = {self.pragma.synchronous}")
        conn.execute(f"PRAGMA busy_timeout = {int(self.pragma.busy_timeout_ms)}")
        return conn

