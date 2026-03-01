"""
SQLite 事务上下文。

设计模式：上下文管理器模式（Context Manager Pattern）
原因：统一事务边界与异常回滚策略，降低仓储层重复代码。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import sqlite3

from app.services.storage.sqlite_engine import SQLiteEngine


@contextmanager
def immediate_transaction(engine: SQLiteEngine) -> Iterator[sqlite3.Connection]:
    """以 BEGIN IMMEDIATE 语义执行事务。"""
    conn = engine.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

