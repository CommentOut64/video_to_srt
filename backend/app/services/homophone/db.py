"""
同音索引数据库访问层。

V3.2.0+dev.20260210.02
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Literal, Optional


IndexStatus = Literal["building", "ready", "failed", "recovering"]


@dataclass(frozen=True)
class PostingRecord:
    job_id: str
    revision: int
    language: str
    chunk_index: int
    sentence_index: int
    token_index: int
    token_text: str
    reading_key: str
    reading_key_fuzzy: str
    reading_key_no_punct: str
    reading_key_fuzzy_no_punct: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class IndexState:
    job_id: str
    revision: int
    status: IndexStatus
    last_committed_chunk: int
    heartbeat_at: str
    updated_at: str


@dataclass(frozen=True)
class GlobalTermRule:
    language: str
    source_text: str
    target_text: str
    match_mode: str
    priority: int
    is_enabled: bool
    note: str


class HomophoneDb:
    """SQLite 持久化层。"""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        sql = schema_path.read_text(encoding="utf-8")
        with self.connect() as conn:
            conn.executescript(sql)

    def upsert_index_state(self, state: IndexState) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO homophone_index_state(
                    job_id, revision, status, last_committed_chunk, heartbeat_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id, revision) DO UPDATE SET
                    status=excluded.status,
                    last_committed_chunk=excluded.last_committed_chunk,
                    heartbeat_at=excluded.heartbeat_at,
                    updated_at=excluded.updated_at
                """,
                (
                    state.job_id,
                    state.revision,
                    state.status,
                    state.last_committed_chunk,
                    state.heartbeat_at,
                    state.updated_at,
                ),
            )

    def get_latest_index_state(self, job_id: str) -> Optional[IndexState]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT job_id, revision, status, last_committed_chunk, heartbeat_at, updated_at
                FROM homophone_index_state
                WHERE job_id=?
                ORDER BY revision DESC
                LIMIT 1
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return IndexState(
            job_id=str(row["job_id"]),
            revision=int(row["revision"]),
            status=str(row["status"]),
            last_committed_chunk=int(row["last_committed_chunk"]),
            heartbeat_at=str(row["heartbeat_at"]),
            updated_at=str(row["updated_at"]),
        )

    def replace_chunk_postings(self, records: Iterable[PostingRecord]) -> None:
        rows = list(records)
        if not rows:
            return
        head = rows[0]
        with self.connect() as conn:
            conn.execute(
                """
                DELETE FROM homophone_postings
                WHERE job_id=? AND revision=? AND chunk_index=?
                """,
                (head.job_id, head.revision, head.chunk_index),
            )
            conn.executemany(
                """
                INSERT INTO homophone_postings(
                    job_id, revision, language, chunk_index, sentence_index, token_index,
                    token_text, reading_key, reading_key_fuzzy,
                    reading_key_no_punct, reading_key_fuzzy_no_punct,
                    char_start, char_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        rec.job_id,
                        rec.revision,
                        rec.language,
                        rec.chunk_index,
                        rec.sentence_index,
                        rec.token_index,
                        rec.token_text,
                        rec.reading_key,
                        rec.reading_key_fuzzy,
                        rec.reading_key_no_punct,
                        rec.reading_key_fuzzy_no_punct,
                        rec.char_start,
                        rec.char_end,
                    )
                    for rec in rows
                ],
            )

    def query_postings(
        self,
        *,
        job_id: str,
        revision: int,
        language: str,
        key_value: str,
        is_fuzzy: bool,
        is_ignore_punctuation: bool,
        limit: int,
    ) -> List[sqlite3.Row]:
        if is_ignore_punctuation:
            key_column = "reading_key_fuzzy_no_punct" if is_fuzzy else "reading_key_no_punct"
        else:
            key_column = "reading_key_fuzzy" if is_fuzzy else "reading_key"
        sql = f"""
            SELECT sentence_index, token_index, token_text, char_start, char_end, reading_key
            FROM homophone_postings
            WHERE job_id=? AND revision=? AND language=? AND {key_column}=?
            ORDER BY sentence_index ASC, token_index ASC
            LIMIT ?
        """
        with self.connect() as conn:
            return list(conn.execute(sql, (job_id, revision, language, key_value, limit)).fetchall())

    def query_postings_by_prefix(
        self,
        *,
        job_id: str,
        revision: int,
        language: str,
        key_prefix: str,
        is_fuzzy: bool,
        is_ignore_punctuation: bool,
        limit: int,
    ) -> List[sqlite3.Row]:
        """按读音前缀召回 postings（用于 fuzzy 场景的近音扩召）。"""
        if not key_prefix:
            return []
        if is_ignore_punctuation:
            key_column = "reading_key_fuzzy_no_punct" if is_fuzzy else "reading_key_no_punct"
        else:
            key_column = "reading_key_fuzzy" if is_fuzzy else "reading_key"
        sql = f"""
            SELECT sentence_index, token_index, token_text, char_start, char_end, reading_key
            FROM homophone_postings
            WHERE job_id=? AND revision=? AND language=? AND {key_column} LIKE ?
            ORDER BY sentence_index ASC, token_index ASC
            LIMIT ?
        """
        with self.connect() as conn:
            return list(
                conn.execute(
                    sql,
                    (job_id, revision, language, f"{key_prefix}%", limit),
                ).fetchall()
            )

    def list_sentence_postings(
        self,
        *,
        job_id: str,
        revision: int,
        language: str,
        sentence_indices: List[int],
    ) -> List[sqlite3.Row]:
        """批量读取句子级 postings（用于词级序列匹配）。"""
        if not sentence_indices:
            return []
        placeholders = ",".join("?" for _ in sentence_indices)
        sql = f"""
            SELECT
                sentence_index,
                token_index,
                token_text,
                char_start,
                char_end,
                reading_key,
                reading_key_fuzzy,
                reading_key_no_punct,
                reading_key_fuzzy_no_punct
            FROM homophone_postings
            WHERE job_id=? AND revision=? AND language=?
              AND sentence_index IN ({placeholders})
            ORDER BY sentence_index ASC, token_index ASC
        """
        params: List[object] = [job_id, revision, language]
        params.extend(sentence_indices)
        with self.connect() as conn:
            return list(conn.execute(sql, tuple(params)).fetchall())

    def get_sentence_indices_by_chunk(self, *, job_id: str, revision: int, chunk_index: int) -> List[int]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT sentence_index
                FROM homophone_postings
                WHERE job_id=? AND revision=? AND chunk_index=?
                ORDER BY sentence_index ASC
                """,
                (job_id, revision, chunk_index),
            ).fetchall()
        return [int(row["sentence_index"]) for row in rows]

    def replace_global_terms(self, items: List[GlobalTermRule], now_iso: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM global_term_replacements")
            conn.executemany(
                """
                INSERT INTO global_term_replacements(
                    language, source_text, target_text, match_mode,
                    priority, is_enabled, note, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.language,
                        item.source_text,
                        item.target_text,
                        item.match_mode,
                        item.priority,
                        1 if item.is_enabled else 0,
                        item.note,
                        now_iso,
                        now_iso,
                    )
                    for item in items
                ],
            )

    def list_global_terms(self) -> List[GlobalTermRule]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT language, source_text, target_text, match_mode, priority, is_enabled, note
                FROM global_term_replacements
                ORDER BY priority ASC, LENGTH(source_text) DESC, id ASC
                """
            ).fetchall()
        return [
            GlobalTermRule(
                language=str(row["language"]),
                source_text=str(row["source_text"]),
                target_text=str(row["target_text"]),
                match_mode=str(row["match_mode"]),
                priority=int(row["priority"]),
                is_enabled=bool(int(row["is_enabled"])),
                note=str(row["note"] or ""),
            )
            for row in rows
        ]
