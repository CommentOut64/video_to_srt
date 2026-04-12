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
    project_id: str
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
    project_id: str
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
            self._migrate_legacy_job_id_schema(conn)

    @staticmethod
    def _table_has_column(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return any(str(row["name"]) == column_name for row in rows)

    def _migrate_legacy_job_id_schema(self, conn: sqlite3.Connection) -> None:
        """兼容旧版 job_id 列，迁移为 project_id 列。"""
        if self._table_has_column(conn, "homophone_index_state", "job_id") and not self._table_has_column(
            conn, "homophone_index_state", "project_id"
        ):
            conn.execute("ALTER TABLE homophone_index_state RENAME TO homophone_index_state_legacy")
            conn.execute(
                """
                CREATE TABLE homophone_index_state (
                  project_id TEXT NOT NULL,
                  revision INTEGER NOT NULL,
                  status TEXT NOT NULL,
                  last_committed_chunk INTEGER NOT NULL DEFAULT -1,
                  heartbeat_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY (project_id, revision)
                )
                """
            )
            conn.execute(
                """
                INSERT INTO homophone_index_state(
                  project_id, revision, status, last_committed_chunk, heartbeat_at, updated_at
                )
                SELECT job_id, revision, status, last_committed_chunk, heartbeat_at, updated_at
                FROM homophone_index_state_legacy
                """
            )
            conn.execute("DROP TABLE homophone_index_state_legacy")

        if self._table_has_column(conn, "homophone_postings", "job_id") and not self._table_has_column(
            conn, "homophone_postings", "project_id"
        ):
            conn.execute("DROP INDEX IF EXISTS idx_posting_lookup_strict")
            conn.execute("DROP INDEX IF EXISTS idx_posting_lookup_fuzzy")
            conn.execute("DROP INDEX IF EXISTS idx_posting_lookup_strict_no_punct")
            conn.execute("DROP INDEX IF EXISTS idx_posting_lookup_fuzzy_no_punct")
            conn.execute("ALTER TABLE homophone_postings RENAME TO homophone_postings_legacy")
            conn.execute(
                """
                CREATE TABLE homophone_postings (
                  project_id TEXT NOT NULL,
                  revision INTEGER NOT NULL,
                  language TEXT NOT NULL,
                  chunk_index INTEGER NOT NULL,
                  sentence_index INTEGER NOT NULL,
                  token_index INTEGER NOT NULL,
                  token_text TEXT NOT NULL,
                  reading_key TEXT NOT NULL,
                  reading_key_fuzzy TEXT NOT NULL,
                  reading_key_no_punct TEXT NOT NULL,
                  reading_key_fuzzy_no_punct TEXT NOT NULL,
                  char_start INTEGER NOT NULL,
                  char_end INTEGER NOT NULL,
                  PRIMARY KEY (
                    project_id, revision, sentence_index, token_index
                  )
                )
                """
            )
            conn.execute(
                """
                INSERT INTO homophone_postings(
                  project_id, revision, language, chunk_index, sentence_index, token_index,
                  token_text, reading_key, reading_key_fuzzy,
                  reading_key_no_punct, reading_key_fuzzy_no_punct,
                  char_start, char_end
                )
                SELECT
                  job_id, revision, language, chunk_index, sentence_index, token_index,
                  token_text, reading_key, reading_key_fuzzy,
                  reading_key_no_punct, reading_key_fuzzy_no_punct,
                  char_start, char_end
                FROM homophone_postings_legacy
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_posting_lookup_strict
                ON homophone_postings(project_id, revision, language, reading_key, sentence_index, token_index)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_posting_lookup_fuzzy
                ON homophone_postings(project_id, revision, language, reading_key_fuzzy, sentence_index, token_index)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_posting_lookup_strict_no_punct
                ON homophone_postings(project_id, revision, language, reading_key_no_punct, sentence_index, token_index)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_posting_lookup_fuzzy_no_punct
                ON homophone_postings(project_id, revision, language, reading_key_fuzzy_no_punct, sentence_index, token_index)
                """
            )
            conn.execute("DROP TABLE homophone_postings_legacy")

        if self._table_has_column(conn, "sentence_revision", "job_id") and not self._table_has_column(
            conn, "sentence_revision", "project_id"
        ):
            conn.execute("ALTER TABLE sentence_revision RENAME TO sentence_revision_legacy")
            conn.execute(
                """
                CREATE TABLE sentence_revision (
                  project_id TEXT NOT NULL,
                  sentence_index INTEGER NOT NULL,
                  revision INTEGER NOT NULL,
                  text_hash TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY (project_id, sentence_index)
                )
                """
            )
            conn.execute(
                """
                INSERT INTO sentence_revision(project_id, sentence_index, revision, text_hash, updated_at)
                SELECT job_id, sentence_index, revision, text_hash, updated_at
                FROM sentence_revision_legacy
                """
            )
            conn.execute("DROP TABLE sentence_revision_legacy")

    def upsert_index_state(self, state: IndexState) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO homophone_index_state(
                    project_id, revision, status, last_committed_chunk, heartbeat_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, revision) DO UPDATE SET
                    status=excluded.status,
                    last_committed_chunk=excluded.last_committed_chunk,
                    heartbeat_at=excluded.heartbeat_at,
                    updated_at=excluded.updated_at
                """,
                (
                    state.project_id,
                    state.revision,
                    state.status,
                    state.last_committed_chunk,
                    state.heartbeat_at,
                    state.updated_at,
                ),
            )

    def get_latest_index_state(self, project_id: str) -> Optional[IndexState]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT project_id, revision, status, last_committed_chunk, heartbeat_at, updated_at
                FROM homophone_index_state
                WHERE project_id=?
                ORDER BY revision DESC
                LIMIT 1
                """,
                (project_id,),
            ).fetchone()
        if row is None:
            return None
        return IndexState(
            project_id=str(row["project_id"]),
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
                WHERE project_id=? AND revision=? AND chunk_index=?
                """,
                (head.project_id, head.revision, head.chunk_index),
            )
            conn.executemany(
                """
                INSERT INTO homophone_postings(
                    project_id, revision, language, chunk_index, sentence_index, token_index,
                    token_text, reading_key, reading_key_fuzzy,
                    reading_key_no_punct, reading_key_fuzzy_no_punct,
                    char_start, char_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        rec.project_id,
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

    def replace_sentence_postings(
        self,
        *,
        project_id: str,
        revision: int,
        sentence_index: int,
        records: Iterable[PostingRecord],
    ) -> None:
        """按句替换 postings，用于编辑后的增量同步。"""
        rows = list(records)
        with self.connect() as conn:
            conn.execute(
                """
                DELETE FROM homophone_postings
                WHERE project_id=? AND revision=? AND sentence_index=?
                """,
                (project_id, revision, sentence_index),
            )
            if not rows:
                return
            conn.executemany(
                """
                INSERT INTO homophone_postings(
                    project_id, revision, language, chunk_index, sentence_index, token_index,
                    token_text, reading_key, reading_key_fuzzy,
                    reading_key_no_punct, reading_key_fuzzy_no_punct,
                    char_start, char_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        rec.project_id,
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

    def delete_sentence_postings(
        self,
        *,
        project_id: str,
        revision: int,
        sentence_indices: Iterable[int],
    ) -> None:
        """删除指定句子的 postings，用于删除字幕后的增量同步。"""
        normalized_indices = sorted({int(item) for item in sentence_indices})
        if not normalized_indices:
            return
        placeholders = ",".join("?" for _ in normalized_indices)
        params: List[object] = [project_id, revision]
        params.extend(normalized_indices)
        sql = f"""
            DELETE FROM homophone_postings
            WHERE project_id=? AND revision=? AND sentence_index IN ({placeholders})
        """
        with self.connect() as conn:
            conn.execute(sql, tuple(params))

    def query_postings(
        self,
        *,
        project_id: str,
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
            WHERE project_id=? AND revision=? AND language=? AND {key_column}=?
            ORDER BY sentence_index ASC, token_index ASC
            LIMIT ?
        """
        with self.connect() as conn:
            return list(conn.execute(sql, (project_id, revision, language, key_value, limit)).fetchall())

    def query_postings_by_prefix(
        self,
        *,
        project_id: str,
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
            WHERE project_id=? AND revision=? AND language=? AND {key_column} LIKE ?
            ORDER BY sentence_index ASC, token_index ASC
            LIMIT ?
        """
        with self.connect() as conn:
            return list(
                conn.execute(
                    sql,
                    (project_id, revision, language, f"{key_prefix}%", limit),
                ).fetchall()
            )

    def list_sentence_postings(
        self,
        *,
        project_id: str,
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
            WHERE project_id=? AND revision=? AND language=?
              AND sentence_index IN ({placeholders})
            ORDER BY sentence_index ASC, token_index ASC
        """
        params: List[object] = [project_id, revision, language]
        params.extend(sentence_indices)
        with self.connect() as conn:
            return list(conn.execute(sql, tuple(params)).fetchall())

    def get_sentence_indices_by_chunk(self, *, project_id: str, revision: int, chunk_index: int) -> List[int]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT sentence_index
                FROM homophone_postings
                WHERE project_id=? AND revision=? AND chunk_index=?
                ORDER BY sentence_index ASC
                """,
                (project_id, revision, chunk_index),
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
