"""
Speaker Store 仓储。

设计模式：Repository Pattern
原因：隔离 SQL 细节，保证 speaker_store.db 的结构真相由单点维护。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Sequence
from pathlib import Path

from app.services.storage import MigrationStep, SQLiteEngine, SQLiteMigrator
from app.services.storage.sqlite_tx import immediate_transaction


class SpeakerStoreRepository:
    """任务级 speaker_store.db 仓储。"""

    DOMAIN = "speaker_store"
    SCHEMA_VERSION = 1
    DEFAULT_COLOR_KEYS: tuple[str, ...] = (
        "speaker-01",
        "speaker-02",
        "speaker-03",
        "speaker-04",
        "speaker-05",
        "speaker-06",
        "speaker-07",
        "speaker-08",
        "speaker-09",
        "speaker-10",
        "speaker-11",
        "speaker-12",
    )

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

    def _pick_default_color_key(self, speaker_id: str) -> str:
        """基于 speaker_id 稳定分配默认颜色键，避免全部落到同一颜色。"""
        normalized = str(speaker_id or "unknown")
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % len(self.DEFAULT_COLOR_KEYS)
        return self.DEFAULT_COLOR_KEYS[index]

    @staticmethod
    def _normalize_binding_source(raw_source: str | None) -> str:
        """绑定来源归一化：外部只暴露 auto/user 两种。"""
        source = str(raw_source or "").strip().lower()
        if source == "user":
            return "user"
        return "auto"

    def _ensure_speaker_profile_exists_on_conn(self, conn: Any, *, speaker_id: str) -> None:
        """在给定连接内确保 speaker_profiles 存在对应主键。"""
        normalized_speaker_id = str(speaker_id or "unknown")
        now = time.time()
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
            (
                normalized_speaker_id,
                normalized_speaker_id,
                self._pick_default_color_key(normalized_speaker_id),
                "candidate",
                now,
                now,
            ),
        )

    def _ensure_speaker_profile_exists(self, *, speaker_id: str) -> None:
        """确保 speaker_profiles 存在对应主键，避免外键写入失败。"""
        with self.engine.connect() as conn:
            self._ensure_speaker_profile_exists_on_conn(conn, speaker_id=speaker_id)

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
        normalized_binding_source = self._normalize_binding_source(binding_source)
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
                    normalized_binding_source,
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
                binding_source=str(item.get("binding_source") or "auto"),
            )

    def list_speaker_profiles(self) -> list[dict[str, object]]:
        """查询所有 speaker profile（含字幕计数）。"""
        with self.engine.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    sp.speaker_id,
                    sp.display_name,
                    sp.color_key,
                    sp.status,
                    sp.is_locked,
                    sp.sample_count,
                    sp.quality_score,
                    sp.created_at,
                    sp.updated_at,
                    COALESCE(link_counts.subtitle_count, 0) AS subtitle_count
                FROM speaker_profiles AS sp
                LEFT JOIN (
                    SELECT speaker_id, COUNT(*) AS subtitle_count
                    FROM subtitle_speaker_links
                    GROUP BY speaker_id
                ) AS link_counts
                ON link_counts.speaker_id = sp.speaker_id
                ORDER BY sp.created_at ASC, sp.speaker_id ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_speaker_profile(self, *, speaker_id: str) -> dict[str, object] | None:
        """查询单个 speaker profile。"""
        with self.engine.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    sp.speaker_id,
                    sp.display_name,
                    sp.color_key,
                    sp.status,
                    sp.is_locked,
                    sp.sample_count,
                    sp.quality_score,
                    sp.created_at,
                    sp.updated_at,
                    COALESCE(link_counts.subtitle_count, 0) AS subtitle_count
                FROM speaker_profiles AS sp
                LEFT JOIN (
                    SELECT speaker_id, COUNT(*) AS subtitle_count
                    FROM subtitle_speaker_links
                    GROUP BY speaker_id
                ) AS link_counts
                ON link_counts.speaker_id = sp.speaker_id
                WHERE sp.speaker_id = ?
                LIMIT 1
                """,
                (str(speaker_id),),
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def update_speaker_profile(
        self,
        *,
        speaker_id: str,
        display_name: str | None = None,
        color_key: str | None = None,
        is_locked: bool | None = None,
        status: str | None = None,
    ) -> dict[str, object] | None:
        """更新 speaker profile 可编辑字段。"""
        updates: dict[str, object] = {}
        if display_name is not None:
            stripped = str(display_name).strip()
            updates["display_name"] = stripped or str(speaker_id)
        if color_key is not None:
            stripped_color = str(color_key).strip()
            updates["color_key"] = stripped_color or self._pick_default_color_key(str(speaker_id))
        if is_locked is not None:
            updates["is_locked"] = 1 if bool(is_locked) else 0
        if status is not None:
            stripped_status = str(status).strip()
            if stripped_status:
                updates["status"] = stripped_status

        if not updates:
            return self.get_speaker_profile(speaker_id=speaker_id)

        updates["updated_at"] = time.time()
        set_sql = ", ".join(f"{key} = ?" for key in updates.keys())
        params = tuple(updates.values()) + (str(speaker_id),)

        with self.engine.connect() as conn:
            row = conn.execute(
                "SELECT speaker_id FROM speaker_profiles WHERE speaker_id = ? LIMIT 1",
                (str(speaker_id),),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                f"UPDATE speaker_profiles SET {set_sql} WHERE speaker_id = ?",
                params,
            )
        return self.get_speaker_profile(speaker_id=speaker_id)

    def list_subtitle_speaker_links(
        self,
        *,
        speaker_id: str | None = None,
        sentence_indices: Sequence[int] | None = None,
    ) -> list[dict[str, object]]:
        """查询字幕-说话人绑定，可按 speaker 或句子索引过滤。"""
        query = """
            SELECT
                ssl.sentence_index,
                ssl.turn_id,
                ssl.speaker_id,
                ssl.start,
                ssl.end,
                ssl.text_hash,
                ssl.binding_source,
                ssl.updated_at,
                COALESCE(sp.display_name, ssl.speaker_id) AS speaker_label,
                COALESCE(sp.color_key, ?) AS speaker_color_key,
                COALESCE(sp.is_locked, 0) AS speaker_is_locked
            FROM subtitle_speaker_links AS ssl
            LEFT JOIN speaker_profiles AS sp
            ON sp.speaker_id = ssl.speaker_id
            WHERE 1 = 1
        """
        params: list[object] = [self._pick_default_color_key("default")]
        if speaker_id is not None:
            query += " AND ssl.speaker_id = ?"
            params.append(str(speaker_id))
        if sentence_indices:
            placeholders = ", ".join("?" for _ in sentence_indices)
            query += f" AND ssl.sentence_index IN ({placeholders})"
            params.extend(int(index) for index in sentence_indices)
        query += " ORDER BY ssl.sentence_index ASC"

        with self.engine.connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def get_subtitle_speaker_link(self, *, sentence_index: int) -> dict[str, object] | None:
        """查询单条字幕 speaker 绑定。"""
        rows = self.list_subtitle_speaker_links(sentence_indices=[int(sentence_index)])
        if not rows:
            return None
        return rows[0]

    def rebind_subtitle_speaker(
        self,
        *,
        sentence_index: int,
        speaker_id: str,
        binding_source: str = "user",
    ) -> dict[str, object] | None:
        """将指定句子改绑到目标 speaker。"""
        normalized_speaker_id = str(speaker_id or "unknown")
        normalized_binding_source = self._normalize_binding_source(binding_source)
        with immediate_transaction(self.engine) as conn:
            existing = conn.execute(
                """
                SELECT sentence_index
                FROM subtitle_speaker_links
                WHERE sentence_index = ?
                LIMIT 1
                """,
                (int(sentence_index),),
            ).fetchone()
            if existing is None:
                return None
            self._ensure_speaker_profile_exists_on_conn(
                conn,
                speaker_id=normalized_speaker_id,
            )
            conn.execute(
                """
                UPDATE subtitle_speaker_links
                SET speaker_id = ?, binding_source = ?, updated_at = ?
                WHERE sentence_index = ?
                """,
                (
                    normalized_speaker_id,
                    normalized_binding_source,
                    time.time(),
                    int(sentence_index),
                ),
            )
        return self.get_subtitle_speaker_link(sentence_index=sentence_index)

    def merge_speakers(
        self,
        *,
        source_speaker_id: str,
        target_speaker_id: str,
    ) -> dict[str, object] | None:
        """将 source speaker 合并到 target speaker。"""
        normalized_source = str(source_speaker_id or "").strip()
        normalized_target = str(target_speaker_id or "").strip()
        if not normalized_source or not normalized_target:
            return None
        if normalized_source == normalized_target:
            target_profile = self.get_speaker_profile(speaker_id=normalized_target)
            if target_profile is None:
                return None
            return {
                "source_speaker_id": normalized_source,
                "target_speaker_id": normalized_target,
                "subtitle_rebind_count": 0,
                "turn_rebind_count": 0,
                "target_profile": target_profile,
            }

        with immediate_transaction(self.engine) as conn:
            source_row = conn.execute(
                "SELECT speaker_id FROM speaker_profiles WHERE speaker_id = ? LIMIT 1",
                (normalized_source,),
            ).fetchone()
            target_row = conn.execute(
                "SELECT speaker_id FROM speaker_profiles WHERE speaker_id = ? LIMIT 1",
                (normalized_target,),
            ).fetchone()
            if source_row is None or target_row is None:
                return None

            subtitle_cursor = conn.execute(
                """
                UPDATE subtitle_speaker_links
                SET speaker_id = ?, updated_at = ?
                WHERE speaker_id = ?
                """,
                (normalized_target, time.time(), normalized_source),
            )
            turn_cursor = conn.execute(
                """
                UPDATE turn_speaker_links
                SET speaker_id = ?, updated_at = ?
                WHERE speaker_id = ?
                """,
                (normalized_target, time.time(), normalized_source),
            )
            conn.execute(
                """
                UPDATE speaker_profiles
                SET status = ?, updated_at = ?
                WHERE speaker_id = ?
                """,
                ("merged", time.time(), normalized_source),
            )

            subtitle_rebind_count = int(subtitle_cursor.rowcount or 0)
            turn_rebind_count = int(turn_cursor.rowcount or 0)

        target_profile = self.get_speaker_profile(speaker_id=normalized_target)
        return {
            "source_speaker_id": normalized_source,
            "target_speaker_id": normalized_target,
            "subtitle_rebind_count": subtitle_rebind_count,
            "turn_rebind_count": turn_rebind_count,
            "target_profile": target_profile,
        }

    def append_audit_log(
        self,
        *,
        action: str,
        speaker_id: str | None = None,
        sentence_index: int | None = None,
        payload: dict[str, object] | str | None = None,
    ) -> int:
        """写入 speaker 审计日志。"""
        payload_json: str | None
        if payload is None:
            payload_json = None
        elif isinstance(payload, str):
            payload_json = payload
        else:
            payload_json = json.dumps(payload, ensure_ascii=False)
        with self.engine.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO speaker_audit_logs (
                    action,
                    speaker_id,
                    sentence_index,
                    payload_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(action),
                    None if speaker_id is None else str(speaker_id),
                    None if sentence_index is None else int(sentence_index),
                    payload_json,
                    time.time(),
                ),
            )
            return int(cursor.lastrowid)
