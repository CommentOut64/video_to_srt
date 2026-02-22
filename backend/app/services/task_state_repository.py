"""
任务状态仓库

使用 Repository 模式集中封装 SQLite 读写，确保状态来源唯一且可测试。
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.models.job_models import JobSettings, JobState


@dataclass(frozen=True)
class QueueState:
    """队列状态快照"""
    queue: List[str]
    running_job_id: Optional[str]
    interrupted_job_id: Optional[str]
    updated_at: Optional[float]


class TaskStateRepository:
    """
    任务状态仓库（Repository 模式）

    统一封装任务状态/队列/事件/检查点/心跳的 SQLite 访问。
    """

    def __init__(self, db_path: Path, logger: Optional[logging.Logger] = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def transaction(self) -> Iterable[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    job_id TEXT PRIMARY KEY,
                    filename TEXT,
                    title TEXT,
                    dir TEXT,
                    input_path TEXT,
                    status TEXT,
                    state_seq INTEGER DEFAULT 0,
                    phase TEXT,
                    progress REAL,
                    phase_percent REAL,
                    message TEXT,
                    error TEXT,
                    processed INTEGER,
                    total INTEGER,
                    language TEXT,
                    srt_path TEXT,
                    canceled INTEGER,
                    paused INTEGER,
                    subtitle_time_offset REAL,
                    settings_json TEXT,
                    updated_at REAL,
                    created_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT,
                    event_type TEXT,
                    from_status TEXT,
                    to_status TEXT,
                    reason TEXT,
                    state_seq INTEGER DEFAULT 0,
                    payload_json TEXT,
                    created_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS queue_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    queue_json TEXT,
                    running_job_id TEXT,
                    interrupted_job_id TEXT,
                    updated_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    job_id TEXT PRIMARY KEY,
                    summary_json TEXT,
                    file_path TEXT,
                    checksum TEXT,
                    updated_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_heartbeats (
                    job_id TEXT PRIMARY KEY,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    last_heartbeat REAL,
                    updated_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            self._ensure_tasks_column(conn, "subtitle_time_offset", "REAL")
            self._ensure_tasks_column(conn, "state_seq", "INTEGER DEFAULT 0")
            self._ensure_task_events_column(conn, "state_seq", "INTEGER DEFAULT 0")

    def _ensure_tasks_column(self, conn: sqlite3.Connection, name: str, column_type: str) -> None:
        cursor = conn.execute("PRAGMA table_info(tasks)")
        columns = {row["name"] for row in cursor.fetchall()}
        if name in columns:
            return
        conn.execute(f"ALTER TABLE tasks ADD COLUMN {name} {column_type}")

    def _ensure_task_events_column(self, conn: sqlite3.Connection, name: str, column_type: str) -> None:
        cursor = conn.execute("PRAGMA table_info(task_events)")
        columns = {row["name"] for row in cursor.fetchall()}
        if name in columns:
            return
        conn.execute(f"ALTER TABLE task_events ADD COLUMN {name} {column_type}")

    def upsert_task(self, job: JobState, conn: Optional[sqlite3.Connection] = None) -> None:
        payload = self._job_to_row(job)
        target_conn = conn or self._connect()
        try:
            target_conn.execute(
                """
                INSERT INTO tasks (
                    job_id, filename, title, dir, input_path, status, state_seq, phase, progress,
                    phase_percent, message, error, processed, total, language, srt_path,
                    canceled, paused, subtitle_time_offset, settings_json, updated_at, created_at
                ) VALUES (
                    :job_id, :filename, :title, :dir, :input_path, :status, :state_seq, :phase, :progress,
                    :phase_percent, :message, :error, :processed, :total, :language, :srt_path,
                    :canceled, :paused, :subtitle_time_offset, :settings_json, :updated_at, :created_at
                )
                ON CONFLICT(job_id) DO UPDATE SET
                    filename=excluded.filename,
                    title=excluded.title,
                    dir=excluded.dir,
                    input_path=excluded.input_path,
                    status=excluded.status,
                    state_seq=excluded.state_seq,
                    phase=excluded.phase,
                    progress=excluded.progress,
                    phase_percent=excluded.phase_percent,
                    message=excluded.message,
                    error=excluded.error,
                    processed=excluded.processed,
                    total=excluded.total,
                    language=excluded.language,
                    srt_path=excluded.srt_path,
                    canceled=excluded.canceled,
                    paused=excluded.paused,
                    subtitle_time_offset=excluded.subtitle_time_offset,
                    settings_json=excluded.settings_json,
                    updated_at=excluded.updated_at,
                    created_at=COALESCE(tasks.created_at, excluded.created_at)
                """,
                payload,
            )
            if conn is None:
                target_conn.commit()
        finally:
            if conn is None:
                target_conn.close()

    def get_task(self, job_id: str) -> Optional[JobState]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_job(row)

    def list_tasks(self, statuses: Optional[List[str]] = None) -> List[JobState]:
        query = "SELECT * FROM tasks"
        params: Tuple[Any, ...] = ()
        if statuses:
            placeholders = ",".join(["?"] * len(statuses))
            query += f" WHERE status IN ({placeholders})"
            params = tuple(statuses)
        query += " ORDER BY updated_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_job(row) for row in rows]

    def delete_task(self, job_id: str, conn: Optional[sqlite3.Connection] = None) -> None:
        target_conn = conn or self._connect()
        try:
            target_conn.execute("DELETE FROM tasks WHERE job_id = ?", (job_id,))
            if conn is None:
                target_conn.commit()
        finally:
            if conn is None:
                target_conn.close()

    def save_queue_state(
        self,
        queue: List[str],
        running_job_id: Optional[str],
        interrupted_job_id: Optional[str],
        conn: Optional[sqlite3.Connection] = None
    ) -> None:
        payload = {
            "queue_json": json.dumps(queue),
            "running_job_id": running_job_id,
            "interrupted_job_id": interrupted_job_id,
            "updated_at": time.time(),
        }
        target_conn = conn or self._connect()
        try:
            target_conn.execute(
                """
                INSERT INTO queue_state (id, queue_json, running_job_id, interrupted_job_id, updated_at)
                VALUES (1, :queue_json, :running_job_id, :interrupted_job_id, :updated_at)
                ON CONFLICT(id) DO UPDATE SET
                    queue_json=excluded.queue_json,
                    running_job_id=excluded.running_job_id,
                    interrupted_job_id=excluded.interrupted_job_id,
                    updated_at=excluded.updated_at
                """,
                payload,
            )
            if conn is None:
                target_conn.commit()
        finally:
            if conn is None:
                target_conn.close()

    def load_queue_state(self) -> Optional[QueueState]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT queue_json, running_job_id, interrupted_job_id, updated_at FROM queue_state WHERE id = 1"
            ).fetchone()
        if not row:
            return None
        queue = json.loads(row["queue_json"]) if row["queue_json"] else []
        return QueueState(
            queue=queue,
            running_job_id=row["running_job_id"],
            interrupted_job_id=row["interrupted_job_id"],
            updated_at=row["updated_at"],
        )

    def save_checkpoint_summary(
        self,
        job_id: str,
        summary: Dict[str, Any],
        file_path: Optional[str],
        checksum: Optional[str],
        conn: Optional[sqlite3.Connection] = None
    ) -> None:
        payload = {
            "job_id": job_id,
            "summary_json": json.dumps(summary),
            "file_path": file_path,
            "checksum": checksum,
            "updated_at": time.time(),
        }
        target_conn = conn or self._connect()
        try:
            target_conn.execute(
                """
                INSERT INTO checkpoints (job_id, summary_json, file_path, checksum, updated_at)
                VALUES (:job_id, :summary_json, :file_path, :checksum, :updated_at)
                ON CONFLICT(job_id) DO UPDATE SET
                    summary_json=excluded.summary_json,
                    file_path=excluded.file_path,
                    checksum=excluded.checksum,
                    updated_at=excluded.updated_at
                """,
                payload,
            )
            if conn is None:
                target_conn.commit()
        finally:
            if conn is None:
                target_conn.close()

    def get_checkpoint_summary(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary_json, file_path, checksum, updated_at FROM checkpoints WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if not row:
            return None
        summary = json.loads(row["summary_json"]) if row["summary_json"] else {}
        summary["_file_path"] = row["file_path"]
        summary["_checksum"] = row["checksum"]
        summary["_updated_at"] = row["updated_at"]
        return summary

    def record_event(
        self,
        job_id: str,
        event_type: str,
        from_status: Optional[str],
        to_status: Optional[str],
        reason: Optional[str],
        state_seq: int = 0,
        payload: Optional[Dict[str, Any]] = None,
        conn: Optional[sqlite3.Connection] = None
    ) -> None:
        payload_json = json.dumps(payload or {})
        target_conn = conn or self._connect()
        try:
            target_conn.execute(
                """
                INSERT INTO task_events (
                    job_id, event_type, from_status, to_status, reason, state_seq, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    event_type,
                    from_status,
                    to_status,
                    reason,
                    max(0, int(state_seq or 0)),
                    payload_json,
                    time.time(),
                ),
            )
            if conn is None:
                target_conn.commit()
        finally:
            if conn is None:
                target_conn.close()

    def list_events(self, job_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_type, from_status, to_status, reason, state_seq, payload_json, created_at
                FROM task_events
                WHERE job_id = ?
                ORDER BY id ASC
                """,
                (job_id,),
            ).fetchall()
        events = []
        for row in rows:
            payload = json.loads(row["payload_json"]) if row["payload_json"] else {}
            events.append(
                {
                    "event_type": row["event_type"],
                    "from_status": row["from_status"],
                    "to_status": row["to_status"],
                    "reason": row["reason"],
                    "state_seq": row["state_seq"] or 0,
                    "payload": payload,
                    "created_at": row["created_at"],
                }
            )
        return events

    def acquire_lease(
        self,
        job_id: str,
        lease_owner: str,
        ttl_seconds: float,
        conn: Optional[sqlite3.Connection] = None
    ) -> bool:
        now = time.time()
        lease_expires_at = now + ttl_seconds
        target_conn = conn or self._connect()
        try:
            row = target_conn.execute(
                "SELECT lease_owner, lease_expires_at FROM task_heartbeats WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row:
                existing_expires_at = row["lease_expires_at"]
                existing_owner = row["lease_owner"]
                is_active = existing_expires_at is not None and existing_expires_at > now
                if is_active and existing_owner and existing_owner != lease_owner:
                    return False
            target_conn.execute(
                """
                INSERT INTO task_heartbeats (job_id, lease_owner, lease_expires_at, last_heartbeat, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    lease_owner=excluded.lease_owner,
                    lease_expires_at=excluded.lease_expires_at,
                    last_heartbeat=excluded.last_heartbeat,
                    updated_at=excluded.updated_at
                """,
                (job_id, lease_owner, lease_expires_at, now, now),
            )
            if conn is None:
                target_conn.commit()
            return True
        finally:
            if conn is None:
                target_conn.close()

    def refresh_heartbeat(
        self,
        job_id: str,
        lease_owner: str,
        ttl_seconds: float,
        conn: Optional[sqlite3.Connection] = None
    ) -> bool:
        now = time.time()
        lease_expires_at = now + ttl_seconds
        target_conn = conn or self._connect()
        try:
            row = target_conn.execute(
                "SELECT lease_owner FROM task_heartbeats WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if not row or row["lease_owner"] != lease_owner:
                return False
            target_conn.execute(
                """
                UPDATE task_heartbeats
                SET lease_expires_at = ?, last_heartbeat = ?, updated_at = ?
                WHERE job_id = ? AND lease_owner = ?
                """,
                (lease_expires_at, now, now, job_id, lease_owner),
            )
            if conn is None:
                target_conn.commit()
            return True
        finally:
            if conn is None:
                target_conn.close()

    def release_lease(
        self,
        job_id: str,
        lease_owner: str,
        conn: Optional[sqlite3.Connection] = None
    ) -> None:
        now = time.time()
        target_conn = conn or self._connect()
        try:
            target_conn.execute(
                """
                UPDATE task_heartbeats
                SET lease_owner = NULL,
                    lease_expires_at = NULL,
                    last_heartbeat = ?,
                    updated_at = ?
                WHERE job_id = ? AND lease_owner = ?
                """,
                (now, now, job_id, lease_owner),
            )
            if conn is None:
                target_conn.commit()
        finally:
            if conn is None:
                target_conn.close()

    def list_expired_leases(self, now: Optional[float] = None) -> List[str]:
        timestamp = now or time.time()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT job_id FROM task_heartbeats
                WHERE lease_expires_at IS NOT NULL AND lease_expires_at < ?
                """,
                (timestamp,),
            ).fetchall()
        return [row["job_id"] for row in rows]

    def list_heartbeat_timeouts(self, timeout_seconds: float, now: Optional[float] = None) -> List[str]:
        timestamp = now or time.time()
        cutoff = timestamp - timeout_seconds
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT job_id FROM task_heartbeats
                WHERE last_heartbeat IS NOT NULL AND last_heartbeat < ?
                """,
                (cutoff,),
            ).fetchall()
        return [row["job_id"] for row in rows]

    def set_metadata(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO metadata (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (key, value),
            )

    def get_metadata(self, key: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM metadata WHERE key = ?",
                (key,),
            ).fetchone()
        return row["value"] if row else None

    def _job_to_row(self, job: JobState) -> Dict[str, Any]:
        created_at = self._normalize_created_at(job.createdAt)
        settings_json = json.dumps(job.settings.to_dict())
        now_seconds = time.time()
        job.updatedAt = int(now_seconds * 1000)
        return {
            "job_id": job.job_id,
            "filename": job.filename,
            "title": job.title,
            "dir": job.dir,
            "input_path": job.input_path,
            "status": job.status,
            "state_seq": max(0, int(job.state_seq or 0)),
            "phase": job.phase,
            "progress": job.progress,
            "phase_percent": job.phase_percent,
            "message": job.message,
            "error": job.error,
            "processed": job.processed,
            "total": job.total,
            "language": job.language,
            "srt_path": job.srt_path,
            "canceled": 1 if job.canceled else 0,
            "paused": 1 if job.paused else 0,
            "subtitle_time_offset": job.subtitle_time_offset,
            "settings_json": settings_json,
            "updated_at": now_seconds,
            "created_at": created_at,
        }

    def _row_to_job(self, row: sqlite3.Row) -> JobState:
        settings_data = json.loads(row["settings_json"]) if row["settings_json"] else {}
        settings = JobSettings.from_dict(settings_data)
        created_at_ms = self._format_created_at(row["created_at"])
        updated_at_ms = None
        if row["updated_at"] is not None:
            updated_at_ms = int(row["updated_at"] * 1000)
        return JobState(
            job_id=row["job_id"],
            filename=row["filename"] or "unknown",
            title=row["title"] or "",
            dir=row["dir"] or "",
            input_path=row["input_path"] or "",
            settings=settings,
            status=row["status"] or "queued",
            state_seq=max(0, int((row["state_seq"] or 0))),
            phase=row["phase"] or "pending",
            progress=row["progress"] or 0.0,
            phase_percent=row["phase_percent"] or 0.0,
            message=row["message"] or "",
            error=row["error"],
            processed=row["processed"] or 0,
            total=row["total"] or 0,
            language=row["language"],
            srt_path=row["srt_path"],
            canceled=bool(row["canceled"]),
            paused=bool(row["paused"]),
            subtitle_time_offset=row["subtitle_time_offset"],
            createdAt=created_at_ms,
            updatedAt=updated_at_ms,
        )

    def _normalize_created_at(self, created_at: Optional[int]) -> Optional[float]:
        if created_at is None:
            return None
        if created_at > 1_000_000_000_000:
            return created_at / 1000.0
        return float(created_at)

    def _format_created_at(self, created_at: Optional[float]) -> Optional[int]:
        if created_at is None:
            return None
        return int(created_at * 1000)
