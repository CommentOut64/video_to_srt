"""
Runtime checkpoint 服务。

设计模式：Service Layer
原因：统一封装 runtime_state 读写与 checkpoint 镜像桥接，
避免业务层直接操作仓储细节。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.services.checkpoint.runtime_checkpoint_models import RuntimeCheckpointSnapshot
from app.services.checkpoint.runtime_checkpoint_repository import RuntimeCheckpointRepository


class RuntimeCheckpointService:
    """运行时断点服务。"""
    _KEY_TRANSCRIPTION_HINT = "transcription_hint"
    _KEY_SUBTITLE_RUNTIME = "subtitle_runtime_v1"

    def __init__(self, job_dir: Path, db_file_name: str = "runtime_state.db") -> None:
        self.job_dir = Path(job_dir)
        self.db_path = self.job_dir / db_file_name
        self.logger = logging.getLogger(__name__)
        self.repository = RuntimeCheckpointRepository(self.db_path, logger=self.logger)

    def record_unit_started(self, *, stage: str, unit_id: str, payload: dict[str, Any] | None = None) -> None:
        """记录单元开始。"""
        payload_json = json.dumps(payload, ensure_ascii=False) if payload is not None else None
        self.repository.append_unit_journal(
            stage=stage,
            unit_id=unit_id,
            status="started",
            payload_json=payload_json,
        )

    def record_unit_committed(self, *, stage: str, unit_id: str, payload: dict[str, Any] | None = None) -> None:
        """记录单元提交并推进提交点。"""
        payload_json = json.dumps(payload, ensure_ascii=False) if payload is not None else None
        # 设计模式：事务脚本（Transaction Script）
        # 原因：日志写入和提交点推进必须原子提交，避免恢复时出现“日志存在但提交点缺失”的不一致
        with self.repository.transaction() as conn:
            self.repository.append_unit_journal(
                stage=stage,
                unit_id=unit_id,
                status="committed",
                payload_json=payload_json,
                conn=conn,
            )
            self.repository.upsert_unit_commit(stage=stage, last_unit_id=unit_id, conn=conn)

    def save_transcription_hint(self, hint: dict[str, Any]) -> None:
        """保存转录摘要提示（仅关键索引，不写入全量快照）。"""
        payload = json.dumps(hint, ensure_ascii=False)
        self.repository.upsert_control_signal(key=self._KEY_TRANSCRIPTION_HINT, value=payload)

    def save_subtitle_runtime(self, runtime_payload: dict[str, Any]) -> None:
        """保存字幕运行态真源快照。"""
        payload = json.dumps(runtime_payload, ensure_ascii=False)
        self.repository.upsert_control_signal(key=self._KEY_SUBTITLE_RUNTIME, value=payload)

    def load_subtitle_runtime(self) -> dict[str, Any] | None:
        """读取字幕运行态真源快照。"""
        return self._load_json_control_signal(self._KEY_SUBTITLE_RUNTIME)

    def mark_pause_requested(self) -> None:
        """写入暂停请求信号。"""
        self.repository.upsert_control_signal(key="pause_requested", value="1")

    def clear_pause_requested(self) -> None:
        """清除暂停请求信号。"""
        self.repository.upsert_control_signal(key="pause_requested", value="0")

    def mark_cancel_requested(self) -> None:
        """写入取消请求信号。"""
        self.repository.upsert_control_signal(key="cancel_requested", value="1")

    def clear_cancel_requested(self) -> None:
        """清除取消请求信号。"""
        self.repository.upsert_control_signal(key="cancel_requested", value="0")

    def is_pause_requested(self) -> bool:
        """读取暂停请求。"""
        return self.repository.get_control_signal("pause_requested") == "1"

    def is_cancel_requested(self) -> bool:
        """读取取消请求。"""
        return self.repository.get_control_signal("cancel_requested") == "1"

    def load_snapshot(self) -> RuntimeCheckpointSnapshot:
        """读取运行时快照。"""
        transcription_hint = self._load_json_control_signal(self._KEY_TRANSCRIPTION_HINT)
        subtitle_runtime = self._load_json_control_signal(self._KEY_SUBTITLE_RUNTIME)
        return RuntimeCheckpointSnapshot(
            last_unit_commits=self.repository.list_unit_commits(),
            transcription_hint=transcription_hint,
            subtitle_runtime=subtitle_runtime,
        )

    def has_runtime_state(self) -> bool:
        """判断是否存在 runtime_state 记录。"""
        return self.repository.has_any_state()

    def _load_json_control_signal(self, key: str) -> dict[str, Any] | None:
        """读取 JSON 控制信号并安全解析。"""
        raw_value = self.repository.get_control_signal(key)
        if not raw_value:
            return None
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            self.logger.warning("解析 %s 失败: %s", key, exc)
            return None
        if not isinstance(parsed, dict):
            self.logger.warning("%s 不是对象，已忽略", key)
            return None
        return parsed

