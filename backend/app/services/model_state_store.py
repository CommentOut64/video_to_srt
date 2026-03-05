"""
模型自愈状态存储。

职责：
1. 记录模型后台校验/下载状态；
2. 原子写入 `models/model_state.json`，避免中途写坏状态文件；
3. 为启动器与 API 提供统一状态快照。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from app.core.config import config

logger = logging.getLogger(__name__)

BootstrapStatus = Literal[
    "unknown",
    "checking",
    "ready",
    "incomplete",
    "downloading",
    "failed",
]


@dataclass
class ModelBootstrapRecord:
    """单模型后台自愈状态。"""

    model_id: str
    status: BootstrapStatus = "unknown"
    local_path: Optional[str] = None
    missing: List[str] = field(default_factory=list)
    message: Optional[str] = None
    is_required_on_boot: bool = True
    attempts: int = 0
    updated_at: float = field(default_factory=time.time)
    checked_at: Optional[float] = None
    ready_at: Optional[float] = None
    last_error_at: Optional[float] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ModelBootstrapRecord":
        return cls(
            model_id=str(payload.get("model_id", "")),
            status=payload.get("status", "unknown"),
            local_path=payload.get("local_path"),
            missing=list(payload.get("missing") or []),
            message=payload.get("message"),
            is_required_on_boot=bool(payload.get("is_required_on_boot", True)),
            attempts=int(payload.get("attempts", 0) or 0),
            updated_at=float(payload.get("updated_at", time.time())),
            checked_at=payload.get("checked_at"),
            ready_at=payload.get("ready_at"),
            last_error_at=payload.get("last_error_at"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "status": self.status,
            "local_path": self.local_path,
            "missing": list(self.missing),
            "message": self.message,
            "is_required_on_boot": self.is_required_on_boot,
            "attempts": self.attempts,
            "updated_at": self.updated_at,
            "checked_at": self.checked_at,
            "ready_at": self.ready_at,
            "last_error_at": self.last_error_at,
        }


class ModelStateStore:
    """模型状态文件存储。"""

    def __init__(self, state_file: Optional[Path] = None):
        self._lock = threading.RLock()
        self._state_file = state_file or (Path(config.MODELS_DIR) / "model_state.json")
        self._records: Dict[str, ModelBootstrapRecord] = {}
        self._last_updated_at: float = time.time()
        self._load()

    def get_record(self, model_id: str) -> ModelBootstrapRecord:
        with self._lock:
            existing = self._records.get(model_id)
            if existing:
                return existing
            record = ModelBootstrapRecord(model_id=model_id)
            self._records[model_id] = record
            return record

    def update_record(
        self,
        model_id: str,
        *,
        status: Optional[BootstrapStatus] = None,
        local_path: Optional[str] = None,
        missing: Optional[List[str]] = None,
        message: Optional[str] = None,
        is_required_on_boot: Optional[bool] = None,
        attempts: Optional[int] = None,
        checked_at: Optional[float] = None,
        ready_at: Optional[float] = None,
        last_error_at: Optional[float] = None,
    ) -> ModelBootstrapRecord:
        with self._lock:
            record = self.get_record(model_id)
            now = time.time()

            if status is not None:
                record.status = status
            if local_path is not None:
                record.local_path = local_path
            if missing is not None:
                record.missing = list(missing)
            if message is not None:
                record.message = message
            if is_required_on_boot is not None:
                record.is_required_on_boot = is_required_on_boot
            if attempts is not None:
                record.attempts = attempts
            if checked_at is not None:
                record.checked_at = checked_at
            if ready_at is not None:
                record.ready_at = ready_at
            if last_error_at is not None:
                record.last_error_at = last_error_at

            record.updated_at = now
            self._last_updated_at = now
            self._save()
            return record

    def mark_record_removed(self, model_id: str) -> None:
        with self._lock:
            if model_id in self._records:
                del self._records[model_id]
                self._last_updated_at = time.time()
                self._save()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            models = {
                model_id: record.to_dict()
                for model_id, record in sorted(self._records.items())
            }
            return {
                "updated_at": self._last_updated_at,
                "models": models,
            }

    def _load(self) -> None:
        with self._lock:
            if not self._state_file.exists():
                self._state_file.parent.mkdir(parents=True, exist_ok=True)
                return
            try:
                payload = json.loads(self._state_file.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("读取模型状态文件失败，忽略旧状态: %s", exc)
                return

            records = payload.get("models", {})
            for model_id, raw in records.items():
                try:
                    record = ModelBootstrapRecord.from_dict(raw)
                except Exception as exc:
                    logger.warning("解析模型状态失败，model=%s error=%s", model_id, exc)
                    continue
                if not record.model_id:
                    record.model_id = model_id
                self._records[record.model_id] = record

            self._last_updated_at = float(payload.get("updated_at", time.time()))

    def _save(self) -> None:
        payload = self.snapshot()
        self._atomic_write_json(self._state_file, payload)

    @staticmethod
    def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)
