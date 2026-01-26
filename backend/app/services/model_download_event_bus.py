"""
模型下载事件总线（SSE 推送 + 状态快照）
V3.2.0+dev.20260119.04
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal, Any, List

from app.services.sse_service import get_sse_manager


DownloadStatus = Literal["starting", "downloading", "complete", "error", "cache_hit"]


@dataclass
class ModelDownloadState:
    """单个模型的下载状态快照。"""

    model_id: str
    repo_id: Optional[str]
    status: DownloadStatus
    downloaded_bytes: int = 0
    total_bytes: Optional[int] = None
    percent: Optional[float] = None
    file: Optional[str] = None
    message: Optional[str] = None
    local_path: Optional[str] = None
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "repo_id": self.repo_id,
            "status": self.status,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "percent": self.percent,
            "file": self.file,
            "message": self.message,
            "local_path": self.local_path,
            "updated_at": self.updated_at,
        }


class ModelDownloadEventBus:
    """模型下载事件总线，负责维护状态并推送 SSE。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._downloads: Dict[str, ModelDownloadState] = {}
        self._sse_manager = get_sse_manager()

    def get_snapshot(self) -> Dict[str, Any]:
        """返回当前下载状态快照（用于 SSE initial_state）。"""
        with self._lock:
            downloads = [state.to_dict() for state in self._downloads.values()]
        return {"downloads": downloads, "timestamp": time.time()}

    def set_sse_manager(self, sse_manager: Any) -> None:
        """测试或特殊场景下替换 SSE 管理器。"""
        self._sse_manager = sse_manager

    def start(self, model_id: str, repo_id: Optional[str]) -> None:
        self._update_state(
            model_id=model_id,
            repo_id=repo_id,
            status="starting",
        )
        self._emit("model.download.start", model_id)

    def progress(
        self,
        model_id: str,
        downloaded_bytes: int,
        total_bytes: Optional[int],
        file: Optional[str] = None,
        repo_id: Optional[str] = None,
    ) -> None:
        percent = None
        if total_bytes:
            percent = round(min(100.0, downloaded_bytes / total_bytes * 100), 2)
        self._update_state(
            model_id=model_id,
            repo_id=repo_id,
            status="downloading",
            downloaded_bytes=downloaded_bytes,
            total_bytes=total_bytes,
            percent=percent,
            file=file,
        )
        self._emit("model.download.progress", model_id)

    def complete(self, model_id: str, local_path: Optional[str], repo_id: Optional[str]) -> None:
        self._update_state(
            model_id=model_id,
            repo_id=repo_id,
            status="complete",
            local_path=local_path,
            percent=100.0,
        )
        self._emit("model.download.complete", model_id)

    def cache_hit(self, model_id: str, local_path: Optional[str], repo_id: Optional[str]) -> None:
        self._update_state(
            model_id=model_id,
            repo_id=repo_id,
            status="cache_hit",
            local_path=local_path,
        )
        self._emit("model.download.cache_hit", model_id)

    def error(self, model_id: str, message: str, repo_id: Optional[str]) -> None:
        self._update_state(
            model_id=model_id,
            repo_id=repo_id,
            status="error",
            message=message,
        )
        self._emit("model.download.error", model_id)

    def _update_state(
        self,
        model_id: str,
        repo_id: Optional[str],
        status: DownloadStatus,
        downloaded_bytes: Optional[int] = None,
        total_bytes: Optional[int] = None,
        percent: Optional[float] = None,
        file: Optional[str] = None,
        message: Optional[str] = None,
        local_path: Optional[str] = None,
    ) -> None:
        with self._lock:
            state = self._downloads.get(model_id)
            if not state:
                state = ModelDownloadState(
                    model_id=model_id,
                    repo_id=repo_id,
                    status=status,
                )
                self._downloads[model_id] = state

            if repo_id is not None:
                state.repo_id = repo_id
            state.status = status
            if downloaded_bytes is not None:
                state.downloaded_bytes = downloaded_bytes
            if total_bytes is not None:
                state.total_bytes = total_bytes
            if percent is not None:
                state.percent = percent
            if file is not None:
                state.file = file
            if message is not None:
                state.message = message
            if local_path is not None:
                state.local_path = local_path
            state.updated_at = time.time()

    def _emit(self, event: str, model_id: str) -> None:
        with self._lock:
            state = self._downloads.get(model_id)
            payload = state.to_dict() if state else {"model_id": model_id}
        self._sse_manager.broadcast_sync("models", event, payload)


_event_bus: Optional[ModelDownloadEventBus] = None


def get_model_download_event_bus() -> ModelDownloadEventBus:
    """获取模型下载事件总线单例。"""
    global _event_bus
    if _event_bus is None:
        _event_bus = ModelDownloadEventBus()
    return _event_bus
