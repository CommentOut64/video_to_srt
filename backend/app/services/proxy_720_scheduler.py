"""
V3.1.2+dev.20260114.01: 新增 720p Proxy 调度器

职责：
1. 统一管理 720p 自动/手动触发入口
2. 维护待触发队列（默认 FIFO，可带优先级）
3. 持久化状态，支持重启恢复和前端状态查询
4. 负责“队列空闲后再启动”的判断，避免编辑器加载时重复触发
"""
import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from app.core.config import config

logger = logging.getLogger(__name__)


def _now() -> float:
    return time.time()


class Proxy720Scheduler:
    """720p 调度器，所有自动/手动触发都走这里"""

    def __init__(self) -> None:
        self.media_prep = None  # 运行时注入，避免循环依赖
        self.lock = threading.Lock()
        # 待触发的任务 {job_id: {priority, requested_at, trigger_type, video_path}}
        self.pending: Dict[str, Dict] = {}

    # ===== 公共方法 =====
    def bind_media_prep(self, media_prep) -> None:
        """由 media_prep_service 在单例初始化后绑定"""
        self.media_prep = media_prep

    def request(self, job_id: str, video_path: Path, trigger_type: str,
                auto_enabled: bool, force: bool = False, priority: int = 100) -> Dict:
        """
        提交一次 720p 触发请求（自动/手动皆可）

        返回值：{accepted: bool, reason: str}
        """
        result = {"accepted": False, "reason": ""}
        proxy_path = config.JOBS_DIR / job_id / "proxy_720p.mp4"
        preview_path = config.JOBS_DIR / job_id / "preview_360p.mp4"

        with self.lock:
            state = self._load_state(job_id)
            version = state.get("version", 0) + 1

            # 自动禁用时直接拒绝（手动 force 或 trigger_type == manual 可绕过）
            if not force and trigger_type != "manual" and not auto_enabled:
                result["reason"] = "auto_disabled"
                return result

            # 已有 720p 文件，直接标记 ready
            if proxy_path.exists():
                self._save_state(job_id, {
                    "state": "ready",
                    "version": version,
                    "updated_at": _now(),
                    "url": f"/api/media/{job_id}/video"
                })
                result["reason"] = "already_exists"
                return result

            # 已在转码/队列中
            if self.media_prep:
                proxy_status = self.media_prep.get_proxy_status(job_id)
                if proxy_status and proxy_status.get("status") in ["queued", "processing"]:
                    result["reason"] = "already_processing"
                    return result

            # 360p 必须完成（除非强制）
            if not force:
                preview_status_ok = False
                if preview_path.exists():
                    preview_status_ok = True
                elif self.media_prep:
                    preview_status = self.media_prep.get_preview_status(job_id)
                    preview_status_ok = preview_status and preview_status.get("status") == "completed"

                if not preview_status_ok:
                    result["reason"] = "preview_not_ready"
                    return result

            # 去重：已有待触发则不重复入队，只更新时间与来源
            pending = self.pending.get(job_id, {})
            pending.update({
                "job_id": job_id,
                "video_path": str(video_path),
                "trigger_type": trigger_type,
                "priority": priority,
                "requested_at": _now()
            })
            self.pending[job_id] = pending

            self._save_state(job_id, {
                "state": "waiting_check",
                "version": version,
                "updated_at": _now(),
                "trigger_type": trigger_type,
                "auto_enabled": auto_enabled
            })

            result["accepted"] = True
            result["reason"] = "queued_for_check"

        # 队列空闲时尝试启动
        self.on_queue_idle()
        return result

    def on_queue_idle(self) -> None:
        """由队列空闲事件触发，从待触发列表里启动一个"""
        if not self.media_prep:
            return

        with self.lock:
            if not self.pending:
                return

            # 排序：优先级 -> 请求时间（默认 FIFO）
            candidates = sorted(
                self.pending.values(),
                key=lambda x: (x.get("priority", 100), x.get("requested_at", 0))
            )

        for item in candidates:
            if self._try_start(item):
                break

    def mark_processing(self, job_id: str) -> None:
        with self.lock:
            state = self._load_state(job_id)
            version = state.get("version", 0) + 1
            self._save_state(job_id, {
                "state": "processing",
                "version": version,
                "updated_at": _now()
            })
            self.pending.pop(job_id, None)

    def mark_complete(self, job_id: str, output_path: Path) -> None:
        with self.lock:
            state = self._load_state(job_id)
            version = state.get("version", 0) + 1
            self._save_state(job_id, {
                "state": "ready",
                "version": version,
                "updated_at": _now(),
                "url": f"/api/media/{job_id}/video",
                "size": output_path.stat().st_size if output_path.exists() else 0
            })
            self.pending.pop(job_id, None)

    def mark_failed(self, job_id: str, message: str) -> None:
        with self.lock:
            state = self._load_state(job_id)
            version = state.get("version", 0) + 1
            self._save_state(job_id, {
                "state": "failed",
                "version": version,
                "updated_at": _now(),
                "error": message
            })
            # 失败可留在 pending，等待下一次空闲重试
            pending = self.pending.get(job_id)
            if pending:
                pending["last_error"] = message

    def ensure_tracked(self, job_id: str, video_path: Optional[Path], trigger_type: str = "restore") -> None:
        """编辑器加载或服务重启时调用，确保状态文件存在但不立即触发"""
        with self.lock:
            state = self._load_state(job_id)
            if state.get("state") in ["ready", "processing", "queued"]:
                return
            pending = self.pending.get(job_id, {})
            if video_path:
                pending["video_path"] = str(video_path)
            pending.update({
                "job_id": job_id,
                "trigger_type": trigger_type,
                "priority": pending.get("priority", 100),
                "requested_at": pending.get("requested_at", _now())
            })
            self.pending[job_id] = pending
            self._save_state(job_id, {
                "state": state.get("state", "waiting_check"),
                "version": state.get("version", 0) + 1,
                "updated_at": _now(),
                "trigger_type": trigger_type
            })

    def get_state(self, job_id: str) -> Dict:
        """V3.1.2+dev.20260114.06: 对外提供只读状态查询"""
        with self.lock:
            return self._load_state(job_id)

    # ===== 内部方法 =====
    def _try_start(self, item: Dict) -> bool:
        job_id = item["job_id"]
        video_path = Path(item["video_path"])
        output_path = config.JOBS_DIR / job_id / "proxy_720p.mp4"

        # 队列繁忙则跳过，等待下次空闲事件
        if self.media_prep and self.media_prep._is_transcription_queue_busy():
            logger.info(f"[Proxy720Scheduler] 队列繁忙，暂不启动: {job_id}")
            return False

        # 360p/文件校验
        preview = config.JOBS_DIR / job_id / "preview_360p.mp4"
        if not preview.exists():
            preview_status = self.media_prep.get_preview_status(job_id) if self.media_prep else None
            if not (preview_status and preview_status.get("status") == "completed"):
                logger.info(f"[Proxy720Scheduler] 360p未完成，暂不启动: {job_id}")
                return False

        if output_path.exists():
            logger.info(f"[Proxy720Scheduler] 720p已存在，跳过: {job_id}")
            return True

        # 最终入队
        logger.info(f"[Proxy720Scheduler] 队列空闲，启动720p: {job_id} ({item.get('trigger_type')})")
        success = self.media_prep.enqueue_proxy(job_id, video_path, output_path, priority=10)

        with self.lock:
            state = self._load_state(job_id)
            version = state.get("version", 0) + 1
            if success:
                self._save_state(job_id, {
                    "state": "queued",
                    "version": version,
                    "updated_at": _now(),
                    "trigger_type": item.get("trigger_type")
                })
                self.pending.pop(job_id, None)
                return True
            else:
                self._save_state(job_id, {
                    "state": "waiting_check",
                    "version": version,
                    "updated_at": _now(),
                    "error": "enqueue_failed"
                })
                return False

    def _load_state(self, job_id: str) -> Dict:
        path = self._state_path(job_id)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.debug(f"[Proxy720Scheduler] 读取状态失败: {job_id}, {e}")
            return {}

    def _save_state(self, job_id: str, state: Dict) -> None:
        path = self._state_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _state_path(self, job_id: str) -> Path:
        return config.JOBS_DIR / job_id / "proxy_state.json"


# ===== 单例访问 =====
_proxy_scheduler: Optional[Proxy720Scheduler] = None


def get_proxy_scheduler() -> Proxy720Scheduler:
    """获取 720p 调度器单例"""
    global _proxy_scheduler
    if _proxy_scheduler is None:
        _proxy_scheduler = Proxy720Scheduler()
    return _proxy_scheduler
