"""
模型自愈服务（支持 strict/background/off）。

设计目标：
1. 严格模式可在启动阶段同步校验必需模型；
2. 后台模式仅调度异步任务，不阻塞启动；
3. 所有模型状态可观测（API + SSE + 本地状态文件）；
4. 模型不完整时自动修复，修复失败在后台模式下不影响后端存活。
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

from app.core.config import config
from app.services.model_state_store import ModelStateStore
from app.services.model_scope_service import is_bootstrap_required_model
from app.services.sse_service import get_sse_manager

if TYPE_CHECKING:
    from app.services.model_manager_v2 import ModelManagerV2

logger = logging.getLogger(__name__)

BootstrapMode = Literal["strict", "background", "off"]
_DEFAULT_REQUIRED_KINDS = {"asr", "separation", "vad"}


class ModelBootstrapService:
    """模型自愈编排器（支持同步校验或后台修复）。"""

    def __init__(
        self,
        model_manager: "ModelManagerV2",
        state_store: Optional[ModelStateStore] = None,
    ):
        self._model_manager = model_manager
        self._state_store = state_store or ModelStateStore()
        self._sse_manager = get_sse_manager()
        self._task_lock = threading.Lock()
        self._task: Optional[asyncio.Task[Any]] = None
        self._is_running = False
        self._last_run_at: Optional[float] = None
        self._run_count = 0
        self._last_summary: Dict[str, Any] = {
            "started_at": None,
            "finished_at": None,
            "model_count": 0,
            "ready_models": [],
            "failed_models": [],
            "repaired_models": [],
        }

    def start_non_blocking(
        self,
        model_ids: Optional[List[str]] = None,
        *,
        is_force: bool = False,
        is_required_on_boot: bool = False,
    ) -> bool:
        """
        非阻塞启动后台自愈任务。

        返回：
        - True: 成功启动新任务
        - False: 已有任务运行中或无可用事件循环
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("模型后台自愈启动失败：当前线程无可用事件循环")
            return False

        with self._task_lock:
            is_task_running = self._task is not None and not self._task.done()
            if is_task_running:
                return False
            self._task = loop.create_task(
                self.run_once(
                    model_ids=model_ids,
                    is_force=is_force,
                    is_required_on_boot=is_required_on_boot,
                ),
                name="model-bootstrap-runner",
            )
            return True

    def cancel_background(self) -> bool:
        """取消当前后台任务。"""
        with self._task_lock:
            if self._task is None or self._task.done():
                return False
            self._task.cancel()
            return True

    async def run_once(
        self,
        model_ids: Optional[List[str]] = None,
        *,
        is_force: bool = False,
        is_required_on_boot: bool = False,
    ) -> Dict[str, Any]:
        """执行一轮后台模型自愈扫描。"""
        selected_ids = self._resolve_model_ids(model_ids)
        self._is_running = True
        self._run_count += 1
        started_at = time.time()
        self._last_run_at = started_at
        ready_models: List[str] = []
        repaired_models: List[str] = []
        failed_models: List[str] = []

        self._emit(
            "model.bootstrap.started",
            {
                "started_at": started_at,
                "model_count": len(selected_ids),
                "is_force": is_force,
                "is_required_on_boot": is_required_on_boot,
                "model_ids": selected_ids,
            },
        )
        logger.info(
            "模型后台自愈任务启动：count=%d force=%s",
            len(selected_ids),
            is_force,
        )

        try:
            for model_id in selected_ids:
                outcome = await self._process_one_model(
                    model_id=model_id,
                    is_force=is_force,
                    is_required_on_boot=is_required_on_boot,
                )
                status = outcome.get("status")
                source = outcome.get("source")
                if status == "failed":
                    failed_models.append(model_id)
                elif status == "ready":
                    ready_models.append(model_id)
                    if source == "repair":
                        repaired_models.append(model_id)
        except asyncio.CancelledError:
            self._emit(
                "model.bootstrap.cancelled",
                {
                    "cancelled_at": time.time(),
                    "model_count": len(selected_ids),
                },
            )
            logger.info("模型后台自愈任务被取消")
            raise
        finally:
            finished_at = time.time()
            self._is_running = False
            summary = {
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_sec": round(finished_at - started_at, 3),
                "model_count": len(selected_ids),
                "ready_models": ready_models,
                "failed_models": failed_models,
                "repaired_models": repaired_models,
            }
            self._last_summary = summary
            self._emit(
                "model.bootstrap.completed",
                summary,
            )
            logger.info(
                "模型后台自愈任务结束：duration=%.2fs count=%d failed=%d repaired=%d",
                finished_at - started_at,
                len(selected_ids),
                len(failed_models),
                len(repaired_models),
            )
        return self._last_summary

    def get_status(self) -> Dict[str, Any]:
        """返回当前自愈服务状态。"""
        task_state = "idle"
        with self._task_lock:
            if self._task is not None:
                if self._task.cancelled():
                    task_state = "cancelled"
                elif self._task.done():
                    task_state = "completed"
                else:
                    task_state = "running"

        snapshot = self._state_store.snapshot()
        return {
            "is_running": self._is_running,
            "task_state": task_state,
            "run_count": self._run_count,
            "last_run_at": self._last_run_at,
            "updated_at": snapshot.get("updated_at"),
            "last_summary": self._last_summary,
            "models": snapshot.get("models", {}),
        }

    def resolve_required_model_ids(self) -> List[str]:
        """解析严格模式下的必需模型列表。"""
        known_specs = self._model_manager.registry.list()
        known_ids = [spec.id for spec in known_specs]
        known_set = set(known_ids)

        env_raw = str(os.getenv("MODEL_BOOTSTRAP_REQUIRED_MODELS", "")).strip()
        if env_raw:
            selected: List[str] = []
            for item in env_raw.split(","):
                model_id = item.strip()
                if not model_id or model_id not in known_set:
                    continue
                if model_id not in selected:
                    selected.append(model_id)
            if selected:
                return selected

        # 默认必需模型：以 backend/models/pretrained 为“启动时需要自动自愈”的强约束边界。
        # Why: 这些模型属于主流程预置模型，即使首启时目录尚未下载出来，也应在启动自愈阶段处理；
        # Whisper-* 等非预置模型必须继续保持“用到时再下载”。
        selected: List[str] = []
        for spec in known_specs:
            if not is_bootstrap_required_model(spec):
                continue
            if spec.id not in selected:
                selected.append(spec.id)

        # 兜底：若未命中任何 pretrained 配置，回退旧行为（按 kind 筛选）。
        if not selected:
            for spec in known_specs:
                if spec.kind in _DEFAULT_REQUIRED_KINDS and spec.id not in selected:
                    selected.append(spec.id)

        return selected

    async def _process_one_model(
        self,
        model_id: str,
        *,
        is_force: bool,
        is_required_on_boot: bool,
    ) -> Dict[str, Any]:
        checked_at = time.time()
        existing = self._state_store.get_record(model_id)
        attempts = int(existing.attempts or 0)

        self._state_store.update_record(
            model_id,
            status="checking",
            checked_at=checked_at,
            message="后台校验中",
            is_required_on_boot=is_required_on_boot,
            attempts=attempts,
        )
        self._emit(
            "model.bootstrap.checking",
            {
                "model_id": model_id,
                "checked_at": checked_at,
                "is_force": is_force,
            },
        )

        try:
            status_payload = await asyncio.to_thread(self._model_manager.model_status, model_id)
        except Exception as exc:
            error_text = f"校验失败: {exc}"
            self._state_store.update_record(
                model_id,
                status="failed",
                message=error_text,
                missing=[error_text],
                last_error_at=time.time(),
                attempts=attempts + 1,
            )
            self._emit(
                "model.bootstrap.failed",
                {"model_id": model_id, "message": error_text},
            )
            logger.error("模型后台校验失败: model=%s error=%s", model_id, exc)
            return {"model_id": model_id, "status": "failed", "source": "check"}

        status = str(status_payload.get("status", "unknown"))
        local_path = status_payload.get("local_path")
        missing = list(status_payload.get("missing") or [])

        if status == "ready" and not is_force:
            self._state_store.update_record(
                model_id,
                status="ready",
                local_path=local_path,
                missing=[],
                message="模型已就绪",
                ready_at=time.time(),
                attempts=attempts,
            )
            self._emit(
                "model.bootstrap.ready",
                {"model_id": model_id, "local_path": local_path, "from": "local"},
            )
            return {"model_id": model_id, "status": "ready", "source": "local"}

        target_status = "incomplete" if status == "incomplete" else "downloading"
        self._state_store.update_record(
            model_id,
            status=target_status,
            local_path=local_path,
            missing=missing,
            message="后台修复中",
            attempts=attempts + 1,
        )
        self._emit(
            "model.bootstrap.repairing",
            {
                "model_id": model_id,
                "status": status,
                "missing": missing,
                "is_force": is_force,
            },
        )

        try:
            repaired_path = await asyncio.to_thread(
                self._model_manager.ensure_available,
                model_id,
                is_allow_download_override=True,
                is_local_files_only_override=False,
            )
        except Exception as exc:
            error_text = str(exc)
            self._state_store.update_record(
                model_id,
                status="failed",
                local_path=local_path,
                missing=missing,
                message=error_text,
                last_error_at=time.time(),
                attempts=attempts + 1,
            )
            self._emit(
                "model.bootstrap.failed",
                {"model_id": model_id, "message": error_text},
            )
            logger.warning("模型后台修复失败: model=%s error=%s", model_id, exc)
            return {"model_id": model_id, "status": "failed", "source": "repair"}

        self._state_store.update_record(
            model_id,
            status="ready",
            local_path=repaired_path,
            missing=[],
            message="后台修复成功",
            ready_at=time.time(),
            attempts=attempts + 1,
        )
        self._emit(
            "model.bootstrap.ready",
            {
                "model_id": model_id,
                "local_path": repaired_path,
                "from": "repair",
            },
        )
        return {"model_id": model_id, "status": "ready", "source": "repair"}

    def _resolve_model_ids(self, model_ids: Optional[List[str]]) -> List[str]:
        known_ids = [spec.id for spec in self._model_manager.registry.list()]
        known_set = set(known_ids)

        if model_ids:
            selected = []
            for model_id in model_ids:
                normalized = model_id.strip()
                if not normalized or normalized not in known_set:
                    continue
                if normalized not in selected:
                    selected.append(normalized)
            return selected

        env_raw = str(os.getenv("MODEL_BOOTSTRAP_MODELS", "")).strip()
        if env_raw:
            selected = []
            for item in env_raw.split(","):
                model_id = item.strip()
                if not model_id or model_id not in known_set:
                    continue
                if model_id not in selected:
                    selected.append(model_id)
            if selected:
                return selected

        # 默认仅扫描“必需模型”，避免后台自愈去拉取非必需模型（如 whisper-*）。
        return self.resolve_required_model_ids()

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        self._sse_manager.broadcast_sync("models", event, payload)


_bootstrap_service: Optional[ModelBootstrapService] = None
_bootstrap_lock = threading.Lock()


def get_model_bootstrap_service(
    model_manager: Optional["ModelManagerV2"] = None,
) -> ModelBootstrapService:
    """获取模型后台自愈服务单例。"""
    global _bootstrap_service
    with _bootstrap_lock:
        if _bootstrap_service is None:
            if model_manager is None:
                from app.services.model_manager_v2 import get_model_manager_v2

                model_manager = get_model_manager_v2()
            _bootstrap_service = ModelBootstrapService(model_manager=model_manager)
    return _bootstrap_service
