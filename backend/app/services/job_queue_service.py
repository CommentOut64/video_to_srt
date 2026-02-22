"""
任务队列管理服务 - V2.4
核心功能: 串行执行，防止并发OOM，队列持久化，插队功能

v3.1.0 更新:
- 集成 CancellationToken 机制，支持协作式取消/暂停
- 在原子区域内的暂停/取消请求会被延迟执行
"""
import threading
import time
import logging
import gc
import json
import os
import sys
import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Literal, Any
from pathlib import Path
import torch

from app.models.job_models import JobState
from app.models.task_state_machine import get_state_guard
from app.config.lifecycle_config import STATE_MACHINE_GUARD_ENABLED, RUNNER_GATE_ENABLED
from app.services.sse_service import get_sse_manager
from app.core.config import config
from app.services.checkpoint import RuntimeCheckpointService
from app.services.task_state_repository import QueueState
from app.utils.cancellation_token import (
    CancellationToken,
    CancelledException,
    PausedException,
    create_cancellation_token
)

logger = logging.getLogger(__name__)

# 插队模式类型
PrioritizeMode = Literal["gentle", "force"]


@dataclass
class CancelResult:
    """取消/删除操作结果。"""

    success: bool
    status: str
    reason_code: str
    message: str
    pending_delete: bool
    state_seq: int = 0


CANCEL_REASON_CODES: Dict[str, str] = {
    "cancel_queued": "排队中任务直接取消",
    "cancel_running": "运行中任务请求取消",
    "cancel_already": "任务已取消或已完成",
    "cancel_not_found": "任务不存在",
    "delete_blocked": "删除被阻塞（任务仍在运行）",
    "delete_failed": "删除失败",
}


def _run_async_safely(coro):
    """
    安全地运行异步协程，处理 Windows ProactorEventLoop 的已知问题。

    在 Windows 上，当使用 asyncio.run() 运行包含子进程的异步代码时，
    事件循环关闭后可能会触发 _ProactorBasePipeTransport._call_connection_lost 回调，
    导致 "Exception in callback" 错误。这是 Python asyncio 在 Windows 上的已知问题。

    解决方案：
    1. 手动创建和管理事件循环
    2. 在关闭前等待所有传输完成
    3. 忽略关闭时的无害异常
    """
    if sys.platform == 'win32':
        # Windows 特殊处理
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            try:
                # 取消所有待处理的任务
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                # 运行一次以让取消生效
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                # 关闭所有异步生成器
                loop.run_until_complete(loop.shutdown_asyncgens())
                # Python 3.9+ 提供了 shutdown_default_executor
                if hasattr(loop, 'shutdown_default_executor'):
                    loop.run_until_complete(loop.shutdown_default_executor())
            except Exception:
                pass
            finally:
                # 关闭事件循环前等待一小段时间，让传输完成清理
                try:
                    # 给 ProactorEventLoop 一点时间完成管道清理
                    import time
                    time.sleep(0.1)
                except Exception:
                    pass
                loop.close()
                asyncio.set_event_loop(None)
    else:
        # 非 Windows 平台使用标准方式
        return asyncio.run(coro)


class JobQueueService:
    """
    任务队列管理器

    职责:
    1. 维护任务队列 (FIFO)
    2. 单线程Worker循环
    3. 串行执行任务（同一时间只有1个running）
    4. 支持两种插队模式：温和插队、强制插队
    """

    def __init__(
        self,
        transcription_service,
        enable_worker: bool = True,
        enable_cancel_monitor: bool = True,
        enable_state_recovery: bool = True
    ):
        """
        初始化队列服务

        Args:
            transcription_service: 转录服务实例
            enable_worker: 是否启动 Worker 线程
            enable_cancel_monitor: 是否启动取消超时监控线程
            enable_state_recovery: 是否从仓库恢复队列状态
        """
        # 核心数据结构
        self.jobs: Dict[str, JobState] = {}  # 任务注册表 {job_id: JobState}
        self.queue: deque = deque()           # 等待队列 [job_id1, job_id2, ...]
        self.running_job_id: Optional[str] = None  # 当前正在执行的任务ID

        # 强制插队相关：记录被中断的任务，用于自动恢复
        self.interrupted_job_id: Optional[str] = None  # 被强制中断的任务ID

        # 插队设置
        self._default_prioritize_mode: PrioritizeMode = "gentle"  # 默认插队模式

        # [v3.1.0] 取消令牌注册表
        self.cancellation_tokens: Dict[str, CancellationToken] = {}

        # V3.1.2+dev.20260114.09: 720p 调度空闲通知延迟定时器
        self._pending_proxy_idle_timer: Optional[threading.Timer] = None

        # [V3.1.0] 取消超时保障机制
        # 用于确保取消操作最终生效，防止任务卡死导致队列阻塞
        self._pending_cancel_requests: Dict[str, float] = {}  # {job_id: cancel_request_time}
        self._pending_delete_after_cancel: set = set()  # 取消后需要删除数据的任务
        self._force_cancel_timeout: float = 60.0  # 超时时间（秒）
        self._current_executing_job_id: Optional[str] = None  # Worker 当前实际执行的任务ID（不受 cancel 影响）
        # V3.2.4+dev.20260222.01: RunnerGate 风险闸门（孤儿执行观测）
        self._orphan_executions: Dict[str, float] = {}  # {job_id: force_cancel_timestamp}
        self._is_gpu_busy_override: bool = False
        self._runner_gate_orphan_timeout_seconds: float = 120.0
        self._runner_gate_warn_interval_seconds: float = 10.0
        self._runner_gate_last_warn_at: float = 0.0

        # 依赖服务
        self.transcription_service = transcription_service
        self.sse_manager = get_sse_manager()
        self.state_guard = get_state_guard()
        # V3.2.0+dev.20260120.05: 队列状态改为仓库驱动 + 心跳租约
        self.state_repo = transcription_service.job_lifecycle.state_repo
        self.event_bus = transcription_service.job_lifecycle.event_bus
        self.heartbeat_service = transcription_service.job_lifecycle.heartbeat_service

        # 控制信号
        self.stop_event = threading.Event()
        self.lock = threading.RLock()  # 使用可重入锁，避免嵌套调用死锁

        # 持久化文件路径
        self.queue_file = Path(config.JOBS_DIR) / "queue_state.json"
        self.settings_file = Path(config.JOBS_DIR) / "queue_settings.json"

        # 加载设置
        self._load_settings()

        # 启动时恢复队列
        if enable_state_recovery:
            self._load_state()

        self._lease_owner = f"job_queue_{os.getpid()}"
        self._heartbeat_ttl_seconds = 30.0
        self._heartbeat_interval_seconds = 10.0
        self._heartbeat_thread: Optional[threading.Thread] = None

        # 启动Worker线程
        if enable_worker:
            self.worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="JobQueueWorker"
            )
            self.worker_thread.start()
            logger.info("任务队列Worker线程已启动")

            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
                name="JobQueueHeartbeat"
            )
            self._heartbeat_thread.start()
            logger.info("任务队列心跳线程已启动")

        # [V3.1.0] 启动取消超时监控线程
        if enable_cancel_monitor:
            self._cancel_timeout_thread = threading.Thread(
                target=self._cancel_timeout_monitor,
                daemon=True,
                name="CancelTimeoutMonitor"
            )
            self._cancel_timeout_thread.start()
            logger.info("[V3.1.0] 取消超时监控线程已启动")

    def _find_video_file(self, job_id: str) -> Optional[Path]:
        """V3.1.2+dev.20260114.11: 查找源视频（跳过 preview/proxy/remux）"""
        job_dir = config.JOBS_DIR / job_id
        if not job_dir.exists():
            return None
        video_exts = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.webm', '.flv', '.m4v']
        for file in job_dir.iterdir():
            if file.is_file() and file.suffix.lower() in video_exts:
                if file.name.startswith(('preview_', 'proxy_', 'remux')):
                    continue
                return file
        return None

    def add_job(self, job: JobState):
        """
        添加任务到队列

        Args:
            job: 任务状态对象
        """
        with self.lock:
            from_status = job.status
            is_transitioned = self._transition_job_status(job, "queued", "queue_add")
            if not is_transitioned:
                logger.error("加入队列失败，状态迁移被拒绝: job=%s", job.job_id)
                return
            self.jobs[job.job_id] = job
            self.queue.append(job.job_id)
            job.message = f"排队中 (位置: {len(self.queue)})"

        logger.info(f"任务已加入队列: {job.job_id} (队列长度: {len(self.queue)})")

        # 保存队列状态和任务元信息
        self._persist_queue_and_jobs([job], {job.job_id: from_status}, reason="queue_add")

        # 推送全局SSE通知
        self._notify_queue_change()
        self._notify_job_status(job.job_id, job.status)

    def get_job(self, job_id: str) -> Optional[JobState]:
        """获取任务状态"""
        return self.jobs.get(job_id)

    def _transition_job_status(self, job: JobState, target_status: str, reason: str) -> bool:
        """
        通过状态守卫执行状态迁移。

        守卫关闭时直接赋值，便于灰度回滚。
        """
        normalized_target = self.state_guard.normalize_status(target_status)
        if not STATE_MACHINE_GUARD_ENABLED:
            job.status = normalized_target
            return True

        self.state_guard.sync_seq(job.job_id, job.state_seq)
        normalized_current = self.state_guard.normalize_status(job.status)
        if normalized_current == normalized_target:
            job.status = normalized_target
            return True

        result = self.state_guard.transition(
            job_id=job.job_id,
            current_status=normalized_current,
            target_status=normalized_target,
            reason=reason,
        )
        if not result.success:
            logger.error(
                "状态迁移被拒绝: job=%s, %s -> %s, reason=%s",
                job.job_id,
                normalized_current,
                normalized_target,
                reason,
            )
            return False

        job.status = result.to_status
        job.state_seq = result.state_seq
        return True

    def pause_job(self, job_id: str) -> bool:
        """
        暂停任务

        v3.1.0 更新: 集成 CancellationToken，触发协作式暂停
        V3.1.0 更新: 区分"正在暂停"和"已暂停"状态
        - 正在运行的任务：推送 pause_pending，等待流水线响应
        - 队列中的任务：立即推送 job_paused

        Args:
            job_id: 任务ID

        Returns:
            bool: 是否成功设置暂停标志
        """
        # V3.2.0+dev.20260120.06: 支持重启后从状态仓库加载暂停任务
        job = self.jobs.get(job_id)
        if not job:
            return False

        from_status = job.status
        is_running = False
        with self.lock:
            if job_id == self.running_job_id:
                # 正在执行的任务：设置暂停标志（pipeline会自己检测并保存checkpoint）
                is_running = True
                job.paused = True
                # V3.1.0: 状态改为 pausing，表示正在等待流水线响应
                if not self._transition_job_status(job, "pausing", "pause_request_running"):
                    logger.error("暂停失败，状态迁移被拒绝: job=%s", job_id)
                    job.paused = False
                    return False
                job.message = "正在暂停，等待当前操作完成..."

                # [v3.1.0] 触发取消令牌的暂停
                token = self.cancellation_tokens.get(job_id)
                if token:
                    token.pause()
                    logger.info(f"[v3.1.0] 已触发取消令牌暂停: {job_id}")
                else:
                    logger.info(f"设置暂停标志: {job_id}")
            elif job_id in self.queue:
                # 还在排队的任务：直接从队列移除
                if not self._transition_job_status(job, "paused", "pause_request_queued"):
                    logger.error("暂停失败，状态迁移被拒绝: job=%s", job_id)
                    return False
                self.queue.remove(job_id)
                job.message = "已暂停（未开始）"
                logger.info(f"从队列移除: {job_id}")

        # 保存队列状态和任务元信息
        self._persist_queue_and_jobs([job], {job.job_id: from_status}, reason="pause_request")

        # 推送全局SSE通知
        self._notify_queue_change()
        self._notify_job_status(job_id, job.status)

        # V3.1.0: 根据任务状态推送不同的信号
        if is_running:
            # 正在运行的任务：推送 pause_pending，前端显示"正在暂停..."
            self._notify_job_signal(job_id, "pause_pending")
        else:
            # 队列中的任务：立即推送 job_paused
            self._notify_job_signal(job_id, "job_paused")

        return True

    def resume_job(self, job_id: str) -> bool:
        """
        恢复暂停的任务

        v3.1.0 更新: 智能恢复逻辑
        - 如果任务仍在运行中（暂停被延迟），只需清除暂停标志
        - 如果任务已完全停止，重新加入队列等待执行

        V3.1.0 更新: 支持 pausing 状态（正在暂停但尚未完全暂停）

        V3.1.0 更新: 恢复时从 checkpoint 恢复进度，避免 SSE 推送 0%

        Args:
            job_id: 任务ID

        Returns:
            bool: 是否成功
        """
        job = self.jobs.get(job_id)
        if not job:
            job = self.transcription_service.load_job_meta(job_id)
            if not job:
                return False
            self.jobs[job_id] = job

        # V3.1.0: 支持 paused 和 pausing 两种状态
        if job.status not in ("paused", "pausing"):
            logger.warning(f"任务未暂停，无法恢复: {job_id}, status={job.status}")
            return False

        # Phase 1: 恢复请求到达后清理暂停/取消控制信号，
        # 防止 PauseBarrier 在下一个单元边界误判并再次停机。
        try:
            if job.dir:
                runtime_service = RuntimeCheckpointService(job_dir=Path(job.dir))
                runtime_service.clear_pause_requested()
                runtime_service.clear_cancel_requested()
        except Exception as exc:
            logger.warning("清理 runtime_state 控制信号失败: %s", exc)

        from_status = job.status
        # V3.1.0: 在推送 SSE 之前，先从 checkpoint 恢复进度
        # 这样 _notify_job_status 推送的进度就是正确的，而非 0
        self._restore_progress_from_checkpoint(job)

        with self.lock:
            # [v3.1.0] 检查任务是否仍在运行中
            # 场景: 用户在原子区域内暂停后立即恢复
            is_still_running = (job_id == self.running_job_id)
            token = self.cancellation_tokens.get(job_id)

            if is_still_running and token:
                # 任务仍在运行，只需清除暂停标志
                # 任务会在原子区域结束后继续正常执行（不会抛出 PausedException）
                token.resume()
                job.paused = False
                if not self._transition_job_status(job, "processing", "resume_running"):
                    logger.error("恢复失败，状态迁移被拒绝: job=%s", job_id)
                    return False
                job.message = "已恢复，继续执行中..."
                logger.info(f"[v3.1.0] 任务仍在运行，清除暂停标志: {job_id}")
            else:
                # 任务已完全停止：优先保留既有队列顺序
                if job_id in self.queue:
                    queue_position = list(self.queue).index(job_id) + 1
                    if not self._transition_job_status(job, "queued", "resume_queued"):
                        logger.error("恢复失败，状态迁移被拒绝: job=%s", job_id)
                        return False
                    job.paused = False
                    job.message = f"已恢复，等待执行 (位置: {queue_position})"
                    logger.info(f"[V3.2.0+dev.20260124.01] 任务已在恢复队列中: {job_id}")
                else:
                    self.queue.append(job_id)
                    if not self._transition_job_status(job, "queued", "resume_queued"):
                        logger.error("恢复失败，状态迁移被拒绝: job=%s", job_id)
                        self.queue.remove(job_id)
                        return False
                    job.paused = False
                    job.message = f"已恢复，排队中 (位置: {len(self.queue)})"

                if token:
                    # Token 还存在但任务不在运行（理论上不应该发生）
                    token.resume()
                    logger.warning(f"[v3.1.0] Token存在但任务未运行，可能是竞态条件: {job_id}")
                else:
                    logger.info(f"[v3.1.0] 任务已停止，重新加入队列: {job_id}")

        # 保存队列状态和任务元信息
        self._persist_queue_and_jobs([job], {job.job_id: from_status}, reason="resume_request")

        # 推送全局SSE通知
        self._notify_queue_change()
        self._notify_job_status(job_id, job.status)

        # 同时推送到单任务频道
        self._notify_job_signal(job_id, "job_resumed")

        return True

    def cancel_job(self, job_id: str, delete_data: bool = False) -> CancelResult:
        """
        取消任务（支持删除已完成的任务）

        V3.1.0 修复：
        - 删除数据时同步清理内存中的 self.jobs[job_id]
        - 广播 job_removed 事件，解决幽灵任务问题

        v3.1.0 更新:
        - 集成 CancellationToken，触发协作式取消

        V3.1.0 更新:
        - 正在运行的任务进入"canceling"状态，不再立即清除 running_job_id
        - 增加超时保障机制，确保任务最终被清除

        Args:
            job_id: 任务ID
            delete_data: 是否删除任务数据

        Returns:
            CancelResult: 结构化取消结果
        """
        job = self.jobs.get(job_id)

        # 任务不在队列服务中（可能是已完成或已删除）
        if not job:
            if delete_data:
                try:
                    result = self.transcription_service.cancel_job(job_id, delete_data=True)
                    success, err = result if isinstance(result, tuple) else (bool(result), None)
                    if success:
                        self._notify_job_removed(job_id)
                        self._remove_cancellation_token(job_id)
                        return CancelResult(
                            success=True,
                            status="removed",
                            reason_code="cancel_already",
                            message="任务已删除",
                            pending_delete=False,
                            state_seq=0,
                        )
                    reason_code = "delete_blocked" if "占用" in (err or "") else "delete_failed"
                    return CancelResult(
                        success=False,
                        status="unknown",
                        reason_code=reason_code,
                        message=err or CANCEL_REASON_CODES[reason_code],
                        pending_delete=False,
                        state_seq=0,
                    )
                except Exception as e:
                    logger.warning(f"删除任务 {job_id} 失败: {e}")
                    return CancelResult(
                        success=False,
                        status="unknown",
                        reason_code="delete_failed",
                        message=str(e),
                        pending_delete=False,
                        state_seq=0,
                    )
            return CancelResult(
                success=False,
                status="not_found",
                reason_code="cancel_not_found",
                message=CANCEL_REASON_CODES["cancel_not_found"],
                pending_delete=False,
                state_seq=0,
            )

        from_status = job.status
        is_running = False  # [V3.1.0] 标记是否为正在运行的任务
        reason_code = "cancel_queued"
        result_message = CANCEL_REASON_CODES["cancel_queued"]
        cancel_request_time = time.time()
        logger.info(
            "[Lifecycle] 取消请求: job=%s, current_status=%s, delete_data=%s, request_time=%s",
            job_id,
            job.status,
            delete_data,
            cancel_request_time,
        )

        if job.status in ("finished", "failed", "canceled", "force_canceled", "removed") and not delete_data:
            return CancelResult(
                success=True,
                status=job.status,
                reason_code="cancel_already",
                message=CANCEL_REASON_CODES["cancel_already"],
                pending_delete=False,
                state_seq=int(job.state_seq or 0),
            )
        if (
            delete_data
            and job.status in ("finished", "failed", "canceled", "force_canceled", "removed")
            and self.running_job_id != job_id
        ):
            result = self.transcription_service.cancel_job(job_id, delete_data=True)
            success, err = result if isinstance(result, tuple) else (bool(result), None)
            if not success:
                reason_code = "delete_blocked" if "占用" in (err or "") else "delete_failed"
                return CancelResult(
                    success=False,
                    status=job.status,
                    reason_code=reason_code,
                    message=err or CANCEL_REASON_CODES[reason_code],
                    pending_delete=False,
                    state_seq=int(job.state_seq or 0),
                )
            with self.lock:
                if job_id in self.jobs:
                    del self.jobs[job_id]
            self._remove_cancellation_token(job_id)
            self._notify_job_removed(job_id)
            return CancelResult(
                success=True,
                status="removed",
                reason_code="cancel_already",
                message="任务已删除",
                pending_delete=False,
                state_seq=int(job.state_seq or 0),
            )

        with self.lock:
            # 设置取消标志
            job.canceled = True

            # [v3.1.0] 触发取消令牌的取消
            token = self.cancellation_tokens.get(job_id)
            if token:
                token.cancel()
                logger.info(f"[v3.1.0] 已触发取消令牌取消: {job_id}")

            # 如果在队列中，直接移除并标记为已取消
            if job_id in self.queue:
                if not self._transition_job_status(job, "canceled", "cancel_queued"):
                    logger.error("取消失败，状态迁移被拒绝: job=%s", job_id)
                    return CancelResult(
                        success=False,
                        status=job.status,
                        reason_code="delete_failed",
                        message="状态迁移被拒绝",
                        pending_delete=False,
                        state_seq=int(job.state_seq or 0),
                    )
                self.queue.remove(job_id)
                job.message = "已取消（未开始）"
                logger.info(
                    "[Lifecycle] 取消终态达成: job=%s, status=%s, total_cancel_duration=%.1fs",
                    job_id,
                    job.status,
                    time.time() - cancel_request_time,
                )
                reason_code = "cancel_queued"
                result_message = CANCEL_REASON_CODES["cancel_queued"]

            # [V3.1.0] 如果是正在运行的任务，进入"取消中"状态
            # 不再立即清除 running_job_id，让 Worker 的 finally 块处理
            elif self.running_job_id == job_id:
                is_running = True
                if not self._transition_job_status(job, "canceling", "cancel_running"):
                    logger.error("取消失败，状态迁移被拒绝: job=%s", job_id)
                    return CancelResult(
                        success=False,
                        status=job.status,
                        reason_code="delete_failed",
                        message="状态迁移被拒绝",
                        pending_delete=delete_data,
                        state_seq=int(job.state_seq or 0),
                    )
                # 运行中删除：提示将延迟自动删除
                job.message = "当前有进程占用，将延迟自动删除"
                # 记录取消请求时间，用于超时保障
                self._pending_cancel_requests[job_id] = cancel_request_time
                if delete_data:
                    self._pending_delete_after_cancel.add(job_id)
                logger.info(f"[V3.1.0] 任务进入取消中状态: {job_id}")
                logger.info(
                    "[Lifecycle] 进入 canceling: job=%s, enter_time=%s",
                    job_id,
                    cancel_request_time,
                )
                reason_code = "cancel_running"
                result_message = "任务正在执行，已请求取消" + ("并将在结束后删除" if delete_data else "")
            else:
                if not self._transition_job_status(job, "canceled", "cancel_direct"):
                    logger.error("取消失败，状态迁移被拒绝: job=%s", job_id)
                    return CancelResult(
                        success=False,
                        status=job.status,
                        reason_code="delete_failed",
                        message="状态迁移被拒绝",
                        pending_delete=False,
                        state_seq=int(job.state_seq or 0),
                    )
                job.message = "已取消"
                reason_code = "cancel_queued"
                result_message = "任务已取消"

        self._persist_queue_and_jobs([job], {job.job_id: from_status}, reason="cancel_request")

        # [V3.1.0] 正在运行的任务：延迟处理删除，由 Worker 或超时监控完成
        if is_running:
            # 不在这里删除数据，等待任务真正结束
            # 推送状态变更
            self._notify_queue_change()
            self._notify_job_status(job_id, job.status)
            self._notify_job_signal(job_id, "job_canceling")  # 新信号
            # 如果需要删除数据，标记完成后再删
            if delete_data:
                with self.lock:
                    self._pending_delete_after_cancel.add(job_id)
            return CancelResult(
                success=True,
                status=job.status,
                reason_code=reason_code,
                message=result_message,
                pending_delete=delete_data,
                state_seq=int(job.state_seq or 0),
            )

        # 非运行中的任务：立即处理
        if delete_data:
            result = self.transcription_service.cancel_job(job_id, delete_data=True)
            success, err = result if isinstance(result, tuple) else (bool(result), None)
            if not success:
                # 删除失败时不广播删除事件，保留任务文件供用户重试
                reason_code = "delete_blocked" if "占用" in (err or "") else "delete_failed"
                return CancelResult(
                    success=False,
                    status=job.status,
                    reason_code=reason_code,
                    message=err or CANCEL_REASON_CODES[reason_code],
                    pending_delete=False,
                    state_seq=int(job.state_seq or 0),
                )

            # [V3.1.0] 从内存中彻底移除任务，防止幽灵任务
            with self.lock:
                if job_id in self.jobs:
                    del self.jobs[job_id]
                    logger.info(f"[幽灵任务修复] 已从内存移除任务: {job_id}")

            # [v3.1.0] 清理取消令牌
            self._remove_cancellation_token(job_id)
        else:
            success, err = True, None

        # [V3.1.0] 根据是否删除数据，推送不同事件
        if delete_data:
            # 推送 job_removed 事件（任务被彻底删除）
            self._notify_job_removed(job_id)
        else:
            # 推送状态变更事件（任务仍存在）
            self._notify_queue_change()
            self._notify_job_status(job_id, job.status)
            # 同时推送到单任务频道，确保 EditorView 能收到
            self._notify_job_signal(job_id, "job_canceled")

        if delete_data and success:
            return CancelResult(
                success=True,
                status="removed",
                reason_code=reason_code,
                message="任务已取消并删除",
                pending_delete=False,
                state_seq=int(job.state_seq or 0),
            )

        return CancelResult(
            success=bool(success),
            status=job.status,
            reason_code=reason_code,
            message=err or result_message,
            pending_delete=False,
            state_seq=int(job.state_seq or 0),
        )

    def _worker_loop(self):
        """
        Worker线程主循环

        核心逻辑:
        1. 从队列取任务
        2. 执行任务（阻塞）
        3. 清理资源
        4. 循环
        """
        logger.info("Worker循环已启动")

        while not self.stop_event.is_set():
            try:
                # 1. 检查队列是否为空
                with self.lock:
                    if not self.queue:
                        # 队列为空，休眠1秒
                        pass
                    else:
                        # 取队头任务（不移除，防止出错丢失）
                        job_id = self.queue[0]
                        job = self.jobs.get(job_id)

                        # 验证任务有效性
                        if not job:
                            logger.warning(f"⚠️ 任务不存在，跳过: {job_id}")
                            self.queue.popleft()
                            continue

                        if job.status == "paused":
                            # V3.2.0+dev.20260124.01: 重启后保留队列顺序，等待用户恢复
                            logger.info(f"[V3.2.0+dev.20260124.01] 队列头任务已暂停，等待恢复: {job_id}")
                        elif job.status in ["canceled", "canceling", "force_canceled", "failed", "removed"]:
                            logger.info(f"⏭️ 跳过已取消/失败的任务: {job_id}")
                            self.queue.popleft()
                            continue
                        else:
                            if self._is_runner_gate_blocking_locked():
                                now = time.time()
                                if now - self._runner_gate_last_warn_at >= self._runner_gate_warn_interval_seconds:
                                    logger.warning(
                                        "[RunnerGate] GPU 忙碌覆盖生效，等待孤儿任务退出后再调度: orphan_jobs=%s",
                                        list(self._orphan_executions.keys()),
                                    )
                                    self._runner_gate_last_warn_at = now
                            else:
                                # 正式从队列移除
                                self.queue.popleft()
                                self.running_job_id = job_id
                                self._current_executing_job_id = job_id  # [V3.1.0] 记录实际执行的任务ID
                                if not self._transition_job_status(job, "processing", "worker_start"):
                                    logger.error("任务启动失败，状态迁移被拒绝: job=%s", job_id)
                                    self.running_job_id = None
                                    self._current_executing_job_id = None
                                    continue
                                job.message = "开始处理"

                                # V3.1.0: 在推送 SSE 之前，先从 checkpoint 恢复进度
                                # 这样断点续传时前端收到的进度是正确的，而非 0
                                self._restore_progress_from_checkpoint(job)

                                # 推送队列变化和任务状态通知（在lock内，避免数据不一致）
                                self._notify_queue_change()
                                self._notify_job_status(job_id, "processing")
                                # 推送初始进度（让前端立即知道任务的初始状态）
                                self._notify_job_progress(job_id)

                    # 任务开始执行前保存状态（确保断电后能恢复 running 任务）
                    if self.running_job_id:
                        self._save_state()
                        # 同时保存任务元信息（记录 processing 状态）
                        job = self.jobs.get(self.running_job_id)
                        if job:
                            self.transcription_service.save_job_meta(job)
                            self.heartbeat_service.acquire_lease(
                                job.job_id,
                                self._lease_owner,
                                self._heartbeat_ttl_seconds,
                            )

                        # [v3.1.0] 创建取消令牌
                        token = self._create_cancellation_token(self.running_job_id)
                        logger.debug(f"[v3.1.0] 已创建取消令牌: {self.running_job_id}")

                        # V3.1.2+dev.20260114.11: 新任务开始前，智能处理正在运行的 720p 转码
                        self._maybe_throttle_or_pause_proxy()

                # 2. 如果没有任务，休眠后继续
                if self.running_job_id is None:
                    time.sleep(1)
                    continue

                # 3. 执行任务（阻塞，直到完成/失败/暂停/取消）
                job = self.jobs[self.running_job_id]
                logger.info(f" 开始执行任务: {self.running_job_id}")

                try:
                    transcription = getattr(job.settings, "transcription", None)
                    transcription_profile = (
                        transcription.transcription_profile
                        if transcription else "sensevoice_only"
                    )
                    preset_id = getattr(job.settings, "preset_id", "balanced")

                    logger.info(
                        "路由决策: profile=%s, preset=%s",
                        transcription_profile,
                        preset_id,
                    )

                    logger.info(
                        "使用双流对齐流水线 (profile=%s, preset=%s)",
                        transcription_profile,
                        preset_id,
                    )
                    _run_async_safely(self._run_dual_alignment_pipeline(job, preset_id))

                    # 检查最终状态
                    if job.canceled:
                        self._transition_job_status(job, "canceled", "pipeline_canceled")
                        job.message = "已取消"
                    elif job.paused:
                        self._transition_job_status(job, "paused", "pipeline_paused")
                        job.message = "已暂停"
                    else:
                        is_finish_valid, finish_error = self._validate_finish_integrity(job)
                        if not is_finish_valid:
                            raise RuntimeError(finish_error)
                        self._transition_job_status(job, "finished", "pipeline_complete")
                        job.message = "完成"
                        logger.info(f"任务完成: {self.running_job_id}")

                except CancelledException as e:
                    # [v3.1.0] 捕获取消异常
                    self._transition_job_status(job, "canceled", "pipeline_canceled")
                    job.message = "已取消"
                    logger.info(f"[v3.1.0] 任务被取消: {e.job_id}")

                except PausedException as e:
                    # [v3.1.0] 捕获暂停异常
                    self._transition_job_status(job, "paused", "pipeline_paused")
                    job.message = "已暂停"
                    self._notify_pause_ack(job)
                    logger.info(f"[v3.1.0] 任务已暂停: {e.job_id}")

                except Exception as e:
                    self._transition_job_status(job, "failed", "pipeline_error")
                    job.message = f"失败: {e}"
                    job.error = str(e)
                    logger.error(f"任务执行失败: {self.running_job_id} - {e}", exc_info=True)

                finally:
                    # 4. 清理资源（关键！）
                    # [V3.1.0] 使用 _current_executing_job_id 而非 running_job_id
                    # 因为 running_job_id 可能被超时监控清除
                    finished_job_id = self._current_executing_job_id
                    cancel_request_time = None
                    with self.lock:
                        self.running_job_id = None
                        self._current_executing_job_id = None
                        # [V3.1.0] 从待取消列表移除
                        cancel_request_time = self._pending_cancel_requests.pop(finished_job_id, None)
                        self._clear_orphan_execution_locked(finished_job_id)

                    if (
                        cancel_request_time is not None
                        and job.status in ("canceled", "force_canceled")
                    ):
                        logger.info(
                            "[Lifecycle] 取消终态达成: job=%s, status=%s, total_cancel_duration=%.1fs",
                            finished_job_id,
                            job.status,
                            time.time() - cancel_request_time,
                        )

                    # [v3.1.0] 清理取消令牌
                    self._remove_cancellation_token(finished_job_id)
                    self.heartbeat_service.release(finished_job_id, self._lease_owner)

                    # 资源大清洗
                    self._cleanup_resources()

                    # [V3.1.0] 处理取消后的延迟删除
                    need_delete_data = finished_job_id in self._pending_delete_after_cancel
                    if need_delete_data:
                        self._pending_delete_after_cancel.discard(finished_job_id)
                        logger.info(f"[V3.1.0] 执行取消后的延迟删除: {finished_job_id}")
                        try:
                            result = self.transcription_service.cancel_job(finished_job_id, delete_data=True)
                            success, err = result if isinstance(result, tuple) else (bool(result), None)
                            if success:
                                with self.lock:
                                    if finished_job_id in self.jobs:
                                        del self.jobs[finished_job_id]
                                self._notify_job_removed(finished_job_id)
                                # 跳过后续的状态保存和通知
                                self._save_state()
                                continue
                            # 删除失败（如外部占用），保留任务并提示稍后重试
                            job.message = err or "当前有进程占用，请稍后再试"
                            self.transcription_service.save_job_meta(job)
                            self._notify_job_status(job.job_id, job.status)
                            logger.error(f"[V3.1.0] 延迟删除失败: {finished_job_id}, {err}")
                        except Exception as e:
                            logger.error(f"[V3.1.0] 延迟删除失败: {finished_job_id}, {e}")

                    # 保存任务最终状态到状态仓库
                    self.transcription_service.save_job_meta(job)

                    # 推送任务结束信号（单任务频道）
                    # 使用统一的命名空间前缀格式：signal.{signal_type}
                    signal_type = "job_complete" if job.status == "finished" else f"job_{job.status}"
                    self.sse_manager.broadcast_sync(
                        f"job:{job.job_id}",
                        f"signal.{signal_type}",
                        {
                            "signal": signal_type,
                            "job_id": job.job_id,
                            "message": job.message,
                            "status": job.status,
                            "percent": round(job.progress, 1)
                        }
                    )

                    # 推送全局SSE通知
                    self._notify_job_status(job.job_id, job.status)
                    self._notify_queue_change()

                    # 5. 检查是否需要恢复被中断的任务（强制插队后的自动恢复）
                    self._try_restore_interrupted_job(finished_job_id, job.status)

                    # 保存队列状态
                    self._save_state()

                    # V3.1.2+dev.20260114.04: 队列变空/任务完成时通知720p调度器
                    if job.status == "finished":
                        self._trigger_720p_check_after_job_complete(job.job_id)

            except Exception as e:
                logger.error(f"Worker循环异常: {e}", exc_info=True)
                time.sleep(1)

        logger.info("Worker循环已停止")

    def _clear_orphan_execution_locked(self, job_id: Optional[str]) -> None:
        """清理已退出执行对应的孤儿标记（需持有 self.lock）。"""
        if not job_id:
            return

        if self._orphan_executions.pop(job_id, None) is not None:
            logger.info("[RunnerGate] 孤儿任务已实际退出并清理: %s", job_id)

        if not self._orphan_executions and self._is_gpu_busy_override:
            self._is_gpu_busy_override = False
            logger.info("[RunnerGate] 孤儿任务全部清理，解除 GPU 忙碌覆盖")

    def _check_orphan_cleanup_locked(self, now: Optional[float] = None) -> bool:
        """检查并清理超时孤儿任务（需持有 self.lock）。"""
        if not self._orphan_executions:
            return True

        checkpoint_time = now if now is not None else time.time()
        expired_job_ids = [
            orphan_job_id
            for orphan_job_id, marked_at in self._orphan_executions.items()
            if checkpoint_time - marked_at >= self._runner_gate_orphan_timeout_seconds
        ]
        for orphan_job_id in expired_job_ids:
            self._orphan_executions.pop(orphan_job_id, None)
            logger.warning("[RunnerGate] 孤儿任务超过保护窗口，按超时清理: %s", orphan_job_id)

        return not self._orphan_executions

    def _is_runner_gate_blocking_locked(self) -> bool:
        """判断 RunnerGate 是否阻断新任务调度（需持有 self.lock）。"""
        if not RUNNER_GATE_ENABLED:
            return False
        if not self._is_gpu_busy_override:
            return False

        if self._check_orphan_cleanup_locked():
            self._is_gpu_busy_override = False
            logger.info("[RunnerGate] 孤儿任务保护窗口结束，恢复队列调度")
            return False
        return True

    def is_runner_gate_blocking(self) -> bool:
        """提供给外部模块的 RunnerGate 状态只读接口。"""
        with self.lock:
            return self._is_runner_gate_blocking_locked()

    def _cancel_timeout_monitor(self):
        """
        [V3.1.0] 取消超时监控线程

        职责：
        1. 定期检查待取消任务是否超时
        2. 超时后强制清除 running_job_id，允许队列继续
        3. 确保任务最终被清除，防止队列阻塞

        设计原则：
        - 最小侵入：不干扰正常的协作式取消流程
        - 超时保障：只在任务卡死时介入
        - 资源安全：等待流水线自然退出后再清理资源
        """
        logger.info("[V3.1.0] 取消超时监控线程已启动")

        while not self.stop_event.is_set():
            try:
                time.sleep(5)  # 每5秒检查一次

                now = time.time()
                force_cancel_list = []

                with self.lock:
                    for job_id, cancel_time in list(self._pending_cancel_requests.items()):
                        elapsed = now - cancel_time
                        if elapsed > self._force_cancel_timeout:
                            force_cancel_list.append((job_id, elapsed))

                # 处理超时的取消请求
                for job_id, elapsed in force_cancel_list:
                    self._force_cancel_timeout_job(job_id, elapsed)

            except Exception as e:
                logger.error(f"[V3.1.0] 超时监控异常: {e}", exc_info=True)

        logger.info("[V3.1.0] 取消超时监控线程已停止")

    def _force_cancel_timeout_job(self, job_id: str, elapsed: float):
        """
        [V3.1.0] 强制取消超时的任务

        当任务取消请求超时（流水线未响应）时调用此方法。
        强制清除 running_job_id，允许队列继续处理其他任务。

        注意：
        - 流水线可能仍在后台运行，但队列不再等待它
        - 当流水线最终退出时，Worker 的 finally 块会完成清理
        - 这是一种"放弃等待"策略，而非"强制终止"

        Args:
            job_id: 超时的任务ID
            elapsed: 已等待时间（秒）
        """
        logger.warning(
            f"[V3.1.0] 任务取消超时，强制放行队列: {job_id} "
            f"(等待了 {elapsed:.1f}s，超时阈值 {self._force_cancel_timeout}s)"
        )

        with self.lock:
            # 从待取消列表移除
            self._pending_cancel_requests.pop(job_id, None)

            job = self.jobs.get(job_id)
            if job:
                from_status = job.status
                self._transition_job_status(job, "force_canceled", "cancel_timeout")
                job.message = f"已强制取消（响应超时 {elapsed:.0f}s）"
                logger.info(f"[V3.1.0] 任务状态更新为 force_canceled: {job_id}")

            # 关键：强制清除 running_job_id，允许下一个任务开始
            # 注意：_current_executing_job_id 保持不变，让 Worker finally 块知道要清理谁
            if self.running_job_id == job_id:
                self.running_job_id = None
                logger.warning(f"[V3.1.0] 强制清除 running_job_id: {job_id}")
                self._orphan_executions[job_id] = time.time()
                if RUNNER_GATE_ENABLED:
                    self._is_gpu_busy_override = True
                    logger.warning(
                        "[RunnerGate] 记录孤儿执行并启用 GPU 忙碌覆盖: %s",
                        job_id,
                    )

        # 保存状态
        if job:
            self._persist_queue_and_jobs([job], {job.job_id: from_status}, reason="force_cancel_timeout")
        else:
            self._save_state()

        # 推送通知
        self._notify_queue_change()
        self._notify_job_status(job_id, "force_canceled")
        self._notify_job_signal(job_id, "job_force_canceled")

        # 如果需要删除数据，保留在 _pending_delete_after_cancel 中
        # 等 Worker 的 finally 块执行时处理
        if job_id in self._pending_delete_after_cancel:
            logger.info(f"[V3.1.0] 任务 {job_id} 的数据将在流水线退出后删除")

    async def _run_dual_alignment_pipeline(self, job: 'JobState', preset_id: str):
        """
        运行双流对齐流水线

        V3.2.0+dev.20260125.08: 重构为委托模式
        - 核心编排逻辑委托给 PipelineOrchestrator
        - 保留：管理器初始化、音频加载、用户编辑叠加、Proxy 触发
        - 移除：预处理、转录、SRT 生成（由 Orchestrator 处理）

        Args:
            job: 任务状态对象
            preset_id: 预设 ID
        """
        from app.pipelines.orchestrator import PipelineOrchestrator
        from app.services.streaming_subtitle import get_streaming_subtitle_manager, remove_streaming_subtitle_manager
        from app.services.progress_tracker import get_progress_tracker, remove_progress_tracker
        from app.services.sse_service import get_sse_manager
        from app.services.job.checkpoint_manager import CheckpointManagerV37
        from app.services.progress_emitter import get_progress_emitter, remove_progress_emitter
        from app.services.subtitle_edit_store import load_deleted_indices, load_edits
        from pathlib import Path
        import librosa
        import soundfile as sf

        def push_signal_event(sse_manager, job_id: str, signal_code: str, message: str = ""):
            """推送信号事件"""
            sse_manager.broadcast_sync(
                f"job:{job_id}",
                f"signal.{signal_code}",
                {"signal": signal_code, "message": message}
            )

        # 初始化管理器
        subtitle_manager = get_streaming_subtitle_manager(job.job_id)
        progress_tracker = get_progress_tracker(job.job_id, preset_id)
        sse_manager = get_sse_manager()

        transcription = getattr(job.settings, "transcription", None)
        transcription_profile = (
            transcription.transcription_profile
            if transcription else "sensevoice_only"
        )
        progress_emitter = get_progress_emitter(
            job, sse_manager,
            transcription_profile=transcription_profile
        )

        cancellation_token = self.get_cancellation_token(job.job_id)

        job_dir = Path(job.dir)
        checkpoint_manager = CheckpointManagerV37(job_dir, logger)
        checkpoint_manager.save_checkpoint({"original_settings": job.settings.to_dict()})

        try:
            logger.info(f"[双流对齐] 开始处理任务: {job.job_id}, preset={preset_id}")

            # 从检查点恢复进度
            checkpoint = checkpoint_manager.load_checkpoint()
            if checkpoint and hasattr(checkpoint, "to_dict"):
                progress_emitter.restore_from_checkpoint(checkpoint.to_dict())
                logger.info(f"[V3.1.0] 已恢复进度: {job.progress:.1f}%")

            # 预触发 Proxy 生成（不阻塞主流程）
            await self._maybe_trigger_proxy_generation(job)

            # 加载完整音频（用于 Audio Overlap）
            full_audio, sr = librosa.load(job.input_path, sr=16000, mono=True)
            audio_path = job_dir / "audio.wav"
            sf.write(str(audio_path), full_audio, sr)
            logger.info(f"音频文件已保存: {audio_path}")

            # 恢复字幕状态并叠加用户编辑（保留在调用方）
            if checkpoint and getattr(checkpoint, "transcription", None):
                transcription_state = checkpoint.transcription
                if transcription_state.sentences_snapshot:
                    subtitle_checkpoint_data = {
                        "sentences_snapshot": transcription_state.sentences_snapshot,
                        "sentence_count": transcription_state.sentence_count,
                        "chunk_sentences_map": transcription_state.chunk_sentences_map,
                    }
                    if subtitle_manager.restore_from_checkpoint(subtitle_checkpoint_data):
                        logger.info(
                            "[V3.1.0] 字幕状态已恢复: %s 个句子",
                            len(transcription_state.sentences_snapshot),
                        )
                        # V3.2.0+dev.20260125.08: 叠加用户编辑
                        try:
                            edits = load_edits(job_dir)
                            deleted_indices = load_deleted_indices(job_dir)
                            subtitle_manager.apply_user_edits(edits)
                            subtitle_manager.apply_user_deletions(list(deleted_indices))
                            subtitle_manager.apply_manual_entries(edits)
                        except Exception as exc:
                            logger.warning("叠加用户编辑失败: %s", exc)
                        subtitle_manager.push_restored_subtitles_to_frontend()
                    else:
                        logger.warning("[V3.1.0] 字幕恢复失败，将从头生成字幕")
                else:
                    logger.info("[V3.1.0] checkpoint 中无字幕快照，字幕将从头生成")

            # V3.2.0+dev.20260125.08: 委托给 PipelineOrchestrator
            orchestrator = PipelineOrchestrator(
                job_lifecycle=self.transcription_service.job_lifecycle,
                sse_manager=sse_manager,
                hardware_profile_provider=self.transcription_service.hardware_profile_provider,
                logger=logger,
            )
            await orchestrator.run_pipeline(
                job,
                cancellation_token=cancellation_token,
                progress_emitter=progress_emitter,
                progress_tracker=progress_tracker,
                subtitle_manager=subtitle_manager,
                checkpoint_manager=checkpoint_manager,
                job_dir=job_dir,
                full_audio_array=full_audio,
                full_audio_sr=sr,
            )

            logger.info(f"[双流对齐] 任务完成: {job.job_id}")

        except CancelledException as e:
            logger.info("[双流对齐] 任务取消: %s", e.job_id)
            raise

        except PausedException as e:
            logger.info("[双流对齐] 任务暂停: %s", e.job_id)
            raise

        except Exception as e:
            logger.error(f"[双流对齐] 任务失败: {e}", exc_info=True)
            self._transition_job_status(job, "failed", "pipeline_error")
            job.error = str(e)
            push_signal_event(sse_manager, job.job_id, "job_failed", str(e))
            raise

        finally:
            # 清理资源
            remove_streaming_subtitle_manager(job.job_id)
            remove_progress_tracker(job.job_id)
            remove_progress_emitter(job.job_id)

    async def _maybe_trigger_proxy_generation(self, job: 'JobState'):
        """
        预检视频格式，提前触发 Proxy 生成（不阻塞主流程）

        在转录任务开始时调用，让 H265 等不兼容格式的视频提前开始转码，
        用户打开编辑器时可能已完成转码。
        """
        from app.services.media_prep_service import get_media_prep_service
        from app.api.routes.media_routes import (
            _get_video_codec, NEED_TRANSCODE_CODECS, NEED_TRANSCODE_FORMATS,
            BROWSER_COMPATIBLE_FORMATS, _find_video_file
        )

        try:
            job_dir = Path(job.dir)

            # 查找视频文件
            video_file = _find_video_file(job_dir)
            if not video_file:
                return

            # 检查是否需要转码
            needs_transcode = False
            if video_file.suffix.lower() in NEED_TRANSCODE_FORMATS:
                needs_transcode = True
            elif video_file.suffix.lower() in BROWSER_COMPATIBLE_FORMATS:
                codec = _get_video_codec(video_file)
                if codec and codec in NEED_TRANSCODE_CODECS:
                    needs_transcode = True

            if needs_transcode:
                # 先生成 360p 预览（高优先级，快速）
                preview_360p = job_dir / "preview_360p.mp4"
                if not preview_360p.exists():
                    media_prep = get_media_prep_service()
                    enqueued = media_prep.enqueue_preview(
                        job.job_id, video_file, preview_360p, priority=5
                    )
                    if enqueued:
                        logger.info(f"[Proxy预生成] 检测到不兼容格式，提前入队360p预览: {job.job_id}")

                # 720p 将在 360p 完成后或转录完成后自动触发（由 media_prep_service 和 transcription_service 处理）
        except Exception as e:
            # 预触发失败不影响主流程
            logger.warning(f"[Proxy预生成] 预触发失败，忽略: {e}")

    def _trigger_720p_check_after_job_complete(self, completed_job_id: str):
        """
        任务完成后触发720p转码检查（V3.1.0新增）

        解决问题:
        - 360p完成时如果队列繁忙（有转录任务正在执行），就不会安排720p检查
        - 之后队列变空闲，但没有任何机制重新触发720p检查
        - 此方法在每个任务完成后检查是否有待处理的720p转码

        策略:
        1. 扫描所有已完成任务的目录
        2. 找到有360p但没有720p的任务
        3. 触发720p转码
        """
        # V3.1.2+dev.20260114.04: 队列空闲时直接通知调度器，由调度器选择待触发任务
        # V3.1.2+dev.20260114.09: 队列空闲后延迟10秒再触发，给用户启动新任务的机会
        try:
            # 若已有未触发的定时器先取消，保证只保留最新的10秒窗口
            if self._pending_proxy_idle_timer:
                self._pending_proxy_idle_timer.cancel()
                self._pending_proxy_idle_timer = None

            def _delayed_notify():
                try:
                    from app.services.proxy_720_scheduler import get_proxy_scheduler
                    scheduler = get_proxy_scheduler()
                    scheduler.on_queue_idle()
                except Exception as e:
                    logger.debug(f"[720p触发] 调度器通知失败（非致命）: {e}")
                finally:
                    self._pending_proxy_idle_timer = None

            timer = threading.Timer(10, _delayed_notify)
            self._pending_proxy_idle_timer = timer
            timer.start()
        except Exception as e:
            logger.debug(f"[720p触发] 延迟触发安排失败（非致命）: {e}")

    # V3.1.2+dev.20260114.11: 新任务开始时，智能决定保留/中断正在运行的 720p 转码
    def _maybe_throttle_or_pause_proxy(self):
        try:
            from app.services.media_prep_service import get_media_prep_service
            from app.services.proxy_720_scheduler import get_proxy_scheduler

            media_prep = get_media_prep_service()
            scheduler = get_proxy_scheduler()

            # 获取正在处理的 proxy_720p 任务
            active = []
            with media_prep.lock:
                for jid, status in media_prep.task_status.items():
                    proxy = status.get("proxy_720p") if isinstance(status, dict) else None
                    if proxy and proxy.get("status") == "processing":
                        active.append((jid, proxy.get("progress", 0)))

            if not active:
                return

            policy = config.PROXY_CONFIG.get("proxy_pause_policy", {})
            # 自适应阈值：基础 50%，长视频稍微提高，短视频稍微降低
            default_base = policy.get("progress_cutoff_base", 0.5)
            min_cutoff = policy.get("progress_cutoff_min", 0.4)
            max_cutoff = policy.get("progress_cutoff_max", 0.6)
            default_time_cutoff = policy.get("time_left_cutoff_seconds")  # 可为空

            for job_id, progress in active:
                video_path = self._find_video_file(job_id)
                duration = 0
                if video_path:
                    try:
                        duration = media_prep._get_video_duration(video_path)
                    except Exception:
                        duration = 0

                # 计算自适应进度阈值
                adapt_factor = min(duration / 7200, 1.0) if duration > 0 else 0
                progress_cutoff = default_base + 0.1 * adapt_factor
                progress_cutoff = max(min_cutoff, min(max_cutoff, progress_cutoff))

                # 剩余时间阈值
                if default_time_cutoff is not None:
                    time_left_cutoff = default_time_cutoff
                else:
                    time_left_cutoff = min(900, duration * 0.2) if duration > 0 else 900

                remaining = duration * max(0, 1 - progress / 100) if duration > 0 else 999999

                logger.info(
                    f"[Proxy调度] 新任务即将开始，检测720p: job={job_id}, progress={progress:.1f}%, "
                    f"duration={duration:.1f}s, remaining≈{remaining:.1f}s, "
                    f"cutoff={progress_cutoff:.2f}, time_cutoff={time_left_cutoff:.1f}s"
                )

                # 决策：进度高且剩余不长 -> 降优先级继续；否则中断并重排队（队首）
                if progress >= progress_cutoff and remaining <= time_left_cutoff:
                    media_prep.set_low_priority_by_job(job_id)
                    logger.info(f"[Proxy调度] 进度高，改为低优先级继续: {job_id}")
                else:
                    paused = media_prep.cancel_proxy_job(job_id, "paused_for_new_job")
                    if paused:
                        # 清理可能的半成品，避免返回坏文件
                        proxy_file = config.JOBS_DIR / job_id / "proxy_720p.mp4"
                        if proxy_file.exists():
                            try:
                                proxy_file.unlink()
                                logger.info(f"[Proxy调度] 已删除半成品720p: {proxy_file}")
                            except Exception as e:
                                logger.warning(f"[Proxy调度] 删除半成品720p失败: {proxy_file}, {e}")

                        if video_path:
                            # 标记暂停（待检查），避免前端误判失败
                            scheduler.mark_paused(job_id, "paused_for_new_job")
                            # 强制接受（即便 auto_trigger 关闭也恢复之前的任务），优先级最高
                            scheduler.request(
                                job_id,
                                video_path,
                                trigger_type="auto_resume",
                                auto_enabled=True,
                                force=True,
                                priority=0  # 队首优先
                            )
                            logger.info(
                                f"[Proxy调度] 进度低/剩余长，终止并重排队(优先级高): {job_id}, progress={progress:.1f}%"
                            )
                        else:
                            logger.warning(f"[Proxy调度] 终止后未找到源视频，无法重排队: {job_id}")
        except Exception as e:
            logger.debug(f"[Proxy调度] 智能暂停/降级失败（非致命）: {e}")

    def _cleanup_resources(self):
        """
        资源大清洗（增强版）

        策略:
        1. 清理 Whisper 模型（1-3GB）
        2. 保留最近使用的3个对齐模型（LRU，共~600MB）
        3. GC + CUDA 清理
        """
        logger.info("开始资源清理（增强版）...")

        # 1. 清空 Whisper 模型缓存
        try:
            from app.services.model_cache_service import get_model_cache_service

            get_model_cache_service(logger=logger).clear_whisper_cache()
        except Exception as e:
            logger.warning(f"清空模型缓存失败: {e}")

        # 2. Python垃圾回收
        gc.collect()
        logger.debug("  - Python GC 完成")

        # 3. CUDA显存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # 记录显存状态（调试用）
            try:
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.debug(f"  - 显存: 已分配 {memory_allocated:.2f}GB, 已保留 {memory_reserved:.2f}GB")
            except:
                pass

            logger.debug("  - CUDA缓存已清空")

        # 4. 等待资源释放
        time.sleep(1)

        logger.info("资源清理完成")

    def _try_restore_interrupted_job(self, finished_job_id: str, finished_status: str):
        """
        尝试恢复被强制中断的任务

        当插队任务完成后，自动将被中断的任务重新加入队列头部

        Args:
            finished_job_id: 刚完成的任务ID
            finished_status: 刚完成任务的状态
        """
        with self.lock:
            # 检查是否有被中断的任务需要恢复
            if not self.interrupted_job_id:
                return

            interrupted_job = self.jobs.get(self.interrupted_job_id)
            if not interrupted_job:
                logger.warning(f"被中断的任务不存在: {self.interrupted_job_id}")
                self.interrupted_job_id = None
                return

            # 只有插队任务正常完成时才自动恢复
            # 如果插队任务失败或被取消，不自动恢复（让用户决定）
            if finished_status == "finished":
                # 将被中断的任务重新加入队列头部
                if self.interrupted_job_id not in self.queue:
                    self.queue.appendleft(self.interrupted_job_id)
                    self._transition_job_status(interrupted_job, "queued", "interrupted_resume")
                    interrupted_job.paused = False
                    interrupted_job.message = "插队任务已完成，自动恢复执行"
                    logger.info(f"[自动恢复] 被中断的任务已恢复到队头: {self.interrupted_job_id}")
            else:
                # 插队任务未正常完成，被中断任务保持暂停状态
                interrupted_job.message = f"插队任务{finished_status}，需手动恢复"
                logger.info(f"[未恢复] 插队任务状态={finished_status}，被中断任务需手动恢复: {self.interrupted_job_id}")

            # 清除中断标记
            self.interrupted_job_id = None

    # ========== 全局SSE通知方法 (V3.0) ==========

    def _notify_queue_change(self):
        """推送队列变化事件到全局SSE"""
        with self.lock:
            data = {
                "queue": list(self.queue),
                "running": self.running_job_id,
                "interrupted": self.interrupted_job_id,
                "timestamp": time.time(),
                "updated_at": int(time.time() * 1000)
            }

        self.sse_manager.broadcast_sync("global", "queue_update", data)
        logger.debug(f"[全局SSE] 推送队列变化: queue={len(data['queue'])}个, running={data['running']}")

    def _notify_job_status(self, job_id: str, status: str):
        """推送任务状态变化到全局SSE"""
        job = self.jobs.get(job_id)
        if not job:
            return

        updated_at_ms = int(time.time() * 1000)
        job.updatedAt = updated_at_ms
        data = {
            "id": job_id,
            "status": status,
            "state_seq": int(job.state_seq or 0),
            "percent": round(job.progress, 1),  # 统一字段名为 percent，保留1位小数
            "message": job.message,
            "filename": job.filename,
            "phase": job.phase,  # 新增：阶段信息
            "timestamp": time.time(),
            "updated_at": updated_at_ms
        }

        self.sse_manager.broadcast_sync("global", "job_status", data)
        logger.debug(f"[全局SSE] 推送任务状态: {job_id[:8]}... -> {status}")

    def _heartbeat_loop(self) -> None:
        """后台刷新运行中任务的心跳"""
        while not self.stop_event.is_set():
            try:
                job_id = self.running_job_id
                if job_id:
                    refreshed = self.heartbeat_service.refresh(
                        job_id,
                        self._lease_owner,
                        self._heartbeat_ttl_seconds,
                    )
                    if not refreshed:
                        self.heartbeat_service.acquire_lease(
                            job_id,
                            self._lease_owner,
                            self._heartbeat_ttl_seconds,
                        )
                time.sleep(self._heartbeat_interval_seconds)
            except Exception as exc:
                logger.debug(f"[心跳] 刷新失败: {exc}")
                time.sleep(self._heartbeat_interval_seconds)

    def _notify_job_progress(self, job_id: str):
        """推送任务进度更新到全局SSE（低频调用，节省带宽）"""
        job = self.jobs.get(job_id)
        if not job:
            return

        updated_at_ms = int(time.time() * 1000)
        job.updatedAt = updated_at_ms
        data = {
            "id": job_id,
            "percent": round(job.progress, 1),  # 统一字段名为 percent，保留1位小数
            "phase": job.phase,
            "phase_percent": round(job.phase_percent, 1),  # 新增：阶段内进度
            "message": job.message,
            "processed": job.processed,
            "total": job.total,
            "timestamp": time.time(),
            "updated_at": updated_at_ms
        }

        self.sse_manager.broadcast_sync("global", "job_progress", data)

    def _notify_job_signal(self, job_id: str, signal: str):
        """
        推送关键信号到单任务SSE频道

        用于暂停/取消/恢复等关键操作，确保 EditorView 能收到状态变更通知

        Args:
            job_id: 任务ID
            signal: 信号类型 (job_paused, job_canceled, job_resumed)
        """
        job = self.jobs.get(job_id)
        if not job:
            return

        data = {
            "signal": signal,
            "job_id": job_id,
            "status": job.status,
            "state_seq": int(job.state_seq or 0),
            "message": job.message,
            "percent": round(job.progress, 1),
            "updated_at": int(time.time() * 1000)
        }

        self.sse_manager.broadcast_sync(f"job:{job_id}", f"signal.{signal}", data)
        logger.debug(f"[单任务SSE] 推送信号: {job_id[:8]}... -> signal.{signal}")

    def _validate_finish_integrity(self, job: "JobState") -> tuple[bool, str]:
        """
        完成态校验（finished 前最后闸门）。

        约束：
        1. finalized 覆盖数必须达到 total_chunks。
        2. sentences_snapshot 不得残留草稿标记。
        """
        try:
            job_dir = Path(job.dir) if job.dir else None
            if not job_dir or not job_dir.exists():
                return True, ""

            checkpoint_path = job_dir / "checkpoint.json"
            snapshot_path = job_dir / "transcription_text.json"
            source = "none"
            transcription: Dict[str, Any] = {}
            total_chunks = int(job.total or 0)

            if checkpoint_path.exists():
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)
                if isinstance(checkpoint_data, dict):
                    transcription = checkpoint_data.get("transcription", {}) or {}
                    total_chunks = int(
                        checkpoint_data.get("preprocessing", {}).get("total_chunks", 0)
                        or transcription.get("total_chunks", 0)
                        or total_chunks
                    )
                source = "checkpoint"
            elif snapshot_path.exists():
                with open(snapshot_path, "r", encoding="utf-8") as f:
                    snapshot_data = json.load(f)
                if isinstance(snapshot_data, dict):
                    transcription = snapshot_data
                    total_chunks = int(transcription.get("total_chunks", 0) or total_chunks)
                source = "snapshot"
            else:
                return True, ""

            finalized_indices = transcription.get("finalized_indices")
            if finalized_indices is None:
                alignment = transcription.get("alignment", {})
                if isinstance(alignment, dict):
                    finalized_indices = alignment.get("finalized_indices", [])
            finalized_set = set()
            for raw_idx in finalized_indices or []:
                try:
                    finalized_set.add(int(raw_idx))
                except (TypeError, ValueError):
                    continue

            if total_chunks > 0 and len(finalized_set) < total_chunks:
                missing = sorted(set(range(total_chunks)) - finalized_set)
                return (
                    False,
                    "完成态校验失败: "
                    f"{source} 定稿覆盖不足 {len(finalized_set)}/{total_chunks}, "
                    f"missing={missing[:20]}",
                )

            sentences_snapshot = transcription.get("sentences_snapshot", [])
            draft_count = 0
            for item in sentences_snapshot if isinstance(sentences_snapshot, list) else []:
                if not isinstance(item, dict):
                    continue
                is_draft = bool(item.get("_is_draft", False))
                is_finalized = item.get("_is_finalized")
                if is_finalized is None:
                    is_finalized = not is_draft
                if is_draft or not bool(is_finalized):
                    draft_count += 1
            if draft_count > 0:
                return (
                    False,
                    f"完成态校验失败: {source} 存在草稿残留 draft_count={draft_count}",
                )

            return True, ""
        except Exception as exc:
            return False, f"完成态校验异常: {exc}"

    def _build_pause_ack_payload(self, job: "JobState") -> Dict[str, Any]:
        """构建暂停握手确认的载荷信息（包含检查点摘要）"""
        payload: Dict[str, Any] = {
            "signal": "pause_ack",
            "job_id": job.job_id,
            "status": job.status,
            "message": job.message,
            "percent": round(job.progress, 1),
            "checkpoint_found": False,
        }
        try:
            job_dir = Path(job.dir) if job.dir else None
            if not job_dir or not job_dir.exists():
                return payload

            # Phase 1: 优先读取 runtime_state.db 的单元提交信息。
            runtime_service = RuntimeCheckpointService(job_dir=job_dir)
            if runtime_service.has_runtime_state():
                snapshot = runtime_service.load_snapshot()
                payload.update(
                    {
                        "checkpoint_found": True,
                        "checkpoint_source": "runtime_state.db",
                        "unit_commits": snapshot.last_unit_commits,
                    }
                )
                preprocess_unit = snapshot.last_unit_commits.get("preprocess")
                if preprocess_unit:
                    payload["phase"] = f"preprocess:{preprocess_unit}"
                return payload

            checkpoint_path = job_dir / "checkpoint.json"
            if not checkpoint_path.exists():
                return payload
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)

            transcription = checkpoint_data.get("transcription", {})
            fast_indices = transcription.get("fast_processed_indices")
            if fast_indices is None:
                fast_indices = transcription.get("fast_worker", {}).get("processed_indices", [])
            slow_indices = transcription.get("slow_processed_indices")
            if slow_indices is None:
                slow_indices = transcription.get("slow_worker", {}).get("processed_indices", [])
            finalized_indices = transcription.get("finalized_indices")
            if finalized_indices is None:
                finalized_indices = transcription.get("alignment", {}).get("finalized_indices", [])

            fast_count = len(fast_indices or [])
            slow_count = len(slow_indices or [])
            finalized_count = len(finalized_indices or [])
            total_chunks = (
                checkpoint_data.get("preprocessing", {}).get("total_chunks", 0)
                or transcription.get("total_chunks", 0)
            )
            if total_chunks <= 0:
                total_chunks = job.total or max(fast_count, slow_count, finalized_count, 0)
            payload.update(
                {
                    "checkpoint_found": True,
                    "checkpoint_updated_at": checkpoint_data.get("updated_at"),
                    "phase": checkpoint_data.get("phase"),
                    "phase_status": checkpoint_data.get("phase_status"),
                    "total_chunks": total_chunks,
                    "fast_processed": fast_count,
                    "slow_processed": slow_count,
                    "finalized": finalized_count,
                }
            )
        except Exception as exc:
            payload["checkpoint_error"] = str(exc)
        return payload

    def _notify_pause_ack(self, job: "JobState") -> None:
        """V3.2.0+dev.20260123.04: 暂停握手确认（检查点落盘完成）"""
        payload = self._build_pause_ack_payload(job)
        self.sse_manager.broadcast_sync(
            f"job:{job.job_id}",
            "signal.pause_ack",
            payload,
        )
        logger.debug(f"[单任务SSE] 推送暂停确认: {job.job_id[:8]}... -> signal.pause_ack")

    def _notify_job_removed(self, job_id: str):
        """
        通知前端任务已被彻底删除（V3.1.0 新增）

        解决幽灵任务问题：当任务被删除时，广播此事件让前端移除任务卡片，
        避免 syncTasksFromBackend 时因缓存数据不一致导致任务"复活"。

        Args:
            job_id: 被删除的任务ID
        """
        data = {
            "job_id": job_id,
            "timestamp": time.time()
        }

        # 广播到全局频道
        self.sse_manager.broadcast_sync("global", "job_removed", data)

        # 同时推送队列变化
        self._notify_queue_change()

        logger.info(f"[幽灵任务修复] 已广播任务删除事件: {job_id}")

    def _restore_progress_from_checkpoint(self, job: "JobState"):
        """
        V3.1.0: 从 checkpoint 恢复任务进度

        在 resume_job 和 worker 开始执行前调用，确保 SSE 推送的进度是正确的。
        这是修复"暂停恢复后进度归零"问题的关键。

        Args:
            job: 任务状态对象
        """
        logger.info(f"[V3.1.0] 尝试从 checkpoint 恢复进度: {job.job_id}, 当前进度={job.progress:.1f}%")

        try:
            job_dir = Path(job.dir) if job.dir else None
            if not job_dir:
                logger.info(f"[V3.1.0] 任务目录为空，跳过恢复: job.dir={job.dir}")
                return

            if not job_dir.exists():
                logger.info(f"[V3.1.0] 任务目录不存在，跳过恢复: path={job_dir}")
                return

            from app.services.job.checkpoint_manager import CheckpointManagerV37

            checkpoint_manager = CheckpointManagerV37(job_dir, logger)
            checkpoint = checkpoint_manager.load_checkpoint()

            if not checkpoint:
                logger.info(f"[V3.1.0] 无 checkpoint 文件，跳过恢复")
                return

            # 从 checkpoint 恢复进度
            checkpoint_dict = checkpoint.to_dict() if hasattr(checkpoint, 'to_dict') else {}
            progress_data = checkpoint_dict.get("progress", {})

            if progress_data:
                # 直接从 progress 字段恢复（ProgressEmitter 保存的格式）
                restored_progress = progress_data.get("total", 0)
            else:
                # 从 CheckpointV37 格式计算进度
                preprocessing = checkpoint_dict.get("preprocessing", {})
                transcription = checkpoint_dict.get("transcription", {})
                total_chunks = preprocessing.get("total_chunks", 0)

                if total_chunks > 0:
                    # 计算各阶段进度
                    preprocess_done = 100 if preprocessing.get("separation_completed") else 0

                    fast_worker = transcription.get("fast_worker", {})
                    fast_count = fast_worker.get("completed_count", 0) or len(fast_worker.get("processed_indices", []))
                    fast_progress = (fast_count / total_chunks * 100) if total_chunks > 0 else 0

                    slow_worker = transcription.get("slow_worker", {})
                    slow_count = slow_worker.get("completed_count", 0) or len(slow_worker.get("processed_indices", []))
                    slow_progress = (slow_count / total_chunks * 100) if total_chunks > 0 else 0

                    alignment = transcription.get("alignment", {})
                    align_count = alignment.get("completed_count", 0) or len(alignment.get("finalized_indices", []))
                    align_progress = (align_count / total_chunks * 100) if total_chunks > 0 else 0

                    # 使用默认权重计算总进度（与 ProgressEmitter 一致）
                    # dual_stream 权重: preprocess=0.05, fast=0.25, slow=0.50, align=0.20
                    restored_progress = (
                        preprocess_done * 0.05 +
                        fast_progress * 0.25 +
                        slow_progress * 0.50 +
                        align_progress * 0.20
                    )
                else:
                    restored_progress = 0

            # 只有当恢复的进度比当前进度高时才更新（单调递增原则）
            if restored_progress > job.progress:
                old_progress = job.progress
                job.progress = round(restored_progress, 1)
                logger.info(f"[V3.1.0] 从 checkpoint 恢复进度: {job.job_id}, {old_progress:.1f}% -> {job.progress:.1f}%")
            else:
                logger.debug(f"[V3.1.0] checkpoint 进度 ({restored_progress:.1f}%) <= 当前进度 ({job.progress:.1f}%)，保持不变")

        except Exception as e:
            logger.warning(f"[V3.1.0] 恢复进度失败，保持当前进度: {job.job_id}, error={e}")

    # ==================== v3.1.0 取消令牌管理 ====================

    def _create_cancellation_token(self, job_id: str) -> CancellationToken:
        """
        创建任务的取消令牌

        Args:
            job_id: 任务ID

        Returns:
            CancellationToken: 新创建的取消令牌
        """
        # 如果已存在，先清理
        if job_id in self.cancellation_tokens:
            logger.warning(f"[v3.1.0] 取消令牌已存在，覆盖: {job_id}")

        token = create_cancellation_token(job_id)
        self.cancellation_tokens[job_id] = token
        return token

    def _remove_cancellation_token(self, job_id: str):
        """
        移除任务的取消令牌

        Args:
            job_id: 任务ID
        """
        if job_id and job_id in self.cancellation_tokens:
            del self.cancellation_tokens[job_id]
            logger.debug(f"[v3.1.0] 已移除取消令牌: {job_id}")

    def get_cancellation_token(self, job_id: str) -> Optional[CancellationToken]:
        """
        获取任务的取消令牌

        供流水线等组件使用。

        Args:
            job_id: 任务ID

        Returns:
            Optional[CancellationToken]: 取消令牌，不存在则返回 None
        """
        return self.cancellation_tokens.get(job_id)

    def _load_settings(self):
        """加载队列设置"""
        if not self.settings_file.exists():
            logger.info("无队列设置文件，使用默认设置")
            return

        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)

            self._default_prioritize_mode = settings.get("default_prioritize_mode", "gentle")
            logger.info(f"加载队列设置: 默认插队模式={self._default_prioritize_mode}")
        except Exception as e:
            logger.warning(f"加载队列设置失败: {e}")

    def _save_settings(self):
        """保存队列设置"""
        try:
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)

            settings = {
                "default_prioritize_mode": self._default_prioritize_mode,
                "timestamp": time.time()
            }

            temp_path = self.settings_file.with_suffix(".tmp")
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2)

            temp_path.replace(self.settings_file)
            logger.debug("队列设置已保存")
        except Exception as e:
            logger.error(f"保存队列设置失败: {e}")

    def get_settings(self) -> dict:
        """获取队列设置"""
        return {
            "default_prioritize_mode": self._default_prioritize_mode
        }

    def update_settings(self, default_prioritize_mode: Optional[str] = None) -> dict:
        """
        更新队列设置

        Args:
            default_prioritize_mode: 默认插队模式 ("gentle" 或 "force")

        Returns:
            更新后的设置
        """
        if default_prioritize_mode is not None:
            if default_prioritize_mode not in ("gentle", "force"):
                raise ValueError(f"无效的插队模式: {default_prioritize_mode}")
            self._default_prioritize_mode = default_prioritize_mode
            logger.info(f"更新默认插队模式: {default_prioritize_mode}")

        self._save_settings()
        return self.get_settings()

    def _save_state(self):
        """
        持久化队列状态到仓库
        """
        with self.lock:
            queue_snapshot = list(self.queue)
            running_snapshot = self.running_job_id
            interrupted_snapshot = self.interrupted_job_id

        try:
            self.state_repo.save_queue_state(
                queue=queue_snapshot,
                running_job_id=running_snapshot,
                interrupted_job_id=interrupted_snapshot,
            )
            logger.debug("队列状态已保存到仓库")
        except Exception as e:
            logger.error(f"保存队列状态失败: {e}")

    def _persist_queue_and_jobs(
        self,
        jobs: list[JobState],
        from_status_map: Dict[str, Optional[str]],
        reason: Optional[str] = None
    ) -> None:
        try:
            with self.state_repo.transaction() as conn:
                self.state_repo.save_queue_state(
                    queue=list(self.queue),
                    running_job_id=self.running_job_id,
                    interrupted_job_id=self.interrupted_job_id,
                    conn=conn,
                )
                for job in jobs:
                    self.state_guard.sync_seq(job.job_id, job.state_seq)
                    job.status = self.state_guard.normalize_status(job.status)
                    self.state_repo.upsert_task(job, conn=conn)

                for job in jobs:
                    from_status = from_status_map.get(job.job_id)
                    normalized_from_status = (
                        self.state_guard.normalize_status(from_status)
                        if from_status else from_status
                    )
                    if normalized_from_status != job.status or reason:
                        self.event_bus.emit_status_event(
                            job_id=job.job_id,
                            from_status=normalized_from_status,
                            to_status=job.status,
                            reason=reason,
                            state_seq=job.state_seq,
                            conn=conn,
                        )
        except Exception as exc:
            logger.error(f"持久化队列与任务状态失败: {exc}")

    def _load_job_for_recovery(self, job_id: str) -> Optional[JobState]:
        """
        加载任务用于恢复（优先从缓存/仓库加载，最后从 checkpoint 加载）

        这是重启恢复的核心方法，确保能正确恢复任务状态

        Args:
            job_id: 任务ID

        Returns:
            Optional[JobState]: 恢复的任务状态对象
        """
        if job_id in self.jobs:
            logger.info(f"从队列缓存获取任务: {job_id}")
            return self.jobs[job_id]

        # 1. 从状态仓库加载（包含完整的任务元信息）
        job = self.transcription_service.load_job_meta(job_id)
        if job:
            logger.info(f"从状态仓库恢复任务: {job_id}")
            return job

        # 2. 降级：从 checkpoint 恢复（兼容旧版本）
        job = self.transcription_service.restore_job_from_checkpoint(job_id)
        if job:
            logger.info(f"从 checkpoint 恢复任务（旧版兼容）: {job_id}")
            # 同时保存到状态仓库，便于下次直接加载
            self.transcription_service.save_job_meta(job)
            return job

        logger.warning(f"无法恢复任务: {job_id}")
        return None

    def _build_recovery_queue(self, state: QueueState) -> list[str]:
        """
        重启恢复时重建队列顺序（running -> interrupted -> queue）。

        保证顺序唯一性，避免重复任务占位。
        """
        ordered: list[str] = []

        def _append(job_id: Optional[str]) -> None:
            if not job_id:
                return
            if job_id not in ordered:
                ordered.append(job_id)

        _append(state.running_job_id)
        _append(state.interrupted_job_id)
        for job_id in state.queue:
            _append(job_id)

        return ordered

    def _apply_restart_pause(
        self,
        job: JobState,
        expired_jobs: set[str],
        timeout_jobs: set[str],
    ) -> None:
        """重启纠偏：统一标记暂停并设置原因提示。"""
        if job.status in ("finished", "failed", "canceled", "force_canceled", "removed"):
            return
        is_transitioned = self._transition_job_status(job, "paused", "restart_correction")
        if not is_transitioned:
            logger.warning("重启纠偏状态迁移被拒绝: job=%s", job.job_id)
            return
        job.paused = True
        if job.job_id in expired_jobs:
            job.message = "租约过期，任务已暂停"
        elif job.job_id in timeout_jobs:
            job.message = "心跳超时，任务已暂停"
        else:
            job.message = "程序重启，任务已暂停，请手动恢复"

    def _load_state(self):
        """
        启动时恢复队列状态

        恢复逻辑:
        1. 读取状态仓库
        2. 兼容旧 queue_state.json（仅迁移一次）
        3. 重启后所有非终态任务统一置为暂停并保留队列顺序
        """
        try:
            state = self.state_repo.load_queue_state()
            if state is None and self.queue_file.exists():
                with open(self.queue_file, "r", encoding="utf-8") as f:
                    legacy_state = json.load(f)
                state = QueueState(
                    queue=legacy_state.get("queue", []),
                    running_job_id=legacy_state.get("running"),
                    interrupted_job_id=legacy_state.get("interrupted"),
                    updated_at=legacy_state.get("timestamp"),
                )
                self.state_repo.save_queue_state(
                    queue=state.queue,
                    running_job_id=state.running_job_id,
                    interrupted_job_id=state.interrupted_job_id,
                )
                logger.info("[迁移] 已从 queue_state.json 迁移到状态仓库")

            if state is None:
                logger.info("无队列状态，从空队列启动")
                return

            recovery_queue = self._build_recovery_queue(state)
            job_ids = set(recovery_queue)
            # V3.2.0+dev.20260120.06: 加载仓库中的暂停任务，避免重启后无法恢复
            for job in self.state_repo.list_tasks():
                is_non_terminal = job.status not in (
                    "finished",
                    "failed",
                    "canceled",
                    "force_canceled",
                    "removed",
                )
                if is_non_terminal:
                    job_ids.add(job.job_id)

            expired_jobs = set(self.heartbeat_service.list_expired_leases())
            timeout_jobs = set(self.heartbeat_service.list_heartbeat_timeouts(60.0))

            self.queue.clear()
            self.running_job_id = None
            self.interrupted_job_id = None

            jobs_to_persist: list[JobState] = []
            from_status_map: Dict[str, Optional[str]] = {}

            for job_id in recovery_queue:
                job = self._load_job_for_recovery(job_id)
                if not job:
                    continue
                self.state_guard.sync_seq(job.job_id, job.state_seq)

                if job.status in ("finished", "failed", "canceled", "force_canceled", "removed"):
                    logger.info(f"[V3.2.0+dev.20260124.01] 过滤终态任务: {job_id}")
                    continue

                from_status_map[job_id] = job.status
                self._apply_restart_pause(job, expired_jobs, timeout_jobs)
                self.queue.append(job_id)
                self.jobs[job_id] = job
                jobs_to_persist.append(job)

            for job_id in job_ids:
                if job_id in self.jobs:
                    continue
                job = self._load_job_for_recovery(job_id)
                if not job:
                    continue
                self.state_guard.sync_seq(job.job_id, job.state_seq)
                if job.status in ("finished", "failed", "canceled", "force_canceled", "removed"):
                    continue
                from_status_map[job_id] = job.status
                self._apply_restart_pause(job, expired_jobs, timeout_jobs)
                self.jobs[job_id] = job
                jobs_to_persist.append(job)

            if jobs_to_persist:
                self._persist_queue_and_jobs(jobs_to_persist, from_status_map, reason="system_restart")
            else:
                self._save_state()

            paused_count = len([j for j in self.jobs.values() if j.status == "paused"])
            logger.info(
                f"[V3.2.0+dev.20260124.01] 队列恢复完成: "
                f"{paused_count}个暂停任务, queue={len(self.queue)}"
            )

        except Exception as e:
            logger.error(f"恢复队列状态失败: {e}")

    def prioritize_job(self, job_id: str, mode: Optional[str] = None) -> dict:
        """
        将任务移到队列头部（插队）

        Args:
            job_id: 要优先的任务ID
            mode: 插队模式
                - "gentle": 温和插队，放到队列头部，等当前任务完成后执行
                - "force": 强制插队，暂停当前任务A -> 执行B -> B完成后自动恢复A
                - None: 使用默认模式

        Returns:
            dict: 操作结果
                - success: 是否成功
                - mode: 实际使用的模式
                - interrupted_job_id: 被中断的任务ID（仅force模式）
        """
        # 使用默认模式
        if mode is None:
            mode = self._default_prioritize_mode

        if mode not in ("gentle", "force"):
            return {"success": False, "error": f"无效的插队模式: {mode}"}

        job = self.jobs.get(job_id)
        if not job:
            return {"success": False, "error": "任务不存在"}

        from_status_map = {job_id: job.status}
        jobs_to_persist = [job]

        with self.lock:
            # 1. 如果任务已经在跑，无法插队
            if job_id == self.running_job_id:
                logger.info(f"任务已在执行，无需插队: {job_id}")
                return {"success": False, "error": "任务已在执行中"}

            # 2. 如果任务在队列中，移除
            if job_id in self.queue:
                self.queue.remove(job_id)

            # 3. 插到队头
            self.queue.appendleft(job_id)
            if not self._transition_job_status(job, "queued", "prioritize"):
                logger.error("插队失败，状态迁移被拒绝: job=%s", job_id)
                self.queue.remove(job_id)
                return {"success": False, "error": "状态迁移被拒绝"}

            result = {
                "success": True,
                "mode": mode,
                "job_id": job_id,
                "interrupted_job_id": None
            }

            if mode == "gentle":
                # 温和插队：只放队头，不影响当前任务
                job.message = "优先执行（队列第1位）"
                logger.info(f"[温和插队] 任务已插队到队头: {job_id}")

            elif mode == "force":
                # 强制插队：暂停当前任务，记录以便自动恢复
                if self.running_job_id:
                    current_job = self.jobs.get(self.running_job_id)
                    if current_job:
                        from_status_map[current_job.job_id] = current_job.status
                        jobs_to_persist.append(current_job)
                        current_job.paused = True
                        current_job.message = "被强制插队暂停，稍后自动恢复..."
                        # 记录被中断的任务，用于自动恢复
                        self.interrupted_job_id = self.running_job_id
                        result["interrupted_job_id"] = self.running_job_id
                        logger.info(f"[强制插队] 暂停当前任务: {self.running_job_id}, 插队任务: {job_id}")

                job.message = "强制插队（等待当前任务暂停）"

        # 保存队列状态
        self._persist_queue_and_jobs(jobs_to_persist, from_status_map, reason="prioritize")

        # 推送全局SSE通知
        self._notify_queue_change()
        self._notify_job_status(job_id, job.status)
        if mode == "force" and result.get("interrupted_job_id"):
            # 通知被中断的任务状态变化
            self._notify_job_status(result["interrupted_job_id"], "pausing")

        return result

    def reorder_queue(self, job_ids: list) -> bool:
        """
        重新排序队列

        Args:
            job_ids: 按新顺序排列的任务ID列表

        Returns:
            bool: 是否成功
        """
        with self.lock:
            # 验证所有job_id都在队列中
            current_queue_set = set(self.queue)
            new_queue_set = set(job_ids)

            if current_queue_set != new_queue_set:
                logger.warning(f"队列重排失败：任务ID不匹配")
                return False

            # 更新队列顺序
            self.queue.clear()
            for job_id in job_ids:
                self.queue.append(job_id)

            # 更新每个任务的消息
            jobs_to_persist: list[JobState] = []
            from_status_map: Dict[str, Optional[str]] = {}
            for idx, job_id in enumerate(self.queue):
                job = self.jobs.get(job_id)
                if job:
                    from_status_map[job_id] = job.status
                    jobs_to_persist.append(job)
                    job.message = f"排队中 (位置: {idx + 1})"

            logger.info(f"队列已重新排序: {list(self.queue)}")

        # 保存队列状态
        if jobs_to_persist:
            self._persist_queue_and_jobs(jobs_to_persist, from_status_map, reason=None)
        else:
            self._save_state()

        # 推送全局SSE通知
        self._notify_queue_change()

        return True

    def get_queue_status(self) -> dict:
        """
        获取队列状态摘要

        Returns:
            dict: 队列状态信息
        """
        with self.lock:
            return {
                "queue": list(self.queue),
                "running": self.running_job_id,
                "queue_length": len(self.queue),
                "jobs": {
                    job_id: {
                        "status": job.status,
                        "message": job.message,
                        "filename": job.filename,
                        "progress": job.progress
                    }
                    for job_id, job in self.jobs.items()
                }
            }

    def shutdown(self):
        """
        停止Worker线程并保存所有任务状态
        
        执行顺序:
        1. 设置停止信号
        2. 保存当前运行任务的状态
        3. 保存队列状态
        4. 等待Worker线程结束
        """
        logger.info("停止队列服务...")
        
        # 1. 设置停止信号
        self.stop_event.set()
        
        # 2. 保存当前运行任务的状态
        with self.lock:
            if self.running_job_id:
                job = self.jobs.get(self.running_job_id)
                if job:
                    # 标记为暂停，这样流水线会保存 checkpoint
                    job.paused = True
                    job.message = "系统关闭，进度已保存"
                    try:
                        self.transcription_service.save_job_meta(job)
                        logger.info(f"已保存运行中任务状态: {self.running_job_id}")
                    except Exception as e:
                        logger.warning(f"保存任务状态失败: {e}")
        
        # 3. 保存队列状态
        try:
            self._save_state()
            logger.info("队列状态已保存")
        except Exception as e:
            logger.warning(f"保存队列状态失败: {e}")
        
        # 4. 等待Worker线程结束
        if hasattr(self, "worker_thread"):
            self.worker_thread.join(timeout=5)
        logger.info("队列服务已停止")


# ========== 单例模式 ==========

_queue_service_instance: Optional[JobQueueService] = None


def get_queue_service(
    transcription_service=None,
    enable_worker: bool = True,
    enable_cancel_monitor: bool = True,
    enable_state_recovery: bool = True
) -> JobQueueService:
    """
    获取队列服务单例

    Args:
        transcription_service: 首次调用时必须提供

    Returns:
        JobQueueService: 队列服务实例
    """
    global _queue_service_instance
    if _queue_service_instance is None:
        if transcription_service is None:
            raise RuntimeError("首次调用必须提供transcription_service")
        _queue_service_instance = JobQueueService(
            transcription_service,
            enable_worker=enable_worker,
            enable_cancel_monitor=enable_cancel_monitor,
            enable_state_recovery=enable_state_recovery,
        )
    return _queue_service_instance
