"""
任务队列管理服务 - V2.4
核心功能: 串行执行，防止并发OOM，队列持久化，插队功能

V3.7 更新:
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
from typing import Dict, Optional, Literal
from pathlib import Path
import torch

from app.models.job_models import JobState
from app.services.sse_service import get_sse_manager
from app.services.config_adapter import ConfigAdapter
from app.utils.cancellation_token import (
    CancellationToken,
    CancelledException,
    PausedException,
    create_cancellation_token
)

logger = logging.getLogger(__name__)

# 插队模式类型
PrioritizeMode = Literal["gentle", "force"]


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

    def __init__(self, transcription_service):
        """
        初始化队列服务

        Args:
            transcription_service: 转录服务实例
        """
        # 核心数据结构
        self.jobs: Dict[str, JobState] = {}  # 任务注册表 {job_id: JobState}
        self.queue: deque = deque()           # 等待队列 [job_id1, job_id2, ...]
        self.running_job_id: Optional[str] = None  # 当前正在执行的任务ID

        # 强制插队相关：记录被中断的任务，用于自动恢复
        self.interrupted_job_id: Optional[str] = None  # 被强制中断的任务ID

        # 插队设置
        self._default_prioritize_mode: PrioritizeMode = "gentle"  # 默认插队模式

        # [V3.7] 取消令牌注册表
        self.cancellation_tokens: Dict[str, CancellationToken] = {}

        # [V3.1.0] 取消超时保障机制
        # 用于确保取消操作最终生效，防止任务卡死导致队列阻塞
        self._pending_cancel_requests: Dict[str, float] = {}  # {job_id: cancel_request_time}
        self._pending_delete_after_cancel: set = set()  # 取消后需要删除数据的任务
        self._force_cancel_timeout: float = 60.0  # 超时时间（秒）
        self._current_executing_job_id: Optional[str] = None  # Worker 当前实际执行的任务ID（不受 cancel 影响）

        # 依赖服务
        self.transcription_service = transcription_service
        self.sse_manager = get_sse_manager()

        # 控制信号
        self.stop_event = threading.Event()
        self.lock = threading.RLock()  # 使用可重入锁，避免嵌套调用死锁

        # 持久化文件路径
        from app.core.config import config
        self.queue_file = Path(config.JOBS_DIR) / "queue_state.json"
        self.settings_file = Path(config.JOBS_DIR) / "queue_settings.json"

        # 加载设置
        self._load_settings()

        # 启动时恢复队列
        self._load_state()

        # 启动Worker线程
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="JobQueueWorker"
        )
        self.worker_thread.start()
        logger.info("任务队列Worker线程已启动")

        # [V3.1.0] 启动取消超时监控线程
        self._cancel_timeout_thread = threading.Thread(
            target=self._cancel_timeout_monitor,
            daemon=True,
            name="CancelTimeoutMonitor"
        )
        self._cancel_timeout_thread.start()
        logger.info("[V3.1.0] 取消超时监控线程已启动")

    def add_job(self, job: JobState):
        """
        添加任务到队列

        Args:
            job: 任务状态对象
        """
        with self.lock:
            self.jobs[job.job_id] = job
            self.queue.append(job.job_id)
            job.status = "queued"
            job.message = f"排队中 (位置: {len(self.queue)})"

        logger.info(f"任务已加入队列: {job.job_id} (队列长度: {len(self.queue)})")

        # 保存队列状态和任务元信息
        self._save_state()
        self.transcription_service.save_job_meta(job)

        # 推送全局SSE通知
        self._notify_queue_change()
        self._notify_job_status(job.job_id, job.status)

    def get_job(self, job_id: str) -> Optional[JobState]:
        """获取任务状态"""
        return self.jobs.get(job_id)

    def pause_job(self, job_id: str) -> bool:
        """
        暂停任务

        V3.7 更新: 集成 CancellationToken，触发协作式暂停
        V3.1.0 更新: 区分"正在暂停"和"已暂停"状态
        - 正在运行的任务：推送 pause_pending，等待流水线响应
        - 队列中的任务：立即推送 job_paused

        Args:
            job_id: 任务ID

        Returns:
            bool: 是否成功设置暂停标志
        """
        job = self.jobs.get(job_id)
        if not job:
            return False

        is_running = False
        with self.lock:
            if job_id == self.running_job_id:
                # 正在执行的任务：设置暂停标志（pipeline会自己检测并保存checkpoint）
                is_running = True
                job.paused = True
                # V3.1.0: 状态改为 pausing，表示正在等待流水线响应
                job.status = "pausing"
                job.message = "正在暂停，等待当前操作完成..."

                # [V3.7] 触发取消令牌的暂停
                token = self.cancellation_tokens.get(job_id)
                if token:
                    token.pause()
                    logger.info(f"[V3.7] 已触发取消令牌暂停: {job_id}")
                else:
                    logger.info(f"设置暂停标志: {job_id}")
            elif job_id in self.queue:
                # 还在排队的任务：直接从队列移除
                self.queue.remove(job_id)
                job.status = "paused"
                job.message = "已暂停（未开始）"
                logger.info(f"从队列移除: {job_id}")

        # 保存队列状态和任务元信息
        self._save_state()
        self.transcription_service.save_job_meta(job)

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

        V3.7 更新: 智能恢复逻辑
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
            return False

        # V3.1.0: 支持 paused 和 pausing 两种状态
        if job.status not in ("paused", "pausing"):
            logger.warning(f"任务未暂停，无法恢复: {job_id}, status={job.status}")
            return False

        # V3.1.0: 在推送 SSE 之前，先从 checkpoint 恢复进度
        # 这样 _notify_job_status 推送的进度就是正确的，而非 0
        self._restore_progress_from_checkpoint(job)

        with self.lock:
            # [V3.7] 检查任务是否仍在运行中
            # 场景: 用户在原子区域内暂停后立即恢复
            is_still_running = (job_id == self.running_job_id)
            token = self.cancellation_tokens.get(job_id)

            if is_still_running and token:
                # 任务仍在运行，只需清除暂停标志
                # 任务会在原子区域结束后继续正常执行（不会抛出 PausedException）
                token.resume()
                job.paused = False
                job.status = "processing"
                job.message = "已恢复，继续执行中..."
                logger.info(f"[V3.7] 任务仍在运行，清除暂停标志: {job_id}")
            else:
                # 任务已完全停止，需要重新加入队列
                if job_id not in self.queue:
                    self.queue.append(job_id)

                job.status = "queued"
                job.paused = False
                job.message = f"已恢复，排队中 (位置: {len(self.queue)})"

                if token:
                    # Token 还存在但任务不在运行（理论上不应该发生）
                    token.resume()
                    logger.warning(f"[V3.7] Token存在但任务未运行，可能是竞态条件: {job_id}")
                else:
                    logger.info(f"[V3.7] 任务已停止，重新加入队列: {job_id}")

        # 保存队列状态和任务元信息
        self._save_state()
        self.transcription_service.save_job_meta(job)

        # 推送全局SSE通知
        self._notify_queue_change()
        self._notify_job_status(job_id, job.status)

        # 同时推送到单任务频道
        self._notify_job_signal(job_id, "job_resumed")

        return True

    def cancel_job(self, job_id: str, delete_data: bool = False):
        """
        取消任务（支持删除已完成的任务）

        V3.1.0 修复：
        - 删除数据时同步清理内存中的 self.jobs[job_id]
        - 广播 job_removed 事件，解决幽灵任务问题

        V3.7 更新:
        - 集成 CancellationToken，触发协作式取消

        V3.1.0 更新:
        - 正在运行的任务进入"canceling"状态，不再立即清除 running_job_id
        - 增加超时保障机制，确保任务最终被清除

        Args:
            job_id: 任务ID
            delete_data: 是否删除任务数据

        Returns:
            Tuple[bool, Optional[str], bool]: (是否成功, 失败原因, 是否处于延迟删除状态)
        """
        job = self.jobs.get(job_id)

        # 如果任务不在队列服务中（可能是已完成的任务），直接调用transcription_service删除
        if not job:
            if delete_data:
                # 尝试通过transcription_service删除已完成的任务
                try:
                    result = self.transcription_service.cancel_job(job_id, delete_data=True)
                    success, err = result if isinstance(result, tuple) else (bool(result), None)
                    if success:
                        # [V3.1.0] 推送任务删除事件（而非仅状态变更）
                        self._notify_job_removed(job_id)
                        # [V3.7] 清理取消令牌
                        self._remove_cancellation_token(job_id)
                        return True, None, False
                    return False, err or "删除失败", False
                except Exception as e:
                    logger.warning(f"删除任务 {job_id} 失败: {e}")
                    return False, str(e), False
            return False, "任务未找到", False

        is_running = False  # [V3.1.0] 标记是否为正在运行的任务

        with self.lock:
            # 设置取消标志
            job.canceled = True

            # [V3.7] 触发取消令牌的取消
            token = self.cancellation_tokens.get(job_id)
            if token:
                token.cancel()
                logger.info(f"[V3.7] 已触发取消令牌取消: {job_id}")

            # 如果在队列中，直接移除并标记为已取消
            if job_id in self.queue:
                self.queue.remove(job_id)
                job.status = "canceled"
                job.message = "已取消（未开始）"

            # [V3.1.0] 如果是正在运行的任务，进入"取消中"状态
            # 不再立即清除 running_job_id，让 Worker 的 finally 块处理
            elif self.running_job_id == job_id:
                is_running = True
                job.status = "canceling"  # 新状态：取消中
                # 运行中删除：提示将延迟自动删除
                job.message = "当前有进程占用，将延迟自动删除"
                # 记录取消请求时间，用于超时保障
                self._pending_cancel_requests[job_id] = time.time()
                if delete_data:
                    self._pending_delete_after_cancel.add(job_id)
                logger.info(f"[V3.1.0] 任务进入取消中状态: {job_id}")

        # [V3.1.0] 正在运行的任务：延迟处理删除，由 Worker 或超时监控完成
        if is_running:
            # 不在这里删除数据，等待任务真正结束
            # 保存队列状态
            self._save_state()
            # 推送状态变更
            self._notify_queue_change()
            self._notify_job_status(job_id, job.status)
            self._notify_job_signal(job_id, "job_canceling")  # 新信号
            # 如果需要删除数据，标记完成后再删
            if delete_data:
                with self.lock:
                    self._pending_delete_after_cancel.add(job_id)
                return True, "任务正在执行，已请求取消并将在结束后删除", True
            return True, None, False

        # 非运行中的任务：立即处理
        if delete_data:
            result = self.transcription_service.cancel_job(job_id, delete_data=True)
            success, err = result if isinstance(result, tuple) else (bool(result), None)
            if not success:
                # 删除失败时不广播删除事件，保留任务文件供用户重试
                return False, err or "删除失败", False

            # [V3.1.0] 从内存中彻底移除任务，防止幽灵任务
            with self.lock:
                if job_id in self.jobs:
                    del self.jobs[job_id]
                    logger.info(f"[幽灵任务修复] 已从内存移除任务: {job_id}")

            # [V3.7] 清理取消令牌
            self._remove_cancellation_token(job_id)
        else:
            success, err = True, None
            # 不删除数据时，保存任务元信息
            self.transcription_service.save_job_meta(job)

        # 保存队列状态
        self._save_state()

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

        return success, err, False

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

                        if job.status in ["paused", "canceled", "canceling", "force_canceled"]:
                            logger.info(f"⏭️ 跳过已暂停/取消的任务: {job_id}")
                            self.queue.popleft()
                            continue

                        # 正式从队列移除
                        self.queue.popleft()
                        self.running_job_id = job_id
                        self._current_executing_job_id = job_id  # [V3.1.0] 记录实际执行的任务ID
                        job.status = "processing"
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

                    # [V3.7] 创建取消令牌
                    token = self._create_cancellation_token(self.running_job_id)
                    logger.debug(f"[V3.7] 已创建取消令牌: {self.running_job_id}")

                # 2. 如果没有任务，休眠后继续
                if self.running_job_id is None:
                    time.sleep(1)
                    continue

                # 3. 执行任务（阻塞，直到完成/失败/暂停/取消）
                job = self.jobs[self.running_job_id]
                logger.info(f" 开始执行任务: {self.running_job_id}")

                try:
                    # 根据引擎和配置选择流水线 (使用 ConfigAdapter 统一新旧配置)
                    engine = getattr(job.settings, 'engine', 'sensevoice')
                    use_dual_alignment = ConfigAdapter.needs_dual_alignment(job.settings)
                    transcription_profile = ConfigAdapter.get_transcription_profile(job.settings)
                    preset_id = ConfigAdapter.get_preset_id(job.settings)

                    # 调试日志: 输出配置来源和关键参数
                    config_source = ConfigAdapter.get_config_source(job.settings)
                    logger.info(f"路由决策: engine={engine}, use_dual_alignment={use_dual_alignment}, profile={transcription_profile}, preset={preset_id}")
                    logger.debug(f"配置来源: {config_source}")

                    if use_dual_alignment:
                        # 双流对齐流水线 (V3.0+ 新架构)
                        # V3.1.0: 所有 SenseVoice 模式都走新架构
                        logger.info(f"使用双流对齐流水线 (profile={transcription_profile}, preset={preset_id})")
                        _run_async_safely(self._run_dual_alignment_pipeline(job, preset_id))
                    elif engine == 'sensevoice':
                        # V3.1.0: 旧架构已废弃，所有 SenseVoice 模式都应该走新架构
                        logger.error(f"错误：SenseVoice 任务未走新架构！profile={transcription_profile}")
                        logger.error(f"这是一个配置错误，请检查 ConfigAdapter.needs_dual_alignment() 方法")
                        raise RuntimeError(f"SenseVoice 任务路由错误：{transcription_profile} 应该走新架构")
                        # 旧代码（已废弃）：
                        # _run_async_safely(self.transcription_service._process_video_sensevoice(job))
                    else:
                        # 新架构 Pipeline 流水线（2025-12-17 架构改造）
                        # 使用 AudioProcessingPipeline + AsyncDualPipeline
                        logger.info(f"使用新架构 Pipeline 流水线")
                        _run_async_safely(self.transcription_service._run_pipeline_v2(job))

                    # 检查最终状态
                    if job.canceled:
                        job.status = "canceled"
                        job.message = "已取消"
                    elif job.paused:
                        job.status = "paused"
                        job.message = "已暂停"
                    else:
                        job.status = "finished"
                        job.message = "完成"
                        logger.info(f"任务完成: {self.running_job_id}")

                except CancelledException as e:
                    # [V3.7] 捕获取消异常
                    job.status = "canceled"
                    job.message = "已取消"
                    logger.info(f"[V3.7] 任务被取消: {e.job_id}")

                except PausedException as e:
                    # [V3.7] 捕获暂停异常
                    job.status = "paused"
                    job.message = "已暂停"
                    logger.info(f"[V3.7] 任务已暂停: {e.job_id}")

                except Exception as e:
                    job.status = "failed"
                    job.message = f"失败: {e}"
                    job.error = str(e)
                    logger.error(f"任务执行失败: {self.running_job_id} - {e}", exc_info=True)

                finally:
                    # 4. 清理资源（关键！）
                    # [V3.1.0] 使用 _current_executing_job_id 而非 running_job_id
                    # 因为 running_job_id 可能被超时监控清除
                    finished_job_id = self._current_executing_job_id
                    with self.lock:
                        self.running_job_id = None
                        self._current_executing_job_id = None
                        # [V3.1.0] 从待取消列表移除
                        self._pending_cancel_requests.pop(finished_job_id, None)

                    # [V3.7] 清理取消令牌
                    self._remove_cancellation_token(finished_job_id)

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
                            job.status = "paused"
                            job.message = err or "当前有进程占用，请稍后再试"
                            self.transcription_service.save_job_meta(job)
                            self._notify_job_status(job.job_id, job.status)
                            logger.error(f"[V3.1.0] 延迟删除失败: {finished_job_id}, {err}")
                        except Exception as e:
                            logger.error(f"[V3.1.0] 延迟删除失败: {finished_job_id}, {e}")

                    # 保存任务最终状态到 job_meta.json
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
                job.status = "force_canceled"
                job.message = f"已强制取消（响应超时 {elapsed:.0f}s）"
                logger.info(f"[V3.1.0] 任务状态更新为 force_canceled: {job_id}")

            # 关键：强制清除 running_job_id，允许下一个任务开始
            # 注意：_current_executing_job_id 保持不变，让 Worker finally 块知道要清理谁
            if self.running_job_id == job_id:
                self.running_job_id = None
                logger.warning(f"[V3.1.0] 强制清除 running_job_id: {job_id}")

        # 保存状态
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

        V3.1.0 新特性：支持两种流水线模式
        - async: 三级异步流水线（错位并行，性能提升 30-50%）
        - sync: 串行流水线（稳定版，V3.0 兼容）

        V3.7 新特性：支持断点续传
        - 集成 CancellationToken 机制
        - 支持从 CheckpointV37 恢复

        Args:
            job: 任务状态对象
            preset_id: 预设 ID
        """
        from app.pipelines import (
            AudioProcessingPipeline,
            AudioProcessingConfig,
            AsyncDualPipeline,
            get_audio_processing_pipeline,
            get_async_dual_pipeline
        )
        from app.services.streaming_subtitle import get_streaming_subtitle_manager, remove_streaming_subtitle_manager
        from app.services.progress_tracker import get_progress_tracker, remove_progress_tracker, ProcessPhase
        from app.services.sse_service import get_sse_manager
        from app.services.job.checkpoint_manager import CheckpointManagerV37
        from app.services.progress_emitter import (
            get_progress_emitter, remove_progress_emitter, ProgressMode
        )
        from pathlib import Path

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

        # V3.1.0: 初始化进度发射器
        transcription_profile = ConfigAdapter.get_transcription_profile(job.settings)
        progress_emitter = get_progress_emitter(
            job, sse_manager,
            transcription_profile=transcription_profile
        )

        # V3.7: 获取取消令牌
        cancellation_token = self.get_cancellation_token(job.job_id)

        # V3.7: 初始化检查点管理器
        job_dir = Path(job.dir)
        checkpoint_manager = CheckpointManagerV37(job_dir, logger)

        # 流水线配置
        from app.core.config import config as project_config
        queue_maxsize = project_config.PIPELINE_QUEUE_MAXSIZE

        try:
            logger.info(f"[双流对齐] 开始处理任务: {job.job_id}, preset={preset_id}")

            # V3.7: 检查是否有检查点需要恢复
            checkpoint = checkpoint_manager.load_checkpoint()
            is_resuming = checkpoint is not None
            if is_resuming:
                logger.info(f"[V3.7] 检测到检查点，准备断点续传: phase={checkpoint.phase}")
                # V3.1.0: 从检查点恢复进度并立即推送 SSE
                if hasattr(checkpoint, 'to_dict'):
                    progress_emitter.restore_from_checkpoint(checkpoint.to_dict())
                    logger.info(f"[V3.1.0] 已恢复进度: {job.progress:.1f}%")

            # === 预触发 Proxy 生成（不阻塞主流程）===
            await self._maybe_trigger_proxy_generation(job)

            # 阶段 1: 音频前处理（使用新架构 PreprocessingPipeline）
            progress_tracker.start_phase(ProcessPhase.EXTRACT, 1, "音频前处理...")
            progress_emitter.update_preprocess(0, "extract", "音频前处理...")

            from app.pipelines.preprocessing_pipeline import PreprocessingPipeline
            import soundfile as sf
            import librosa

            logger.info("使用新架构 PreprocessingPipeline（Stage模式）")

            # V3.1.0: 根据语言选择VAD配置（迁移自旧架构）
            from app.services.audio.vad_service import VADConfig
            language = getattr(job.settings, 'language', 'auto')
            is_english = language in {'en', 'english'}

            if is_english:
                # Whisper模式：合并VAD，避免幻觉
                vad_config = VADConfig(
                    merge_max_gap=1.0,
                    merge_max_duration=12.0,
                    smart_target_duration=12.0
                )
                logger.info(f"VAD配置: Whisper模式（合并），language={language}")
            else:
                # SenseVoice模式：保留停顿信息，获得更自然的断句
                vad_config = VADConfig(
                    merge_max_gap=0.3,
                    merge_max_duration=8.0,
                    smart_target_duration=8.0
                )
                logger.info(f"VAD配置: SenseVoice模式（保留停顿），language={language}")

            # 创建预处理流水线（V3.7: 传递取消令牌，V3.1.0: 传递VAD配置）
            preprocessing_pipeline = PreprocessingPipeline(
                config=job.settings.preprocessing,
                vad_config=vad_config,  # V3.1.0: 新增
                logger=logger,
                cancellation_token=cancellation_token  # V3.7
            )

            # V3.7: 检查是否需要跳过预处理阶段
            skip_preprocessing = False
            preprocessing_state = None
            if is_resuming and checkpoint.preprocessing:
                preprocessing_state = checkpoint.preprocessing
                if preprocessing_state.separation_completed:
                    skip_preprocessing = True
                    logger.info("[V3.7] 预处理阶段已完成，跳过")
                    progress_emitter.update_preprocess(100, "completed", "预处理已完成")

            if not skip_preprocessing:
                # 执行预处理（包含：音频提取、VAD、频谱分诊、按需分离）
                audio_chunks = await preprocessing_pipeline.process(
                    video_path=job.input_path,
                    job_state=job,
                    job_dir=job_dir  # V3.7: 传递 job_dir 用于检查点保存
                )

                # 获取预处理统计信息
                stats = preprocessing_pipeline.get_statistics(audio_chunks)
                logger.info(
                    f"PreprocessingPipeline 完成: "
                    f"总chunk数={stats['total_chunks']}, "
                    f"需要分离={stats['need_separation']}, "
                    f"已分离={stats['separated']}, "
                    f"分离比例={stats['separation_ratio']:.2%}"
                )
            else:
                # V3.1.0: 从检查点恢复 AudioChunk（传递 checkpoint 数据给预处理流水线）
                # 预处理流水线会根据 checkpoint 中的 chunks_metadata 跳过 VAD
                logger.info("[V3.1.0] 从检查点恢复预处理状态...")

                # 将 checkpoint 转换为字典格式供预处理流水线使用
                checkpoint_dict = checkpoint.to_dict() if hasattr(checkpoint, 'to_dict') else None

                audio_chunks = await preprocessing_pipeline.process(
                    video_path=job.input_path,
                    job_state=job,
                    job_dir=job_dir,
                    checkpoint=checkpoint_dict  # V3.1.0: 传递 checkpoint 用于跳过 VAD
                )

            # 加载完整音频（用于双流对齐的 Audio Overlap 功能）
            if audio_chunks:
                sr = audio_chunks[0].sample_rate
                full_audio, _ = librosa.load(job.input_path, sr=sr, mono=True)

                # 保存音频文件供波形图使用
                audio_path = Path(job.dir) / "audio.wav"
                sf.write(str(audio_path), full_audio, sr)
                logger.info(f"音频文件已保存: {audio_path}")
            else:
                raise RuntimeError("PreprocessingPipeline 未返回任何 AudioChunk")

            progress_tracker.complete_phase(ProcessPhase.EXTRACT)
            # V3.1.0: 预处理完成
            progress_emitter.update_preprocess(100, "completed", "预处理完成")

            # V3.1.0: 预处理→转录过渡检查点
            # 在开始转录前检查是否有待处理的暂停/取消请求
            if cancellation_token and job_dir:
                checkpoint_data = {
                    "preprocessing": {
                        "completed": True,
                        "total_chunks": len(audio_chunks)
                    }
                }
                cancellation_token.check_and_save(checkpoint_data, job_dir)
                logger.debug("[V3.1.0] 预处理→转录过渡检查点已保存")

            # 阶段 2: 双流对齐处理
            total_chunks = len(audio_chunks)
            progress_tracker.start_phase(ProcessPhase.SENSEVOICE, total_chunks, "双流对齐...")

            # V3.7: 检查是否需要恢复转录状态
            # V3.1.0: 使用 min(fast, slow) 作为安全恢复点
            # 原因：finalized_indices 在当前实现中未被保存到 checkpoint，始终为空
            # 使用 min 确保不会跳过任何需要处理的 chunk
            fast_processed_indices = set()
            slow_processed_indices = set()
            previous_whisper_text = None
            if is_resuming and checkpoint.transcription:
                transcription_state = checkpoint.transcription

                # 获取各 Worker 的已处理索引
                fast_indices = set(transcription_state.fast_processed_indices) if transcription_state.fast_processed_indices else set()
                slow_indices = set(transcription_state.slow_processed_indices) if transcription_state.slow_processed_indices else set()
                finalized = set(transcription_state.finalized_indices) if transcription_state.finalized_indices else set()

                # V3.1.0: 使用安全恢复策略
                # 优先使用 finalized_indices（如果有）
                # 否则使用 fast 和 slow 的交集（两者都已处理的 chunk）
                if finalized:
                    # V3.1.0: 使用 finalized 的最大索引+1 作为安全恢复点
                    # finalized_indices 包含已完成对齐的 chunk 索引
                    # 例如：{0, 1, ..., 15}，我们应该跳过 0-15，从 16 开始
                    max_finalized = max(finalized)
                    safe_indices = set(range(max_finalized + 1))
                    logger.info(
                        f"[V3.1.0] 使用 finalized_indices 的最大值作为恢复点: "
                        f"max={max_finalized}, 跳过 0-{max_finalized} 共 {len(safe_indices)} 个 Chunk"
                    )
                elif fast_indices and slow_indices:
                    # 使用交集：只有两个 Worker 都处理过的 chunk 才能跳过
                    safe_indices = fast_indices & slow_indices
                    logger.info(f"[V3.1.0] 使用 fast & slow 交集作为恢复点: {len(safe_indices)} 个")
                elif slow_indices:
                    # 只有 slow 数据（不太可能，但以防万一）
                    safe_indices = slow_indices
                    logger.info(f"[V3.1.0] 使用 slow_indices 作为恢复点: {len(slow_indices)} 个")
                else:
                    # 没有可靠的恢复点，从头开始
                    safe_indices = set()
                    logger.info("[V3.1.0] 无可靠恢复点，从头开始")

                fast_processed_indices = safe_indices
                slow_processed_indices = safe_indices

                previous_whisper_text = transcription_state.previous_whisper_text
                logger.info(
                    f"[V3.1.0] 恢复转录状态: safe={len(safe_indices)}, "
                    f"checkpoint.fast={len(fast_indices)}, "
                    f"checkpoint.slow={len(slow_indices)}, "
                    f"finalized={len(finalized)}"
                )
                # V3.1.0: 更新进度发射器的已处理数（使用 safe_indices）
                progress_emitter.update_fast(len(safe_indices), total_chunks, force_push=True)

                # V3.1.0: 同步 progress_tracker 的已完成数量
                # 修复进度归零问题：start_phase 会将 completed_items 重置为 0
                # 这里需要恢复正确的已完成数量
                progress_tracker.update_phase(ProcessPhase.SENSEVOICE, completed=len(safe_indices))
                logger.info(f"[V3.1.0] progress_tracker 已同步: {len(safe_indices)}/{total_chunks} 个 Chunk")

                # V3.1.0: 恢复字幕状态（核心修复）
                # 从 checkpoint 恢复已生成的字幕，确保新字幕索引不会与已有字幕冲突
                if transcription_state.sentences_snapshot:
                    subtitle_checkpoint_data = {
                        "sentences_snapshot": transcription_state.sentences_snapshot,
                        "sentence_count": transcription_state.sentence_count,
                        "chunk_sentences_map": transcription_state.chunk_sentences_map
                    }
                    if subtitle_manager.restore_from_checkpoint(subtitle_checkpoint_data):
                        logger.info(f"[V3.1.0] 字幕状态已恢复: {len(transcription_state.sentences_snapshot)} 个句子")
                        # 推送已恢复的字幕到前端
                        subtitle_manager.push_restored_subtitles_to_frontend()
                    else:
                        logger.warning("[V3.1.0] 字幕恢复失败，将从头生成字幕")
                else:
                    logger.info("[V3.1.0] checkpoint 中无字幕快照，字幕将从头生成")

            pipeline_sentences = []  # 用于收集本轮流水线产出的句子，作为无字幕快照时的兜底

            # V3.1.0: 异步流水线（三级流水线，错位并行）
            logger.info(f"[双流对齐] 使用异步流水线处理 {total_chunks} 个 Chunk (queue_maxsize={queue_maxsize})")
            async_pipeline = AsyncDualPipeline(
                job_id=job.job_id,
                queue_maxsize=queue_maxsize,
                sensevoice_language=getattr(job.settings, 'sensevoice_language', 'auto'),
                whisper_language=getattr(job.settings, 'whisper_language', 'auto'),
                user_glossary=getattr(job.settings, 'user_glossary', None),
                transcription_profile=ConfigAdapter.get_transcription_profile(job.settings),
                logger=logger,
                cancellation_token=cancellation_token,  # V3.7
                progress_emitter=progress_emitter  # V3.1.0: 传递进度发射器
            )

            # V3.7: 如果有历史上下文，恢复 SlowWorker 状态
            if previous_whisper_text and async_pipeline.slow_worker:
                async_pipeline.slow_worker.restore_prompt_cache(previous_whisper_text)
                logger.info(f"[V3.7] 已恢复 SlowWorker 上下文: {len(previous_whisper_text)} 字符")

            # V3.1.0: 分别计算各 Worker 的基准偏移量
            # FastWorker 使用 safe_indices（用于跳过已处理的 chunk）
            # SlowWorker 和 AlignmentWorker 使用各自实际处理的数量（用于进度计算）
            base_slow_count = 0
            base_align_count = 0
            if is_resuming and checkpoint.transcription:
                # SlowWorker 的基准 = checkpoint 中保存的 slow_indices 数量
                base_slow_count = len(slow_indices) if slow_indices else len(safe_indices)
                # AlignmentWorker 的基准 = checkpoint 中保存的 finalized_indices 数量
                base_align_count = len(finalized) if finalized else len(safe_indices)
                logger.info(
                    f"[V3.1.0] Worker 基准偏移量: "
                    f"FastWorker={len(fast_processed_indices)}, "
                    f"SlowWorker={base_slow_count}, "
                    f"AlignmentWorker={base_align_count}"
                )

            # 处理所有 Chunks（流水线并行，传递完整音频数组用于 Audio Overlap）
            # V3.1.0: 传递初始索引集合，修复恢复后进度不准确问题
            contexts = await async_pipeline.run(
                audio_chunks=audio_chunks,
                full_audio_array=full_audio,
                full_audio_sr=sr,
                job_dir=job_dir,  # V3.7
                processed_indices=fast_processed_indices,  # V3.1.0: FastWorker 跳过的索引
                base_slow_count=base_slow_count,  # V3.1.0: SlowWorker 的基准偏移量（已废弃）
                base_align_count=base_align_count,  # V3.1.0: AlignmentWorker 的基准偏移量（已废弃）
                initial_slow_processed_indices=slow_indices if is_resuming else None,  # V3.1.0: SlowWorker 初始索引
                initial_finalized_indices=finalized if is_resuming else None  # V3.1.0: AlignmentWorker 初始索引
            )

            # 提取结果
            for ctx in contexts:
                pipeline_sentences.extend(ctx.final_sentences)

            progress_tracker.complete_phase(ProcessPhase.SENSEVOICE)
            # V3.1.0: 双流对齐完成
            progress_emitter.update_fast(total_chunks, total_chunks, force_push=True)
            progress_emitter.update_slow(total_chunks, total_chunks, force_push=True)
            progress_emitter.update_align(total_chunks, total_chunks, force_push=True)

            # 阶段 3: 生成字幕文件
            progress_tracker.start_phase(ProcessPhase.SRT, 1, "生成字幕...")

            # 选取用于导出的字幕集合：优先使用字幕管理器的完整快照，确保恢复场景下包含暂停前的句子
            final_sentences = pipeline_sentences
            if subtitle_manager:
                subtitle_snapshot = subtitle_manager.get_all_sentences()
                if subtitle_snapshot:
                    final_sentences = subtitle_snapshot
                    logger.info(
                        f"[V3.1.0] 使用字幕管理器快照生成 SRT: sentences={len(final_sentences)}"
                    )
                else:
                    logger.info("[V3.1.0] 字幕管理器无有效句子，回退到流水线结果")

            # 按时间排序
            final_sentences.sort(key=lambda s: s.start)

            # 生成 SRT
            output_path = str(Path(job.dir) / f"{job.job_id}.srt")
            self.transcription_service._generate_subtitle_from_sentences(
                final_sentences,
                output_path,
                include_translation=False
            )

            progress_tracker.complete_phase(ProcessPhase.SRT)

            # V3.1.2+dev.20260114.01: 任务完成前持久化字幕快照，供后台转录后显示准确率
            # 只保存精简版的 sentences_snapshot，避免删除 checkpoint 后丢失 display_confidence
            if subtitle_manager and job_dir:
                try:
                    snapshot_data = subtitle_manager.to_checkpoint_data()
                    snapshot_path = job_dir / "transcription_text.json"
                    with open(snapshot_path, "w", encoding="utf-8") as f:
                        json.dump(snapshot_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"[V3.1.2] 已保存字幕快照供编辑器复用: {snapshot_path}")
                except Exception as e:
                    logger.warning(f"[V3.1.2] 保存字幕快照失败: {e}")

            # V3.7: 任务完成，清理检查点
            checkpoint_manager.delete_checkpoint()
            logger.info("[V3.7] 任务完成，检查点已清理")

            # V3.1.0: 使用 progress_emitter 标记完成
            progress_emitter.complete("处理完成")

            # 完成
            job.status = 'completed'
            # push_signal_event 已在 progress_emitter.complete() 中调用

            logger.info(f"[双流对齐] 任务完成: {job.job_id}")

        except Exception as e:
            logger.error(f"[双流对齐] 任务失败: {e}", exc_info=True)
            job.status = 'failed'
            job.error = str(e)
            push_signal_event(sse_manager, job.job_id, "job_failed", str(e))
            raise

        finally:
            # 清理资源
            remove_streaming_subtitle_manager(job.job_id)
            remove_progress_tracker(job.job_id)
            remove_progress_emitter(job.job_id)  # V3.1.0: 清理进度发射器

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
        try:
            from app.services.proxy_720_scheduler import get_proxy_scheduler

            scheduler = get_proxy_scheduler()
            scheduler.on_queue_idle()
        except Exception as e:
            logger.debug(f\"[720p触发] 调度器通知失败（非致命）: {e}\")

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
            self.transcription_service.clear_model_cache()
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
                    interrupted_job.status = "queued"
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
                "timestamp": time.time()
            }

        self.sse_manager.broadcast_sync("global", "queue_update", data)
        logger.debug(f"[全局SSE] 推送队列变化: queue={len(data['queue'])}个, running={data['running']}")

    def _notify_job_status(self, job_id: str, status: str):
        """推送任务状态变化到全局SSE"""
        job = self.jobs.get(job_id)
        if not job:
            return

        data = {
            "id": job_id,
            "status": status,
            "percent": round(job.progress, 1),  # 统一字段名为 percent，保留1位小数
            "message": job.message,
            "filename": job.filename,
            "phase": job.phase,  # 新增：阶段信息
            "timestamp": time.time()
        }

        self.sse_manager.broadcast_sync("global", "job_status", data)
        logger.debug(f"[全局SSE] 推送任务状态: {job_id[:8]}... -> {status}")

    def _notify_job_progress(self, job_id: str):
        """推送任务进度更新到全局SSE（低频调用，节省带宽）"""
        job = self.jobs.get(job_id)
        if not job:
            return

        data = {
            "id": job_id,
            "percent": round(job.progress, 1),  # 统一字段名为 percent，保留1位小数
            "phase": job.phase,
            "phase_percent": round(job.phase_percent, 1),  # 新增：阶段内进度
            "message": job.message,
            "processed": job.processed,
            "total": job.total,
            "timestamp": time.time()
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
            "message": job.message,
            "percent": round(job.progress, 1)
        }

        self.sse_manager.broadcast_sync(f"job:{job_id}", f"signal.{signal}", data)
        logger.debug(f"[单任务SSE] 推送信号: {job_id[:8]}... -> signal.{signal}")

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

    # ==================== V3.7 取消令牌管理 ====================

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
            logger.warning(f"[V3.7] 取消令牌已存在，覆盖: {job_id}")

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
            logger.debug(f"[V3.7] 已移除取消令牌: {job_id}")

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
        持久化队列状态到磁盘

        格式:
        {
          "queue": ["job_id1", "job_id2"],
          "running": "job_id3",
          "interrupted": "job_id4",  // 被强制中断的任务
          "paused": ["job_id5", "job_id6"],  // 暂停的任务列表
          "timestamp": 1234567890.0
        }
        """
        with self.lock:
            # 收集所有暂停状态的任务
            paused_jobs = [
                job_id for job_id, job in self.jobs.items()
                if job.status == "paused" or job.paused
            ]
            state = {
                "queue": list(self.queue),
                "running": self.running_job_id,
                "interrupted": self.interrupted_job_id,
                "paused": paused_jobs,  # 新增：保存暂停的任务列表
                "timestamp": time.time()
            }

        try:
            # 确保目录存在
            self.queue_file.parent.mkdir(parents=True, exist_ok=True)

            # 原子写入（临时文件 + rename）
            temp_path = self.queue_file.with_suffix(".tmp")
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)

            # 原子替换
            temp_path.replace(self.queue_file)
            logger.debug("队列状态已保存")
        except Exception as e:
            logger.error(f"保存队列状态失败: {e}")

    def _load_job_for_recovery(self, job_id: str) -> Optional[JobState]:
        """
        加载任务用于恢复（优先从已有缓存获取，其次从 job_meta.json 加载，最后从 checkpoint 加载）

        这是重启恢复的核心方法，确保能正确恢复任务状态

        Args:
            job_id: 任务ID

        Returns:
            Optional[JobState]: 恢复的任务状态对象
        """
        # 0. 优先检查 transcription_service.jobs（可能已经被 _load_all_jobs_from_disk 加载）
        if job_id in self.transcription_service.jobs:
            job = self.transcription_service.jobs[job_id]
            logger.info(f"从 transcription_service 缓存获取任务: {job_id}")
            return job

        # 1. 从 job_meta.json 加载（包含完整的任务元信息）
        job = self.transcription_service.load_job_meta(job_id)
        if job:
            logger.info(f"从 job_meta.json 恢复任务: {job_id}")
            return job

        # 2. 降级：从 checkpoint 恢复（兼容旧版本）
        job = self.transcription_service.restore_job_from_checkpoint(job_id)
        if job:
            logger.info(f"从 checkpoint 恢复任务（旧版兼容）: {job_id}")
            # 同时保存 job_meta.json 以便下次直接加载
            self.transcription_service.save_job_meta(job)
            return job

        logger.warning(f"无法恢复任务: {job_id}")
        return None

    def _load_state(self):
        """
        启动时恢复队列状态

        恢复逻辑:
        1. 读取 queue_state.json
        2. 优先从 job_meta.json 恢复任务（包含完整状态）
        3. 如果有 running 任务，自动加入队列头部继续执行
        4. 恢复队列中的其他任务
        5. 恢复 interrupted 任务（被强制中断的任务）
        """
        if not self.queue_file.exists():
            logger.info("无队列状态文件，从空队列启动")
            return

        try:
            with open(self.queue_file, 'r', encoding='utf-8') as f:
                state = json.load(f)

            logger.info(f"加载队列状态: {state}")

            # 1. 恢复 running 任务（如果有）- 意外断电/崩溃场景
            running_id = state.get("running")
            if running_id:
                job = self._load_job_for_recovery(running_id)
                if job:
                    # 系统重启后强制暂停，需要用户手动恢复
                    job.status = "paused"
                    job.paused = True
                    job.message = "程序重启，任务已暂停，请手动恢复"
                    self.jobs[running_id] = job
                    # 同步到 transcription_service.jobs（确保 SSE 路由能找到任务）
                    self.transcription_service.jobs[running_id] = job
                    # 更新 job_meta.json 中的状态
                    self.transcription_service.save_job_meta(job)
                    logger.info(f"恢复中断任务为暂停状态（需手动恢复）: {running_id}")

            # 2. 恢复队列中的任务
            for job_id in state.get("queue", []):
                # 避免重复（running任务已经加入队列了）
                if job_id == running_id:
                    continue

                job = self._load_job_for_recovery(job_id)
                if job:
                    # 系统重启后强制暂停，需要用户手动恢复
                    job.status = "paused"
                    job.paused = True
                    job.message = "程序重启，任务已暂停，请手动恢复"
                    self.jobs[job_id] = job
                    # 同步到 transcription_service.jobs（确保 SSE 路由能找到任务）
                    self.transcription_service.jobs[job_id] = job
                    # 更新 job_meta.json 中的状态
                    self.transcription_service.save_job_meta(job)
                    logger.info(f"恢复排队任务为暂停状态（需手动恢复）: {job_id}")

            # 3. 恢复 interrupted 任务（被强制中断的任务）
            interrupted_id = state.get("interrupted")
            if interrupted_id and interrupted_id not in self.jobs:
                job = self._load_job_for_recovery(interrupted_id)
                if job:
                    # 系统重启后强制暂停，需要用户手动恢复
                    job.status = "paused"
                    job.paused = True
                    job.message = "程序重启，被中断任务已暂停，请手动恢复"
                    self.jobs[interrupted_id] = job
                    # 同步到 transcription_service.jobs（确保 SSE 路由能找到任务）
                    self.transcription_service.jobs[interrupted_id] = job
                    # 更新 job_meta.json 中的状态
                    self.transcription_service.save_job_meta(job)
                    logger.info(f"恢复被中断任务为暂停状态（需手动恢复）: {interrupted_id}")

            # 4. 恢复暂停的任务（保持暂停状态）
            for job_id in state.get("paused", []):
                # 避免重复（可能已经被前面的逻辑处理过）
                if job_id in self.jobs:
                    continue

                job = self._load_job_for_recovery(job_id)
                if job:
                    # 保持暂停状态，不加入队列
                    job.status = "paused"
                    job.paused = True
                    job.message = "程序重启，暂停任务已恢复"
                    self.jobs[job_id] = job
                    # 同步到 transcription_service.jobs（确保 SSE 路由能找到任务）
                    self.transcription_service.jobs[job_id] = job
                    # 更新 job_meta.json 中的状态
                    self.transcription_service.save_job_meta(job)
                    logger.info(f"恢复暂停任务（保持暂停）: {job_id}")

            # 统计恢复情况
            paused_count = len([j for j in self.jobs.values() if j.status == "paused"])
            logger.info(f"队列恢复完成: {len(self.queue)}个排队任务, {paused_count}个暂停任务")

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
            job.status = "queued"

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
                        current_job.paused = True
                        current_job.message = "被强制插队暂停，稍后自动恢复..."
                        # 记录被中断的任务，用于自动恢复
                        self.interrupted_job_id = self.running_job_id
                        result["interrupted_job_id"] = self.running_job_id
                        logger.info(f"[强制插队] 暂停当前任务: {self.running_job_id}, 插队任务: {job_id}")

                job.message = "强制插队（等待当前任务暂停）"

        # 保存队列状态
        self._save_state()

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
            for idx, job_id in enumerate(self.queue):
                job = self.jobs.get(job_id)
                if job:
                    job.message = f"排队中 (位置: {idx + 1})"

            logger.info(f"队列已重新排序: {list(self.queue)}")

        # 保存队列状态
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
        self.worker_thread.join(timeout=5)
        logger.info("队列服务已停止")


# ========== 单例模式 ==========

_queue_service_instance: Optional[JobQueueService] = None


def get_queue_service(transcription_service=None) -> JobQueueService:
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
        _queue_service_instance = JobQueueService(transcription_service)
    return _queue_service_instance
