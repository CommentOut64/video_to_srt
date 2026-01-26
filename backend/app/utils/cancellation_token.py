"""
CancellationToken - 协作式取消令牌

v3.1.0 断点续传基础设施 - Phase 1

核心职责：
1. 提供协作式任务取消/暂停能力，贯穿整个流水线
2. 支持原子区域标记（禁止中断区域）
3. 注册清理回调（如终止子进程）
4. 与 CheckpointManager 集成，支持检查点保存

设计原则：
- 原子性原则：每个阶段定义明确的原子操作单位，中断只能发生在原子操作之间
- 幂等性原则：恢复时可以安全地重新执行未完成的操作
- 最小损失原则：中断后仅需重做当前原子操作，不影响已完成部分
"""

import asyncio
import threading
import time
import logging
from typing import Optional, Callable, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.services.job.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


class CancelledException(Exception):
    """
    任务取消异常

    当任务被取消时抛出此异常，流水线应捕获并优雅退出。
    """
    def __init__(self, job_id: str, message: str = None):
        self.job_id = job_id
        self.message = message or f"任务 {job_id} 已被取消"
        super().__init__(self.message)


class PausedException(Exception):
    """
    任务暂停异常

    当任务被暂停时抛出此异常，流水线应捕获并保存状态后退出。
    """
    def __init__(self, job_id: str, message: str = None):
        self.job_id = job_id
        self.message = message or f"任务 {job_id} 已暂停"
        super().__init__(self.message)


class CancellationToken:
    """
    取消令牌

    用于在多线程/异步环境中传递取消和暂停信号。
    支持原子区域标记和清理回调注册。

    使用示例:
    ```python
    token = CancellationToken(job_id="xxx")

    # 阶段1: 音频提取（禁止中断）
    token.enter_atomic_region("ffmpeg_extract")
    try:
        output_path = await run_ffmpeg(video_path)
    finally:
        if token.exit_atomic_region():
            # 有待处理的暂停/取消
            token.raise_if_canceled()
            token.raise_if_paused()

    # 阶段5: SenseVoice（每 Chunk 可中断）
    for i, chunk in enumerate(chunks):
        token.enter_atomic_region(f"sensevoice_chunk_{i}")
        try:
            result = await sensevoice_infer(chunk)
        finally:
            token.exit_atomic_region()

        # 在 Chunk 之间检查并保存进度
        token.check_and_save(checkpoint_data)
    ```
    """

    def __init__(
        self,
        job_id: str,
        checkpoint_manager: Optional["CheckpointManager"] = None
    ):
        """
        初始化取消令牌

        Args:
            job_id: 任务ID
            checkpoint_manager: 检查点管理器（可选，用于自动保存进度）
        """
        self.job_id = job_id
        self.checkpoint_manager = checkpoint_manager

        # 状态标志
        self._canceled = threading.Event()
        self._paused = threading.Event()

        # 延迟执行标志（用于原子区域内的请求）
        self._pending_cancel = False
        self._pending_pause = False

        # 原子区域状态
        self._in_atomic_region = False
        self._atomic_region_name: Optional[str] = None
        self._atomic_region_start_time: Optional[float] = None

        # 清理回调
        self._cleanup_callbacks: List[Callable] = []

        # 线程安全锁
        self._lock = threading.RLock()

        # 统计信息
        self._stats = {
            "cancel_requests": 0,
            "pause_requests": 0,
            "atomic_regions_entered": 0,
            "pending_operations_deferred": 0,
            "checkpoints_saved": 0
        }

        logger.debug(f"[CancellationToken] 创建令牌: {job_id}")

    # ==================== 状态属性 ====================

    @property
    def is_canceled(self) -> bool:
        """检查是否已取消"""
        return self._canceled.is_set()

    @property
    def is_paused(self) -> bool:
        """检查是否已暂停"""
        return self._paused.is_set()

    @property
    def is_in_atomic_region(self) -> bool:
        """检查是否在原子区域内"""
        with self._lock:
            return self._in_atomic_region

    @property
    def has_pending_operations(self) -> bool:
        """检查是否有待处理的暂停/取消请求"""
        with self._lock:
            return self._pending_cancel or self._pending_pause

    @property
    def stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return self._stats.copy()

    # ==================== 取消/暂停控制 ====================

    def cancel(self):
        """
        触发取消

        如果当前在原子区域内，请求会被延迟到区域结束后执行。
        会执行所有注册的清理回调。
        """
        with self._lock:
            self._stats["cancel_requests"] += 1

            if self._in_atomic_region:
                logger.info(
                    f"[CancellationToken] 在原子区域 '{self._atomic_region_name}' "
                    f"内收到取消请求，延迟执行"
                )
                self._pending_cancel = True
                self._stats["pending_operations_deferred"] += 1
            else:
                logger.info(f"[CancellationToken] 触发取消: {self.job_id}")
                self._canceled.set()
                self._execute_cleanup()

    def pause(self):
        """
        触发暂停

        如果当前在原子区域内，请求会被延迟到区域结束后执行。
        """
        with self._lock:
            self._stats["pause_requests"] += 1

            if self._in_atomic_region:
                logger.info(
                    f"[CancellationToken] 在原子区域 '{self._atomic_region_name}' "
                    f"内收到暂停请求，延迟执行"
                )
                self._pending_pause = True
                self._stats["pending_operations_deferred"] += 1
            else:
                logger.info(f"[CancellationToken] 触发暂停: {self.job_id}")
                self._paused.set()

    def resume(self):
        """恢复运行（取消暂停状态）"""
        with self._lock:
            logger.info(f"[CancellationToken] 恢复运行: {self.job_id}")
            self._paused.clear()
            self._pending_pause = False

    # ==================== 原子区域管理 ====================

    def enter_atomic_region(self, region_name: str):
        """
        进入禁止中断区域

        在此区域内，cancel/pause 请求会被延迟到区域结束。

        Args:
            region_name: 区域名称（用于日志和调试）
        """
        with self._lock:
            if self._in_atomic_region:
                logger.debug(
                    f"[CancellationToken] 嵌套进入原子区域: "
                    f"{self._atomic_region_name} -> {region_name}"
                )

            self._in_atomic_region = True
            self._atomic_region_name = region_name
            self._atomic_region_start_time = time.time()
            self._stats["atomic_regions_entered"] += 1

            logger.debug(f"[CancellationToken] 进入原子区域: {region_name}")

    def exit_atomic_region(self) -> bool:
        """
        退出禁止中断区域

        检查是否有待处理的暂停/取消请求，若有则执行。

        Returns:
            bool: True 表示有待处理请求已被执行，调用方应检查状态
        """
        with self._lock:
            if not self._in_atomic_region:
                logger.warning("[CancellationToken] 未在原子区域内调用 exit_atomic_region")
                return False

            region_name = self._atomic_region_name
            duration = time.time() - (self._atomic_region_start_time or 0)

            self._in_atomic_region = False
            self._atomic_region_name = None
            self._atomic_region_start_time = None

            logger.debug(
                f"[CancellationToken] 退出原子区域: {region_name} "
                f"(耗时 {duration:.2f}s)"
            )

            has_pending = self._pending_cancel or self._pending_pause

            # 处理延迟的取消请求（优先级高于暂停）
            if self._pending_cancel:
                logger.info(
                    f"[CancellationToken] 原子区域 '{region_name}' 结束，执行待处理取消"
                )
                self._canceled.set()
                self._pending_cancel = False
                self._execute_cleanup()

            # 处理延迟的暂停请求
            elif self._pending_pause:
                logger.info(
                    f"[CancellationToken] 原子区域 '{region_name}' 结束，执行待处理暂停"
                )
                self._paused.set()
                self._pending_pause = False

            return has_pending

    # ==================== 检查点 ====================

    def check_canceled(self) -> bool:
        """
        检查点：检查是否应该取消

        在流水线关键位置调用此方法。
        如果已取消，抛出 CancelledException。

        Returns:
            bool: False（如果未取消）

        Raises:
            CancelledException: 如果已取消
        """
        if self._canceled.is_set():
            raise CancelledException(self.job_id)
        return False

    async def check_canceled_async(self) -> bool:
        """
        异步版本的取消检查

        Returns:
            bool: False（如果未取消）

        Raises:
            CancelledException: 如果已取消
        """
        if self._canceled.is_set():
            raise CancelledException(self.job_id)
        # 让出控制权，允许事件循环处理其他任务
        await asyncio.sleep(0)
        return False

    def raise_if_canceled(self):
        """如果已取消，抛出异常"""
        if self._canceled.is_set():
            raise CancelledException(self.job_id)

    def raise_if_paused(self):
        """如果已暂停，抛出异常"""
        if self._paused.is_set():
            raise PausedException(self.job_id)

    def check_and_save(
        self,
        checkpoint_data: Dict[str, Any],
        job_dir: Optional[Any] = None
    ) -> bool:
        """
        检查点：检查暂停/取消，并保存进度

        在每个原子操作完成后调用此方法。

        v3.1.0 更新：支持直接传递 job_dir，自动创建 CheckpointManagerV37

        Args:
            checkpoint_data: 要保存的检查点数据
            job_dir: 任务目录（Path 或 str）

        Returns:
            bool: False 表示正常继续

        Raises:
            CancelledException: 如果已取消
            PausedException: 如果已暂停
        """
        # v3.1.0: 直接使用 CheckpointManagerV37 保存
        if job_dir is not None:
            from app.services.job.checkpoint_manager import CheckpointManagerV37
            from pathlib import Path

            checkpoint_mgr = CheckpointManagerV37(Path(job_dir), logger)
            checkpoint_mgr.save_checkpoint(checkpoint_data)
            with self._lock:
                self._stats["checkpoints_saved"] += 1
        elif self.checkpoint_manager:
            # 兼容旧版本
            self.checkpoint_manager.save_checkpoint(job_dir, checkpoint_data)
            with self._lock:
                self._stats["checkpoints_saved"] += 1

        # 检查取消
        if self._canceled.is_set():
            raise CancelledException(self.job_id)

        # 检查暂停
        if self._paused.is_set():
            raise PausedException(self.job_id)

        return False

    async def check_and_save_async(
        self,
        checkpoint_data: Dict[str, Any],
        job_dir: Optional[Any] = None
    ) -> bool:
        """
        异步版本的检查并保存

        Args:
            checkpoint_data: 要保存的检查点数据
            job_dir: 任务目录

        Returns:
            bool: False 表示正常继续

        Raises:
            CancelledException: 如果已取消
            PausedException: 如果已暂停
        """
        # 保存检查点（同步操作）
        if self.checkpoint_manager and job_dir:
            self.checkpoint_manager.save_checkpoint(job_dir, checkpoint_data)
            with self._lock:
                self._stats["checkpoints_saved"] += 1

        # 让出控制权
        await asyncio.sleep(0)

        # 检查取消
        if self._canceled.is_set():
            raise CancelledException(self.job_id)

        # 检查暂停
        if self._paused.is_set():
            raise PausedException(self.job_id)

        return False

    def wait_if_paused(self, timeout: float = 0.5, check_interval: float = 0.1) -> bool:
        """
        暂停检查点

        如果已暂停，阻塞等待恢复或取消。

        Args:
            timeout: 单次等待超时时间（秒）
            check_interval: 检查间隔（秒）

        Returns:
            bool: True 表示已恢复，可以继续

        Raises:
            CancelledException: 如果在等待期间被取消
        """
        while self._paused.is_set():
            if self._canceled.is_set():
                raise CancelledException(self.job_id)

            # 等待暂停状态改变
            self._paused.wait(check_interval)

            if not self._paused.is_set():
                logger.info(f"[CancellationToken] 任务已恢复: {self.job_id}")
                return True

        return True

    async def wait_if_paused_async(self, check_interval: float = 0.1) -> bool:
        """
        异步版本的暂停等待

        Args:
            check_interval: 检查间隔（秒）

        Returns:
            bool: True 表示已恢复，可以继续

        Raises:
            CancelledException: 如果在等待期间被取消
        """
        while self._paused.is_set():
            if self._canceled.is_set():
                raise CancelledException(self.job_id)

            await asyncio.sleep(check_interval)

            if not self._paused.is_set():
                logger.info(f"[CancellationToken] 任务已恢复: {self.job_id}")
                return True

        return True

    # ==================== 清理回调 ====================

    def register_cleanup(self, callback: Callable):
        """
        注册清理回调

        取消时会调用这些回调（如终止子进程）。

        Args:
            callback: 清理回调函数（无参数）
        """
        with self._lock:
            self._cleanup_callbacks.append(callback)
            logger.debug(
                f"[CancellationToken] 注册清理回调: {callback.__name__ if hasattr(callback, '__name__') else callback}"
            )

    def unregister_cleanup(self, callback: Callable):
        """
        注销清理回调

        Args:
            callback: 要注销的回调函数
        """
        with self._lock:
            if callback in self._cleanup_callbacks:
                self._cleanup_callbacks.remove(callback)
                logger.debug(
                    f"[CancellationToken] 注销清理回调: {callback.__name__ if hasattr(callback, '__name__') else callback}"
                )

    def _execute_cleanup(self):
        """执行所有清理回调"""
        with self._lock:
            callbacks = self._cleanup_callbacks.copy()

        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(
                    f"[CancellationToken] 清理回调执行失败: {e}",
                    exc_info=True
                )

    # ==================== 上下文管理器 ====================

    def atomic_region(self, region_name: str):
        """
        原子区域上下文管理器

        使用示例:
        ```python
        with token.atomic_region("ffmpeg_extract"):
            await run_ffmpeg(video_path)
        # 退出时自动检查待处理操作
        ```
        """
        return AtomicRegionContext(self, region_name)

    # ==================== 重置 ====================

    def reset(self):
        """
        重置令牌状态

        用于任务重试或恢复。
        """
        with self._lock:
            self._canceled.clear()
            self._paused.clear()
            self._pending_cancel = False
            self._pending_pause = False
            self._in_atomic_region = False
            self._atomic_region_name = None
            self._atomic_region_start_time = None
            self._cleanup_callbacks.clear()
            logger.info(f"[CancellationToken] 重置令牌: {self.job_id}")


class AtomicRegionContext:
    """
    原子区域上下文管理器

    用于 with 语句，自动管理原子区域的进入和退出。
    """

    def __init__(self, token: CancellationToken, region_name: str):
        self.token = token
        self.region_name = region_name

    def __enter__(self):
        self.token.enter_atomic_region(self.region_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        has_pending = self.token.exit_atomic_region()

        # 如果没有异常且有待处理操作，检查并抛出
        if exc_type is None and has_pending:
            self.token.raise_if_canceled()
            self.token.raise_if_paused()

        # 不抑制异常
        return False

    async def __aenter__(self):
        self.token.enter_atomic_region(self.region_name)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        has_pending = self.token.exit_atomic_region()

        # 如果没有异常且有待处理操作，检查并抛出
        if exc_type is None and has_pending:
            self.token.raise_if_canceled()
            self.token.raise_if_paused()

        # 不抑制异常
        return False


# ==================== 便捷函数 ====================

def create_cancellation_token(
    job_id: str,
    checkpoint_manager: Optional["CheckpointManager"] = None
) -> CancellationToken:
    """
    创建取消令牌

    Args:
        job_id: 任务ID
        checkpoint_manager: 检查点管理器

    Returns:
        CancellationToken 实例
    """
    return CancellationToken(job_id, checkpoint_manager)
