"""
AsyncDualPipeline - 三级异步流水线控制器

核心架构：
    AudioChunk → [FastWorker (CPU)]
                   ↓ Queue1 (maxsize=5)
                 [SlowWorker (GPU)]
                   ↓ Queue2 (maxsize=5)
                 [AlignmentWorker (CPU)]
                   ↓ 完成

设计决策：
- 生产者-消费者模型：数据单向流动
- 队列背压：asyncio.Queue(maxsize=5) 防止内存溢出
- 错位并行：当 SlowWorker 处理 Chunk N 时，FastWorker 同时处理 Chunk N+1
- 异常传播：任何 Worker 的异常都会传播到 run() 方法
- 结束信号：使用 ProcessingContext.is_end 通知下游停止

V3.1.0 更新：
- 集成 CancellationToken 支持暂停/取消
- 支持断点续传检查点保存
- SlowWorker 保存上文状态 (previous_whisper_text)
- 集成 ProgressEventEmitter 统一进度发射器
- 实时同步 job.progress 并推送 SSE 事件
"""
import asyncio
import logging
from typing import List, Optional, Any, TYPE_CHECKING, Set
from pathlib import Path

from app.schemas.pipeline_context import ProcessingContext
from app.services.audio.chunk_engine import AudioChunk
from app.services.sse_service import get_sse_manager
from app.pipelines.workers import FastWorker, SlowWorker, AlignmentWorker
from app.utils.cancellation_token import CancelledException, PausedException  # V3.1.0: 捕获取消/暂停异常

# V3.7: 导入取消令牌和异常
if TYPE_CHECKING:
    from app.utils.cancellation_token import CancellationToken
    from app.services.progress_emitter import ProgressEventEmitter  # V3.1.0


class AsyncDualPipeline:
    """
    三级异步流水线控制器

    职责：
    1. 编排三个 Worker 的生命周期
    2. 管理队列和背压
    3. 处理异常传播
    4. 推送 SSE 事件

    V3.5 更新：
    - 支持 transcription_profile 参数
    - sensevoice_only 模式下跳过 SlowWorker，FastWorker 直接输出定稿
    """

    def __init__(
        self,
        job_id: str,
        queue_maxsize: int = 5,
        sensevoice_language: str = "auto",
        whisper_language: str = "auto",
        user_glossary: Optional[list] = None,
        enable_semantic_grouping: bool = True,
        alignment_score_threshold: float = 0.3,
        enable_fallback: bool = True,
        transcription_profile: str = "sv_whisper_patch",
        logger: Optional[logging.Logger] = None,
        cancellation_token: Optional["CancellationToken"] = None,  # V3.7: 新增
        progress_emitter: Optional["ProgressEventEmitter"] = None  # V3.1.0: 新增
    ):
        """
        初始化流水线

        Args:
            job_id: 任务 ID
            queue_maxsize: 队列最大长度（背压控制）
            sensevoice_language: SenseVoice 语言设置
            whisper_language: Whisper 语言设置
            user_glossary: 用户词表
            enable_semantic_grouping: 是否启用语义分组
            alignment_score_threshold: 对齐质量阈值
            enable_fallback: 是否启用降级策略
            transcription_profile: 转录模式 (sensevoice_only/sv_whisper_patch/sv_whisper_dual)
            logger: 日志记录器
            cancellation_token: 取消令牌（可选，V3.7）
            progress_emitter: 进度发射器（可选，V3.1.0）
        """
        self.job_id = job_id
        self.logger = logger or logging.getLogger(__name__)
        self.transcription_profile = transcription_profile
        self.cancellation_token = cancellation_token  # V3.7
        self.progress_emitter = progress_emitter  # V3.1.0

        # 判断是否为纯 SenseVoice 模式
        self.is_sensevoice_only = (transcription_profile == "sensevoice_only")
        # V3.10: 判断是否为智能补刀模式
        self.is_patching_mode = (transcription_profile == "sv_whisper_patch")

        if self.is_sensevoice_only:
            self.logger.info("极速模式: 纯 SenseVoice 流水线，跳过 Whisper")
        elif self.is_patching_mode:
            self.logger.info("智能补刀模式: 根据 SenseVoice 质量决定是否调用 Whisper")
        else:
            self.logger.info(f"双流精校模式: 全量 Whisper 转录")

        # 创建队列（带背压）
        self.queue_inter = asyncio.Queue(maxsize=queue_maxsize)  # FastWorker -> SlowWorker
        self.queue_final = asyncio.Queue(maxsize=queue_maxsize)  # SlowWorker -> AlignmentWorker

        # 实例化 FastWorker（总是需要）
        self.fast_worker = FastWorker(
            job_id=job_id,
            sensevoice_language=sensevoice_language,
            enable_semantic_grouping=enable_semantic_grouping,
            is_final_output=self.is_sensevoice_only,  # 极速模式下 FastWorker 输出为定稿
            enable_cross_chunk_merge=True,  # V3.1.0: 启用跨chunk合并（仅SenseVoice模式）
            logger=self.logger
        )

        # SlowWorker 和 AlignmentWorker 仅在非极速模式下创建
        if self.is_sensevoice_only:
            self.slow_worker = None
            self.alignment_worker = None
        else:
            # V3.10: 智能补刀模式下设置 is_patching_mode=True
            self.slow_worker = SlowWorker(
                whisper_language=whisper_language,
                user_glossary=user_glossary,
                is_patching_mode=self.is_patching_mode,  # V3.10: 智能补刀模式
                logger=self.logger
            )

            self.alignment_worker = AlignmentWorker(
                job_id=job_id,
                enable_semantic_grouping=enable_semantic_grouping,
                alignment_score_threshold=alignment_score_threshold,
                enable_fallback=enable_fallback,
                logger=self.logger
            )

        # 获取 SSE 管理器
        self.sse_manager = get_sse_manager()

        # 错误收集
        self.errors: List[Exception] = []
        # V3.1.0: 记录暂停异常，待数据排空后统一抛出
        self.pause_exception: Optional[PausedException] = None

    async def run(
        self,
        audio_chunks: List[AudioChunk],
        full_audio_array: Optional[Any] = None,
        full_audio_sr: int = 16000,
        job_dir: Optional[Path] = None,  # V3.7: 用于保存检查点
        processed_indices: Optional[Set[int]] = None,  # V3.7: 已处理的索引（用于恢复）
        base_slow_count: int = 0,  # V3.1.0: SlowWorker 的基准偏移量（已废弃）
        base_align_count: int = 0,  # V3.1.0: AlignmentWorker 的基准偏移量（已废弃）
        initial_slow_processed_indices: Optional[set] = None,  # V3.1.0: SlowWorker 初始索引
        initial_finalized_indices: Optional[set] = None  # V3.1.0: AlignmentWorker 初始索引
    ) -> List[ProcessingContext]:
        """
        运行流水线

        流程：
        - 极速模式 (sensevoice_only): 仅运行 FastWorker，直接输出定稿
        - 补刀/双流模式: 运行完整三级流水线

        V3.1.0: 支持分别设置各 Worker 的基准偏移量和初始索引，修复恢复后进度跳变问题

        Args:
            audio_chunks: AudioChunk 列表
            full_audio_array: 完整音频数组（用于 Audio Overlap）
            full_audio_sr: 完整音频采样率
            job_dir: 任务目录（可选，V3.7 用于保存检查点）
            processed_indices: 已处理的chunk索引集合（可选，V3.7 用于 FastWorker 跳过）
            base_slow_count: SlowWorker 的基准偏移量（V3.1.0，已废弃）
            base_align_count: AlignmentWorker 的基准偏移量（V3.1.0，已废弃）
            initial_slow_processed_indices: SlowWorker 初始已处理索引集合（V3.1.0）
            initial_finalized_indices: AlignmentWorker 初始已完成索引集合（V3.1.0）

        Returns:
            List[ProcessingContext]: 处理结果列表
        """
        if self.is_sensevoice_only:
            return await self._run_sensevoice_only(
                audio_chunks, full_audio_array, full_audio_sr,
                job_dir, processed_indices
            )
        else:
            return await self._run_full_pipeline(
                audio_chunks, full_audio_array, full_audio_sr,
                job_dir, processed_indices,
                base_slow_count, base_align_count,
                initial_slow_processed_indices, initial_finalized_indices
            )

    async def _run_sensevoice_only(
        self,
        audio_chunks: List[AudioChunk],
        full_audio_array: Optional[Any] = None,
        full_audio_sr: int = 16000,
        job_dir: Optional[Path] = None,  # V3.7
        processed_indices: Optional[Set[int]] = None  # V3.7
    ) -> List[ProcessingContext]:
        """
        极速模式: 仅运行 FastWorker

        FastWorker 输出直接作为定稿推送，跳过 Whisper 和对齐。

        V3.7: 支持逐 Chunk 中断和检查点保存
        V3.1.0: 集成进度发射器，实时推送 SSE 进度
        """
        self.logger.info(f"极速模式开始: {len(audio_chunks)} 个 Chunk")

        results: List[ProcessingContext] = []
        token = self.cancellation_token  # V3.7
        processed_indices = processed_indices or set()
        total_chunks = len(audio_chunks)

        for i, chunk in enumerate(audio_chunks):
            # V3.7: 跳过已处理的 chunk（用于恢复）
            if i in processed_indices:
                self.logger.debug(f"跳过已处理的 chunk {i}")
                continue

            ctx = ProcessingContext(
                job_id=self.job_id,
                chunk_index=i,
                audio_chunk=chunk,
                full_audio_array=full_audio_array,
                full_audio_sr=full_audio_sr
            )

            # V3.7: 进入原子区域（单个 Chunk + SSE 推送）
            if token:
                token.enter_atomic_region(f"fast_chunk_{i}")

            try:
                # FastWorker 处理（SenseVoice 推理 + 分句 + 推送定稿）
                await self.fast_worker.process(ctx)
                results.append(ctx)

                # V3.1.0: 更新进度（极速模式只有 fast 阶段）
                if self.progress_emitter:
                    processed_count = len(results)
                    self.progress_emitter.update_fast(
                        processed_count, total_chunks,
                        message=f"SenseVoice: {processed_count}/{total_chunks}"
                    )
            except Exception as e:
                self.logger.error(f"Chunk {i} 处理失败: {e}", exc_info=True)
                self.errors.append(e)
            finally:
                # V3.7: 退出原子区域
                if token:
                    has_pending = token.exit_atomic_region()
                    if has_pending:
                        self.logger.info(f"[V3.7] Chunk {i} 处理完成后检测到待处理请求")

            # V3.7: 每个 Chunk 处理完成后检查暂停/取消并保存检查点
            if token and job_dir:
                processed_indices.add(i)

                # V3.1.0: 获取字幕快照用于实时持久化（极速模式）
                subtitle_checkpoint_data = {}
                if self.fast_worker and self.fast_worker.subtitle_manager:
                    subtitle_checkpoint_data = self.fast_worker.subtitle_manager.to_checkpoint_data()

                checkpoint_data = {
                    "transcription": {
                        "mode": "sensevoice_only",
                        "processed_indices": list(processed_indices),
                        "processed_count": len(processed_indices),
                        "total_chunks": len(audio_chunks),
                        # V3.1.0: 保存 finalized_indices（极速模式下所有 processed 都是 finalized）
                        "finalized_indices": list(processed_indices),
                        # V3.1.0: 字幕快照（实时持久化核心）
                        **subtitle_checkpoint_data
                    }
                }
                try:
                    token.check_and_save(checkpoint_data, job_dir)
                except PausedException as e:
                    # V3.1.0: 捕获取消暂停，停止派发新 Chunk，等待上层处理
                    if not self.pause_exception:
                        self.pause_exception = e
                    self.logger.info(
                        f"[V3.1.0] 极速模式捕获暂停信号，已处理 {len(processed_indices)} / {total_chunks} 个 Chunk"
                    )
                    break

        if self.errors:
            self.logger.error(f"极速模式执行中发生 {len(self.errors)} 个错误")
            raise self.errors[0]

        if self.pause_exception:
            # V3.1.0: 触发暂停时抛出异常，保持上层状态机一致
            raise self.pause_exception

        self.logger.info(f"极速模式完成: {len(results)} 个 Chunk 已处理")
        return results

    async def _run_full_pipeline(
        self,
        audio_chunks: List[AudioChunk],
        full_audio_array: Optional[Any] = None,
        full_audio_sr: int = 16000,
        job_dir: Optional[Path] = None,  # V3.7
        processed_indices: Optional[Set[int]] = None,  # V3.7
        base_slow_count: int = 0,  # V3.1.0: SlowWorker 的基准偏移量
        base_align_count: int = 0,  # V3.1.0: AlignmentWorker 的基准偏移量
        initial_slow_processed_indices: Optional[set] = None,  # V3.1.0: SlowWorker 初始索引
        initial_finalized_indices: Optional[set] = None  # V3.1.0: AlignmentWorker 初始索引
    ) -> List[ProcessingContext]:
        """
        运行完整三级流水线（补刀/双流模式）

        流程：
        1. 启动三个并行任务（FastWorker, SlowWorker, AlignmentWorker）
        2. FastWorker 遍历 audio_chunks，每个 chunk 包装为 ProcessingContext
        3. 数据通过两个队列单向流动
        4. 等待所有任务完成
        5. 检查异常

        V3.7: 支持检查点保存和恢复
        V3.1.0: 集成进度发射器
        V3.1.0: 支持分别设置各 Worker 的基准偏移量和初始索引，修复恢复后进度跳变问题

        Args:
            audio_chunks: AudioChunk 列表
            full_audio_array: 完整音频数组（用于 Audio Overlap）
            full_audio_sr: 完整音频采样率
            job_dir: 任务目录（可选，V3.7 用于保存检查点）
            processed_indices: 已处理的chunk索引集合（可选，V3.7 用于 FastWorker 跳过）
            base_slow_count: SlowWorker 的基准偏移量（V3.1.0，已废弃，使用索引集合代替）
            base_align_count: AlignmentWorker 的基准偏移量（V3.1.0，已废弃，使用索引集合代替）
            initial_slow_processed_indices: SlowWorker 初始已处理索引集合（V3.1.0）
            initial_finalized_indices: AlignmentWorker 初始已完成索引集合（V3.1.0）

        Returns:
            List[ProcessingContext]: 处理结果列表
        """
        # V3.1.0: 清理历史状态，避免重复抛出旧异常
        self.errors.clear()
        self.pause_exception = None

        total_chunks = len(audio_chunks)  # V3.1.0: 保存总数用于进度计算
        self.logger.info(f"开始三级流水线: {total_chunks} 个 Chunk")

        # 存储结果
        results: List[ProcessingContext] = []

        # V3.7: 初始化已处理索引集合
        processed_indices = processed_indices or set()

        # 启动三个并行任务（V3.1.0: 传递初始索引集合）
        task_fast = asyncio.create_task(
            self._fast_loop(audio_chunks, full_audio_array, full_audio_sr, job_dir, processed_indices, total_chunks)
        )
        task_slow = asyncio.create_task(
            self._slow_loop(job_dir, total_chunks, base_slow_count, initial_slow_processed_indices)
        )
        task_align = asyncio.create_task(
            self._align_loop(results, job_dir, total_chunks, base_align_count, initial_finalized_indices)
        )

        # 等待所有任务结束（return_exceptions=True 确保一个挂了不会立刻抛出）
        await_results = await asyncio.gather(
            task_fast, task_slow, task_align,
            return_exceptions=True
        )

        # 检查是否有异常
        for res in await_results:
            if isinstance(res, Exception):
                self.logger.error(f"任务异常: {res}", exc_info=res)
                raise res

        # 检查收集的错误
        if self.errors:
            self.logger.error(f"流水线执行中发生 {len(self.errors)} 个错误")
            raise self.errors[0]

        if self.pause_exception:
            # V3.1.0: 等待队列排空后再通知上层暂停，避免进度回退
            raise self.pause_exception

        self.logger.info(f"三级流水线完成: {len(results)} 个 Chunk 已处理")

        return results

    async def _fast_loop(
        self,
        chunks: List[AudioChunk],
        full_audio_array: Optional[Any] = None,
        full_audio_sr: int = 16000,
        job_dir: Optional[Path] = None,  # V3.7
        processed_indices: Optional[Set[int]] = None,  # V3.7
        total_chunks: int = 0  # V3.1.0
    ):
        """
        FastWorker 循环（生产者）

        职责：
        1. 遍历所有 audio_chunks
        2. 每个 chunk 包装为 ProcessingContext
        3. 调用 FastWorker.process()
        4. 将 context 放入 queue_inter
        5. 发送结束信号

        V3.7: 支持原子区域和检查点保存
        V3.1.0: 集成进度发射器

        Args:
            chunks: AudioChunk 列表
            full_audio_array: 完整音频数组（用于 Audio Overlap）
            full_audio_sr: 完整音频采样率
            job_dir: 任务目录（可选，V3.7）
            processed_indices: 已处理的chunk索引集合（可选，V3.7）
            total_chunks: 总 Chunk 数（V3.1.0）
        """
        token = self.cancellation_token  # V3.7
        processed_indices = processed_indices or set()
        total_chunks = total_chunks or len(chunks)

        # V3.1.0: 计算基准偏移量（已完成的 chunk 数量）
        # 修复进度归零问题：恢复后 fast_processed_count 应该从已完成数量开始累加
        base_fast_count = len(processed_indices)
        fast_processed_count = 0  # V3.1.0: 追踪本次新处理的数量
        pause_requested = False  # V3.1.0: 捕获暂停后进入排空模式
        should_send_end_signal = True  # V3.1.0: 控制是否需要发送正常结束信号

        try:
            for i, chunk in enumerate(chunks):
                # V3.7: 跳过已处理的 chunk（用于恢复）
                if i in processed_indices:
                    self.logger.debug(f"[FastWorker] 跳过已处理的 chunk {i}")
                    continue

                # 创建处理上下文（包含完整音频数组）
                ctx = ProcessingContext(
                    job_id=self.job_id,
                    chunk_index=i,
                    audio_chunk=chunk,
                    full_audio_array=full_audio_array,
                    full_audio_sr=full_audio_sr
                )

                # V3.7: 进入原子区域（单个 Chunk 处理 + SSE 推送）
                if token:
                    token.enter_atomic_region(f"fast_worker_chunk_{i}")

                try:
                    # FastWorker 处理（SenseVoice 推理 + 分句 + 推送草稿）
                    await self.fast_worker.process(ctx)

                    # 放入队列（如果队列满了，会自动阻塞，实现背压）
                    await self.queue_inter.put(ctx)
                    fast_processed_count += 1  # V3.1.0

                    # V3.1.0: 更新 FastWorker 进度（叠加基准偏移量）
                    if self.progress_emitter:
                        total_processed = base_fast_count + fast_processed_count
                        self.progress_emitter.update_fast(
                            total_processed, total_chunks,
                            message=f"SenseVoice: {total_processed}/{total_chunks}"
                        )
                finally:
                    # V3.7: 退出原子区域
                    if token:
                        has_pending = token.exit_atomic_region()
                        if has_pending:
                            self.logger.info(f"[V3.7] FastWorker chunk {i} 完成后检测到待处理请求")

                # V3.7: 每个 Chunk 处理完成后保存检查点
                if token and job_dir:
                    processed_indices.add(i)
                    checkpoint_data = {
                        "transcription": {
                            "fast_processed_indices": list(processed_indices),
                            "fast_processed_count": len(processed_indices),
                            "total_chunks": len(chunks)
                        }
                    }
                    try:
                        token.check_and_save(checkpoint_data, job_dir)
                    except PausedException as e:
                        # V3.1.0: 捕获暂停信号，停止派发新 Chunk，但允许下游排空
                        pause_requested = True
                        if not self.pause_exception:
                            self.pause_exception = e
                        self.logger.info(
                            f"[V3.1.0] FastWorker 捕获暂停信号，已完成 {len(processed_indices)} / {len(chunks)} 个 Chunk"
                        )
                        break

        except CancelledException as e:
            self.logger.error(f"FastWorker 循环取消: {e}", exc_info=True)
            self.errors.append(e)
            should_send_end_signal = False

            error_ctx = ProcessingContext(
                job_id=self.job_id,
                chunk_index=-1,
                audio_chunk=None,
                is_end=True,
                error=e
            )
            await self.queue_inter.put(error_ctx)
            return

        except Exception as e:
            self.logger.error(f"FastWorker 循环异常: {e}", exc_info=True)
            self.errors.append(e)

            # 发送错误信号
            error_ctx = ProcessingContext(
                job_id=self.job_id,
                chunk_index=-1,
                audio_chunk=None,
                is_end=True,
                error=e
            )
            await self.queue_inter.put(error_ctx)
            should_send_end_signal = False
            return
        finally:
            if should_send_end_signal:
                end_ctx = ProcessingContext(
                    job_id=self.job_id,
                    chunk_index=-1,
                    audio_chunk=None,
                    is_end=True
                )
                await self.queue_inter.put(end_ctx)
                if pause_requested:
                    self.logger.info("[V3.1.0] FastWorker 已发送暂停结束信号，等待下游排空")
                else:
                    self.logger.info("FastWorker 循环完成")

    async def _slow_loop(
        self,
        job_dir: Optional[Path] = None,
        total_chunks: int = 0,
        base_slow_count: int = 0,  # V3.1.0: 基准偏移量（已完成的 chunk 数量）
        initial_slow_processed_indices: Optional[set] = None  # V3.1.0: 初始已处理索引集合
    ):
        """
        SlowWorker 循环（中间消费者-生产者）

        职责：
        1. 从 queue_inter 取 context
        2. 调用 SlowWorker.process()
        3. 将 context 放入 queue_final
        4. 透传结束/错误信号

        V3.7: 支持原子区域和检查点保存（包括关键的 previous_whisper_text）
        V3.1.0: 集成进度发射器
        V3.1.0: 保存 slow_processed_indices 用于断点续传
        V3.1.0: 使用累计索引集合，修复恢复后进度不准确问题
        """
        token = self.cancellation_token  # V3.7
        slow_processed_count = 0  # V3.7: 追踪本次新处理的数量
        # V3.1.0: 使用累计索引集合（类似 FastWorker）
        slow_processed_indices = set(initial_slow_processed_indices) if initial_slow_processed_indices else set()
        pause_requested = False  # V3.1.0: 捕获暂停后继续排空队列

        try:
            while True:
                # 从队列取 context
                ctx = await self.queue_inter.get()

                # 检查结束信号或错误
                if ctx.is_end or ctx.error:
                    await self.queue_final.put(ctx)  # 透传
                    break

                chunk_index = ctx.chunk_index  # V3.7: 获取 chunk 索引

                # V3.7: 进入原子区域（单个 Chunk 处理 + 上下文更新）
                if token:
                    token.enter_atomic_region(f"slow_worker_chunk_{chunk_index}")

                try:
                    # SlowWorker 处理（Whisper 推理 + 幻觉检测）
                    await self.slow_worker.process(ctx)

                    # 更新 SlowWorker 的 Prompt 缓存
                    if ctx.whisper_result:
                        whisper_text = ctx.whisper_result.get('text', '')
                        self.slow_worker.update_prompt_cache(whisper_text)

                    # 放入队列
                    await self.queue_final.put(ctx)
                    slow_processed_count += 1
                    slow_processed_indices.add(chunk_index)  # V3.1.0: 累加到集合（类似 FastWorker）

                    # V3.1.0: 更新 SlowWorker 进度（使用累计索引数量）
                    if self.progress_emitter and total_chunks > 0:
                        total_processed = len(slow_processed_indices)
                        self.progress_emitter.update_slow(
                            total_processed, total_chunks,
                            message=f"Whisper: {total_processed}/{total_chunks}"
                        )
                finally:
                    # V3.7: 退出原子区域
                    if token:
                        has_pending = token.exit_atomic_region()
                        if has_pending:
                            self.logger.info(f"[V3.7] SlowWorker chunk {chunk_index} 完成后检测到待处理请求")

                # V3.7: 每个 Chunk 处理完成后保存检查点（包含关键的 previous_whisper_text）
                if token and job_dir:
                    # 获取当前的 prompt_cache 作为 previous_whisper_text
                    previous_whisper_text = getattr(self.slow_worker, 'prompt_cache', '')
                    checkpoint_data = {
                        "transcription": {
                            "slow_processed_count": len(slow_processed_indices),  # V3.1.0: 使用累计数量
                            "slow_processed_indices": list(slow_processed_indices),  # V3.1.0: 保存累计索引集合
                            "previous_whisper_text": previous_whisper_text,  # 关键：保存上文状态
                            "last_slow_chunk_index": chunk_index
                        }
                    }
                    try:
                        token.check_and_save(checkpoint_data, job_dir)
                    except PausedException as e:
                        if not pause_requested:
                            self.logger.info("[V3.1.0] SlowWorker 捕获暂停信号，继续排空 queue_inter")
                        pause_requested = True
                        if not self.pause_exception:
                            self.pause_exception = e
                        # 不抛异常，等待队列排空

            self.logger.info("SlowWorker 循环完成")

        except Exception as e:
            self.logger.error(f"SlowWorker 循环异常: {e}", exc_info=True)
            self.errors.append(e)

            # 发送错误信号
            error_ctx = ProcessingContext(
                job_id=self.job_id,
                chunk_index=-1,
                audio_chunk=None,
                is_end=True,
                error=e
            )
            await self.queue_final.put(error_ctx)
        finally:
            if pause_requested:
                self.logger.info("[V3.1.0] SlowWorker 已完成排空，等待 AlignmentWorker 同步完成")

    async def _align_loop(
        self,
        results: List[ProcessingContext],
        job_dir: Optional[Path] = None,
        total_chunks: int = 0,
        base_align_count: int = 0,  # V3.1.0: 基准偏移量（已完成的 chunk 数量）
        initial_finalized_indices: Optional[set] = None  # V3.1.0: 初始已完成索引集合
    ):
        """
        AlignmentWorker 循环（最终消费者）

        职责：
        1. 从 queue_final 取 context
        2. 调用 AlignmentWorker.process()
        3. 收集结果到 results 列表
        4. 检测结束信号

        V3.7: 支持原子区域和检查点保存
        V3.1.0: 集成进度发射器
        V3.1.0: 使用累计索引集合，修复恢复后进度不准确问题

        Args:
            results: 结果列表（用于收集 context）
            job_dir: 任务目录（可选，V3.7）
            total_chunks: 总 Chunk 数（V3.1.0）
            base_align_count: 基准偏移量（V3.1.0）
            initial_finalized_indices: 初始已完成索引集合（V3.1.0）
        """
        token = self.cancellation_token  # V3.7
        align_processed_count = 0  # V3.7: 追踪本次新处理的数量
        # V3.1.0: 使用累计索引集合（类似 FastWorker 和 SlowWorker）
        finalized_indices = set(initial_finalized_indices) if initial_finalized_indices else set()
        pause_requested = False  # V3.1.0: 捕获暂停后继续排空 queue_final

        try:
            while True:
                # 从队列取 context
                ctx = await self.queue_final.get()

                # 检查结束信号
                if ctx.is_end:
                    if ctx.error:
                        self.logger.error(f"上游错误: {ctx.error}")
                        raise ctx.error
                    break

                chunk_index = ctx.chunk_index  # V3.7: 获取 chunk 索引

                # V3.7: 进入原子区域（单个 Chunk 对齐 + SSE 推送）
                if token:
                    token.enter_atomic_region(f"align_worker_chunk_{chunk_index}")

                try:
                    # AlignmentWorker 处理（双流对齐 + 分句 + 推送定稿）
                    await self.alignment_worker.process(ctx)

                    # 收集结果
                    results.append(ctx)
                    align_processed_count += 1
                    finalized_indices.add(chunk_index)  # V3.1.0: 累加到集合

                    # V3.1.0: 更新 AlignmentWorker 进度（使用累计索引数量）
                    if self.progress_emitter and total_chunks > 0:
                        total_processed = len(finalized_indices)
                        self.progress_emitter.update_align(
                            total_processed, total_chunks,
                            message=f"对齐: {total_processed}/{total_chunks}"
                        )
                finally:
                    # V3.7: 退出原子区域
                    if token:
                        has_pending = token.exit_atomic_region()
                        if has_pending:
                            self.logger.info(f"[V3.7] AlignmentWorker chunk {chunk_index} 完成后检测到待处理请求")

                # V3.7: 每个 Chunk 处理完成后保存检查点
                if token and job_dir:
                    # V3.1.0: 获取字幕快照用于实时持久化
                    subtitle_checkpoint_data = {}
                    if self.alignment_worker and self.alignment_worker.subtitle_manager:
                        subtitle_checkpoint_data = self.alignment_worker.subtitle_manager.to_checkpoint_data()

                    checkpoint_data = {
                        "transcription": {
                            "align_processed_count": len(finalized_indices),  # V3.1.0: 使用累计数量
                            "last_align_chunk_index": chunk_index,
                            "completed_chunks": len(results),
                            # V3.1.0: 保存累计的 finalized_indices
                            "finalized_indices": list(finalized_indices),
                            # V3.1.0: 字幕快照（实时持久化核心）
                            **subtitle_checkpoint_data
                        }
                    }
                    try:
                        token.check_and_save(checkpoint_data, job_dir)
                    except PausedException as e:
                        if not pause_requested:
                            self.logger.info("[V3.1.0] AlignmentWorker 捕获暂停信号，继续排空 queue_final")
                        pause_requested = True
                        if not self.pause_exception:
                            self.pause_exception = e
                        # 继续排空，待数据全部写入后再暂停

            self.logger.info("AlignmentWorker 循环完成")

        except Exception as e:
            self.logger.error(f"AlignmentWorker 循环异常: {e}", exc_info=True)
            self.errors.append(e)
        finally:
            if pause_requested:
                self.logger.info("[V3.1.0] AlignmentWorker 已排空所有上下文，等待上层暂停")

    def get_statistics(self) -> dict:
        """
        获取流水线统计信息

        Returns:
            dict: 统计信息
        """
        return {
            "queue_inter_size": self.queue_inter.qsize(),
            "queue_final_size": self.queue_final.qsize(),
            "errors": len(self.errors)
        }


# 便捷函数
def get_async_dual_pipeline(
    job_id: str,
    queue_maxsize: int = 5,
    logger: Optional[logging.Logger] = None,
    cancellation_token: Optional["CancellationToken"] = None  # V3.7: 新增
) -> AsyncDualPipeline:
    """
    获取异步双流流水线实例

    Args:
        job_id: 任务 ID
        queue_maxsize: 队列最大长度
        logger: 日志记录器
        cancellation_token: 取消令牌（可选，V3.7）

    Returns:
        AsyncDualPipeline 实例
    """
    return AsyncDualPipeline(
        job_id=job_id,
        queue_maxsize=queue_maxsize,
        logger=logger,
        cancellation_token=cancellation_token  # V3.7
    )
