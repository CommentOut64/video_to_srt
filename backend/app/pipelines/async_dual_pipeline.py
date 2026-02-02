"""
AsyncDualPipeline - 三级异步流水线控制器

核心架构：
    AudioChunk → [FastWorker (CPU)]
                   ↓ Queue1 (maxsize=5)
                 [SlowWorker (GPU)]
                   ↓ Queue2 (maxsize=5)
                 [Alignment Stage (Pipeline/CPU)]
                   ↓ 完成

设计决策：
- 生产者-消费者模型：数据单向流动
- 队列背压：asyncio.Queue(maxsize=5) 防止内存溢出
- 错位并行：当 SlowWorker 处理 Chunk N 时，FastWorker 同时处理 Chunk N+1
- 异常传播：任何阶段异常都会传播到 run() 方法
- 结束信号：使用 ProcessingContext.is_end 通知下游停止

V3.1.0 更新：
- 集成 CancellationToken 支持暂停/取消
- 支持断点续传检查点保存
- 流水线保存 Whisper 上文状态 (previous_whisper_text)
- 集成 ProgressEventEmitter 统一进度发射器
- 实时同步 job.progress 并推送 SSE 事件

V3.2.0+dev.20260123.05 更新：
- 修复暂停快照实例变量未同步问题
- 确保 _fast_processed_indices, _slow_processed_indices, _finalized_indices 实时更新
- 解决恢复时"无可靠恢复点"导致从头开始的问题
"""
import asyncio
import copy
import logging
from typing import List, Optional, Any, TYPE_CHECKING, Set, Dict, Tuple
from pathlib import Path

from app.core.asr.engine import ASREngine
from app.core.thresholds import ThresholdConfig, needs_whisper_patch
from app.schemas.pipeline_context import ProcessingContext
from app.models.sensevoice_models import SentenceSegment
from app.services.audio.chunk_engine import AudioChunk
from app.services.alignment.default_aligner import DefaultAligner
from app.services.alignment.text_normalizer import get_alignment_text_normalizer
from app.services.alignment.types import NormalizationResult
from app.services.arbitration.arbiter import Arbiter, ArbitrationResult
from app.services.arbitration.hallucination_detector import HallucinationDetector
from app.services.model_runtime_config_service import get_model_runtime_config_service
from app.services.sse_service import get_sse_manager
from app.services.segmentation.default_segmenter import DefaultSegmenter
from app.services.streaming_subtitle import get_streaming_subtitle_manager
from app.services.punctuation.fast_punctuation_pipeline import FastPunctuationPipeline
from app.services.punctuation.base import PunctuationResult, PuncPosition, SplitPoint
from app.services.bridge.bridge_controller import BridgeBatch, BridgeController
from app.services.punctuation.semantic_buffer import (
    PunctuationDecision,
    SemanticBuffer,
    SemanticBufferInput,
    SemanticChunk,
)
from app.services.punctuation.semantic_injector import SemanticInjector
from app.services.whisper.whisper_text_sanitizer import WhisperTextSanitizer
from app.pipelines.workers import FastWorker, SlowWorker, SlowWorkerResult
from app.utils.prompt_builder import get_prompt_builder
from app.utils.cancellation_token import CancelledException, PausedException  # V3.1.0: 捕获取消/暂停异常

# v3.1.0: 导入取消令牌和异常
if TYPE_CHECKING:
    from app.utils.cancellation_token import CancellationToken
    from app.services.progress_emitter import ProgressEventEmitter  # V3.1.0
    from app.services.punctuation.service import PunctuationService


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
        draft_engine: ASREngine,
        patch_engine: Optional[ASREngine] = None,
        queue_maxsize: int = 5,
        sensevoice_language: str = "auto",
        punctuation_service: Optional["PunctuationService"] = None,
        whisper_language: str = "auto",
        user_glossary: Optional[list] = None,
        enable_semantic_grouping: bool = True,
        enable_semantic_buffer: bool = True,
        semantic_buffer: Optional[SemanticBuffer] = None,
        alignment_score_threshold: float = 0.3,
        enable_fallback: bool = True,
        transcription_profile: str = "sv_whisper_patch",
        segmenter: Optional[DefaultSegmenter] = None,
        aligner: Optional[DefaultAligner] = None,
        patching_threshold: Optional[ThresholdConfig] = None,
        enable_cross_chunk_merge: bool = True,
        enable_bridge_controller: bool = True,
        bridge_controller: Optional[BridgeController] = None,
        debug_punctuation: bool = False,
        logger: Optional[logging.Logger] = None,
        cancellation_token: Optional["CancellationToken"] = None,  # v3.1.0: 新增
        progress_emitter: Optional["ProgressEventEmitter"] = None  # V3.1.0: 新增
    ):
        """
        初始化流水线

        Args:
            job_id: 任务 ID
            queue_maxsize: 队列最大长度（背压控制）
            sensevoice_language: SenseVoice 语言设置
            punctuation_service: 标点服务（可选）
            whisper_language: Whisper 语言设置
            user_glossary: 用户词表
            enable_semantic_grouping: 是否启用语义分组
            enable_semantic_buffer: 是否启用 SemanticBuffer 语义缓冲
            semantic_buffer: 语义缓冲器实例（可选）
            alignment_score_threshold: 对齐质量阈值
            enable_fallback: 是否启用降级策略
            transcription_profile: 转录模式 (sensevoice_only/sv_whisper_patch/sv_whisper_dual)
            draft_engine: 草稿引擎实例（必须提供）
            patch_engine: 复核引擎实例（非极速模式必须提供）
            segmenter: 分句服务实例（可选）
            aligner: 对齐服务实例（可选）
            patching_threshold: 复核阈值配置（可选）
            enable_cross_chunk_merge: 是否启用跨 chunk 合并
            enable_bridge_controller: 是否启用 Bridge 控制器
            bridge_controller: Bridge 控制器实例（可选）
            debug_punctuation: 是否启用标点调试输出
            logger: 日志记录器
            cancellation_token: 取消令牌（可选，v3.1.0）
            progress_emitter: 进度发射器（可选，V3.1.0）
        """
        self.job_id = job_id
        self.logger = logger or logging.getLogger(__name__)
        self.transcription_profile = transcription_profile
        self.cancellation_token = cancellation_token  # v3.1.0
        self.progress_emitter = progress_emitter  # V3.1.0
        if not draft_engine:
            raise ValueError("AsyncDualPipeline 需要提供 draft_engine")
        self.draft_engine = draft_engine
        self.patch_engine = patch_engine
        self.patching_threshold = patching_threshold
        self.enable_cross_chunk_merge = enable_cross_chunk_merge
        self.debug_punctuation = debug_punctuation
        self.user_glossary = user_glossary
        self.previous_whisper_text: Optional[str] = None
        self._job_dir: Optional[Path] = None
        self._bridge_prompt_hint: Optional[str] = None
        self._bridge_last_batch: Optional[BridgeBatch] = None
        self._enable_bridge_batches: bool = False
        self._context_cache: Dict[int, ProcessingContext] = {}
        self._audio_chunks_by_index: Dict[int, AudioChunk] = {}
        self._vad_intervals: Optional[List[Tuple[float, float]]] = None
        self._full_audio_array: Optional[Any] = None
        self._full_audio_sr: int = 16000

        # 判断是否为纯 SenseVoice 模式
        self.is_sensevoice_only = (transcription_profile == "sensevoice_only")
        # V3.10: 判断是否为智能复核模式
        self.is_patching_mode = (transcription_profile == "sv_whisper_patch")

        if self.is_sensevoice_only:
            self.logger.info("极速模式: 纯 SenseVoice 流水线，跳过 Whisper")
        elif self.is_patching_mode:
            self.logger.info("智能复核模式: 根据 SenseVoice 质量决定是否调用 Whisper")
        else:
            self.logger.info(f"双流精校模式: 全量 Whisper 转录")

        if not self.is_sensevoice_only and not self.patch_engine:
            raise ValueError("非极速模式下必须提供 patch_engine")

        # 创建队列（带背压）
        self.queue_inter = asyncio.Queue(maxsize=queue_maxsize)  # FastWorker -> SlowWorker
        self.queue_final = asyncio.Queue(maxsize=queue_maxsize)  # SlowWorker -> 对齐阶段

        # 初始化分句服务与字幕管理器
        if segmenter is None:
            segmenter = DefaultSegmenter(
                is_enable_semantic_grouping=enable_semantic_grouping,
                is_enable_cross_chunk_merge=enable_cross_chunk_merge,
                logger=self.logger,
            )
        self.segmenter = segmenter
        self.subtitle_manager = get_streaming_subtitle_manager(job_id)

        # V3.2.0+dev.20260130.03: SemanticBuffer 语义缓冲（Phase C 接入）
        self.enable_semantic_buffer = enable_semantic_buffer
        self.semantic_buffer = semantic_buffer if enable_semantic_buffer else None
        if self.semantic_buffer is None and enable_semantic_buffer:
            self.semantic_buffer = SemanticBuffer(logger=self.logger)

        # V3.2.0+dev.20260201.04: Bridge 控制器（Phase E 完整实现）
        self.bridge_controller = bridge_controller if enable_bridge_controller else None
        if self.bridge_controller is None and enable_bridge_controller:
            self.bridge_controller = BridgeController(logger=self.logger)
        self._enable_bridge_batches = bool(
            self.bridge_controller and self.semantic_buffer and not self.is_sensevoice_only
        )

        # V3.2.0+dev.20260129.01: 标点服务注入
        if punctuation_service is None:
            from app.services.punctuation.service import get_punctuation_service

            punctuation_service = get_punctuation_service()
        self.punctuation_service = punctuation_service

        # V3.2.0+dev.20260202.03: 统一规范化与 Whisper 最小清洗器
        self._text_normalizer = get_alignment_text_normalizer(logger=self.logger)
        self._whisper_sanitizer = WhisperTextSanitizer(logger=self.logger)
        # V3.2.0+dev.20260202.08: 仲裁器与幻觉检测器
        self._hallucination_detector = HallucinationDetector(logger=self.logger)
        self._arbiter = Arbiter(
            logger=self.logger,
            hallucination_detector=self._hallucination_detector,
        )
        self._fast_punctuator = FastPunctuationPipeline(
            job_id=self.job_id,
            punctuation_service=self.punctuation_service,
            logger=self.logger,
        )

        # 实例化 FastWorker（仅推理）
        self.fast_worker = FastWorker(
            job_id=job_id,
            draft_engine=self.draft_engine,
            sensevoice_language=sensevoice_language,
            logger=self.logger
        )

        # SlowWorker 仅在非极速模式下创建；对齐阶段由流水线负责
        if self.is_sensevoice_only:
            self.slow_worker = None
            self.aligner = None
        else:
            # V3.10: 智能复核模式下设置 is_patching_mode=True
            self.slow_worker = SlowWorker(
                patch_engine=self.patch_engine,
                whisper_language=whisper_language,
                punctuation_service=self.punctuation_service,
                logger=self.logger
            )

            if aligner is None:
                aligner = DefaultAligner(
                    alignment_score_threshold=alignment_score_threshold,
                    is_enable_fallback=enable_fallback,
                    is_enable_semantic_grouping=enable_semantic_grouping,
                    logger=self.logger,
                )

            self.aligner = aligner

        # 获取 SSE 管理器
        self.sse_manager = get_sse_manager()

        # 错误收集
        self.errors: List[Exception] = []
        # V3.1.0: 记录暂停异常，待数据排空后统一抛出
        self.pause_exception: Optional[PausedException] = None
        # V3.2.0+dev.20260123.04: 暂停快照所需索引汇总（用于强制保存）
        self._fast_processed_indices: Set[int] = set()
        self._slow_processed_indices: Set[int] = set()
        self._finalized_indices: Set[int] = set()
        self._last_slow_chunk_index = -1
        self._last_align_chunk_index = -1
        self.is_pause_snapshot_saved = False

    async def run(
        self,
        audio_chunks: List[AudioChunk],
        full_audio_array: Optional[Any] = None,
        full_audio_sr: int = 16000,
        job_dir: Optional[Path] = None,  # v3.1.0: 用于保存检查点
        vad_intervals: Optional[List[Tuple[float, float]]] = None,
        processed_indices: Optional[Set[int]] = None,  # v3.1.0: 已处理的索引（用于恢复）
        base_slow_count: int = 0,  # V3.1.0: SlowWorker 的基准偏移量（已废弃）
        base_align_count: int = 0,  # V3.1.0: 对齐阶段的基准偏移量（已废弃）
        initial_slow_processed_indices: Optional[set] = None,  # V3.1.0: SlowWorker 初始索引
        initial_finalized_indices: Optional[set] = None  # V3.1.0: 对齐阶段初始索引
    ) -> List[ProcessingContext]:
        """
        运行流水线

        流程：
        - 极速模式 (sensevoice_only): 仅运行 FastWorker，直接输出定稿
        - 复核/双流模式: 运行完整三级流水线

        V3.1.0: 支持分别设置各 Worker 的基准偏移量和初始索引，修复恢复后进度跳变问题

        Args:
            audio_chunks: AudioChunk 列表
            full_audio_array: 完整音频数组（用于 Audio Overlap）
            full_audio_sr: 完整音频采样率
            job_dir: 任务目录（可选，v3.1.0 用于保存检查点）
            processed_indices: 已处理的chunk索引集合（可选，v3.1.0 用于 FastWorker 跳过）
            base_slow_count: SlowWorker 的基准偏移量（V3.1.0，已废弃）
            base_align_count: 对齐阶段的基准偏移量（V3.1.0，已废弃）
            initial_slow_processed_indices: SlowWorker 初始已处理索引集合（V3.1.0）
            initial_finalized_indices: 对齐阶段初始已完成索引集合（V3.1.0）

        Returns:
            List[ProcessingContext]: 处理结果列表
        """
        self._job_dir = job_dir
        if self.is_sensevoice_only:
            return await self._run_sensevoice_only(
                audio_chunks, full_audio_array, full_audio_sr,
                job_dir, processed_indices
            )
        else:
            return await self._run_full_pipeline(
                audio_chunks, full_audio_array, full_audio_sr,
                job_dir, vad_intervals, processed_indices,
                base_slow_count, base_align_count,
                initial_slow_processed_indices, initial_finalized_indices
            )

    def _force_save_pause_checkpoint(self, job_dir: Optional[Path], total_chunks: int) -> bool:
        """暂停时强制落盘检查点，避免排空后进度丢失。"""
        if not job_dir:
            return False
        try:
            from app.services.job.checkpoint_manager import CheckpointManagerV37

            checkpoint_mgr = CheckpointManagerV37(job_dir, self.logger)
            subtitle_data = {}
            if self.subtitle_manager:
                subtitle_data = self.subtitle_manager.to_checkpoint_data()

            fast_indices = sorted(self._fast_processed_indices)
            slow_indices = sorted(self._slow_processed_indices)
            finalized_indices = sorted(self._finalized_indices)

            checkpoint_data = {
                "preprocessing": {
                    "total_chunks": total_chunks,
                },
                "transcription": {
                    "fast_processed_indices": fast_indices,
                    "fast_processed_count": len(fast_indices),
                    "slow_processed_indices": slow_indices,
                    "slow_processed_count": len(slow_indices),
                    "previous_whisper_text": self.previous_whisper_text or "",
                    "last_slow_chunk_index": self._last_slow_chunk_index,
                    "finalized_indices": finalized_indices,
                    "align_processed_count": len(finalized_indices),
                    "last_align_chunk_index": self._last_align_chunk_index,
                    "total_chunks": total_chunks,
                    **subtitle_data,
                }
            }
            checkpoint_mgr.save_checkpoint(checkpoint_data)
            self.logger.info("[V3.2.0+dev.20260123.05] 暂停快照已写入检查点")
            return True
        except Exception as exc:
            self.logger.warning("[V3.2.0+dev.20260123.05] 暂停快照写入失败: %s", exc)
            return False

    async def _run_sensevoice_only(
        self,
        audio_chunks: List[AudioChunk],
        full_audio_array: Optional[Any] = None,
        full_audio_sr: int = 16000,
        job_dir: Optional[Path] = None,  # v3.1.0
        processed_indices: Optional[Set[int]] = None  # v3.1.0
    ) -> List[ProcessingContext]:
        """
        极速模式: 仅运行 FastWorker

        FastWorker 输出直接作为定稿推送，跳过 Whisper 和对齐。

        v3.1.0: 支持逐 Chunk 中断和检查点保存
        V3.1.0: 集成进度发射器，实时推送 SSE 进度
        """
        self.logger.info(f"极速模式开始: {len(audio_chunks)} 个 Chunk")

        results: List[ProcessingContext] = []
        token = self.cancellation_token  # v3.1.0
        processed_indices = processed_indices or set()
        self._fast_processed_indices = processed_indices
        self._finalized_indices = processed_indices
        total_chunks = len(audio_chunks)
        last_chunk_index: Optional[int] = None

        for i, chunk in enumerate(audio_chunks):
            # v3.1.0: 跳过已处理的 chunk（用于恢复）
            if i in processed_indices:
                self.logger.debug(f"跳过已处理的 chunk {i}")
                continue

            ctx = ProcessingContext(
                job_id=self.job_id,
                chunk_index=i,
                audio_chunk=chunk,
                job_dir=job_dir,
                debug_punctuation=self.debug_punctuation,
                full_audio_array=full_audio_array,
                full_audio_sr=full_audio_sr
            )

            # v3.1.0: 进入原子区域（单个 Chunk + SSE 推送）
            if token:
                token.enter_atomic_region(f"fast_chunk_{i}")

            try:
                # FastWorker 处理（仅推理）
                await self.fast_worker.process(ctx)
                normalized = self._normalize_sensevoice_result(ctx)
                await self._apply_fast_punctuation(ctx, normalized)
                await self._emit_draft_sentences(ctx, is_final_output=True)
                results.append(ctx)
                last_chunk_index = i

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
                # v3.1.0: 退出原子区域
                if token:
                    has_pending = token.exit_atomic_region()
                    if has_pending:
                        self.logger.info(f"[v3.1.0] Chunk {i} 处理完成后检测到待处理请求")

            # v3.1.0: 每个 Chunk 处理完成后检查暂停/取消并保存检查点
            if token and job_dir:
                processed_indices.add(i)
                self._last_align_chunk_index = i

                # V3.1.0: 获取字幕快照用于实时持久化（极速模式）
                subtitle_checkpoint_data = {}
                if self.subtitle_manager:
                    subtitle_checkpoint_data = self.subtitle_manager.to_checkpoint_data()

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
            self.is_pause_snapshot_saved = self._force_save_pause_checkpoint(job_dir, total_chunks)
            # V3.1.0: 触发暂停时抛出异常，保持上层状态机一致
            raise self.pause_exception

        if last_chunk_index is not None:
            await self._flush_semantic_buffer(is_final_output=True, chunk_index=last_chunk_index)

        self.logger.info(f"极速模式完成: {len(results)} 个 Chunk 已处理")
        return results

    async def _run_full_pipeline(
        self,
        audio_chunks: List[AudioChunk],
        full_audio_array: Optional[Any] = None,
        full_audio_sr: int = 16000,
        job_dir: Optional[Path] = None,  # v3.1.0
        vad_intervals: Optional[List[Tuple[float, float]]] = None,
        processed_indices: Optional[Set[int]] = None,  # v3.1.0
        base_slow_count: int = 0,  # V3.1.0: SlowWorker 的基准偏移量
        base_align_count: int = 0,  # V3.1.0: 对齐阶段的基准偏移量
        initial_slow_processed_indices: Optional[set] = None,  # V3.1.0: SlowWorker 初始索引
        initial_finalized_indices: Optional[set] = None  # V3.1.0: 对齐阶段初始索引
    ) -> List[ProcessingContext]:
        """
        运行完整三级流水线（复核/双流模式）

        流程：
        1. 启动三个并行任务（FastWorker, SlowWorker, 对齐阶段）
        2. FastWorker 遍历 audio_chunks，每个 chunk 包装为 ProcessingContext
        3. 数据通过两个队列单向流动
        4. 等待所有任务完成
        5. 检查异常

        v3.1.0: 支持检查点保存和恢复
        V3.1.0: 集成进度发射器
        V3.1.0: 支持分别设置各 Worker 的基准偏移量和初始索引，修复恢复后进度跳变问题

        Args:
            audio_chunks: AudioChunk 列表
            full_audio_array: 完整音频数组（用于 Audio Overlap）
            full_audio_sr: 完整音频采样率
            job_dir: 任务目录（可选，v3.1.0 用于保存检查点）
            vad_intervals: VAD 语音区间（可选，用于 GapResolver 锚点约束）
            processed_indices: 已处理的chunk索引集合（可选，v3.1.0 用于 FastWorker 跳过）
            base_slow_count: SlowWorker 的基准偏移量（V3.1.0，已废弃，使用索引集合代替）
            base_align_count: 对齐阶段的基准偏移量（V3.1.0，已废弃，使用索引集合代替）
            initial_slow_processed_indices: SlowWorker 初始已处理索引集合（V3.1.0）
            initial_finalized_indices: 对齐阶段初始已完成索引集合（V3.1.0）

        Returns:
            List[ProcessingContext]: 处理结果列表
        """
        # V3.1.0: 清理历史状态，避免重复抛出旧异常
        self.errors.clear()
        self.pause_exception = None

        # V3.2.0+dev.20260201.07: 记录完整音频与 Chunk 映射，供 Bridge 批次使用
        self._full_audio_array = full_audio_array
        self._full_audio_sr = full_audio_sr
        self._audio_chunks_by_index = {chunk.index: chunk for chunk in audio_chunks}
        self._context_cache = {}
        self._vad_intervals = list(vad_intervals) if vad_intervals else None

        total_chunks = len(audio_chunks)  # V3.1.0: 保存总数用于进度计算
        self.logger.info(f"开始三级流水线: {total_chunks} 个 Chunk")

        # 存储结果
        results: List[ProcessingContext] = []

        # v3.1.0: 初始化已处理索引集合
        processed_indices = processed_indices or set()
        self._fast_processed_indices = processed_indices

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
            self.is_pause_snapshot_saved = self._force_save_pause_checkpoint(job_dir, total_chunks)
            # V3.1.0: 等待队列排空后再通知上层暂停，避免进度回退
            raise self.pause_exception

        self.logger.info(f"三级流水线完成: {len(results)} 个 Chunk 已处理")

        return results

    async def _emit_draft_sentences(self, ctx: ProcessingContext, is_final_output: bool) -> bool:
        """
        使用 DefaultSegmenter 生成句子并推送字幕事件。

        Args:
            ctx: 处理上下文
            is_final_output: 是否为定稿输出
        """
        if not ctx.sv_result or not ctx.audio_chunk:
            return False

        semantic_sentences = await self._emit_semantic_sentences(ctx, is_final_output)
        if semantic_sentences is not None:
            return True

        is_draft = not is_final_output
        sentences = self.segmenter.split_draft(
            ctx.sv_result,
            ctx.audio_chunk,
            is_draft=is_draft
        )

        if is_final_output:
            self.subtitle_manager.add_finalized_sentences(ctx.chunk_index, sentences)
            self.logger.debug(
                f"Chunk {ctx.chunk_index}: 定稿已推送 ({len(sentences)} 个句子)"
            )
        else:
            self.subtitle_manager.add_draft_sentences(ctx.chunk_index, sentences)
            self.logger.debug(
                f"Chunk {ctx.chunk_index}: 草稿已推送 ({len(sentences)} 个句子)"
            )
        return False

    async def _emit_semantic_sentences(
        self,
        ctx: ProcessingContext,
        is_final_output: bool,
    ) -> Optional[List[SentenceSegment]]:
        """使用 SemanticBuffer 生成句子并推送字幕。"""
        if not self.semantic_buffer or not ctx.sv_result or not ctx.audio_chunk:
            return None

        semantic_input = self._build_semantic_input(ctx)
        if semantic_input is None:
            return None

        chunks = self.semantic_buffer.add(semantic_input)
        if not chunks:
            return []

        await self._ingest_bridge_chunks(chunks)

        total_sentences = 0
        for chunk in chunks:
            sentences = chunk.sentences
            if is_final_output:
                for sentence in sentences:
                    sentence.is_finalized = True
                    sentence.is_draft = False
                self.subtitle_manager.add_finalized_sentences(ctx.chunk_index, sentences)
            else:
                self.subtitle_manager.add_draft_sentences(ctx.chunk_index, sentences)
            total_sentences += len(sentences)

        phase = "定稿" if is_final_output else "草稿"
        self.logger.debug(
            "Chunk %s: SemanticBuffer %s推送 (%d 个句子)",
            ctx.chunk_index,
            phase,
            total_sentences,
        )
        return [sentence for chunk in chunks for sentence in chunk.sentences]

    async def _ingest_bridge_chunks(self, chunks: List[SemanticChunk]) -> None:
        """将语义 Chunk 送入 Bridge 控制器，更新提示词缓存。"""
        if not self.bridge_controller:
            return
        for chunk in chunks:
            batch = await self.bridge_controller.add_semantic_chunk(chunk)
            if batch:
                self._bridge_last_batch = batch
                self._bridge_prompt_hint = batch.prompt or None
                self.logger.debug(
                    "Bridge 批次就绪: batch_id=%s, prompt_len=%d",
                    batch.batch_id,
                    len(batch.prompt or ""),
                )
                await self._enqueue_bridge_batch(batch)

    async def _flush_bridge_controller(self) -> None:
        """强制刷新 Bridge 控制器缓冲。"""
        if not self.bridge_controller:
            return
        batch = await self.bridge_controller.flush(reason="pipeline_flush")
        if batch:
            self._bridge_last_batch = batch
            self._bridge_prompt_hint = batch.prompt or None
            self.logger.debug(
                "Bridge 强制刷新: batch_id=%s, prompt_len=%d",
                batch.batch_id,
                len(batch.prompt or ""),
            )
            await self._enqueue_bridge_batch(batch)

    async def _enqueue_bridge_batch(self, batch: BridgeBatch) -> None:
        """将 Bridge 批次送入 SlowWorker 队列。"""
        if not self._enable_bridge_batches:
            return
        await self.queue_inter.put(batch)

    def _normalize_sensevoice_result(
        self,
        ctx: ProcessingContext,
    ) -> Optional[NormalizationResult]:
        """统一规范化 SenseVoice 输出（V3.2.0+dev.20260202.05）。"""
        if not ctx.sv_result or not ctx.audio_chunk:
            return None
        sv_result = ctx.sv_result
        raw_text = str(sv_result.get("text_clean") or sv_result.get("text") or "")
        language = ctx.audio_chunk.language or sv_result.get("language") or "auto"
        normalized = self._text_normalizer.normalize(raw_text, language)
        sv_result["text_itn_raw"] = normalized.text_itn_raw
        sv_result["text_clean"] = normalized.text_clean or normalized.text_itn_raw
        return normalized

    def _apply_whisper_full_sanitize(self, ctx: ProcessingContext) -> None:
        """Whisper 清洗增强与规范化重算（V3.2.0+dev.20260202.07）。"""
        if not ctx.whisper_result or not ctx.audio_chunk:
            return
        whisper_result = ctx.whisper_result
        raw_text = str(whisper_result.get("text") or "")
        if not raw_text:
            return
        prompt = whisper_result.get("prompt")
        sanitized = self._whisper_sanitizer.sanitize_full(raw_text, prompt=prompt)
        if sanitized == raw_text:
            return

        language = whisper_result.get("language") or ctx.audio_chunk.language or "auto"
        normalized = self._text_normalizer.normalize(sanitized, language)
        whisper_result["text_itn_raw"] = normalized.text_itn_raw
        whisper_result["text_clean"] = normalized.text_clean
        whisper_result["text"] = normalized.text_clean or sanitized

    def _is_arbitration_enabled(self) -> bool:
        runtime = get_model_runtime_config_service().get_effective_runtime_global()
        punct_config = runtime.get("effective", {}).get("punctuation", {})
        return bool(punct_config.get("enable_arbitration", True))

    def _run_arbitration(
        self,
        ctx: ProcessingContext,
        sv_result: Dict[str, Any],
        whisper_result: Dict[str, Any],
    ) -> ArbitrationResult:
        if not self._is_arbitration_enabled():
            return ArbitrationResult(
                text_source="whisper",
                punct_source="sv",
                reason="disabled",
                coverage=1.0,
                sv_score=0.0,
                wh_score=0.0,
                gap_positions=[],
            )
        prompt = whisper_result.get("prompt")
        return self._arbiter.arbitrate(sv_result, whisper_result, prompt=prompt)

    @staticmethod
    def _select_text_for_alignment(
        sv_result: Dict[str, Any],
        whisper_result: Dict[str, Any],
        arbitration_result: ArbitrationResult,
    ) -> str:
        if arbitration_result.text_source == "sv":
            return str(
                sv_result.get("text_clean")
                or sv_result.get("text_itn_raw")
                or sv_result.get("text")
                or ""
            )
        return str(whisper_result.get("text_clean") or whisper_result.get("text") or "")

    def _apply_arbitration_text(
        self,
        whisper_result: Dict[str, Any],
        sv_result: Dict[str, Any],
        arbitration_result: ArbitrationResult,
        chosen_text_clean: str,
    ) -> None:
        if arbitration_result.text_source == "sv":
            whisper_result["text"] = chosen_text_clean
            whisper_result["text_clean"] = chosen_text_clean
            whisper_result["text_itn_raw"] = sv_result.get("text_itn_raw") or chosen_text_clean
        else:
            whisper_result["text"] = chosen_text_clean
            whisper_result["text_clean"] = chosen_text_clean

    def _resolve_punctuation_positions(
        self,
        sv_result: Dict[str, Any],
        whisper_result: Dict[str, Any],
        arbitration_result: ArbitrationResult,
        chosen_text_clean: str,
    ) -> Tuple[Optional[List[PuncPosition]], Optional[str]]:
        punct_source = arbitration_result.punct_source
        clean_text = chosen_text_clean
        if not clean_text:
            return None, clean_text

        sv_positions = self._extract_sv_punctuation_positions(sv_result, clean_text)
        if punct_source == "sv":
            return (sv_positions or None), clean_text

        if punct_source == "whisper":
            raw_text = str(whisper_result.get("text_itn_raw") or whisper_result.get("text") or "")
            clean_text, wh_positions = SemanticInjector.extract_raw_punctuation_positions(raw_text)
            if not clean_text:
                clean_text = chosen_text_clean
            return (wh_positions or None), clean_text

        raw_text = str(whisper_result.get("text_itn_raw") or whisper_result.get("text") or "")
        clean_text, wh_positions = SemanticInjector.extract_raw_punctuation_positions(raw_text)
        if not clean_text:
            clean_text = chosen_text_clean
        if sv_positions and clean_text == (sv_result.get("text_clean") or clean_text):
            merged = self._merge_punctuation_positions(sv_positions, wh_positions)
            return (merged or None), clean_text
        return (wh_positions or None), clean_text

    def _extract_sv_punctuation_positions(
        self,
        sv_result: Dict[str, Any],
        clean_text: str,
    ) -> List[PuncPosition]:
        if not sv_result:
            return []
        metadata = sv_result.get("metadata", {}) if isinstance(sv_result, dict) else {}
        punctuation_meta = metadata.get("punctuation")
        if not isinstance(punctuation_meta, dict):
            return []
        if clean_text and sv_result.get("text_clean") and clean_text != sv_result.get("text_clean"):
            return []
        result = self._build_punctuation_result(punctuation_meta, clean_text)
        return result.punctuation_positions if result else []

    @staticmethod
    def _merge_punctuation_positions(
        base: List[PuncPosition],
        extra: List[PuncPosition],
    ) -> List[PuncPosition]:
        merged: Dict[Tuple[int, str], float] = {}
        for pos in base + extra:
            key = (pos.char_index, pos.punctuation)
            merged[key] = max(merged.get(key, 0.0), pos.confidence)
        return [
            PuncPosition(char_index=idx, punctuation=punct, confidence=conf)
            for (idx, punct), conf in sorted(merged.items(), key=lambda item: item[0][0])
        ]

    async def _apply_fast_punctuation(
        self,
        ctx: ProcessingContext,
        normalized: Optional[NormalizationResult],
    ) -> None:
        """快流标点恢复与后处理（V3.2.0+dev.20260202.03）。"""
        if not normalized or not ctx.sv_result or not ctx.audio_chunk:
            return
        await self._fast_punctuator.apply(
            ctx.sv_result,
            chunk=ctx.audio_chunk,
            ctx=ctx,
            normalization=normalized,
        )

    def _build_semantic_input(self, ctx: ProcessingContext) -> Optional[SemanticBufferInput]:
        """构建 SemanticBuffer 输入。"""
        sv_result = ctx.sv_result or {}
        chunk = ctx.audio_chunk
        metadata = sv_result.get("metadata", {}) if isinstance(sv_result, dict) else {}
        punctuation_meta = metadata.get("punctuation")
        decision_meta = metadata.get("punctuation_decision")

        raw_text = sv_result.get("text_itn_raw") or sv_result.get("text") or ""
        text = sv_result.get("text_clean") or raw_text
        words = sv_result.get("words") if isinstance(sv_result, dict) else None
        raw_tokens = sv_result.get("raw_tokens") if isinstance(sv_result, dict) else None
        punctuation_result = None
        # V3.2.0+dev.20260201.10: 即使缺少标点元信息，也走语义缓冲与 Bridge 批次，旧逐Chunk仅作备份
        if isinstance(punctuation_meta, dict):
            punctuation_result = self._build_punctuation_result(punctuation_meta, raw_text)
            if punctuation_result and punctuation_result.text:
                text = punctuation_result.text

        if not text:
            return None

        decision = self._build_punctuation_decision(decision_meta)
        source_chunks = [f"chunk-{chunk.index}"]
        return SemanticBufferInput(
            chunk_id=f"chunk-{ctx.chunk_index}",
            text=text,
            audio_range=(chunk.start, chunk.end),
            language=chunk.language or sv_result.get("language") or "auto",
            punctuation_result=punctuation_result,
            punctuation_decision=decision,
            word_timestamps=words if isinstance(words, list) else None,
            raw_tokens=raw_tokens if isinstance(raw_tokens, list) else None,
            source_chunks=source_chunks,
            speaker_id=None,
        )

    @staticmethod
    def _build_punctuation_decision(meta: Optional[Dict[str, Any]]) -> Optional[PunctuationDecision]:
        if not meta or not isinstance(meta, dict):
            return None
        return PunctuationDecision.from_dict(meta)

    @staticmethod
    def _build_punctuation_result(
        meta: Optional[Dict[str, Any]],
        fallback_text: str,
    ) -> Optional[PunctuationResult]:
        if not meta or not isinstance(meta, dict):
            return None

        text = meta.get("text") or fallback_text
        raw_split_points = meta.get("split_points", []) or []
        raw_positions = meta.get("punctuation_positions", []) or []
        split_points: List[SplitPoint] = []
        for item in raw_split_points:
            if not isinstance(item, dict):
                continue
            split_points.append(
                SplitPoint(
                    char_index=int(item.get("char_index", 0)),
                    relative_time=float(item.get("relative_time", 0.0)),
                    punctuation=str(item.get("punctuation", "")),
                    confidence=float(item.get("confidence", 1.0)),
                )
            )
        positions: List[PuncPosition] = []
        for item in raw_positions:
            if not isinstance(item, dict):
                continue
            positions.append(
                PuncPosition(
                    char_index=int(item.get("char_index", 0)),
                    punctuation=str(item.get("punctuation", "")),
                    confidence=float(item.get("confidence", 1.0)),
                )
            )

        return PunctuationResult(
            text=str(text),
            model_id=str(meta.get("model_id", "unknown")),
            split_points=split_points,
            punctuation_positions=positions,
            confidence=float(meta.get("confidence", 1.0)),
            processing_time_ms=float(meta.get("processing_time_ms", 0.0)),
        )

    async def _flush_semantic_buffer(self, *, is_final_output: bool, chunk_index: int) -> None:
        """刷新 SemanticBuffer 尾部内容并推送字幕。"""
        if not self.semantic_buffer:
            return
        chunks = self.semantic_buffer.flush(reason="pipeline_end")
        if not chunks:
            return
        await self._ingest_bridge_chunks(chunks)
        total_sentences = 0
        for chunk in chunks:
            sentences = chunk.sentences
            if is_final_output:
                for sentence in sentences:
                    sentence.is_finalized = True
                    sentence.is_draft = False
                self.subtitle_manager.add_finalized_sentences(chunk_index, sentences)
            else:
                self.subtitle_manager.add_draft_sentences(chunk_index, sentences)
            total_sentences += len(sentences)
        phase = "定稿" if is_final_output else "草稿"
        self.logger.info(
            "SemanticBuffer 尾部刷新完成: %s %d 个句子",
            phase,
            total_sentences,
        )
        await self._flush_bridge_controller()

    def _should_skip_whisper(self, sv_result: Dict[str, Any], chunk: AudioChunk) -> bool:
        """
        智能复核模式下判断是否跳过 Whisper。
        """
        confidence = sv_result.get("confidence", 0.0)
        text_clean = sv_result.get("text_clean", "")
        words = sv_result.get("words", [])
        duration = chunk.duration
        return not needs_whisper_patch(
            confidence=confidence,
            duration=duration,
            text_length=len(text_clean),
            words=words,
            config=self.patching_threshold or ThresholdConfig()
        )

    def _build_whisper_prompt(self, sv_context: Optional[str]) -> str:
        """
        构建 Whisper Prompt（关键词 + 语义线索）。
        """
        prompt_builder = get_prompt_builder()
        base_prompt = prompt_builder.build_prompt(
            previous_text=self.previous_whisper_text,
            user_glossary=self.user_glossary
        )

        context_segments: List[str] = []
        if self._bridge_prompt_hint:
            context_segments.append(self._bridge_prompt_hint.strip())
        if sv_context:
            semantic_hint = sv_context[-50:] if len(sv_context) > 50 else sv_context
            semantic_hint = semantic_hint.lstrip()
            if semantic_hint:
                context_segments.append(semantic_hint)

        if context_segments:
            context_text = " ".join(segment for segment in context_segments if segment)
            if base_prompt:
                return f"Context: {context_text}. {base_prompt}"
            return f"Context: {context_text}."
        return base_prompt

    @staticmethod
    def _parse_source_chunk_indices(source_chunks: List[str]) -> List[int]:
        """解析 source_chunks 中的 chunk 索引。"""
        indices: List[int] = []
        for chunk_id in source_chunks or []:
            if not isinstance(chunk_id, str):
                continue
            parts = chunk_id.split("+")
            for part in parts:
                part = part.strip()
                if not part.startswith("chunk-"):
                    continue
                try:
                    indices.append(int(part.split("-")[-1]))
                except ValueError:
                    continue
        return sorted(set(indices))

    @staticmethod
    def _distance_to_range(value: float, span: tuple[float, float]) -> float:
        start, end = span
        if start <= value <= end:
            return 0.0
        if value < start:
            return start - value
        return value - end

    @staticmethod
    def _estimate_segment_confidence(
        segments: List[Dict[str, Any]],
        fallback: float,
    ) -> float:
        if not segments:
            return float(fallback or 0.0)
        avg_logprob = sum(float(seg.get("avg_logprob", -1.0)) for seg in segments) / len(segments)
        avg_no_speech = sum(float(seg.get("no_speech_prob", 0.0)) for seg in segments) / len(segments)
        confidence = min(1.0, max(0.0, 1.0 + avg_logprob))
        confidence *= (1.0 - avg_no_speech)
        return float(confidence)

    def _split_whisper_result_by_chunks(
        self,
        whisper_result: Dict[str, Any],
        chunk_indices: List[int],
        batch_start: float,
        language_override: Optional[str] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """按 Chunk 时间范围拆分 Whisper 结果。"""
        if not whisper_result or not chunk_indices:
            return {}

        chunk_ranges: Dict[int, tuple[float, float]] = {}
        for idx in chunk_indices:
            chunk = self._audio_chunks_by_index.get(idx)
            if chunk:
                chunk_ranges[idx] = (float(chunk.start), float(chunk.end))

        raw = whisper_result.get("raw_result", {}) if isinstance(whisper_result, dict) else {}
        raw_segments = raw.get("segments", []) if isinstance(raw, dict) else []

        if not raw_segments:
            first = chunk_indices[0]
            return {
                first: {
                    "text": str(whisper_result.get("text", "")),
                    "confidence": float(whisper_result.get("confidence", 0.0) or 0.0),
                    "language": str(whisper_result.get("language", "auto")),
                    "raw_result": {"segments": []},
                }
            }

        assignments: Dict[int, List[Dict[str, Any]]] = {idx: [] for idx in chunk_ranges}
        ordered_indices = sorted(chunk_ranges.keys())

        for seg in raw_segments:
            seg_start = float(seg.get("start", 0.0) or 0.0) + batch_start
            seg_end = float(seg.get("end", 0.0) or 0.0) + batch_start
            seg_mid = (seg_start + seg_end) / 2.0

            target_idx = None
            for idx in ordered_indices:
                start, end = chunk_ranges[idx]
                if start <= seg_mid <= end:
                    target_idx = idx
                    break

            if target_idx is None and ordered_indices:
                target_idx = min(
                    ordered_indices,
                    key=lambda idx: self._distance_to_range(seg_mid, chunk_ranges[idx]),
                )

            if target_idx is not None:
                assignments[target_idx].append(seg)

        results: Dict[int, Dict[str, Any]] = {}
        language = str(language_override or whisper_result.get("language", "auto"))
        fallback_conf = float(whisper_result.get("confidence", 0.0) or 0.0)

        for idx in ordered_indices:
            segs = assignments.get(idx, [])
            text = "".join(str(seg.get("text", "")) for seg in segs).strip()
            confidence = self._estimate_segment_confidence(segs, fallback_conf)
            results[idx] = {
                "text": text,
                "confidence": confidence,
                "language": language,
                "raw_result": {"segments": segs},
            }
        return results

    async def _process_bridge_batch(
        self,
        batch: BridgeBatch,
        *,
        job_dir: Optional[Path],
        total_chunks: int,
        slow_processed_indices: Set[int],
        token: Optional["CancellationToken"],
    ) -> bool:
        """处理 Bridge 批次并推送到对齐阶段。"""
        if not self.slow_worker:
            return False

        chunk_indices = self._parse_source_chunk_indices(batch.source_chunks)
        if not chunk_indices:
            self.logger.warning("Bridge 批次缺少 source_chunks: batch_id=%s", batch.batch_id)
            return False

        contexts: List[tuple[int, ProcessingContext]] = []
        for idx in chunk_indices:
            ctx = self._context_cache.get(idx)
            if ctx:
                contexts.append((idx, ctx))

        if not contexts:
            self.logger.warning("Bridge 批次缺少上下文缓存: batch_id=%s", batch.batch_id)
            return False

        skip_map: Dict[int, bool] = {}
        if self.is_patching_mode:
            for idx, ctx in contexts:
                if ctx.sv_result and ctx.audio_chunk:
                    skip_map[idx] = self._should_skip_whisper(ctx.sv_result, ctx.audio_chunk)
                else:
                    skip_map[idx] = False

        if self.is_patching_mode and all(skip_map.values()):
            for _, ctx in contexts:
                ctx.whisper_skipped = True
                ctx.whisper_result = {}
            return await self._push_batch_contexts(
                contexts,
                slow_processed_indices,
                total_chunks=total_chunks,
                job_dir=job_dir,
                token=token,
            )

        if self._full_audio_array is None:
            self.logger.warning("Bridge 批次缺少完整音频，跳过慢流: batch_id=%s", batch.batch_id)
            for _, ctx in contexts:
                ctx.whisper_skipped = True
                ctx.whisper_result = {}
            return await self._push_batch_contexts(
                contexts,
                slow_processed_indices,
                total_chunks=total_chunks,
                job_dir=job_dir,
                token=token,
            )

        slow_result: SlowWorkerResult = await self.slow_worker.process_batch(
            batch,
            full_audio_array=self._full_audio_array,
            full_audio_sr=self._full_audio_sr,
        )
        if self.bridge_controller:
            self.bridge_controller.record_slow_result(batch.batch_id, slow_result)

        whisper_result = slow_result.whisper_result or {}
        prompt = batch.prompt or None
        whisper_text_raw = str(whisper_result.get("text", "") or "")
        whisper_result["text_raw"] = whisper_text_raw
        whisper_result["prompt"] = prompt
        whisper_result["text"] = self._whisper_sanitizer.sanitize_minimal(whisper_text_raw, prompt=prompt)
        if whisper_result and self._hallucination_detector.is_hallucination(whisper_result, prompt):
            self.logger.warning("Bridge 批次检测到 Whisper 幻觉，回退快流: batch_id=%s", batch.batch_id)
            for _, ctx in contexts:
                ctx.whisper_skipped = True
                ctx.whisper_result = {}
            return await self._push_batch_contexts(
                contexts,
                slow_processed_indices,
                total_chunks=total_chunks,
                job_dir=job_dir,
                token=token,
            )

        batch_language = batch.language or whisper_result.get("language") or "auto"
        normalized_whisper = self._text_normalizer.normalize(whisper_result.get("text", ""), batch_language)
        whisper_result["text_itn_raw"] = normalized_whisper.text_itn_raw
        whisper_result["text_clean"] = normalized_whisper.text_clean
        whisper_result["text"] = normalized_whisper.text_clean or whisper_result.get("text", "")
        whisper_result["language"] = batch_language

        if whisper_result.get("text"):
            self._update_prompt_cache(str(whisper_result.get("text", "")))

        batch_start = min((seg[0] for seg in batch.audio_segments), default=0.0)
        chunk_results = self._split_whisper_result_by_chunks(
            whisper_result,
            [idx for idx, _ in contexts],
            batch_start=batch_start,
            language_override=batch_language,
        )

        for idx, ctx in contexts:
            if skip_map.get(idx):
                ctx.whisper_skipped = True
                ctx.whisper_result = {}
                continue
            ctx.whisper_skipped = False
            chunk_result = chunk_results.get(
                idx,
                {
                    "text": "",
                    "confidence": float(whisper_result.get("confidence", 0.0) or 0.0),
                    "language": batch_language,
                    "raw_result": {"segments": []},
                },
            )
            chunk_text_raw = str(chunk_result.get("text", "") or "")
            chunk_result["text_raw"] = chunk_text_raw
            chunk_result["text"] = self._whisper_sanitizer.sanitize_minimal(chunk_text_raw, prompt=None)
            normalized_chunk = self._text_normalizer.normalize(chunk_result.get("text", ""), batch_language)
            chunk_result["text_itn_raw"] = normalized_chunk.text_itn_raw
            chunk_result["text_clean"] = normalized_chunk.text_clean
            chunk_result["text"] = normalized_chunk.text_clean or chunk_result.get("text", "")
            chunk_result["language"] = batch_language
            ctx.whisper_result = chunk_result

        return await self._push_batch_contexts(
            contexts,
            slow_processed_indices,
            total_chunks=total_chunks,
            job_dir=job_dir,
            token=token,
        )

    async def _push_batch_contexts(
        self,
        contexts: List[tuple[int, ProcessingContext]],
        slow_processed_indices: Set[int],
        *,
        total_chunks: int,
        job_dir: Optional[Path],
        token: Optional["CancellationToken"],
    ) -> bool:
        """批量推送上下文并更新进度/检查点。"""
        pause_requested = False
        for idx, ctx in sorted(contexts, key=lambda item: item[0]):
            await self.queue_final.put(ctx)
            self._context_cache.pop(idx, None)
            slow_processed_indices.add(idx)
            self._last_slow_chunk_index = idx

            if self.progress_emitter and total_chunks > 0:
                total_processed = len(slow_processed_indices)
                self.progress_emitter.update_slow(
                    total_processed,
                    total_chunks,
                    message=f"Whisper: {total_processed}/{total_chunks}",
                )

            if token and job_dir:
                previous_whisper_text = self.previous_whisper_text or ""
                self._slow_processed_indices = slow_processed_indices
                checkpoint_data = {
                    "transcription": {
                        "slow_processed_count": len(slow_processed_indices),
                        "slow_processed_indices": list(slow_processed_indices),
                        "previous_whisper_text": previous_whisper_text,
                        "last_slow_chunk_index": idx,
                    }
                }
                try:
                    token.check_and_save(checkpoint_data, job_dir)
                except PausedException as e:
                    if not pause_requested:
                        self.logger.info("[V3.1.0] SlowWorker 捕获暂停信号，继续排空队列")
                    pause_requested = True
                    if not self.pause_exception:
                        self.pause_exception = e

        return pause_requested

    def _extract_audio_with_overlap(self, ctx: ProcessingContext) -> Any:
        """
        提取 Whisper 使用的音频（包含前向重叠）。
        """
        chunk = ctx.audio_chunk
        full_audio = ctx.full_audio_array
        sr = ctx.full_audio_sr
        overlap_sec = 0.5

        if full_audio is None:
            return chunk.audio

        overlap_start = max(0.0, chunk.start - overlap_sec)
        start_sample = max(0, int(overlap_start * sr))
        end_sample = min(len(full_audio), int(chunk.end * sr))
        if overlap_start < chunk.start:
            self.logger.debug(
                f"Whisper 添加 {chunk.start - overlap_start:.2f}s 前向重叠: "
                f"[{overlap_start:.2f}s, {chunk.end:.2f}s]"
            )
        return full_audio[start_sample:end_sample]

    def _update_prompt_cache(self, whisper_text: str) -> None:
        """更新 Whisper 上下文缓存。"""
        self.previous_whisper_text = whisper_text

    def restore_prompt_cache(self, previous_text: Optional[str]) -> None:
        """恢复 Whisper 上下文缓存（断点续传使用）。"""
        self.previous_whisper_text = previous_text
        if previous_text:
            self.logger.debug(f"[v3.1.0] 已恢复 Whisper 上下文: {len(previous_text)} 字符")
        else:
            self.logger.debug("[v3.1.0] Whisper 上下文为空")

    async def _fast_loop(
        self,
        chunks: List[AudioChunk],
        full_audio_array: Optional[Any] = None,
        full_audio_sr: int = 16000,
        job_dir: Optional[Path] = None,  # v3.1.0
        processed_indices: Optional[Set[int]] = None,  # v3.1.0
        total_chunks: int = 0  # V3.1.0
    ):
        """
        FastWorker 循环（生产者）

        职责：
        1. 遍历所有 audio_chunks
        2. 每个 chunk 包装为 ProcessingContext
        3. 调用 FastWorker.process()（仅推理）
        4. 将 context 放入 queue_inter
        5. 发送结束信号

        v3.1.0: 支持原子区域和检查点保存
        V3.1.0: 集成进度发射器

        Args:
            chunks: AudioChunk 列表
            full_audio_array: 完整音频数组（用于 Audio Overlap）
            full_audio_sr: 完整音频采样率
            job_dir: 任务目录（可选，v3.1.0）
            processed_indices: 已处理的chunk索引集合（可选，v3.1.0）
            total_chunks: 总 Chunk 数（V3.1.0）
        """
        token = self.cancellation_token  # v3.1.0
        processed_indices = processed_indices or set()
        total_chunks = total_chunks or len(chunks)

        # V3.1.0: 计算基准偏移量（已完成的 chunk 数量）
        # 修复进度归零问题：恢复后 fast_processed_count 应该从已完成数量开始累加
        base_fast_count = len(processed_indices)
        fast_processed_count = 0  # V3.1.0: 追踪本次新处理的数量
        pause_requested = False  # V3.1.0: 捕获暂停后进入排空模式
        should_send_end_signal = True  # V3.1.0: 控制是否需要发送正常结束信号
        last_chunk_index: Optional[int] = None

        try:
            for i, chunk in enumerate(chunks):
                # v3.1.0: 跳过已处理的 chunk（用于恢复）
                if i in processed_indices:
                    self.logger.debug(f"[FastWorker] 跳过已处理的 chunk {i}")
                    continue

                # 创建处理上下文（包含完整音频数组）
                ctx = ProcessingContext(
                    job_id=self.job_id,
                    chunk_index=i,
                    audio_chunk=chunk,
                    job_dir=job_dir,
                    debug_punctuation=self.debug_punctuation,
                    full_audio_array=full_audio_array,
                    full_audio_sr=full_audio_sr
                )

                # v3.1.0: 进入原子区域（单个 Chunk 处理 + SSE 推送）
                if token:
                    token.enter_atomic_region(f"fast_worker_chunk_{i}")

                try:
                    # FastWorker 处理（仅推理）
                    await self.fast_worker.process(ctx)
                    normalized = self._normalize_sensevoice_result(ctx)
                    await self._apply_fast_punctuation(ctx, normalized)
                    # V3.2.0+dev.20260201.07: 缓存上下文，供 Bridge 批次使用
                    self._context_cache[ctx.chunk_index] = ctx
                    used_semantic = await self._emit_draft_sentences(ctx, is_final_output=False)

                    # 放入队列（如果队列满了，会自动阻塞，实现背压）
                    if not self._enable_bridge_batches or not used_semantic:
                        await self.queue_inter.put(ctx)
                    fast_processed_count += 1  # V3.1.0
                    last_chunk_index = i

                    # V3.1.0: 更新 FastWorker 进度（叠加基准偏移量）
                    if self.progress_emitter:
                        total_processed = base_fast_count + fast_processed_count
                        self.progress_emitter.update_fast(
                            total_processed, total_chunks,
                            message=f"SenseVoice: {total_processed}/{total_chunks}"
                        )
                finally:
                    # v3.1.0: 退出原子区域
                    if token:
                        has_pending = token.exit_atomic_region()
                        if has_pending:
                            self.logger.info(f"[v3.1.0] FastWorker chunk {i} 完成后检测到待处理请求")

                # v3.1.0: 每个 Chunk 处理完成后保存检查点
                if token and job_dir:
                    processed_indices.add(i)
                    # V3.2.0+dev.20260123.05: 同步到实例变量，用于暂停快照
                    self._fast_processed_indices = processed_indices
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
                job_dir=job_dir,
                debug_punctuation=self.debug_punctuation,
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
                job_dir=job_dir,
                debug_punctuation=self.debug_punctuation,
                is_end=True,
                error=e
            )
            await self.queue_inter.put(error_ctx)
            should_send_end_signal = False
            return
        finally:
            if should_send_end_signal:
                if last_chunk_index is not None:
                    await self._flush_semantic_buffer(is_final_output=False, chunk_index=last_chunk_index)
                end_ctx = ProcessingContext(
                    job_id=self.job_id,
                    chunk_index=-1,
                    audio_chunk=None,
                    job_dir=job_dir,
                    debug_punctuation=self.debug_punctuation,
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
        2. 调用 SlowWorker.infer()（仅推理）
        3. 将 context 放入 queue_final
        4. 透传结束/错误信号

        v3.1.0: 支持原子区域和检查点保存（包括关键的 previous_whisper_text）
        V3.1.0: 集成进度发射器
        V3.1.0: 保存 slow_processed_indices 用于断点续传
        V3.1.0: 使用累计索引集合，修复恢复后进度不准确问题
        """
        token = self.cancellation_token  # v3.1.0
        slow_processed_count = 0  # v3.1.0: 追踪本次新处理的数量
        # V3.1.0: 使用累计索引集合（类似 FastWorker）
        slow_processed_indices = set(initial_slow_processed_indices) if initial_slow_processed_indices else set()
        self._slow_processed_indices = slow_processed_indices
        if slow_processed_indices:
            # 恢复态时同步最后进度索引，避免暂停快照缺失
            self._last_slow_chunk_index = max(slow_processed_indices)
        pause_requested = False  # V3.1.0: 捕获暂停后继续排空队列

        try:
            while True:
                # 从队列取输入（ProcessingContext / BridgeBatch）
                payload = await self.queue_inter.get()

                if isinstance(payload, BridgeBatch):
                    if await self._process_bridge_batch(
                        payload,
                        job_dir=job_dir,
                        total_chunks=total_chunks,
                        slow_processed_indices=slow_processed_indices,
                        token=token,
                    ):
                        pause_requested = True
                    continue

                ctx = payload

                # 检查结束信号或错误
                if ctx.is_end or ctx.error:
                    await self.queue_final.put(ctx)  # 透传
                    break

                chunk_index = ctx.chunk_index  # v3.1.0: 获取 chunk 索引

                # v3.1.0: 进入原子区域（单个 Chunk 处理 + 上下文更新）
                if token:
                    token.enter_atomic_region(f"slow_worker_chunk_{chunk_index}")

                try:
                    sv_result = ctx.sv_result or {}
                    chunk = ctx.audio_chunk

                    skip_whisper = False
                    whisper_result: Optional[Dict[str, Any]] = None
                    prompt: Optional[str] = None

                    # 智能复核：质量足够则跳过 Whisper
                    if self.is_patching_mode and sv_result:
                        if self._should_skip_whisper(sv_result, chunk):
                            skip_whisper = True
                            ctx.whisper_skipped = True
                            ctx.whisper_result = {}
                            self.logger.info(
                                f"Chunk {chunk_index}: SenseVoice 质量足够，跳过 Whisper "
                                f"(confidence={sv_result.get('confidence', 0):.2f})"
                            )

                    if not skip_whisper:
                        sv_context = sv_result.get("text_clean", "") if sv_result else None
                        prompt = self._build_whisper_prompt(sv_context)
                        audio_with_overlap = self._extract_audio_with_overlap(ctx)

                        whisper_result = await self.slow_worker.infer(
                            audio_with_overlap,
                            initial_prompt=prompt,
                        )

                        whisper_text_raw = str(whisper_result.get("text", "") or "")
                        whisper_result["text_raw"] = whisper_text_raw
                        whisper_result["prompt"] = prompt
                        whisper_result["text"] = self._whisper_sanitizer.sanitize_minimal(
                            whisper_text_raw,
                            prompt=prompt,
                        )
                        if self._hallucination_detector.is_hallucination(whisper_result, prompt):
                            self.logger.warning(
                                f"Chunk {chunk_index}: 检测到 Whisper 幻觉，回退到 SenseVoice"
                            )
                            fallback_text = sv_result.get("text_clean", "")
                            whisper_result["text"] = fallback_text
                            whisper_result["text_itn_raw"] = sv_result.get("text_itn_raw") or fallback_text
                            whisper_result["text_clean"] = fallback_text
                            whisper_result["language"] = sv_result.get("language", "auto")
                            whisper_result["is_hallucination"] = True
                        else:
                            whisper_language = chunk.language or whisper_result.get("language") or "auto"
                            normalized = self._text_normalizer.normalize(
                                whisper_result.get("text", ""),
                                whisper_language,
                            )
                            whisper_result["text_itn_raw"] = normalized.text_itn_raw
                            whisper_result["text_clean"] = normalized.text_clean
                            whisper_result["text"] = normalized.text_clean or whisper_result.get("text", "")
                            whisper_result["language"] = whisper_language

                        ctx.whisper_result = copy.deepcopy(whisper_result)
                        ctx.whisper_skipped = False
                        self._update_prompt_cache(whisper_result.get("text", ""))

                        self.logger.info(
                            f"Chunk {chunk_index}: Whisper 推理完成 "
                            f"(text_length={len(whisper_result.get('text', ''))})"
                        )

                    # 放入队列
                    await self.queue_final.put(ctx)
                    self._context_cache.pop(chunk_index, None)
                    slow_processed_count += 1
                    slow_processed_indices.add(chunk_index)  # V3.1.0: 累加到集合（类似 FastWorker）
                    self._last_slow_chunk_index = chunk_index

                    # V3.1.0: 更新 SlowWorker 进度（使用累计索引数量）
                    if self.progress_emitter and total_chunks > 0:
                        total_processed = len(slow_processed_indices)
                        self.progress_emitter.update_slow(
                            total_processed, total_chunks,
                            message=f"Whisper: {total_processed}/{total_chunks}"
                        )
                finally:
                    # v3.1.0: 退出原子区域
                    if token:
                        has_pending = token.exit_atomic_region()
                        if has_pending:
                            self.logger.info(f"[v3.1.0] SlowWorker chunk {chunk_index} 完成后检测到待处理请求")

                # v3.1.0: 每个 Chunk 处理完成后保存检查点（包含关键的 previous_whisper_text）
                if token and job_dir:
                    # 获取当前的 Whisper 上文
                    previous_whisper_text = self.previous_whisper_text or ""
                    # V3.2.0+dev.20260123.05: 同步到实例变量，用于暂停快照
                    self._slow_processed_indices = slow_processed_indices
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
                job_dir=job_dir,
                debug_punctuation=self.debug_punctuation,
                is_end=True,
                error=e
            )
            await self.queue_final.put(error_ctx)
        finally:
            if pause_requested:
                self.logger.info("[V3.1.0] SlowWorker 已完成排空，等待对齐阶段同步完成")

    # V3.2.0+dev.20260120.03: 对齐阶段下放到流水线
    async def _run_alignment_stage(self, ctx: ProcessingContext) -> None:
        """
        对齐阶段（流水线内执行）

        负责：
        1. 双流对齐（含降级兜底）
        2. 推送定稿
        3. 填充 ctx.final_sentences
        """
        if not self.aligner:
            raise RuntimeError("对齐阶段未初始化 aligner")

        chunk = ctx.audio_chunk

        # V3.10: 快速路径 - SlowWorker 跳过时直接使用 SenseVoice
        if ctx.whisper_skipped:
            if ctx.sv_result is None:
                raise ValueError("对齐阶段缺少 SenseVoice 推理结果")
            self.logger.info(f"Chunk {ctx.chunk_index}: Whisper 跳过，直接使用 SenseVoice 定稿")
            final_sentences = self.aligner.split_sensevoice_only(ctx.sv_result, chunk)

            ctx.final_sentences = final_sentences

            # 推送定稿
            sentences_for_manager = copy.deepcopy(final_sentences)
            self.subtitle_manager.replace_chunk(ctx.chunk_index, sentences_for_manager)

            self.logger.debug(
                f"Chunk {ctx.chunk_index}: SenseVoice 定稿已推送 "
                f"({len(final_sentences)} 个句子) [智能复核-跳过]"
            )
            return

        # 阶段 1: 双流对齐（三级降级策略）
        self.logger.debug(f"Chunk {ctx.chunk_index}: 双流对齐")

        if ctx.whisper_result is None or ctx.sv_result is None:
            raise ValueError("对齐阶段缺少必要的推理结果")

        whisper_result = ctx.whisper_result
        sv_result = ctx.sv_result

        # V3.2.0+dev.20260202.07: Whisper 清洗增强 + 规范化重算
        self._apply_whisper_full_sanitize(ctx)

        arbitration_result = self._run_arbitration(ctx, sv_result, whisper_result)
        ctx.arbitration_result = arbitration_result
        chosen_text_clean = self._select_text_for_alignment(
            sv_result,
            whisper_result,
            arbitration_result,
        )
        self._apply_arbitration_text(
            whisper_result,
            sv_result,
            arbitration_result,
            chosen_text_clean,
        )
        punctuation_positions, punctuation_clean_text = self._resolve_punctuation_positions(
            sv_result,
            whisper_result,
            arbitration_result,
            chosen_text_clean,
        )
        if self.bridge_controller:
            self.bridge_controller.record_arbitration_result(arbitration_result)
        self.logger.info(
            "Chunk %s: 仲裁完成 text_source=%s punct_source=%s reason=%s coverage=%.2f",
            ctx.chunk_index,
            arbitration_result.text_source,
            arbitration_result.punct_source,
            arbitration_result.reason,
            arbitration_result.coverage,
        )

        final_sentences, alignment_level = await self.aligner.align(
            whisper_result,
            sv_result,
            chunk,
            vad_intervals=self._vad_intervals,
            punctuation_positions=punctuation_positions,
            punctuation_clean_text=punctuation_clean_text,
        )

        ctx.final_sentences = final_sentences
        if ctx.arbitration_result and self.aligner.last_alignment_stats:
            ctx.arbitration_result.gap_positions = list(
                self.aligner.last_alignment_stats.get("gap_positions", [])
            )
            self.logger.debug(
                "Chunk %s: 仲裁统计 coverage=%.2f gap_positions=%s",
                ctx.chunk_index,
                ctx.arbitration_result.coverage,
                ctx.arbitration_result.gap_positions,
            )

        # 阶段 2: 推送定稿（使用 Chunk 级别的批量替换）
        # V3.8 调试日志：记录对齐结果状态
        self.logger.debug(
            f"Chunk {ctx.chunk_index}: 对齐完成 - "
            f"final_sentences={len(final_sentences)}, "
            f"alignment_level={alignment_level.value}, "
            f"whisper_text_len={len(whisper_result.get('text', ''))}, "
            f"sv_text_clean_len={len(sv_result.get('text_clean', ''))}"
        )

        # V3.8 修复：如果定稿句子为空，尝试从草稿中恢复
        if not final_sentences:
            self.logger.error(
                f"Chunk {ctx.chunk_index}: 定稿句子为空！"
                f"Whisper文本长度={len(whisper_result.get('text', ''))}, "
                f"SenseVoice文本长度={len(sv_result.get('text_clean', ''))}, "
                f"对齐级别={alignment_level.value}"
            )

            # V3.8 修复：尝试从 subtitle_manager 中获取草稿句子作为兜底
            draft_indices = self.subtitle_manager.chunk_sentences.get(ctx.chunk_index, [])
            if draft_indices:
                draft_sentences = []
                for idx in draft_indices:
                    if idx in self.subtitle_manager.sentences:
                        # 深拷贝草稿句子，设置为定稿状态
                        draft_sentence = copy.deepcopy(self.subtitle_manager.sentences[idx])
                        draft_sentence.is_finalized = True
                        draft_sentence.is_draft = False
                        draft_sentences.append(draft_sentence)

                if draft_sentences:
                    self.logger.warning(
                        f"Chunk {ctx.chunk_index}: 使用草稿句子作为兜底 ({len(draft_sentences)} 个句子)"
                    )
                    final_sentences = draft_sentences
                    ctx.final_sentences = final_sentences

        # V3.8 修复竞态条件：深拷贝 final_sentences 再传给 subtitle_manager
        # 避免 subtitle_manager 修改句子对象影响 ctx.final_sentences
        sentences_for_manager = copy.deepcopy(final_sentences)
        self.subtitle_manager.replace_chunk(ctx.chunk_index, sentences_for_manager)

        self.logger.debug(
            f"Chunk {ctx.chunk_index}: 定稿已推送 "
            f"({len(final_sentences)} 个句子, 对齐级别={alignment_level.value})"
        )

    async def _align_loop(
        self,
        results: List[ProcessingContext],
        job_dir: Optional[Path] = None,
        total_chunks: int = 0,
        base_align_count: int = 0,  # V3.1.0: 基准偏移量（已完成的 chunk 数量）
        initial_finalized_indices: Optional[set] = None  # V3.1.0: 初始已完成索引集合
    ):
        """
        对齐阶段循环（最终消费者）

        职责：
        1. 从 queue_final 取 context
        2. 调用对齐阶段处理逻辑
        3. 收集结果到 results 列表
        4. 检测结束信号

        v3.1.0: 支持原子区域和检查点保存
        V3.1.0: 集成进度发射器
        V3.1.0: 使用累计索引集合，修复恢复后进度不准确问题

        Args:
            results: 结果列表（用于收集 context）
            job_dir: 任务目录（可选，v3.1.0）
            total_chunks: 总 Chunk 数（V3.1.0）
            base_align_count: 基准偏移量（V3.1.0）
            initial_finalized_indices: 初始已完成索引集合（V3.1.0）
        """
        token = self.cancellation_token  # v3.1.0
        # V3.1.0: 使用累计索引集合（类似 FastWorker 和 SlowWorker）
        finalized_indices = set(initial_finalized_indices) if initial_finalized_indices else set()
        self._finalized_indices = finalized_indices
        if finalized_indices:
            # 恢复态时同步最后进度索引，避免暂停快照缺失
            self._last_align_chunk_index = max(finalized_indices)
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

                chunk_index = ctx.chunk_index  # v3.1.0: 获取 chunk 索引

                # v3.1.0: 进入原子区域（单个 Chunk 对齐 + SSE 推送）
                if token:
                    token.enter_atomic_region(f"align_stage_chunk_{chunk_index}")

                try:
                    # 对齐阶段处理（双流对齐 + 推送定稿）
                    await self._run_alignment_stage(ctx)

                    # 收集结果
                    results.append(ctx)
                    finalized_indices.add(chunk_index)  # V3.1.0: 累加到集合
                    self._last_align_chunk_index = chunk_index

                    # V3.1.0: 更新对齐阶段进度（使用累计索引数量）
                    if self.progress_emitter and total_chunks > 0:
                        total_processed = len(finalized_indices)
                        self.progress_emitter.update_align(
                            total_processed, total_chunks,
                            message=f"对齐: {total_processed}/{total_chunks}"
                        )
                finally:
                    # v3.1.0: 退出原子区域
                    if token:
                        has_pending = token.exit_atomic_region()
                        if has_pending:
                            self.logger.info(f"[v3.1.0] 对齐阶段 chunk {chunk_index} 完成后检测到待处理请求")

                # v3.1.0: 每个 Chunk 处理完成后保存检查点
                if token and job_dir:
                    # V3.1.0: 获取字幕快照用于实时持久化
                    subtitle_checkpoint_data = {}
                    if self.subtitle_manager:
                        subtitle_checkpoint_data = self.subtitle_manager.to_checkpoint_data()

                    # V3.2.0+dev.20260123.05: 同步到实例变量，用于暂停快照
                    self._finalized_indices = finalized_indices
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
                            self.logger.info("[V3.1.0] 对齐阶段捕获暂停信号，继续排空 queue_final")
                        pause_requested = True
                        if not self.pause_exception:
                            self.pause_exception = e
                        # 继续排空，待数据全部写入后再暂停

            self.logger.info("对齐阶段循环完成")

        except Exception as e:
            self.logger.error(f"对齐阶段循环异常: {e}", exc_info=True)
            self.errors.append(e)
        finally:
            if pause_requested:
                self.logger.info("[V3.1.0] 对齐阶段已排空所有上下文，等待上层暂停")

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
    cancellation_token: Optional["CancellationToken"] = None  # v3.1.0: 新增
) -> AsyncDualPipeline:
    """
    获取异步双流流水线实例

    Args:
        job_id: 任务 ID
        queue_maxsize: 队列最大长度
        logger: 日志记录器
        cancellation_token: 取消令牌（可选，v3.1.0）

    Returns:
        AsyncDualPipeline 实例
    """
    return AsyncDualPipeline(
        job_id=job_id,
        queue_maxsize=queue_maxsize,
        logger=logger,
        cancellation_token=cancellation_token  # v3.1.0
    )
