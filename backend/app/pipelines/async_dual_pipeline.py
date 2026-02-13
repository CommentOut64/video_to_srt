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
import json
import logging
from dataclasses import dataclass, replace
from typing import List, Optional, Any, TYPE_CHECKING, Set, Dict, Tuple, Sequence
from pathlib import Path

from app.core.asr.engine import ASREngine
from app.core.logging import resolve_loguru_logger
from app.core.thresholds import ThresholdConfig, needs_whisper_patch
from app.schemas.pipeline_context import ProcessingContext
from app.models.confidence_models import AlignmentStatus
from app.models.sensevoice_models import SentenceSegment, TextSource, WordTimestamp
from app.services.audio.chunk_engine import AudioChunk
from app.services.alignment.alignment_processor import AlignmentProcessor
from app.services.alignment.default_aligner import DefaultAligner, _strip_trailing_punct_smart
from app.services.alignment.text_normalizer import get_alignment_text_normalizer
from app.services.alignment.text_normalizer_processor import TextNormalizerProcessor
from app.services.alignment.types import (
    AnnotatedWord,
    AlignmentResult,
    L1Input,
    L2Input,
    L2Output,
    L3Input,
    L4Input,
    L5Input,
    L6Input,
    L7Input,
    NormalizationResult,
    PunctSource,
    PunctTrack,
    QualitySignals,
    TextTrack,
    TextTrackBundle,
)
from app.services.arbitration.arbiter import TextArbiterProcessor
from app.services.arbitration.hallucination_detector import HallucinationDetector
from app.services.model_runtime_config_service import get_model_runtime_config_service
from app.services.sse_service import get_sse_manager
from app.services.segmentation.default_segmenter import DefaultSegmenter, DraftSegmenter
from app.services.segmentation.segmentation_processor import SegmentationProcessor
from app.services.streaming_subtitle import get_streaming_subtitle_manager
from app.services.streaming.output_processor import OutputProcessor
from app.services.punctuation.fast_punctuation_pipeline import FastPunctuationPipeline
from app.services.punctuation.punctuation_processor import PunctuationProcessor
from app.services.punctuation.debug_utils import (
    append_debug_dual_time_compare_line,
    append_debug_layer_diag_line,
    append_debug_layer_trace_line,
    append_debug_whisper_line,
    write_debug_json_payload,
)
from app.services.punctuation.base import PunctuationResult, PuncPosition, SplitPoint
from app.services.punctuation.scheduler import get_punctuation_scheduler
from app.services.punctuation.semantic_injection_processor import SemanticInjectionProcessor
from app.services.punctuation.final_splitter import FinalSplitter, FinalSplitConfig
from app.services.semantic_grouper import SemanticGrouper, GroupConfig
from app.services.bridge.bridge_controller import BridgeBatch, BridgeController
from app.services.punctuation.semantic_buffer import (
    PunctuationDecision,
    SemanticBuffer,
    SemanticBufferInput,
    SemanticChunk,
)
from app.services.text_pipeline_config import TextPipelineConfig
from app.services.text_normalizer import TextNormalizer
from app.services.whisper.whisper_text_sanitizer import WhisperTextSanitizer
from app.pipelines.workers import FastWorker, SlowWorker, SlowWorkerResult
from app.utils.prompt_builder import get_prompt_builder
from app.utils.text_utils import format_srt_timestamp
from app.utils.cancellation_token import CancelledException, PausedException  # V3.1.0: 捕获取消/暂停异常

# v3.1.0: 导入取消令牌和异常
if TYPE_CHECKING:
    from app.utils.cancellation_token import CancellationToken
    from app.services.progress_emitter import ProgressEventEmitter  # V3.1.0
    from app.services.punctuation.service import PunctuationService


@dataclass
class _Layer456RunResult:
    alignment_result: AlignmentResult
    words_for_split: List[WordTimestamp]
    injection_stats: Dict[str, Any]
    split_stats: Dict[str, Any]
    final_sentences: List[SentenceSegment]
    alignment_time_source: str = "sv"
    alignment_time_word_count: int = 0


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
        segmenter: Optional[DraftSegmenter] = None,
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
        self.logger = resolve_loguru_logger(
            logger,
            __name__,
            job_id=job_id,
            layer="L0",
        )
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
            segmenter = DraftSegmenter(
                is_enable_semantic_grouping=enable_semantic_grouping,
                is_enable_cross_chunk_merge=enable_cross_chunk_merge,
                logger=self.logger,
            )
        # 仅用于草稿链：定稿链严禁通过 DefaultSegmenter 产出结果。
        self._draft_segmenter = segmenter
        # 兼容外部历史访问（不建议新代码继续使用）。
        self.segmenter = self._draft_segmenter
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
        # V3.2.0+dev.20260204.02: L1 规范化处理器（Processor 入口）
        self._l1_processor = TextNormalizerProcessor(
            normalizer=self._text_normalizer,
            logger=self.logger,
        )
        self._whisper_sanitizer = WhisperTextSanitizer(logger=self.logger)
        # V3.2.0+dev.20260204.03: L2 仲裁处理器与幻觉检测器
        self._hallucination_detector = HallucinationDetector(logger=self.logger)
        self._l2_processor = TextArbiterProcessor(logger=self.logger)
        self._fast_punctuator = FastPunctuationPipeline(
            job_id=self.job_id,
            punctuation_service=self.punctuation_service,
            logger=self.logger,
        )
        # V3.2.0+dev.20260204.05: L3 标点处理器
        self._l3_processor = PunctuationProcessor(
            punctuation_service=self.punctuation_service,
            logger=self.logger,
        )
        # V3.2.0+dev.20260205.09: L4/L5/L6 处理器接入（对齐/注入/切分）
        self._l4_processor = AlignmentProcessor(logger=self.logger)
        self._text_pipeline_config = TextPipelineConfig.from_runtime()
        self._segmentation_layer_config = self._text_pipeline_config.segmentation
        final_split_config = FinalSplitConfig(
            min_tokens=max(1, int(self._segmentation_layer_config.final_min_tokens)),
            max_tokens=max(1, int(self._segmentation_layer_config.final_max_tokens)),
            min_duration=max(0.0, float(self._segmentation_layer_config.final_min_duration)),
            max_duration=max(0.5, float(self._segmentation_layer_config.final_max_duration)),
            soft_pause=max(0.0, float(self._segmentation_layer_config.final_soft_pause)),
            long_pause=max(
                float(self._segmentation_layer_config.final_soft_pause),
                float(self._segmentation_layer_config.final_long_pause),
            ),
            min_mapping_coverage=max(
                0.0,
                min(1.0, float(self._segmentation_layer_config.final_min_mapping_coverage)),
            ),
        )
        self._final_splitter = FinalSplitter(final_split_config, logger=self.logger)
        if enable_semantic_grouping:
            final_group_config = GroupConfig(
                max_group_gap=2.0,
                max_group_duration=10.0,
                max_group_sentences=5,
                enable_overlap_detection=True,
            )
            self._final_grouper = SemanticGrouper(final_group_config)
        else:
            self._final_grouper = None
        self._l5_processor = SemanticInjectionProcessor(
            logger=self.logger,
            min_mapping_coverage=final_split_config.min_mapping_coverage,
        )
        self._l6_processor = SegmentationProcessor(
            final_splitter=self._final_splitter,
            logger=self.logger,
            is_keep_sentence_end_punct=bool(
                self._segmentation_layer_config.is_keep_sentence_end_punct
            ),
        )
        self._l7_processor = OutputProcessor(
            subtitle_manager=self.subtitle_manager,
            logger=self.logger,
        )
        self._keep_sentence_end_punct = bool(
            self._segmentation_layer_config.is_keep_sentence_end_punct
        )
        # V3.2.0+dev.20260206.02: 双轨实验配置（shadow/active 仅串行执行 L4-L6，避免并行占用 GPU）。
        self._alignment_layer_config = self._text_pipeline_config.alignment
        dual_time_mode = str(self._alignment_layer_config.dual_time_mode or "off").lower()
        if dual_time_mode not in {"off", "shadow", "active"}:
            dual_time_mode = "off"
        if dual_time_mode == "off" and self._alignment_layer_config.is_enable_dual_time_experiment:
            dual_time_mode = "shadow"
        self._dual_time_mode = dual_time_mode
        self._is_dual_time_experiment_enabled = dual_time_mode in {"shadow", "active"}
        self._is_dual_time_write_debug_srt = bool(
            self._alignment_layer_config.is_dual_time_write_debug_srt
        )
        self._dual_time_boundary_tolerance_sec = (
            max(50, int(self._alignment_layer_config.dual_time_boundary_tolerance_ms)) / 1000.0
        )
        self._dual_time_active_min_boundary_f1 = float(
            self._alignment_layer_config.dual_time_active_min_boundary_f1
        )
        self._dual_time_compare_accumulator: Dict[str, Any] = {
            "chunk_count": 0,
            "boundary_precision_sum": 0.0,
            "boundary_recall_sum": 0.0,
            "boundary_f1_sum": 0.0,
            "word_start_mae_ms_sum": 0.0,
            "word_end_mae_ms_sum": 0.0,
            "sentence_start_mae_ms_sum": 0.0,
            "sentence_end_mae_ms_sum": 0.0,
            "selected_experiment_count": 0,
        }
        self._dual_time_legacy_sentences_by_chunk: Dict[int, List[SentenceSegment]] = {}
        self._dual_time_experiment_sentences_by_chunk: Dict[int, List[SentenceSegment]] = {}
        if self._is_dual_time_experiment_enabled:
            self.logger.info(
                "双轨实验开启: mode={} tolerance_ms={} gpu_strategy=serial_postprocess",
                self._dual_time_mode,
                int(self._dual_time_boundary_tolerance_sec * 1000),
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
        else:
            # V3.10: 智能复核模式下设置 is_patching_mode=True
            self.slow_worker = SlowWorker(
                patch_engine=self.patch_engine,
                whisper_language=whisper_language,
                punctuation_service=self.punctuation_service,
                logger=self.logger
            )
        # Legacy aligner 仅保留为兼容注入入口（默认不启用主路径）。
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

    def _bind_log(
        self,
        *,
        chunk_index: Optional[int] = None,
        batch_id: Optional[str] = None,
    ):
        log = self.logger
        if chunk_index is not None:
            log = log.bind(chunk_index=chunk_index)
        if batch_id is not None:
            log = log.bind(batch_id=batch_id)
        return log

    @staticmethod
    def _extract_primary_speaker_id(chunk: AudioChunk) -> Optional[str]:
        """从 AudioChunk 中提取主说话人标识（L0 透传）。"""
        if getattr(chunk, "primary_speaker_id", None):
            return chunk.primary_speaker_id
        if getattr(chunk, "speaker_id", None):
            return chunk.speaker_id
        speaker_tracks = getattr(chunk, "speaker_tracks", None)
        if speaker_tracks:
            for track in speaker_tracks:
                if isinstance(track, dict) and track.get("is_primary"):
                    return track.get("speaker_id")
            first = speaker_tracks[0]
            if isinstance(first, dict):
                return first.get("speaker_id")
        return None

    @staticmethod
    def _normalize_chunk_language(language: Optional[str]) -> str:
        """规范化 chunk 语言标签。"""
        value = str(language or "").strip().lower()
        return value if value else "auto"

    @classmethod
    def _resolve_chunk_languages_inplace(cls, audio_chunks: List[AudioChunk]) -> None:
        """
        统一修正 Chunk 语言标签。

        规则：
        1. 若当前 chunk 语言非 auto，直接保留并作为后续基准。
        2. 若当前 chunk 语言为 auto，则沿用前一个 chunk 的有效语言。
        3. 第一个 chunk 为 auto 时，回环使用最后一个 chunk 的有效语言。
        """
        if not audio_chunks:
            return

        normalized_languages = [
            cls._normalize_chunk_language(getattr(chunk, "language", None))
            for chunk in audio_chunks
        ]
        has_effective_language = any(lang != "auto" for lang in normalized_languages)
        if not has_effective_language:
            return

        last_effective = "auto"
        for lang in reversed(normalized_languages):
            if lang != "auto":
                last_effective = lang
                break

        for index, chunk in enumerate(audio_chunks):
            current = normalized_languages[index]
            if current == "auto":
                if getattr(chunk, "language", None) != last_effective:
                    chunk.language = last_effective
                continue
            last_effective = current

    def _validate_l0_result(
        self,
        result: Dict[str, Any],
        *,
        source: str,
        chunk_index: Optional[int] = None,
    ) -> None:
        """L0 仅做完整性校验，不做补全或降级。"""
        if not isinstance(result, dict):
            return
        missing_fields: List[str] = []
        raw_text = result.get("raw_text")
        if raw_text is None or str(raw_text).strip() == "":
            missing_fields.append("raw_text")
        if source == "fast" and result.get("words") is None:
            missing_fields.append("words")
        if source == "slow" and result.get("segments") is None:
            missing_fields.append("segments")
        if missing_fields:
            result["l0_error_code"] = "E_L0_MISSING_INPUT"
            result["l0_missing_fields"] = list(missing_fields)
            log = self._bind_log(chunk_index=chunk_index)
            log.warning(
                "L0 输入缺失: source=%s missing=%s",
                source,
                ",".join(missing_fields),
            )

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
        # V3.2.0+dev.20260212.04: 进入 L0 前先收敛语言标签。
        # 若 chunk.language=auto，则沿用前一个 chunk；第一个 chunk 回环取最后一个有效语言。
        self._resolve_chunk_languages_inplace(audio_chunks)
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
                if ctx.sv_result:
                    self._validate_l0_result(ctx.sv_result, source="fast", chunk_index=i)
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
                        self.logger.debug(f"[v3.1.0] Chunk {i} 处理完成后检测到待处理请求")

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
                    self.logger.debug(
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
        if self._is_dual_time_experiment_enabled:
            self._dual_time_compare_accumulator = {
                "chunk_count": 0,
                "boundary_precision_sum": 0.0,
                "boundary_recall_sum": 0.0,
                "boundary_f1_sum": 0.0,
                "word_start_mae_ms_sum": 0.0,
                "word_end_mae_ms_sum": 0.0,
                "sentence_start_mae_ms_sum": 0.0,
                "sentence_end_mae_ms_sum": 0.0,
                "selected_experiment_count": 0,
            }
            self._dual_time_legacy_sentences_by_chunk.clear()
            self._dual_time_experiment_sentences_by_chunk.clear()

        # V3.2.0+dev.20260201.07: 记录完整音频与 Chunk 映射，供 Bridge 批次使用
        self._full_audio_array = full_audio_array
        self._full_audio_sr = full_audio_sr
        self._audio_chunks_by_index = {chunk.index: chunk for chunk in audio_chunks}
        self._context_cache = {}
        self._vad_intervals = list(vad_intervals) if vad_intervals else None
        self._l6_processor.reset_state()

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
        sentences = self._draft_segmenter.split_draft(
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

    @staticmethod
    def _ensure_text_tracks(ctx: ProcessingContext) -> TextTrackBundle:
        if ctx.text_tracks is None:
            ctx.text_tracks = TextTrackBundle()
        return ctx.text_tracks

    @staticmethod
    def _build_text_track(
        raw_text: str,
        normalized: NormalizationResult,
        source: str,
        language: str,
    ) -> TextTrack:
        return TextTrack(
            raw_text=raw_text,
            text_itn_raw=normalized.text_itn_raw,
            text_clean=normalized.text_clean or normalized.text_itn_raw,
            char_mapping=normalized.char_mapping,
            raw_to_clean=normalized.raw_to_clean,
            clean_to_raw=normalized.clean_to_raw,
            language=language or "auto",
            source=source,
            itn_fallback=normalized.itn_fallback,
            itn_fallback_reason=normalized.itn_fallback_reason,
            mapping_coverage=normalized.mapping_coverage,
        )

    @staticmethod
    def _build_normalization_from_track(track: TextTrack) -> NormalizationResult:
        return NormalizationResult(
            text_itn_raw=track.text_itn_raw,
            text_clean=track.text_clean,
            char_mapping=track.char_mapping,
            raw_to_clean=track.raw_to_clean,
            clean_to_raw=track.clean_to_raw,
            itn_fallback=track.itn_fallback,
            itn_fallback_reason=track.itn_fallback_reason,
            mapping_coverage=track.mapping_coverage,
        )

    @staticmethod
    def _clone_text_track(track: TextTrack, source: str) -> TextTrack:
        return replace(
            track,
            source=source,
            clean_to_word=list(track.clean_to_word),
            word_confidences=list(track.word_confidences),
            punct_positions=list(track.punct_positions),
        )

    def _normalize_sensevoice_result(
        self,
        ctx: ProcessingContext,
    ) -> Optional[NormalizationResult]:
        """统一规范化 SenseVoice 输出（V3.2.0+dev.20260202.05）。"""
        if not ctx.sv_result or not ctx.audio_chunk:
            return None
        sv_result = ctx.sv_result
        # V3.2.0+dev.20260203.10: L0 使用 raw_text 作为规范化入口
        raw_text = sv_result.get("raw_text")
        if raw_text is None or str(raw_text).strip() == "":
            self.logger.warning("L1 规范化缺失 raw_text，跳过规范化")
            return None
        if not sv_result.get("confidence_source"):
            sv_result["confidence_source"] = "fast"
        language = ctx.audio_chunk.language or sv_result.get("language") or "auto"
        l1_output = self._l1_processor.process(
            L1Input(
                sv_raw_result=sv_result,
                wh_raw_result=None,
                language_hint=language,
            )
        )
        if not l1_output.sv_track:
            return None
        sv_track = l1_output.sv_track
        sv_result["text_itn_raw"] = sv_track.text_itn_raw
        sv_result["text_clean"] = sv_track.text_clean or sv_track.text_itn_raw
        tracks = self._ensure_text_tracks(ctx)
        tracks.sv_track = sv_track
        return self._build_normalization_from_track(sv_track)

    def _apply_whisper_full_sanitize(self, ctx: ProcessingContext) -> None:
        """Whisper 清洗增强与规范化重算（V3.2.0+dev.20260202.07）。"""
        if not ctx.whisper_result or not ctx.audio_chunk:
            return
        whisper_result = ctx.whisper_result
        raw_text = whisper_result.get("raw_text") or whisper_result.get("text_raw") or ""
        min_clean_text = whisper_result.get("min_clean_text") or ""
        base_text = raw_text or min_clean_text
        if not base_text:
            return
        if raw_text and not whisper_result.get("text_raw"):
            whisper_result["text_raw"] = raw_text
        prompt = whisper_result.get("prompt")
        sanitized = self._whisper_sanitizer.sanitize_full(base_text, prompt=prompt)

        language = whisper_result.get("language") or ctx.audio_chunk.language or "auto"
        temp_result = dict(whisper_result)
        temp_result["raw_text"] = sanitized
        temp_result["min_clean_text"] = None
        temp_result["text_clean"] = None
        temp_result["text"] = None
        l1_output = self._l1_processor.process(
            L1Input(
                sv_raw_result=None,
                wh_raw_result=temp_result,
                language_hint=language,
            )
        )
        if not l1_output.whisper_track:
            return
        wh_track = l1_output.whisper_track
        whisper_result["text_itn_raw"] = wh_track.text_itn_raw
        whisper_result["text_clean"] = wh_track.text_clean
        whisper_result["text"] = wh_track.text_clean or sanitized
        whisper_result["language"] = language
        tracks = self._ensure_text_tracks(ctx)
        tracks.whisper_track = wh_track

    def _run_arbitration(
        self,
        ctx: ProcessingContext,
        sv_result: Dict[str, Any],
        whisper_result: Dict[str, Any],
    ) -> L2Output:
        tracks = self._ensure_text_tracks(ctx)
        quality_signals = self._build_quality_signals(
            sv_result=sv_result,
            whisper_result=whisper_result,
            tracks=tracks,
        )
        return self._l2_processor.process(
            L2Input(
                sv_track=tracks.sv_track,
                whisper_track=tracks.whisper_track,
                quality_signals=quality_signals,
            )
        )

    @staticmethod
    def _select_text_for_alignment(
        chosen_track: Optional[TextTrack],
    ) -> str:
        if not chosen_track:
            return ""
        return str(
            chosen_track.text_clean
            or chosen_track.text_itn_raw
            or chosen_track.raw_text
            or ""
        )

    def _apply_arbitration_text(
        self,
        whisper_result: Dict[str, Any],
        sv_result: Dict[str, Any],
        chosen_source: str,
        chosen_text_clean: str,
    ) -> None:
        if chosen_source == "fast":
            whisper_result["text"] = chosen_text_clean
            whisper_result["text_clean"] = chosen_text_clean
            whisper_result["text_itn_raw"] = sv_result.get("text_itn_raw") or chosen_text_clean
        else:
            whisper_result["text"] = chosen_text_clean
            whisper_result["text_clean"] = chosen_text_clean

    def _build_quality_signals(
        self,
        *,
        sv_result: Dict[str, Any],
        whisper_result: Dict[str, Any],
        tracks: TextTrackBundle,
    ) -> QualitySignals:
        sv_text = tracks.sv_track.text_clean if tracks.sv_track else ""
        wh_text = tracks.whisper_track.text_clean if tracks.whisper_track else ""
        length_ratio = 0.0
        if sv_text and wh_text:
            length_ratio = len(sv_text) / max(len(wh_text), 1)
        confidence_fast = float(sv_result.get("confidence", 0.0) or 0.0)
        confidence_slow = float(whisper_result.get("confidence", 0.0) or 0.0)

        is_hallucination = bool(whisper_result.get("is_hallucination")) if whisper_result else False

        is_repetition = False
        if wh_text:
            is_repetition, reason = TextNormalizer.detect_intra_block_repetition(wh_text)
            if is_repetition:
                self.logger.debug("L2 仲裁检测到重复: reason=%s", reason)

        is_itn_fallback = bool(
            (tracks.whisper_track and tracks.whisper_track.itn_fallback)
            or (tracks.sv_track and tracks.sv_track.itn_fallback)
        )
        coverage_values = [
            track.mapping_coverage
            for track in (tracks.sv_track, tracks.whisper_track)
            if track
        ]
        mapping_coverage = min(coverage_values) if coverage_values else 0.0

        return QualitySignals(
            length_ratio=length_ratio,
            confidence_fast=confidence_fast,
            confidence_slow=confidence_slow,
            is_repetition=is_repetition,
            is_hallucination=is_hallucination,
            is_itn_fallback=is_itn_fallback,
            mapping_coverage=mapping_coverage,
            alignment_score=0.0,
            gap_ratio=0.0,
        )

    def _build_punct_source_from_metadata(
        self,
        sv_result: Dict[str, Any],
        track: Optional[TextTrack],
    ) -> Optional[PunctSource]:
        if not sv_result or not track:
            return None
        metadata = sv_result.get("metadata", {}) if isinstance(sv_result, dict) else {}
        punct_meta = metadata.get("punctuation")
        if not isinstance(punct_meta, dict):
            return None
        raw_positions = punct_meta.get("punctuation_positions", []) or []
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
        if not positions:
            return None
        return PunctSource(
            clean_text_ref=track.text_clean or "",
            positions=positions,
            source="fast",
            confidence=float(punct_meta.get("confidence", 0.0) or 0.0),
            model_id=str(punct_meta.get("model_id", "") or ""),
        )

    def _build_punct_source_from_whisper(
        self,
        whisper_result: Dict[str, Any],
        track: Optional[TextTrack],
    ) -> Optional[PunctSource]:
        if not whisper_result or not track:
            return None
        # V3.2.0+dev.20260204.11: slow_raw 来源由 L1 预先抽取并写入 TextTrack.punct_positions。
        clean_text = str(track.text_clean or "")
        if not clean_text:
            return None
        positions = list(track.punct_positions or [])
        if not positions:
            return None
        avg_conf = sum(float(pos.confidence or 0.0) for pos in positions) / max(len(positions), 1)
        # V3.2.0+dev.20260205.03: 记录 slow_raw 来源细节，定位词级逗号密集问题
        self.logger.debug(
            "L1 slow_raw 标点候选: clean_len=%d positions=%d avg_conf=%.2f sample=%s",
            len(clean_text),
            len(positions),
            avg_conf,
            positions[:20],
        )
        return PunctSource(
            clean_text_ref=clean_text,
            positions=positions,
            source="slow_raw",
            confidence=float(avg_conf),
            model_id="asr_raw_words",
        )

    @staticmethod
    def _extract_whisper_words(whisper_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw = whisper_result.get("raw_result") if isinstance(whisper_result, dict) else None
        segments = raw.get("segments", []) if isinstance(raw, dict) else []
        words: List[Dict[str, Any]] = []
        edge_punct = set(",.!?;:\"()[]{}，。！？；：、（）【】《》“”‘’「」『』")
        for seg in segments:
            for word in seg.get("words", []) or []:
                token = str(word.get("word", "") or "")
                token = token.replace("▁", " ").strip()
                start = 0
                end = len(token)
                while start < end and token[start] in edge_punct:
                    start += 1
                while end > start and token[end - 1] in edge_punct:
                    end -= 1
                token = token[start:end].strip()
                if not token:
                    continue
                words.append(
                    {
                        "word": token,
                        "start": float(word.get("start", 0.0) or 0.0),
                        "end": float(word.get("end", 0.0) or 0.0),
                        "confidence": float(word.get("probability", 0.0) or 0.0),
                    }
                )
        return words

    async def _run_punctuation_layer(
        self,
        ctx: ProcessingContext,
        sv_result: Dict[str, Any],
        whisper_result: Dict[str, Any],
        chosen_source: str,
    ) -> Optional[PunctTrack]:
        tracks = self._ensure_text_tracks(ctx)
        chosen_track = tracks.chosen_track
        if not chosen_track:
            return None
        word_timestamps: List[Dict[str, Any]] = []
        if chosen_source == "fast":
            word_timestamps = list(sv_result.get("words", []) or [])
        elif whisper_result:
            word_timestamps = self._extract_whisper_words(whisper_result)
        sv_source = self._build_punct_source_from_metadata(sv_result, tracks.sv_track)
        wh_source: Optional[PunctSource] = None
        # V3.2.0+dev.20260204.11: slow_raw 候选由 L1 预先抽取并透传，仅在文本一致时启用
        if (
            chosen_source != "fast"
            and tracks.whisper_track
            and tracks.whisper_track.text_clean
            and tracks.whisper_track.text_clean == chosen_track.text_clean
        ):
            wh_source = self._build_punct_source_from_whisper(
                whisper_result,
                tracks.whisper_track,
            )
        setattr(
            ctx,
            "_trace_l3_input",
            {
                "chosen_source": chosen_source,
                "word_timestamps": list(word_timestamps),
                "sv_punct_source": sv_source,
                "wh_punct_source": wh_source,
                "chosen_text_track": chosen_track,
            },
        )
        output = await self._l3_processor.process(
            L3Input(
                chosen_text_track=chosen_track,
                sv_punct_source=sv_source,
                wh_punct_source=wh_source,
                word_timestamps=word_timestamps,
            )
        )
        setattr(ctx, "_trace_l3_output", output.punct_track)
        return output.punct_track

    def _resolve_gap_ratio_mid(self) -> float:
        gap_ratio_mid = 0.3
        if not self.aligner:
            return gap_ratio_mid
        gap_resolver = getattr(self.aligner.alignment_service, "gap_resolver", None)
        value = getattr(gap_resolver, "_gap_ratio_mid", None)
        if isinstance(value, (int, float)):
            gap_ratio_mid = float(value)
        return gap_ratio_mid

    def _resolve_injection_min_mapping_coverage(self) -> float:
        if not self.aligner:
            return 0.6
        config = getattr(self.aligner.final_splitter, "config", None)
        value = getattr(config, "min_mapping_coverage", None)
        if isinstance(value, (int, float)):
            return float(value)
        return 0.6

    def _record_punct_retry_candidates(self, ctx: ProcessingContext) -> None:
        """L4/L5 补跑候选仅做埋点，不触发补跑。"""
        stats = ctx.finalization_metrics
        if not stats:
            return

        reasons_l4: List[str] = []
        coverage = stats.get("coverage")
        coverage_threshold = get_punctuation_scheduler().policy.alignment_coverage_threshold
        if coverage is not None and float(coverage) < coverage_threshold:
            reasons_l4.append("alignment_coverage_low")
        gap_ratio = stats.get("gap_ratio")
        gap_ratio_mid = self._resolve_gap_ratio_mid()
        if gap_ratio is not None and float(gap_ratio) >= gap_ratio_mid:
            reasons_l4.append("gap_ratio_high")
        stats["punct_retry_candidate_l4"] = 1.0 if reasons_l4 else 0.0
        stats["punct_retry_reason_l4"] = "|".join(reasons_l4)
        stats["punct_retry_alignment_coverage_threshold"] = coverage_threshold
        stats["punct_retry_gap_ratio_mid"] = gap_ratio_mid

        reasons_l5: List[str] = []
        min_mapping = self._resolve_injection_min_mapping_coverage()
        mapping_cov = stats.get("injection_mapping_coverage")
        if mapping_cov is not None and float(mapping_cov) < min_mapping:
            reasons_l5.append("injection_coverage_low")
        miss_ratio = stats.get("injection_miss_ratio")
        miss_threshold = 0.3
        if miss_ratio is not None and float(miss_ratio) >= miss_threshold:
            reasons_l5.append("injection_miss_ratio_high")
        blocked = stats.get("injection_blocked")
        if blocked is not None and float(blocked) > 0.0:
            reasons_l5.append("injection_blocked")
        stats["punct_retry_candidate_l5"] = 1.0 if reasons_l5 else 0.0
        stats["punct_retry_reason_l5"] = "|".join(reasons_l5)
        stats["punct_retry_injection_min_mapping_coverage"] = min_mapping
        stats["punct_retry_injection_miss_ratio_threshold"] = miss_threshold

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
        punct_track = self._build_punct_track_from_metadata(ctx)
        if punct_track:
            ctx.punct_track = punct_track

    def _build_punct_track_from_metadata(self, ctx: ProcessingContext) -> Optional[PunctTrack]:
        """从快流标点元数据构建 PunctTrack（仅用于草稿路径）。"""
        if not ctx.sv_result:
            return None
        tracks = self._ensure_text_tracks(ctx)
        sv_track = tracks.sv_track
        if not sv_track:
            return None
        source = self._build_punct_source_from_metadata(ctx.sv_result, sv_track)
        if not source:
            return None
        return PunctTrack(
            clean_text_ref=source.clean_text_ref,
            positions=list(source.positions),
            source="fast",
            confidence_stats={
                "count": float(len(source.positions)),
                "avg_confidence": float(source.confidence or 0.0),
                "min_confidence": float(source.confidence or 0.0),
                "max_confidence": float(source.confidence or 0.0),
                "model_confidence": float(source.confidence or 0.0),
            },
        )

    @staticmethod
    def _is_semantic_clean_text(text: str) -> bool:
        """判断文本是否包含明显的 ASR 标签/分词符号污染。"""
        if not text:
            return False
        return "<|" not in text and "▁" not in text

    def _build_semantic_input(self, ctx: ProcessingContext) -> Optional[SemanticBufferInput]:
        """构建 SemanticBuffer 输入。"""
        sv_result = ctx.sv_result or {}
        chunk = ctx.audio_chunk
        metadata = sv_result.get("metadata", {}) if isinstance(sv_result, dict) else {}
        punctuation_meta = metadata.get("punctuation")
        decision_meta = metadata.get("punctuation_decision")

        raw_text = str(sv_result.get("raw_text") or "")
        sv_text_clean = str(sv_result.get("text_clean") or "")
        track_text: Optional[str] = None
        tracks = getattr(ctx, "text_tracks", None)
        if tracks and getattr(tracks, "sv_track", None):
            track_text = str(tracks.sv_track.text_clean or "")

        # V3.2.0+dev.20260205.08:
        # SemanticBuffer 的文本输入必须优先使用 L1 的 clean 口径，禁止在 clean 为空时回退到带 <|...|>/▁ 的 raw_text。
        if track_text is not None:
            text = track_text
        elif sv_text_clean:
            text = sv_text_clean
        elif raw_text and self._is_semantic_clean_text(raw_text):
            text = raw_text
        else:
            if raw_text:
                self.logger.debug("SemanticBuffer 跳过污染 raw_text: %s", raw_text[:50])
            text = ""
        words = sv_result.get("words") if isinstance(sv_result, dict) else None
        raw_tokens = sv_result.get("raw_tokens") if isinstance(sv_result, dict) else None
        punctuation_result = None
        # V3.2.0+dev.20260201.10: 即使缺少标点元信息，也走语义缓冲与 Bridge 批次，旧逐Chunk仅作备份
        if isinstance(punctuation_meta, dict):
            punctuation_result = self._build_punctuation_result(punctuation_meta, raw_text)
            if punctuation_result and punctuation_result.text:
                candidate = punctuation_result.text
                if self._is_semantic_clean_text(candidate):
                    text = candidate
                else:
                    self.logger.debug("SemanticBuffer 忽略污染标点文本: %s", candidate[:50])

        if not text:
            return None

        decision = self._build_punctuation_decision(decision_meta)
        source_chunks = [f"chunk-{chunk.index}"]
        speaker_id = self._extract_primary_speaker_id(chunk)
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
            speaker_id=speaker_id,
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
        self.logger.debug(
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
    def _compose_whisper_text_from_words(words: Sequence[Dict[str, Any]]) -> str:
        """按 Whisper 词序列重建文本，避免跨 Chunk 时整段串入。"""
        if not words:
            return ""
        parts: List[str] = []
        for item in words:
            token = str(item.get("word", "") or "")
            if not token:
                continue
            parts.append(token.replace("▁", " "))
        if not parts:
            return ""
        merged = "".join(parts).strip()
        if not merged:
            return ""
        return " ".join(merged.split())

    @staticmethod
    def _find_target_chunk_by_overlap(
        start_abs: float,
        end_abs: float,
        chunk_ranges: Dict[int, tuple[float, float]],
        ordered_indices: Sequence[int],
    ) -> Optional[int]:
        """按时间重叠优先选择词/片段所属 Chunk。"""
        if not ordered_indices:
            return None
        if end_abs < start_abs:
            end_abs = start_abs

        best_idx: Optional[int] = None
        best_overlap = 0.0
        best_distance = float("inf")
        mid = (start_abs + end_abs) / 2.0

        for idx in ordered_indices:
            span_start, span_end = chunk_ranges[idx]
            overlap = max(0.0, min(end_abs, span_end) - max(start_abs, span_start))
            distance = AsyncDualPipeline._distance_to_range(mid, (span_start, span_end))
            if overlap > best_overlap or (overlap == best_overlap and distance < best_distance):
                best_idx = idx
                best_overlap = overlap
                best_distance = distance

        if best_idx is not None:
            return best_idx
        return min(ordered_indices, key=lambda idx: AsyncDualPipeline._distance_to_range(mid, chunk_ranges[idx]))

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
            raw_text = str(whisper_result.get("text_raw") or whisper_result.get("raw_text") or whisper_result.get("text", ""))
            min_clean_text = str(whisper_result.get("min_clean_text") or "")
            if not min_clean_text and raw_text:
                min_clean_text = self._whisper_sanitizer.sanitize_minimal(raw_text, prompt=None)
            return {
                first: {
                    "text": str(whisper_result.get("text", "")),
                    "raw_text": raw_text,
                    "min_clean_text": min_clean_text,
                    "confidence": float(whisper_result.get("confidence", 0.0) or 0.0),
                    "language": str(whisper_result.get("language", "auto")),
                    "raw_result": {"segments": []},
                    "word_time_base": "batch_local",
                    "word_time_offset": float(batch_start),
                }
            }

        assignments: Dict[int, List[Dict[str, Any]]] = {idx: [] for idx in chunk_ranges}
        ordered_indices = sorted(chunk_ranges.keys())

        for seg in raw_segments:
            seg_start = float(seg.get("start", 0.0) or 0.0) + batch_start
            seg_end = float(seg.get("end", 0.0) or 0.0) + batch_start
            seg_words = seg.get("words", []) if isinstance(seg, dict) else []
            if isinstance(seg_words, list) and seg_words:
                words_by_chunk: Dict[int, List[Dict[str, Any]]] = {}
                for word in seg_words:
                    if not isinstance(word, dict):
                        continue
                    word_start_abs = float(word.get("start", 0.0) or 0.0) + batch_start
                    word_end_abs = float(word.get("end", word.get("start", 0.0)) or word.get("start", 0.0)) + batch_start
                    target_idx = self._find_target_chunk_by_overlap(
                        word_start_abs,
                        word_end_abs,
                        chunk_ranges,
                        ordered_indices,
                    )
                    if target_idx is None:
                        continue
                    words_by_chunk.setdefault(target_idx, []).append(word)

                for idx, words in words_by_chunk.items():
                    seg_fragment = dict(seg)
                    seg_fragment["words"] = list(words)
                    fragment_text = self._compose_whisper_text_from_words(words)
                    if fragment_text:
                        seg_fragment["text"] = fragment_text
                    assignments[idx].append(seg_fragment)
                continue

            target_idx = self._find_target_chunk_by_overlap(
                seg_start,
                seg_end,
                chunk_ranges,
                ordered_indices,
            )
            if target_idx is not None:
                assignments[target_idx].append(seg)

        results: Dict[int, Dict[str, Any]] = {}
        language = str(language_override or whisper_result.get("language", "auto"))
        fallback_conf = float(whisper_result.get("confidence", 0.0) or 0.0)

        for idx in ordered_indices:
            segs = assignments.get(idx, [])
            # V3.2.0+dev.20260205.01: 分段文本拼接必须保留段间空格，否则会产生 "lamp.Who" 这类错误粘连
            parts = [str(seg.get("text", "") or "").strip() for seg in segs]
            text = " ".join(part for part in parts if part).strip()
            raw_text = text
            confidence = self._estimate_segment_confidence(segs, fallback_conf)
            results[idx] = {
                "text": text,
                "raw_text": raw_text,
                "min_clean_text": self._whisper_sanitizer.sanitize_minimal(raw_text, prompt=None) if raw_text else "",
                "confidence": confidence,
                "language": language,
                "raw_result": {"segments": segs},
                "word_time_base": "batch_local",
                "word_time_offset": float(batch_start),
            }
        return results

    def _emit_whisper_debug(
        self,
        job_dir: Optional[Path],
        batch: BridgeBatch,
        whisper_result: Dict[str, Any],
        chunk_indices: List[int],
    ) -> None:
        """记录 Whisper 批次调试信息。"""
        raw_result = whisper_result.get("raw_result", {}) if isinstance(whisper_result, dict) else {}
        raw_segments = raw_result.get("segments", []) if isinstance(raw_result, dict) else []
        seg_count = len(raw_segments)
        avg_logprob = None
        avg_no_speech = None
        if seg_count > 0:
            avg_logprob = sum(
                float(seg.get("avg_logprob", 0.0) or 0.0) for seg in raw_segments
            ) / seg_count
            avg_no_speech = sum(
                float(seg.get("no_speech_prob", 0.0) or 0.0) for seg in raw_segments
            ) / seg_count
        raw_text = str(whisper_result.get("text_raw", "") or "")
        sanitized_text = str(whisper_result.get("text", "") or "")
        seg_text = "".join(str(seg.get("text", "")) for seg in raw_segments).strip()
        payload = {
            "batch_id": batch.batch_id,
            "chunk_indices": chunk_indices,
            "text_len": len(sanitized_text),
            "raw_text_len": len(raw_text),
            "seg_text_len": len(seg_text),
            "segments_count": seg_count,
            "avg_logprob": avg_logprob,
            "avg_no_speech_prob": avg_no_speech,
            "language": whisper_result.get("language", "auto"),
        }
        append_debug_whisper_line(job_dir, payload, logger=self.logger)

    def _emit_layer_diagnostics(
        self,
        ctx: ProcessingContext,
        *,
        tracks: TextTrackBundle,
        punct_track: Optional[PunctTrack],
        alignment_result: AlignmentResult,
        injection_stats: Dict[str, Any],
        split_stats: Dict[str, Any],
        final_sentences: List[SentenceSegment],
    ) -> None:
        """输出 L3/L4/L5/L6 诊断到独立文件。"""
        chosen_clean = tracks.chosen_track.text_clean if tracks.chosen_track else ""
        punct_ref = punct_track.clean_text_ref if punct_track else ""
        chosen_compact = str(chosen_clean or "").replace("\n", " ").strip()
        punct_compact = str(punct_ref or "").replace("\n", " ").strip()

        def _clip_head(text: str, limit: int = 40) -> str:
            if len(text) <= limit:
                return text
            return text[:limit]

        def _clip_tail(text: str, limit: int = 40) -> str:
            if len(text) <= limit:
                return text
            return text[-limit:]

        unmatched_prefix_len = 0
        for aligned_word in alignment_result.aligned_words:
            status = aligned_word.alignment_status
            is_prefix_unmatched = bool(aligned_word.is_pseudo) or status in {
                AlignmentStatus.INSERTED,
                AlignmentStatus.PSEUDO,
            }
            if not is_prefix_unmatched:
                break
            unmatched_prefix_len += len(str(aligned_word.word or ""))

        is_ref_mismatch = bool(chosen_compact and punct_compact and chosen_compact != punct_compact)
        is_slow_chosen = bool(
            ctx.arbitration_result and ctx.arbitration_result.chosen_source == "slow"
        )
        punct_positions_count = len(punct_track.positions) if punct_track and punct_track.positions else 0
        is_cross_chunk_boundary_suspected = bool(
            is_slow_chosen
            and punct_positions_count == 0
            and (
                unmatched_prefix_len > 0
                or float(alignment_result.gap_ratio) >= 0.25
                or float(alignment_result.coverage) <= 0.70
            )
        )

        payload: Dict[str, Any] = {
            "job_id": ctx.job_id,
            "chunk_index": int(ctx.chunk_index),
            "layer": "L3-L6",
            "chosen_source": ctx.arbitration_result.chosen_source if ctx.arbitration_result else "",
            "arbitration_reason": ctx.arbitration_result.reason if ctx.arbitration_result else "",
            "chosen_clean_len": len(chosen_clean or ""),
            "l2_chosen_text_head": _clip_head(chosen_compact),
            "l2_chosen_text_tail": _clip_tail(chosen_compact),
            "l3_positions_total": len(punct_track.positions) if punct_track and punct_track.positions else 0,
            "l3_source": punct_track.source if punct_track else "",
            "l3_clean_text_match": bool(chosen_clean and punct_ref and chosen_clean == punct_ref),
            "l3_clean_ref_head": _clip_head(punct_compact),
            "l3_clean_ref_tail": _clip_tail(punct_compact),
            "is_l3_match_blocked_by_ref_mismatch": is_ref_mismatch,
            "l4_alignment_score": float(alignment_result.alignment_score),
            "l4_gap_ratio": float(alignment_result.gap_ratio),
            "l4_coverage": float(alignment_result.coverage),
            "l4_gap_positions": list(alignment_result.gap_positions),
            "l4_unmatched_prefix_len": int(unmatched_prefix_len),
            "is_cross_chunk_boundary_suspected": is_cross_chunk_boundary_suspected,
            "l5_injection_positions_total": int(injection_stats.get("injection_positions_total", 0) or 0),
            "l5_injection_unmatched_total": int(injection_stats.get("injection_unmatched_total", 0) or 0),
            "l5_injection_mapping_coverage": float(injection_stats.get("injection_mapping_coverage", 0.0) or 0.0),
            "l5_injection_blocked": bool(injection_stats.get("injection_blocked", 0.0)),
            "l6_sentence_count": len(final_sentences),
            "l6_split_mapping_coverage": float(split_stats.get("mapping_coverage", 0.0) or 0.0),
            "l6_split_writeback_ratio": float(split_stats.get("writeback_ratio", 0.0) or 0.0),
            "l6_split_writeback_used": bool(split_stats.get("writeback_used", 0.0)),
            "l6_split_writeback_blocked": bool(split_stats.get("writeback_blocked", 0.0)),
            "l6_sentence_texts": [str(sentence.text or "") for sentence in final_sentences],
        }
        append_debug_layer_diag_line(ctx.job_dir, payload, logger=self.logger)

    def _emit_layer_trace_full(
        self,
        ctx: ProcessingContext,
        *,
        tracks: TextTrackBundle,
        sv_result: Dict[str, Any],
        whisper_result: Dict[str, Any],
        arbitration_output: L2Output,
        punct_track: Optional[PunctTrack],
        alignment_result: AlignmentResult,
        words_for_split: Sequence[WordTimestamp],
        injection_stats: Dict[str, Any],
        split_stats: Dict[str, Any],
        final_sentences: Sequence[SentenceSegment],
    ) -> None:
        """输出 L0-L6 全量追踪（逐层 + 逐 token）到独立文件。"""

        def _serialize_punc_positions(positions: Optional[Sequence[PuncPosition]]) -> List[Dict[str, Any]]:
            if not positions:
                return []
            return [
                {
                    "char_index": int(pos.char_index),
                    "punctuation": str(pos.punctuation),
                    "confidence": float(pos.confidence),
                }
                for pos in positions
            ]

        def _serialize_word_timestamps(words: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
            if not words:
                return []
            items: List[Dict[str, Any]] = []
            for word in words:
                if isinstance(word, dict):
                    items.append(
                        {
                            "word": str(word.get("word", "") or ""),
                            "start": float(word.get("start", 0.0) or 0.0),
                            "end": float(word.get("end", 0.0) or 0.0),
                            "confidence": (
                                float(word.get("confidence"))
                                if word.get("confidence") is not None
                                else (
                                    float(word.get("probability"))
                                    if word.get("probability") is not None
                                    else None
                                )
                            ),
                        }
                    )
                    continue
                items.append(
                    {
                        "word": str(getattr(word, "word", "") or ""),
                        "start": float(getattr(word, "start", 0.0) or 0.0),
                        "end": float(getattr(word, "end", 0.0) or 0.0),
                        "confidence": (
                            float(getattr(word, "confidence"))
                            if getattr(word, "confidence", None) is not None
                            else None
                        ),
                        "confidence_source": str(getattr(word, "confidence_source", "") or ""),
                        "is_pseudo": bool(getattr(word, "is_pseudo", False)),
                    }
                )
            return items

        def _serialize_text_track(track: Optional[TextTrack]) -> Dict[str, Any]:
            if not track:
                return {}
            return {
                "source": str(track.source or ""),
                "language": str(track.language or ""),
                "raw_text": str(track.raw_text or ""),
                "text_itn_raw": str(track.text_itn_raw or ""),
                "text_clean": str(track.text_clean or ""),
                "mapping_coverage": float(track.mapping_coverage),
                "itn_fallback": bool(track.itn_fallback),
                "itn_fallback_reason": str(track.itn_fallback_reason or ""),
                "punct_positions": _serialize_punc_positions(track.punct_positions),
                "clean_to_word": list(track.clean_to_word or []),
                "word_confidences": list(track.word_confidences or []),
            }

        def _serialize_aligned_words(words: Sequence[Any]) -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []
            for word in words:
                rows.append(
                    {
                        "word": str(getattr(word, "word", "") or ""),
                        "start": float(getattr(word, "start", 0.0) or 0.0),
                        "end": float(getattr(word, "end", 0.0) or 0.0),
                        "alignment_status": str(getattr(getattr(word, "alignment_status", None), "value", "")),
                        "is_pseudo": bool(getattr(word, "is_pseudo", False)),
                        "sv_confidence": (
                            float(getattr(word, "sv_confidence"))
                            if getattr(word, "sv_confidence", None) is not None
                            else None
                        ),
                        "whisper_confidence": (
                            float(getattr(word, "whisper_confidence"))
                            if getattr(word, "whisper_confidence", None) is not None
                            else None
                        ),
                        "final_confidence": (
                            float(getattr(word, "final_confidence"))
                            if getattr(word, "final_confidence", None) is not None
                            else None
                        ),
                        "confidence_source": str(getattr(word, "confidence_source", "") or ""),
                    }
                )
            return rows

        def _serialize_sentences(sentences: Sequence[SentenceSegment]) -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []
            for sentence in sentences:
                rows.append(
                    {
                        "text": str(sentence.text or ""),
                        "text_clean": str(sentence.text_clean or ""),
                        "start": float(sentence.start),
                        "end": float(sentence.end),
                        "word_count": len(sentence.words or []),
                        "words": _serialize_word_timestamps(sentence.words),
                        "alignment_score": (
                            float(sentence.alignment_score)
                            if sentence.alignment_score is not None
                            else None
                        ),
                        "matched_ratio": (
                            float(sentence.matched_ratio)
                            if sentence.matched_ratio is not None
                            else None
                        ),
                        "confidence_source": str(sentence.confidence_source or ""),
                    }
                )
            return rows

        l3_input = getattr(ctx, "_trace_l3_input", {}) or {}
        l3_output = getattr(ctx, "_trace_l3_output", None)
        slow_raw = _serialize_word_timestamps(
            [
                word
                for seg in (whisper_result.get("raw_result", {}) or {}).get("segments", []) or []
                for word in (seg.get("words", []) or [])
                if isinstance(word, dict)
            ]
        )
        l0_batch_trace = getattr(ctx, "_trace_l0_batch_whisper", {}) or {}
        payload: Dict[str, Any] = {
            "job_id": ctx.job_id,
            "chunk_index": int(ctx.chunk_index),
            "layer": "L0-L6-FULL",
            "arbitration": {
                "chosen_source": arbitration_output.arbitration_result.chosen_source,
                "reason": arbitration_output.arbitration_result.reason,
                "coverage": float(arbitration_output.arbitration_result.coverage),
                "sv_score": float(arbitration_output.arbitration_result.sv_score),
                "wh_score": float(arbitration_output.arbitration_result.wh_score),
                "gap_positions": list(arbitration_output.arbitration_result.gap_positions or []),
            },
            "l0": {
                "bridge_batch": {
                    "batch_id": str(l0_batch_trace.get("batch_id", "")),
                    "chunk_indices": list(l0_batch_trace.get("chunk_indices", []) or []),
                    "prompt": str(l0_batch_trace.get("prompt", "")),
                    "flush_reason": str(l0_batch_trace.get("flush_reason", "")),
                    "whisper_text_raw": str(l0_batch_trace.get("whisper_text_raw", "")),
                    "whisper_text_clean": str(l0_batch_trace.get("whisper_text_clean", "")),
                    "raw_result": dict(l0_batch_trace.get("raw_result", {}) or {}),
                },
                "sv_result": {
                    "raw_text": str(sv_result.get("raw_text") or ""),
                    "text_clean": str(sv_result.get("text_clean") or ""),
                    "words": _serialize_word_timestamps(sv_result.get("words") or []),
                },
                "whisper_result": {
                    "text_raw": str(whisper_result.get("text_raw") or whisper_result.get("raw_text") or ""),
                    "min_clean_text": str(whisper_result.get("min_clean_text") or ""),
                    "text_clean": str(whisper_result.get("text_clean") or whisper_result.get("text") or ""),
                    "prompt": str(whisper_result.get("prompt") or ""),
                    "segments": list((whisper_result.get("raw_result", {}) or {}).get("segments", []) or []),
                    "words_from_segments": slow_raw,
                },
            },
            "l1": {
                "sv_track": _serialize_text_track(tracks.sv_track),
                "whisper_track": _serialize_text_track(tracks.whisper_track),
                "chosen_track": _serialize_text_track(tracks.chosen_track),
            },
            "l3": {
                "input": {
                    "chosen_source": l3_input.get("chosen_source"),
                    "word_timestamps": _serialize_word_timestamps(l3_input.get("word_timestamps") or []),
                    "sv_punct_source": {
                        "source": str(getattr(l3_input.get("sv_punct_source"), "source", "") or ""),
                        "clean_text_ref": str(getattr(l3_input.get("sv_punct_source"), "clean_text_ref", "") or ""),
                        "positions": _serialize_punc_positions(getattr(l3_input.get("sv_punct_source"), "positions", []) or []),
                    },
                    "wh_punct_source": {
                        "source": str(getattr(l3_input.get("wh_punct_source"), "source", "") or ""),
                        "clean_text_ref": str(getattr(l3_input.get("wh_punct_source"), "clean_text_ref", "") or ""),
                        "positions": _serialize_punc_positions(getattr(l3_input.get("wh_punct_source"), "positions", []) or []),
                    },
                },
                "output": {
                    "source": str(getattr(l3_output, "source", "") or ""),
                    "clean_text_ref": str(getattr(l3_output, "clean_text_ref", "") or ""),
                    "positions": _serialize_punc_positions(getattr(l3_output, "positions", []) or []),
                    "confidence_stats": dict(getattr(l3_output, "confidence_stats", {}) or {}),
                },
            },
            "l4": {
                "alignment_score": float(alignment_result.alignment_score),
                "gap_ratio": float(alignment_result.gap_ratio),
                "coverage": float(alignment_result.coverage),
                "gap_positions": list(alignment_result.gap_positions),
                "aligned_words": _serialize_aligned_words(alignment_result.aligned_words),
            },
            "l5": {
                "injection_stats": dict(injection_stats),
                "words_for_split": _serialize_word_timestamps(words_for_split),
            },
            "l6": {
                "split_stats": dict(split_stats),
                "final_sentences": _serialize_sentences(final_sentences),
            },
            "current_punct_track": {
                "source": str(punct_track.source if punct_track else ""),
                "clean_text_ref": str(punct_track.clean_text_ref if punct_track else ""),
                "positions": _serialize_punc_positions(punct_track.positions if punct_track else []),
            },
        }
        append_debug_layer_trace_line(ctx.job_dir, payload, logger=self.logger)

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
        self._validate_l0_result(whisper_result, source="slow")
        prompt = batch.prompt or None
        whisper_text_raw = str(whisper_result.get("raw_text") or "")
        whisper_result["text_raw"] = whisper_text_raw
        whisper_result["prompt"] = prompt
        base_text = whisper_result.get("min_clean_text") or whisper_text_raw
        whisper_result["text"] = self._whisper_sanitizer.sanitize_minimal(
            str(base_text or ""),
            prompt=prompt,
        )
        self._emit_whisper_debug(job_dir, batch, whisper_result, chunk_indices)
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
            setattr(
                ctx,
                "_trace_l0_batch_whisper",
                {
                    "batch_id": batch.batch_id,
                    "chunk_indices": list(chunk_indices),
                    "prompt": str(prompt or ""),
                    "flush_reason": str(getattr(batch, "flush_reason", "") or ""),
                    "whisper_text_raw": str(whisper_result.get("text_raw") or ""),
                    "whisper_text_clean": str(whisper_result.get("text") or ""),
                    "raw_result": dict(whisper_result.get("raw_result", {}) or {}),
                },
            )
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
                    "word_time_base": "batch_local",
                    "word_time_offset": float(batch_start),
                },
            )
            chunk_text_raw = str(chunk_result.get("raw_text") or chunk_result.get("text", "") or "")
            chunk_result["raw_text"] = chunk_text_raw
            chunk_result["text_raw"] = chunk_text_raw
            if not chunk_result.get("min_clean_text"):
                chunk_result["min_clean_text"] = self._whisper_sanitizer.sanitize_minimal(chunk_text_raw, prompt=None)
            chunk_result["text"] = str(chunk_result.get("min_clean_text") or "")
            raw_text_for_track = chunk_text_raw
            normalized_chunk = self._text_normalizer.normalize(chunk_result.get("text", ""), batch_language)
            chunk_result["text_itn_raw"] = normalized_chunk.text_itn_raw
            chunk_result["text_clean"] = normalized_chunk.text_clean
            chunk_result["text"] = normalized_chunk.text_clean or raw_text_for_track
            chunk_result["language"] = batch_language
            ctx.whisper_result = chunk_result
            tracks = self._ensure_text_tracks(ctx)
            tracks.whisper_track = self._build_text_track(
                raw_text_for_track,
                normalized_chunk,
                source="whisper",
                language=batch_language,
            )

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
                        self.logger.debug("[V3.1.0] SlowWorker 捕获暂停信号，继续排空队列")
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
                    if ctx.sv_result:
                        self._validate_l0_result(ctx.sv_result, source="fast", chunk_index=i)
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
                            self.logger.debug(f"[v3.1.0] FastWorker chunk {i} 完成后检测到待处理请求")

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
                        self.logger.debug(
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
                    self.logger.debug("[V3.1.0] FastWorker 已发送暂停结束信号，等待下游排空")
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
        idle_poll_interval = 0.5  # V3.2.0+dev.20260204.01: GPU 空闲检测轮询间隔（秒）

        try:
            while True:
                # 从队列取输入（ProcessingContext / BridgeBatch）
                try:
                    payload = await asyncio.wait_for(
                        self.queue_inter.get(),
                        timeout=idle_poll_interval,
                    )
                except asyncio.TimeoutError:
                    if self.bridge_controller and self.bridge_controller.should_flush_on_idle(
                        self.queue_inter.qsize()
                    ):
                        await self._flush_bridge_controller()
                    continue

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
                            self.logger.debug(
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
                        self._validate_l0_result(whisper_result, source="slow", chunk_index=chunk_index)
                        whisper_text_raw = str(whisper_result.get("raw_text") or "")
                        whisper_result["text_raw"] = whisper_text_raw
                        whisper_result["prompt"] = prompt
                        base_text = whisper_result.get("min_clean_text") or whisper_text_raw
                        whisper_result["text"] = self._whisper_sanitizer.sanitize_minimal(
                            str(base_text or ""),
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

                        self.logger.debug(
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
                            self.logger.debug(f"[v3.1.0] SlowWorker chunk {chunk_index} 完成后检测到待处理请求")

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
                            self.logger.debug("[V3.1.0] SlowWorker 捕获暂停信号，继续排空 queue_inter")
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
                self.logger.debug("[V3.1.0] SlowWorker 已完成排空，等待对齐阶段同步完成")

    # V3.2.0+dev.20260120.03: 对齐阶段下放到流水线
    async def _run_alignment_stage(self, ctx: ProcessingContext) -> None:
        """
        对齐阶段（流水线内执行）

        负责：
        1. 双流对齐（含降级兜底）
        2. 推送定稿
        3. 填充 ctx.final_sentences
        """
        chunk = ctx.audio_chunk

        # V3.10: 快速路径 - SlowWorker 跳过时直接使用 SenseVoice
        if ctx.whisper_skipped:
            if ctx.sv_result is None:
                raise ValueError("对齐阶段缺少 SenseVoice 推理结果")
            self.logger.debug(f"Chunk {ctx.chunk_index}: Whisper 跳过，直接使用 SenseVoice 定稿")
            final_sentences = self._finalize_sensevoice_only(ctx)
            ctx.final_sentences = final_sentences

            # L7 薄层输出：即便走快路，也统一通过输出层分发。
            l7_output = self._l7_processor.process(
                L7Input(
                    chunk_index=ctx.chunk_index,
                    sentence_segments=final_sentences,
                    injection_report={
                        "mapping_coverage": 0.0,
                        "mismatch_count": 0.0,
                        "error_code": "",
                        "blocked": 0.0,
                    },
                    segmentation_report={
                        "boundary_score_stats": {},
                        "forced_split_count": 0.0,
                        "error_code": "",
                    },
                )
            )
            ctx.finalization_metrics["l7_error_count"] = float(
                len(l7_output.output_payload.get("errors", []))
            )

            self.logger.debug(
                f"Chunk {ctx.chunk_index}: SenseVoice 定稿已推送 "
                f"({len(final_sentences)} 个句子) [智能复核-跳过]"
            )
            return

        # 阶段 1: 双流对齐（L2/L3/L4/L5/L6）
        self.logger.debug(f"Chunk {ctx.chunk_index}: 双流对齐")

        if ctx.whisper_result is None or ctx.sv_result is None:
            raise ValueError("对齐阶段缺少必要的推理结果")

        whisper_result = ctx.whisper_result
        sv_result = ctx.sv_result

        # V3.2.0+dev.20260202.07: Whisper 清洗增强 + 规范化重算
        self._apply_whisper_full_sanitize(ctx)

        arbitration_output = self._run_arbitration(ctx, sv_result, whisper_result)
        arbitration_result = arbitration_output.arbitration_result
        ctx.arbitration_result = arbitration_result
        tracks = self._ensure_text_tracks(ctx)
        if arbitration_output.chosen_text_track:
            tracks.chosen_track = arbitration_output.chosen_text_track
        elif tracks.chosen_track is None:
            fallback_track = tracks.sv_track or tracks.whisper_track
            if fallback_track:
                tracks.chosen_track = self._clone_text_track(fallback_track, source="chosen")

        chosen_text_clean = self._select_text_for_alignment(tracks.chosen_track)
        self._apply_arbitration_text(
            whisper_result,
            sv_result,
            arbitration_result.chosen_source,
            chosen_text_clean,
        )
        # V3.2.0+dev.20260204.10: 删除慢流补跑分支，定稿标点仅走统一 L3 入口
        punct_track: Optional[PunctTrack] = None
        # V3.2.0+dev.20260204.11: 若仲裁选择 fast 且快流已产出同一 clean_text_ref 的 PunctTrack，直接复用避免重复跑模型
        if (
            arbitration_result.chosen_source == "fast"
            and ctx.punct_track is not None
            and tracks.chosen_track is not None
            and ctx.punct_track.clean_text_ref == tracks.chosen_track.text_clean
        ):
            punct_track = ctx.punct_track
        else:
            punct_track = await self._run_punctuation_layer(
                ctx,
                sv_result,
                whisper_result,
                arbitration_result.chosen_source,
            )
        ctx.punct_track = punct_track
        punctuation_positions = punct_track.positions if punct_track and punct_track.positions else None
        punctuation_clean_text = punct_track.clean_text_ref if punct_track else None
        if tracks.chosen_track:
            if chosen_text_clean and tracks.chosen_track.text_clean != chosen_text_clean:
                tracks.chosen_track.text_clean = chosen_text_clean
            if (
                punctuation_positions
                and punctuation_clean_text
                and punctuation_clean_text == tracks.chosen_track.text_clean
            ):
                tracks.chosen_track.punct_positions = list(punctuation_positions)
            else:
                if punctuation_positions:
                    self.logger.debug("L3 标点 clean_text 不一致，清空回写位置")
                tracks.chosen_track.punct_positions = []
        if self.bridge_controller:
            self.bridge_controller.record_arbitration_result(arbitration_result)
        self.logger.debug(
            "Chunk %s: 仲裁完成 chosen_source=%s reason=%s coverage=%.2f",
            ctx.chunk_index,
            arbitration_result.chosen_source,
            arbitration_result.reason,
            arbitration_result.coverage,
        )

        # V3.2.0+dev.20260205.09: L4 对齐层（仅对齐与 gap 修复）
        sv_words = self._build_sv_word_timestamps(sv_result, chunk)
        legacy_run = self._run_l4_to_l6_once(
            tracks=tracks,
            sv_result=sv_result,
            whisper_result=whisper_result,
            sv_words=sv_words,
            punctuation_positions=punctuation_positions,
            punctuation_clean_text=punctuation_clean_text,
            variant="legacy",
            speaker_id=ctx.audio_chunk.primary_speaker_id if ctx.audio_chunk else None,
        )

        experiment_run: Optional[_Layer456RunResult] = None
        selected_variant = "legacy"
        selected_reason = "shadow_default" if self._dual_time_mode == "shadow" else "legacy_default"
        compare_payload: Optional[Dict[str, Any]] = None

        if self._is_dual_time_experiment_enabled:
            # Shadow/Active 都采用串行后处理：先 legacy，再 experiment，不并行占用 GPU。
            experiment_run = self._run_l4_to_l6_once(
                tracks=tracks,
                sv_result=sv_result,
                whisper_result=whisper_result,
                sv_words=sv_words,
                punctuation_positions=punctuation_positions,
                punctuation_clean_text=punctuation_clean_text,
                variant="experiment",
                speaker_id=ctx.audio_chunk.primary_speaker_id if ctx.audio_chunk else None,
            )
            compare_payload = self._build_dual_time_compare_payload(
                ctx=ctx,
                legacy_run=legacy_run,
                experiment_run=experiment_run,
            )
            if self._dual_time_mode == "active":
                selected_variant, selected_reason = self._select_dual_time_variant(
                    compare_payload=compare_payload,
                    experiment_run=experiment_run,
                )
            else:
                selected_variant = "legacy"
                selected_reason = "shadow_force_legacy"
            compare_payload["selected_variant"] = selected_variant
            compare_payload["selected_reason"] = selected_reason
            append_debug_dual_time_compare_line(ctx.job_dir, compare_payload, logger=self.logger)
            self._update_dual_time_summary(ctx, compare_payload)
            if self._is_dual_time_write_debug_srt:
                self._dual_time_legacy_sentences_by_chunk[ctx.chunk_index] = copy.deepcopy(legacy_run.final_sentences)
                self._dual_time_experiment_sentences_by_chunk[ctx.chunk_index] = copy.deepcopy(experiment_run.final_sentences)
                self._write_dual_time_debug_srt(ctx.job_dir)

        run_result = legacy_run
        if selected_variant == "experiment" and experiment_run is not None:
            run_result = experiment_run

        alignment_result = run_result.alignment_result
        words_for_split = run_result.words_for_split
        injection_stats = dict(run_result.injection_stats)
        split_stats = dict(run_result.split_stats)
        final_sentences = run_result.final_sentences


        ctx.finalization_metrics = {
            "coverage": alignment_result.coverage,
            "gap_ratio": alignment_result.gap_ratio,
            "alignment_score": alignment_result.alignment_score,
            "gap_positions": list(alignment_result.gap_positions),
            "gap_resolution": alignment_result.resolution.value if alignment_result.resolution else None,
            **injection_stats,
        }
        for key, value in split_stats.items():
            ctx.finalization_metrics[f"split_{key}"] = value

        if compare_payload is not None:
            ctx.finalization_metrics["dual_time_mode"] = self._dual_time_mode
            ctx.finalization_metrics["dual_time_selected_variant"] = selected_variant
            ctx.finalization_metrics["dual_time_selected_reason"] = selected_reason
            ctx.finalization_metrics["dual_time_boundary_f1"] = compare_payload["comparison"]["boundary_f1"]
            ctx.finalization_metrics["dual_time_word_start_mae_ms"] = compare_payload["comparison"]["word_start_mae_ms"]
            ctx.finalization_metrics["dual_time_legacy_time_source"] = compare_payload["legacy"]["alignment_time_source"]
            ctx.finalization_metrics["dual_time_experiment_time_source"] = compare_payload["experiment"]["alignment_time_source"]

        if tracks.chosen_track:
            mapping_cov = split_stats.get("mapping_coverage")
            if mapping_cov is not None:
                tracks.chosen_track.mapping_coverage = float(mapping_cov)
            ctx.finalization_metrics["itn_fallback"] = (
                1.0 if tracks.chosen_track.itn_fallback else 0.0
            )
        if tracks.sv_track:
            ctx.finalization_metrics["sv_itn_fallback"] = (
                1.0 if tracks.sv_track.itn_fallback else 0.0
            )
        if tracks.whisper_track:
            ctx.finalization_metrics["whisper_itn_fallback"] = (
                1.0 if tracks.whisper_track.itn_fallback else 0.0
            )

        self._emit_layer_diagnostics(
            ctx,
            tracks=tracks,
            punct_track=punct_track,
            alignment_result=alignment_result,
            injection_stats=injection_stats,
            split_stats=split_stats,
            final_sentences=final_sentences,
        )
        self._emit_layer_trace_full(
            ctx,
            tracks=tracks,
            sv_result=sv_result,
            whisper_result=whisper_result,
            arbitration_output=arbitration_output,
            punct_track=punct_track,
            alignment_result=alignment_result,
            words_for_split=words_for_split,
            injection_stats=injection_stats,
            split_stats=split_stats,
            final_sentences=final_sentences,
        )

        # V3.2.0+dev.20260204.08: L4/L5 补跑候选埋点（仅统计，不触发补跑）
        self._record_punct_retry_candidates(ctx)

        ctx.final_sentences = final_sentences
        if ctx.arbitration_result:
            ctx.arbitration_result.gap_positions = list(alignment_result.gap_positions)
            self.logger.debug(
                "Chunk %s: 仲裁统计 coverage=%.2f gap_positions=%s",
                ctx.chunk_index,
                ctx.arbitration_result.coverage,
                ctx.arbitration_result.gap_positions,
            )

        # 阶段 2: 推送定稿（使用 Chunk 级别的批量替换）
        self.logger.debug(
            f"Chunk {ctx.chunk_index}: 对齐完成 - "
            f"final_sentences={len(final_sentences)}, "
            f"whisper_text_len={len(whisper_result.get('text', ''))}, "
            f"sv_text_clean_len={len(sv_result.get('text_clean', ''))}"
        )

        # 定稿链内兜底：若 L6 仍未产出句子，基于 L6 输入词流构建单句，禁止回读草稿链。
        if not final_sentences:
            self.logger.error(
                f"Chunk {ctx.chunk_index}: 定稿句子为空！"
                f"Whisper文本长度={len(whisper_result.get('text', ''))}, "
                f"SenseVoice文本长度={len(sv_result.get('text_clean', ''))}"
            )
            fallback_sentence = self._build_final_fallback_sentence(words_for_split)
            if fallback_sentence is not None:
                final_sentences = [fallback_sentence]
                ctx.final_sentences = final_sentences
                split_stats["error_code"] = "E_L6_SPLIT_EMPTY"
                self.logger.warning(
                    "Chunk {}: 触发定稿链内单句兜底（禁止草稿回退）",
                    ctx.chunk_index,
                )

        l7_output = self._l7_processor.process(
            L7Input(
                chunk_index=ctx.chunk_index,
                sentence_segments=final_sentences,
                injection_report={
                    "mapping_coverage": float(injection_stats.get("injection_mapping_coverage", 0.0)),
                    "mismatch_count": float(injection_stats.get("injection_unmatched_total", 0.0)),
                    "error_code": str(injection_stats.get("injection_error_code", "") or ""),
                    "blocked": float(injection_stats.get("injection_blocked", 0.0)),
                },
                segmentation_report={
                    "boundary_score_stats": dict(split_stats),
                    "forced_split_count": float(split_stats.get("force_split_count", 0.0)),
                    "error_code": str(split_stats.get("error_code", "") or ""),
                },
            )
        )
        ctx.finalization_metrics["l7_error_count"] = float(
            len(l7_output.output_payload.get("errors", []))
        )

        self.logger.debug(
            f"Chunk {ctx.chunk_index}: 定稿已推送 "
            f"({len(final_sentences)} 个句子)"
        )

    def _run_l4_to_l6_once(
        self,
        *,
        tracks: TextTrackBundle,
        sv_result: Dict[str, Any],
        whisper_result: Dict[str, Any],
        sv_words: List[WordTimestamp],
        punctuation_positions: Optional[List[PuncPosition]],
        punctuation_clean_text: Optional[str],
        variant: str = "legacy",
        speaker_id: Optional[str] = None,
    ) -> _Layer456RunResult:
        """执行一次 L4-L6 主路径（仅 CPU 后处理）。"""
        time_words, time_source = self._resolve_alignment_time_words(
            variant=variant,
            sv_words=sv_words,
            whisper_result=whisper_result,
        )
        l4_output = self._l4_processor.process(
            L4Input(
                chosen_text_track=tracks.chosen_track,
                sv_words=time_words,
                vad_intervals=self._vad_intervals,
            )
        )
        alignment_result = l4_output.alignment_result

        detected_language = whisper_result.get("language") or (
            tracks.chosen_track.language if tracks.chosen_track else "auto"
        )
        self._final_splitter.set_language(detected_language)
        injection_stats: Dict[str, Any] = {
            "injection_positions_total": len(punctuation_positions or []),
            "injection_unmatched_total": 0,
            "injection_miss_ratio": 0.0,
            "injection_mapping_coverage": 0.0,
            "injection_blocked": 0.0,
            "injection_error_code": "",
        }
        l5_punct_track = PunctTrack(
            clean_text_ref=punctuation_clean_text or "",
            positions=list(punctuation_positions or []),
            source="l3",
            confidence_stats={},
        )
        l5_output = self._l5_processor.process(
            L5Input(
                alignment_result=alignment_result,
                punct_track=l5_punct_track,
                language=detected_language,
                speaker_id=speaker_id,  # V3.2.0+dev.20260207.03: P0 speaker 信号链
            )
        )
        injection_report = dict(l5_output.injection_report or {})
        injection_stats["injection_unmatched_total"] = int(injection_report.get("mismatch_count", 0.0))
        injection_stats["injection_miss_ratio"] = (
            injection_stats["injection_unmatched_total"] / max(len(punctuation_positions or []), 1)
        )
        injection_stats["injection_mapping_coverage"] = float(injection_report.get("mapping_coverage", 0.0))
        injection_stats["injection_blocked"] = float(injection_report.get("blocked", 0.0))
        injection_stats["injection_error_code"] = str(injection_report.get("error_code", "") or "")

        l6_output = self._l6_processor.process(
            L6Input(
                annotated_words=l5_output.annotated_words,
                vad_intervals=self._vad_intervals,
            ),
            stream_id=f"{variant}:{speaker_id or 'main'}",
            chunk_index=self._resolve_chunk_index_from_words(words=time_words),
            is_last_chunk=self._is_last_chunk_for_words(words=time_words),
        )
        words_for_split = list(l6_output.words_for_split)
        final_sentences = list(l6_output.sentence_segments)
        split_stats = dict(self._final_splitter.last_split_stats or {})
        split_stats.update(dict(l6_output.segmentation_report.get("boundary_score_stats", {})))
        segmentation_error = str(l6_output.segmentation_report.get("error_code", "") or "")
        if segmentation_error:
            split_stats["error_code"] = segmentation_error

        if self._final_grouper:
            final_sentences = self._final_grouper.group(final_sentences)

        matched_ratio = self._compute_matched_ratio(alignment_result.aligned_words)
        for sentence in final_sentences:
            sentence.source = TextSource.WHISPER_PATCH
            sentence.is_finalized = True
            sentence.is_draft = False
            sentence.alignment_score = alignment_result.alignment_score
            sentence.matched_ratio = matched_ratio
            sentence.whisper_text = whisper_result.get("text", "")
            sentence.sv_original_text = sv_result.get("text_clean")
            sentence.confidence_source = self._resolve_sentence_confidence_source(sentence.words)

        return _Layer456RunResult(
            alignment_result=alignment_result,
            words_for_split=words_for_split,
            injection_stats=injection_stats,
            split_stats=dict(split_stats),
            final_sentences=final_sentences,
            alignment_time_source=time_source,
            alignment_time_word_count=len(time_words),
        )

    def _resolve_alignment_time_words(
        self,
        *,
        variant: str,
        sv_words: List[WordTimestamp],
        whisper_result: Dict[str, Any],
    ) -> Tuple[List[WordTimestamp], str]:
        """根据轨道选择 L4 时间锚点词流。"""
        if variant != "experiment":
            return list(sv_words), "sv"

        slow_words = self._build_whisper_word_timestamps(whisper_result)
        if not slow_words:
            return list(sv_words), "sv_fallback_no_slow_words"

        resolved_source = "whisper_slow_words"
        if sv_words:
            sv_start = float(sv_words[0].start)
            sv_end = float(sv_words[-1].end)
            guard_start = sv_start - 0.5
            guard_end = sv_end + 0.5
            bounded = [
                word
                for word in slow_words
                if guard_start <= float(word.start) <= guard_end
                and guard_start <= float(word.end) <= guard_end
            ]
            if bounded:
                slow_words = bounded
            else:
                # 兜底：若慢流词仍未落入快流时间窗，按首词差值推断并平移时间基。
                inferred_offset = sv_start - float(slow_words[0].start)
                is_large_offset = abs(inferred_offset) >= 1.0
                if is_large_offset:
                    shifted = self._shift_word_timestamps(slow_words, inferred_offset)
                    bounded_shifted = [
                        word
                        for word in shifted
                        if guard_start <= float(word.start) <= guard_end
                        and guard_start <= float(word.end) <= guard_end
                    ]
                    if bounded_shifted:
                        slow_words = bounded_shifted
                        resolved_source = "whisper_slow_words_shifted"

        min_required = max(3, int(len(sv_words) * 0.4)) if sv_words else 3
        if len(slow_words) < min_required:
            return list(sv_words), "sv_fallback_insufficient_slow_words"

        return slow_words, resolved_source

    @staticmethod
    def _shift_word_timestamps(words: Sequence[WordTimestamp], offset: float) -> List[WordTimestamp]:
        """平移词时间戳，用于统一时间基。"""
        has_effective_offset = abs(offset) >= 1e-6
        if not has_effective_offset:
            return list(words)
        shifted: List[WordTimestamp] = []
        for item in words:
            shifted.append(
                WordTimestamp(
                    word=item.word,
                    start=float(item.start) + offset,
                    end=float(item.end) + offset,
                    confidence=item.confidence,
                    confidence_raw=item.confidence_raw,
                    confidence_display_raw=item.confidence_display_raw,
                    confidence_source=item.confidence_source,
                    token_type=item.token_type,
                    is_pseudo=item.is_pseudo,
                    warning_type=item.warning_type,
                    perplexity=item.perplexity,
                )
            )
        return shifted

    @staticmethod
    def _build_whisper_word_timestamps(whisper_result: Dict[str, Any]) -> List[WordTimestamp]:
        """将 Whisper 原始词流转换为 WordTimestamp（实验轨时间锚点）。"""
        words: List[WordTimestamp] = []
        raw = whisper_result.get("raw_result") if isinstance(whisper_result, dict) else None
        segments = raw.get("segments", []) if isinstance(raw, dict) else []
        time_base = str(whisper_result.get("word_time_base", "") or "").strip().lower()
        time_offset = float(whisper_result.get("word_time_offset", 0.0) or 0.0)
        has_time_offset = abs(time_offset) >= 1e-6
        is_relative_base = time_base in {"batch_local", "chunk_local", "relative"}
        should_apply_offset = has_time_offset and is_relative_base

        for segment in segments or []:
            for item in segment.get("words", []) or []:
                if not isinstance(item, dict):
                    continue
                token = str(item.get("word", "") or "").replace("▁", " ").strip()
                if not token:
                    continue
                start = float(item.get("start", 0.0) or 0.0)
                end = float(item.get("end", start) or start)
                if should_apply_offset:
                    start += time_offset
                    end += time_offset
                if end < start:
                    start, end = end, start
                if abs(end - start) < 1e-6:
                    continue
                words.append(
                    WordTimestamp(
                        word=token,
                        start=start,
                        end=end,
                        confidence=float(item.get("probability", 0.0) or 0.0),
                        confidence_source="slow",
                    )
                )
        words.sort(key=lambda row: (row.start, row.end))
        return words

    def _build_dual_time_compare_payload(
        self,
        *,
        ctx: ProcessingContext,
        legacy_run: _Layer456RunResult,
        experiment_run: _Layer456RunResult,
    ) -> Dict[str, Any]:
        """构建双轨实验 chunk 级对比数据。"""
        boundary_stats = self._compute_boundary_metrics(
            legacy_run.final_sentences,
            experiment_run.final_sentences,
            tolerance_sec=self._dual_time_boundary_tolerance_sec,
        )
        word_mae = self._compute_word_mae_ms(legacy_run.words_for_split, experiment_run.words_for_split)
        sentence_mae = self._compute_sentence_mae_ms(
            legacy_run.final_sentences,
            experiment_run.final_sentences,
        )
        return {
            "job_id": ctx.job_id,
            "chunk_index": int(ctx.chunk_index),
            "mode": self._dual_time_mode,
            "legacy": {
                "sentence_count": len(legacy_run.final_sentences),
                "alignment_score": float(legacy_run.alignment_result.alignment_score),
                "gap_ratio": float(legacy_run.alignment_result.gap_ratio),
                "sentence_texts": [str(sentence.text or "") for sentence in legacy_run.final_sentences],
                "alignment_time_source": legacy_run.alignment_time_source,
                "alignment_time_word_count": int(legacy_run.alignment_time_word_count),
            },
            "experiment": {
                "sentence_count": len(experiment_run.final_sentences),
                "alignment_score": float(experiment_run.alignment_result.alignment_score),
                "gap_ratio": float(experiment_run.alignment_result.gap_ratio),
                "sentence_texts": [str(sentence.text or "") for sentence in experiment_run.final_sentences],
                "alignment_time_source": experiment_run.alignment_time_source,
                "alignment_time_word_count": int(experiment_run.alignment_time_word_count),
            },
            "comparison": {
                **boundary_stats,
                **word_mae,
                **sentence_mae,
            },
        }

    def _select_dual_time_variant(
        self,
        *,
        compare_payload: Dict[str, Any],
        experiment_run: _Layer456RunResult,
    ) -> Tuple[str, str]:
        """Active 模式下选择最终输出轨道，门控失败自动回退 legacy。"""
        if not experiment_run.final_sentences:
            return "legacy", "active_gate_empty_experiment"
        boundary_f1 = float(compare_payload.get("comparison", {}).get("boundary_f1", 0.0) or 0.0)
        if boundary_f1 < self._dual_time_active_min_boundary_f1:
            return (
                "legacy",
                f"active_gate_boundary_f1<{self._dual_time_active_min_boundary_f1:.2f}",
            )
        return "experiment", "active_gate_pass"

    def _update_dual_time_summary(self, ctx: ProcessingContext, payload: Dict[str, Any]) -> None:
        """更新任务级双轨对比汇总并落盘。"""
        comparison = payload.get("comparison", {}) if isinstance(payload, dict) else {}
        acc = self._dual_time_compare_accumulator
        acc["chunk_count"] += 1
        acc["boundary_precision_sum"] += float(comparison.get("boundary_precision", 0.0) or 0.0)
        acc["boundary_recall_sum"] += float(comparison.get("boundary_recall", 0.0) or 0.0)
        acc["boundary_f1_sum"] += float(comparison.get("boundary_f1", 0.0) or 0.0)
        acc["word_start_mae_ms_sum"] += float(comparison.get("word_start_mae_ms", 0.0) or 0.0)
        acc["word_end_mae_ms_sum"] += float(comparison.get("word_end_mae_ms", 0.0) or 0.0)
        acc["sentence_start_mae_ms_sum"] += float(comparison.get("sentence_start_mae_ms", 0.0) or 0.0)
        acc["sentence_end_mae_ms_sum"] += float(comparison.get("sentence_end_mae_ms", 0.0) or 0.0)
        if payload.get("selected_variant") == "experiment":
            acc["selected_experiment_count"] += 1

        chunk_count = max(int(acc["chunk_count"]), 1)
        summary = {
            "job_id": ctx.job_id,
            "mode": self._dual_time_mode,
            "chunk_count": int(acc["chunk_count"]),
            "selected_experiment_count": int(acc["selected_experiment_count"]),
            "selected_experiment_ratio": float(acc["selected_experiment_count"] / chunk_count),
            "avg_boundary_precision": float(acc["boundary_precision_sum"] / chunk_count),
            "avg_boundary_recall": float(acc["boundary_recall_sum"] / chunk_count),
            "avg_boundary_f1": float(acc["boundary_f1_sum"] / chunk_count),
            "avg_word_start_mae_ms": float(acc["word_start_mae_ms_sum"] / chunk_count),
            "avg_word_end_mae_ms": float(acc["word_end_mae_ms_sum"] / chunk_count),
            "avg_sentence_start_mae_ms": float(acc["sentence_start_mae_ms_sum"] / chunk_count),
            "avg_sentence_end_mae_ms": float(acc["sentence_end_mae_ms_sum"] / chunk_count),
        }
        write_debug_json_payload(
            ctx.job_dir,
            "dual_time_eval_summary.json",
            summary,
            logger=self.logger,
        )

    def _write_dual_time_debug_srt(self, job_dir: Optional[Path]) -> None:
        """输出双轨实验快照 SRT（legacy/experiment）。"""
        if not job_dir:
            return
        try:
            debug_dir = job_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            legacy_srt = self._build_srt_from_chunk_map(self._dual_time_legacy_sentences_by_chunk)
            experiment_srt = self._build_srt_from_chunk_map(self._dual_time_experiment_sentences_by_chunk)
            (debug_dir / "dual_time_legacy.srt").write_text(legacy_srt, encoding="utf-8")
            (debug_dir / "dual_time_experiment.srt").write_text(experiment_srt, encoding="utf-8")
        except Exception as exc:
            self.logger.debug("写入双轨实验 SRT 失败（忽略）: %s", exc)

    @staticmethod
    def _build_srt_from_chunk_map(chunk_map: Dict[int, List[SentenceSegment]]) -> str:
        rows: List[SentenceSegment] = []
        for chunk_idx in sorted(chunk_map.keys()):
            rows.extend(sorted(chunk_map[chunk_idx], key=lambda item: (item.start, item.end)))
        lines: List[str] = []
        for index, sentence in enumerate(rows, 1):
            text = str(sentence.text or sentence.text_clean or "").strip()
            lines.append(str(index))
            lines.append(
                f"{format_srt_timestamp(float(sentence.start))} --> {format_srt_timestamp(float(sentence.end))}"
            )
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _extract_sentence_boundaries(sentences: Sequence[SentenceSegment]) -> List[float]:
        if len(sentences) <= 1:
            return []
        return [float(sentence.end) for sentence in sentences[:-1]]

    @classmethod
    def _compute_boundary_metrics(
        cls,
        legacy_sentences: Sequence[SentenceSegment],
        experiment_sentences: Sequence[SentenceSegment],
        *,
        tolerance_sec: float,
    ) -> Dict[str, float]:
        reference = cls._extract_sentence_boundaries(legacy_sentences)
        candidate = cls._extract_sentence_boundaries(experiment_sentences)
        matched = 0
        used = [False] * len(reference)
        for point in candidate:
            best_idx = -1
            best_delta = tolerance_sec + 1.0
            for idx, target in enumerate(reference):
                if used[idx]:
                    continue
                delta = abs(point - target)
                if delta <= tolerance_sec and delta < best_delta:
                    best_idx = idx
                    best_delta = delta
            if best_idx >= 0:
                used[best_idx] = True
                matched += 1

        if candidate:
            precision = matched / len(candidate)
        else:
            precision = 1.0 if not reference else 0.0
        if reference:
            recall = matched / len(reference)
        else:
            recall = 1.0 if not candidate else 0.0
        f1 = 0.0 if (precision + recall) <= 0.0 else (2.0 * precision * recall / (precision + recall))
        return {
            "boundary_matched": float(matched),
            "boundary_reference": float(len(reference)),
            "boundary_candidate": float(len(candidate)),
            "boundary_precision": float(precision),
            "boundary_recall": float(recall),
            "boundary_f1": float(f1),
        }

    @staticmethod
    def _compute_word_mae_ms(
        legacy_words: Sequence[WordTimestamp],
        experiment_words: Sequence[WordTimestamp],
    ) -> Dict[str, float]:
        overlap = min(len(legacy_words), len(experiment_words))
        if overlap <= 0:
            return {
                "word_overlap": 0.0,
                "word_start_mae_ms": 0.0,
                "word_end_mae_ms": 0.0,
            }
        start_errors = [
            abs(float(legacy_words[i].start) - float(experiment_words[i].start)) * 1000.0
            for i in range(overlap)
        ]
        end_errors = [
            abs(float(legacy_words[i].end) - float(experiment_words[i].end)) * 1000.0
            for i in range(overlap)
        ]
        return {
            "word_overlap": float(overlap),
            "word_start_mae_ms": float(sum(start_errors) / overlap),
            "word_end_mae_ms": float(sum(end_errors) / overlap),
        }

    @staticmethod
    def _compute_sentence_mae_ms(
        legacy_sentences: Sequence[SentenceSegment],
        experiment_sentences: Sequence[SentenceSegment],
    ) -> Dict[str, float]:
        overlap = min(len(legacy_sentences), len(experiment_sentences))
        if overlap <= 0:
            return {
                "sentence_overlap": 0.0,
                "sentence_start_mae_ms": 0.0,
                "sentence_end_mae_ms": 0.0,
            }
        start_errors = [
            abs(float(legacy_sentences[i].start) - float(experiment_sentences[i].start)) * 1000.0
            for i in range(overlap)
        ]
        end_errors = [
            abs(float(legacy_sentences[i].end) - float(experiment_sentences[i].end)) * 1000.0
            for i in range(overlap)
        ]
        return {
            "sentence_overlap": float(overlap),
            "sentence_start_mae_ms": float(sum(start_errors) / overlap),
            "sentence_end_mae_ms": float(sum(end_errors) / overlap),
        }

    def _finalize_sensevoice_only(self, ctx: ProcessingContext) -> List[SentenceSegment]:
        """Whisper 跳过时的定稿输出（仅使用 SenseVoice 结果）。"""
        if not ctx.sv_result or not ctx.audio_chunk:
            return []
        sv_words = self._build_sv_word_timestamps(ctx.sv_result, ctx.audio_chunk)
        annotated_words: List[AnnotatedWord] = [
            AnnotatedWord(
                word=word.word,
                start=word.start,
                end=word.end,
                trailing_punct="",
                confidence=word.confidence,
                confidence_source=word.confidence_source or "fast",
                speaker_id=None,
                track_id="main",
            )
            for word in sv_words
        ]
        l6_output = self._l6_processor.process(
            L6Input(
                annotated_words=annotated_words,
                vad_intervals=self._vad_intervals,
            ),
            stream_id="sensevoice_only",
            chunk_index=ctx.chunk_index,
            is_last_chunk=self._is_last_chunk_index(ctx.chunk_index),
        )
        sentences = list(l6_output.sentence_segments)
        if not sentences:
            fallback = self._build_final_fallback_sentence(l6_output.words_for_split)
            if fallback is not None:
                sentences = [fallback]
                self.logger.warning("Whisper 跳过路径触发 L6 单句兜底")
        for sentence in sentences:
            sentence.source = TextSource.SENSEVOICE
            sentence.is_finalized = True
            sentence.is_draft = False
            sentence.confidence_source = "fast"
        if self._final_grouper and sentences:
            sentences = self._final_grouper.group(sentences)
        return sentences

    @staticmethod
    def _build_sv_word_timestamps(
        sv_result: Dict[str, Any],
        chunk: AudioChunk,
    ) -> List[WordTimestamp]:
        """将 SenseVoice words 转换为 WordTimestamp（补齐 chunk 偏移）。"""
        words = sv_result.get("words", []) if isinstance(sv_result, dict) else []
        timestamps: List[WordTimestamp] = []
        for item in words or []:
            if isinstance(item, WordTimestamp):
                start = item.start + chunk.start
                end = item.end + chunk.start
                timestamps.append(
                    WordTimestamp(
                        word=item.word,
                        start=start,
                        end=end,
                        confidence=item.confidence,
                        confidence_raw=item.confidence_raw,
                        confidence_display_raw=item.confidence_display_raw,
                        confidence_source=item.confidence_source or "fast",
                        token_type=item.token_type,
                        is_pseudo=item.is_pseudo,
                    )
                )
                continue
            if not isinstance(item, dict):
                continue
            timestamps.append(
                WordTimestamp(
                    word=str(item.get("word", "")),
                    start=float(item.get("start", 0.0)) + chunk.start,
                    end=float(item.get("end", 0.0)) + chunk.start,
                    confidence=item.get("confidence"),
                    confidence_raw=item.get("confidence_raw"),
                    confidence_display_raw=item.get("confidence_display_raw"),
                    confidence_source=item.get("confidence_source") or "fast",
                    token_type=item.get("token_type"),
                    is_pseudo=bool(item.get("is_pseudo", False)),
                )
            )
        return timestamps

    def _resolve_chunk_index_from_words(self, *, words: Sequence[WordTimestamp]) -> Optional[int]:
        if not words or not self._audio_chunks_by_index:
            return None
        first_word_start = float(words[0].start)
        for chunk_index, chunk in self._audio_chunks_by_index.items():
            if float(chunk.start) <= first_word_start <= float(chunk.end) + 1e-6:
                return int(chunk_index)
        return None

    def _is_last_chunk_for_words(self, *, words: Sequence[WordTimestamp]) -> bool:
        chunk_index = self._resolve_chunk_index_from_words(words=words)
        if chunk_index is None:
            return False
        return self._is_last_chunk_index(chunk_index)

    def _is_last_chunk_index(self, chunk_index: int) -> bool:
        if not self._audio_chunks_by_index:
            return False
        return int(chunk_index) >= max(self._audio_chunks_by_index.keys())

    @staticmethod
    def _compute_matched_ratio(aligned_words: Sequence[Any]) -> float:
        """计算对齐匹配比例（MATCHED / 总词数）。"""
        if not aligned_words:
            return 0.0
        matched = sum(
            1 for word in aligned_words
            if getattr(word, "alignment_status", None) == AlignmentStatus.MATCHED
        )
        return matched / len(aligned_words)

    @staticmethod
    def _build_final_fallback_sentence(words_for_split: Sequence[WordTimestamp]) -> Optional[SentenceSegment]:
        """基于 L6 输入词流构建定稿单句兜底。"""
        if not words_for_split:
            return None
        text = "".join((word.word or "") for word in words_for_split).strip()
        if not text:
            return None
        confidences = [float(word.confidence) for word in words_for_split if word.confidence is not None]
        confidence = sum(confidences) / len(confidences) if confidences else None
        return SentenceSegment(
            text=text,
            text_clean=text,
            start=float(words_for_split[0].start),
            end=float(words_for_split[-1].end),
            words=list(words_for_split),
            confidence=confidence,
            source=TextSource.WHISPER_PATCH,
            is_draft=False,
            is_finalized=True,
        )

    @staticmethod
    def _resolve_sentence_confidence_source(words: Sequence[WordTimestamp]) -> str:
        """根据词级置信度来源汇总句级来源。"""
        if not words:
            return "unknown"
        sources = {
            word.confidence_source
            for word in words
            if word.confidence_source in ("fast", "slow")
        }
        if not sources:
            return "unknown"
        if len(sources) > 1:
            return "merged"
        return sources.pop()

    @staticmethod
    def _load_keep_sentence_end_punct() -> bool:
        runtime = get_model_runtime_config_service().get_effective_runtime_global()
        punct = runtime.get("effective", {}).get("punctuation", {})
        return bool(punct.get("keep_sentence_end_punct", False))

    @staticmethod
    def _strip_sentence_end_punct(sentences: List[SentenceSegment]) -> None:
        """清理句末标点（保留问号/感叹号）。"""
        for sentence in sentences:
            sentence.text = _strip_trailing_punct_smart(sentence.text)
            if sentence.text_clean:
                sentence.text_clean = _strip_trailing_punct_smart(sentence.text_clean)
            if sentence.words:
                last_word = sentence.words[-1]
                last_word.word = _strip_trailing_punct_smart(last_word.word)

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
                            self.logger.debug(f"[v3.1.0] 对齐阶段 chunk {chunk_index} 完成后检测到待处理请求")

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
                            self.logger.debug("[V3.1.0] 对齐阶段捕获暂停信号，继续排空 queue_final")
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
                self.logger.debug("[V3.1.0] 对齐阶段已排空所有上下文，等待上层暂停")

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












