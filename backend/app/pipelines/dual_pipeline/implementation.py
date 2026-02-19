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

V3.2.0+dev.20260215.08 更新：
- 新增评分层->裁决层 soft-cut 计划生成与透传（可配置开关）
- soft-cut 改为即时裁决路径，不再维护 deferred 跨 chunk 状态
- 新增短 turn 续段与词内触发守门，抑制“尾词前错切”
- 收紧句级跨speaker修复门限，避免误切连续文本
"""
import asyncio
import copy
import json
import logging
import time
from dataclasses import replace
from typing import List, Optional, Any, TYPE_CHECKING, Set, Dict, Tuple, Sequence
from pathlib import Path

import numpy as np

from app.core.asr.engine import ASREngine
from app.core.logging import resolve_loguru_logger
from app.core.thresholds import ThresholdConfig, needs_whisper_patch
from app.schemas.pipeline_context import ProcessingContext
from app.models.confidence_models import AlignmentStatus
from app.models.sensevoice_models import SentenceSegment, TextSource, WordTimestamp
from app.services.audio.chunk_engine import AudioChunk
from app.services.alignment.default_aligner import DefaultAligner, _strip_trailing_punct_smart
from app.services.alignment.text_normalizer import get_alignment_text_normalizer
from app.services.alignment.text_normalizer_processor import TextNormalizerProcessor
from app.services.alignment.types import (
    AlignedFacts,
    AnnotatedWord,
    AlignmentResult,
    FusedEvidence,
    L1Input,
    L2Input,
    L2Output,
    NormalizationResult,
    OutputLayerInput,
    OutputTrace,
    PunctuationPreInput,
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
from app.services.segmentation.boundary_mapper import WordBoundaryMapper
from app.services.segmentation.soft_cut import (
    AnchorCandidate,
    AnchorType,
    CutPlan,
    CutWindow,
    DecisionEngineConfig,
    EvidenceLevel,
    EvidenceBuilder,
    EvidenceBuilderConfig,
    SoftCutPlanProvider,
    SoftCutDecisionEngine,
    SpeakerChangeFact,
    SpeakerChangeTag,
    WindowDecisionContext,
    resolve_soft_cut_plan_provider,
)
from app.services.streaming_subtitle import get_streaming_subtitle_manager
from app.services.punctuation.fast_punctuation_pipeline import FastPunctuationPipeline
from app.services.punctuation.punctuation_processor import PunctuationProcessor
from app.services.punctuation.debug_utils import (
    append_debug_dual_time_compare_line,
    append_debug_layer_diag_line,
    append_debug_layer_trace_line,
    append_debug_m2_stage0_line,
    append_debug_whisper_line,
    write_debug_json_payload,
)
from app.services.punctuation.base import PunctuationResult, PuncPosition, SplitPoint
from app.services.punctuation.scheduler import get_punctuation_scheduler
from app.services.punctuation.final_splitter import FinalSplitter, FinalSplitConfig
from app.services.bridge.bridge_controller import BridgeController
from app.services.bridge.flush_policy import FlushPolicyConfig
from app.services.bridge.turn_group_builder import TurnGroupBuilder, TurnGroupEnvelope
from app.services.bridge.turn_group_models import TurnGroup
from app.services.punctuation.semantic_buffer import (
    PunctuationDecision,
    SemanticBuffer,
    SemanticBufferInput,
    SemanticChunk,
)
from app.services.text_protection import is_sentence_end_punct
from app.pipelines.dual_pipeline.services import (
    AlignLoopService,
    AlignmentStageService,
    DualTimeDiagnosticsService,
    FastLoopService,
    FullPipelineOrchestratorService,
    Layer456RunResult,
    SensevoiceOrchestratorService,
    SlowLoopService,
    TextflowFacadeService,
)
from app.services.timeline import (
    ClusterManagerConfig,
    PyannoteDiarizationConfig,
    PyannoteSegmentationConfig,
    SpeakerChunkInput,
    SpeakerTimelineService,
    SpeakerTimelineServiceConfig,
    TurnBuilderConfig,
)
from app.services.text_pipeline_config import TextPipelineConfig
from app.services.text_normalizer import TextNormalizer
from app.services.textflow import (
    CollectionAlignmentProcessor,
    CollectionFactBuilder,
    CollectionFactBuilderConfig,
    ScoringEvidenceFusion,
    ScoringEvidenceFusionConfig,
    ScoringSemanticInjectionProcessor,
    DecisionSegmentationProcessor,
    OutputLayerProcessor,
)
from app.services.whisper.whisper_prompt_policy import WhisperPromptPolicy
from app.services.whisper.whisper_text_sanitizer import WhisperTextSanitizer
from app.pipelines.workers import FastWorker, SlowWorker
from app.utils.cancellation_token import CancelledException, PausedException  # V3.1.0: 捕获取消/暂停异常

try:
    from app.services.semantic_grouper import SemanticGrouper, GroupConfig
except ImportError:
    SemanticGrouper = None
    GroupConfig = None

# v3.1.0: 导入取消令牌和异常
if TYPE_CHECKING:
    from app.utils.cancellation_token import CancellationToken
    from app.services.progress_emitter import ProgressEventEmitter  # V3.1.0
    from app.services.punctuation.service import PunctuationService
    from app.services.speaker_store import SpeakerStoreService


class AsyncDualPipelineKernel:
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
    # Why: pyannote 边界在快语速下会出现亚词级抖动，直接参与软切会导致单词级碎句。
    _SOFT_CUT_MIN_TURN_DURATION_SEC = 0.45
    _SOFT_CUT_SHORT_WORD_DURATION_SEC = 0.09
    _SOFT_CUT_SHORT_WORD_MIN_PAUSE_SEC = 0.30
    _SOFT_CUT_CHUNK_EDGE_GUARD_SEC = 0.18
    _SOFT_CUT_LEADING_WORD_MAX_DURATION_SEC = 0.12
    _SOFT_CUT_LEADING_WORD_MAX_GAP_SEC = 0.60
    _SOFT_CUT_LEADING_WORD_MAX_OFFSET_SEC = 0.08
    _SOFT_CUT_SHORT_TURN_CONTINUATION_SEC = 0.75
    _SOFT_CUT_SHORT_TURN_CONTINUATION_GAP_SEC = 0.80
    _SOFT_CUT_IN_WORD_BODY_MARGIN_SEC = 0.02
    _SOFT_CUT_PAUSE_ANCHOR_TRIGGER_SEC = 0.40
    _SOFT_CUT_PAUSE_ANCHOR_MIN_GAP_SEC = 1.00
    _SOFT_CUT_PUNCT_ANCHOR_CONFIDENCE_WEAK = 0.62
    _SOFT_CUT_PUNCT_ANCHOR_CONFIDENCE_STRONG = 0.82
    _SPEAKER_REPAIR_MIN_TURN_DURATION_SEC = 0.45
    _SPEAKER_REPAIR_STRONG_BREAK_MIN_PAUSE_SEC = 0.30
    _SPEAKER_REPAIR_SENTENCE_END_PUNCT = {"。", "！", "？", ".", "!", "?"}
    _SPEAKER_REPAIR_EDGE_GUARD_SEC = 0.15
    _WHISPER_REBALANCE_MAX_TAIL_WORDS = 2
    _WHISPER_REBALANCE_MAX_TAIL_GAP_SEC = 0.08
    _WHISPER_REBALANCE_MAX_TAIL_DURATION_SEC = 0.85
    _WHISPER_REBALANCE_SINGLE_CHAR_CARRY = {
        "是",
        "才",
        "经",
        "过",
        "直",
        "因",
        "检",
        "警",
        "一",
    }
    _WHISPER_REBALANCE_MULTI_CHAR_CARRY = {
        "因为",
        "以及",
        "直到",
        "经过",
        "检测",
        "都有",
        "然而",
        "但是",
        "不过",
    }

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
        bridge_controller: Optional[BridgeController] = None,
        debug_punctuation: bool = False,
        is_enable_speaker_detection: bool = True,
        is_enable_speaker_guided_split: bool = True,
        speaker_count: int = 0,
        speaker_min_count: int = 0,
        speaker_max_count: int = 0,
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
            layer="编排层",
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
        self._last_prompt_audio_end: Optional[float] = None
        self._whisper_prompt_policy = WhisperPromptPolicy(
            logger=self.logger,
            user_glossary=self.user_glossary,
        )
        self._enable_bridge_batches: bool = False
        self._consume_turn_groups_only: bool = True
        self._turn_group_builder = TurnGroupBuilder()
        self._runtime_checkpoint_service = None
        self._speaker_store_service: Optional["SpeakerStoreService"] = None
        self._context_cache: Dict[int, ProcessingContext] = {}
        self._audio_chunks_by_index: Dict[int, AudioChunk] = {}
        self._timeline_speaker_by_chunk_index: Dict[int, str] = {}
        self._timeline_turn_by_chunk_index: Dict[int, str] = {}
        self._timeline_turns: List[Any] = []
        self._vad_intervals: Optional[List[Tuple[float, float]]] = None
        self._full_audio_array: Optional[Any] = None
        self._full_audio_sr: int = 16000
        self._is_enable_speaker_detection = bool(is_enable_speaker_detection)
        self._is_enable_speaker_guided_split = bool(
            is_enable_speaker_detection and is_enable_speaker_guided_split
        )
        self._speaker_count = max(0, int(speaker_count or 0))
        self._speaker_min_count = max(0, int(speaker_min_count or 0))
        self._speaker_max_count = max(0, int(speaker_max_count or 0))
        if self._speaker_count > 0:
            self._speaker_min_count = 0
            self._speaker_max_count = 0
        elif self._speaker_min_count > 0 and self._speaker_max_count > 0:
            self._speaker_max_count = max(self._speaker_min_count, self._speaker_max_count)
        self.logger.info(
            "任务级 speaker 策略: detect={} guided_split={} speaker_count={} speaker_min={} speaker_max={}",
            self._is_enable_speaker_detection,
            self._is_enable_speaker_guided_split,
            self._speaker_count,
            self._speaker_min_count,
            self._speaker_max_count,
        )
        self.logger.info(
            "Whisper 提示词策略: {}",
            self._whisper_prompt_policy.describe_config(),
        )

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

        # V3.2.0+dev.20260211.04: Bridge 控制器（Phase 3 精简版，仅用于空闲 flush 判定）
        self.bridge_controller = bridge_controller
        if self.bridge_controller is None:
            self.bridge_controller = BridgeController(logger=self.logger)
        self._enable_bridge_batches = bool(
            self.semantic_buffer and not self.is_sensevoice_only
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
        # V3.2.0+dev.20260204.05: 标点前置域处理器
        self._punctuation_pre_processor = PunctuationProcessor(
            punctuation_service=self.punctuation_service,
            logger=self.logger,
        )
        # V3.2.0+dev.20260215.23: 阶段C接入四层统一目录入口（集合/评分/裁决/输出）。
        self._collection_processor = CollectionAlignmentProcessor(logger=self.logger)
        self._text_pipeline_config = TextPipelineConfig.from_runtime()
        self._m2_stage_config = self._text_pipeline_config.m2
        self._is_m2_enabled = bool(self._m2_stage_config.is_enabled)
        self._is_m2_nw_v2_enabled = bool(self._m2_stage_config.is_nw_v2_enabled)
        self._is_m2_time_mapping_enabled = bool(self._m2_stage_config.is_time_mapping_enabled)
        self._m2_shadow_sample_rate = float(self._m2_stage_config.shadow_sample_rate)
        self._m2_shadow_provider_class = str(
            self._m2_stage_config.shadow_provider_class or ""
        ).strip()
        self._fact_builder = CollectionFactBuilder(
            CollectionFactBuilderConfig(
                anchor_snap_tolerance_sec=0.22,
                is_enable_time_mapping=self._is_m2_time_mapping_enabled,
                time_axis_version="m2_nw_v2",
            )
        )
        self._segmentation_layer_config = self._text_pipeline_config.segmentation
        self._is_enable_soft_cut = bool(self._segmentation_layer_config.is_enable_soft_cut)
        self._is_enable_soft_cut_overlap_degrade = bool(
            self._segmentation_layer_config.is_enable_soft_cut_overlap_degrade
        )
        self._soft_cut_plan_provider_name = str(
            self._segmentation_layer_config.soft_cut_plan_provider
            or "m1_internal"
        ).strip()
        self._soft_cut_plan_provider_class = str(
            self._segmentation_layer_config.soft_cut_plan_provider_class
            or ""
        ).strip()
        self._soft_cut_priority_active_profile = str(
            self._segmentation_layer_config.soft_cut_priority_active_profile
            or "punct_boost_transition"
        ).strip()
        self._soft_cut_priority_profiles = dict(
            self._segmentation_layer_config.soft_cut_priority_profiles or {}
        )
        # 兼容旧 orchestrator 清理调用，实际已不再承载 deferred 状态。
        self._soft_cut_pending_deferred_by_stream: Dict[str, List[Any]] = {}
        # Why: 锚点距离惩罚过弱会让“远处大停顿”压过“近处词边界”，表现为尾词前错切。
        self._soft_cut_evidence_builder = EvidenceBuilder(
            config=EvidenceBuilderConfig(
                window_before_sec=0.40,
                window_after_sec=0.25,
                anchor_distance_penalty_factor=0.55,
                priority_active_profile=self._soft_cut_priority_active_profile,
                priority_profiles=self._soft_cut_priority_profiles,
            )
        )
        self._soft_cut_evidence_fusion = ScoringEvidenceFusion(
            config=ScoringEvidenceFusionConfig(
                priority_active_profile=self._soft_cut_priority_active_profile,
                priority_profiles=self._soft_cut_priority_profiles,
            )
        )
        self._soft_cut_decision_engine = SoftCutDecisionEngine(
            config=DecisionEngineConfig()
        )
        self._soft_cut_plan_provider: SoftCutPlanProvider = resolve_soft_cut_plan_provider(
            pipeline=self,
            provider_name=self._soft_cut_plan_provider_name,
            provider_class_path=self._soft_cut_plan_provider_class,
            logger=self.logger,
        )
        self._word_boundary_mapper = WordBoundaryMapper()
        self._soft_cut_semantic_conjunctions = {
            "但是",
            "然后",
            "所以",
            "不过",
            "but",
            "then",
            "so",
            "however",
        }
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
            is_force_split_on_sentence_end_punct=bool(
                self._segmentation_layer_config.is_force_split_on_sentence_end_punct
            ),
            min_mapping_coverage=max(
                0.0,
                min(1.0, float(self._segmentation_layer_config.final_min_mapping_coverage)),
            ),
        )
        self._final_splitter = FinalSplitter(final_split_config, logger=self.logger)
        if enable_semantic_grouping and SemanticGrouper is not None and GroupConfig is not None:
            final_group_config = GroupConfig(
                max_group_gap=2.0,
                max_group_duration=10.0,
                max_group_sentences=5,
                enable_overlap_detection=True,
            )
            self._final_grouper = SemanticGrouper(final_group_config)
        else:
            self._final_grouper = None
        self._scoring_processor = ScoringSemanticInjectionProcessor(
            logger=self.logger,
            min_mapping_coverage=final_split_config.min_mapping_coverage,
        )
        self._decision_processor = DecisionSegmentationProcessor(
            final_splitter=self._final_splitter,
            logger=self.logger,
            is_keep_sentence_end_punct=bool(
                self._segmentation_layer_config.is_keep_sentence_end_punct
            ),
            is_enable_soft_cut_overlap_degrade=bool(
                self._segmentation_layer_config.is_enable_soft_cut_overlap_degrade
            ),
            is_enable_speaker_guided_split=self._is_enable_speaker_guided_split,
        )
        self._output_processor = OutputLayerProcessor(
            subtitle_manager=self.subtitle_manager,
            speaker_store_service_getter=self._get_speaker_store_service,
            logger=self.logger,
        )
        self._fast_loop_service = FastLoopService(host=self)
        self._slow_loop_service = SlowLoopService(host=self)
        self._align_loop_service = AlignLoopService(host=self)
        self._full_pipeline_orchestrator_service = FullPipelineOrchestratorService(host=self)
        self._alignment_stage_service = AlignmentStageService(host=self)
        self._sensevoice_orchestrator_service = SensevoiceOrchestratorService(host=self)
        # 四层门面：收口 collection/scoring/decision/output 串联。
        self._textflow_facade_service = TextflowFacadeService(host=self)
        self._dual_time_diagnostics_service = DualTimeDiagnosticsService(host=self)
        self._keep_sentence_end_punct = bool(
            self._segmentation_layer_config.is_keep_sentence_end_punct
        )
        # V3.2.0+dev.20260206.02: 双轨实验配置（shadow/active 仅串行执行四层主链，避免并行占用 GPU）。
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
        if self._is_m2_enabled:
            self.logger.info(
                "M2阶段0观测开启: nw_v2_enable={} time_mapping_enable={} sample_rate={:.2f}",
                self._is_m2_nw_v2_enabled,
                self._is_m2_time_mapping_enabled,
                self._m2_shadow_sample_rate,
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
        """从 AudioChunk 中提取主说话人标识（兼容无 Timeline 场景）。"""
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

    def _resolve_speaker_id_for_chunk(self, chunk: AudioChunk) -> Optional[str]:
        """优先从 Timeline 域读取 speaker_id，缺失时回退到 chunk 主说话人。"""
        timeline_speaker = self._timeline_speaker_by_chunk_index.get(chunk.index)
        if timeline_speaker:
            return timeline_speaker
        return self._extract_primary_speaker_id(chunk)

    def _resolve_turn_id_for_chunk(self, chunk: AudioChunk) -> Optional[str]:
        """获取 chunk 对应 turn_id（Phase 2 内部追踪）。"""
        return self._timeline_turn_by_chunk_index.get(chunk.index)

    def _get_speaker_store_service(self) -> Optional[Any]:
        """懒加载 SpeakerStoreService，避免非任务目录场景初始化失败。"""
        if self._speaker_store_service is not None:
            return self._speaker_store_service
        if self._job_dir is None:
            return None
        try:
            from app.services.speaker_store import SpeakerStoreService

            self._speaker_store_service = SpeakerStoreService(job_dir=self._job_dir)
            return self._speaker_store_service
        except Exception as exc:
            self.logger.warning("SpeakerStoreService 初始化失败，降级继续: {}", exc)
            return None

    def _persist_timeline_turn_links(self, *, timeline: Any) -> None:
        """将 Timeline turn 结果写入 speaker_store.turn_speaker_links。"""
        speaker_store_service = self._get_speaker_store_service()
        if speaker_store_service is None:
            return

        turns = list(getattr(timeline, "turns", []) or [])
        if not turns:
            return

        payload_items: List[Dict[str, object]] = []
        for turn in turns:
            payload_items.append(
                {
                    "turn_id": str(getattr(turn, "turn_id", "") or ""),
                    "speaker_id": str(getattr(turn, "speaker_id", "") or "unknown"),
                    "block_id": str(getattr(turn, "block_id", "") or "unknown"),
                    "start": float(getattr(turn, "start", 0.0) or 0.0),
                    "end": float(getattr(turn, "end", 0.0) or 0.0),
                    "boundary_confidence": float(
                        getattr(turn, "boundary_confidence", 0.0) or 0.0
                    ),
                    "source": str(getattr(turn, "source", "") or "unknown"),
                }
            )

        if payload_items:
            speaker_store_service.upsert_turn_speaker_links(payload_items)

    async def _prepare_timeline_domain(
        self,
        *,
        audio_chunks: List[AudioChunk],
        full_audio_array: Optional[Any],
        full_audio_sr: int,
        job_dir: Optional[Path],
    ) -> None:
        """S-Epoch：在双流前预构建 SpeakerTimeline 并建立 chunk 映射。"""
        self._timeline_speaker_by_chunk_index = {}
        self._timeline_turn_by_chunk_index = {}
        self._timeline_turns = []

        if not audio_chunks:
            return

        timeline_audio_array, timeline_audio_sr, timeline_audio_source = (
            self._resolve_timeline_audio_input(
                audio_chunks=audio_chunks,
                full_audio_array=full_audio_array,
                full_audio_sr=full_audio_sr,
                job_dir=job_dir,
            )
        )
        if timeline_audio_array is None:
            self.logger.info("Timeline 域跳过：无可用音频输入")
            return

        if not self._is_enable_speaker_detection:
            self.logger.info("Timeline 域跳过：任务级已关闭说话人检测")
            return

        runtime_cfg = get_model_runtime_config_service().get_effective_runtime_global()
        timeline_cfg = runtime_cfg.get("effective", {}).get("timeline", {})
        raw_timeline_enabled = timeline_cfg.get("enabled", True)
        if isinstance(raw_timeline_enabled, str):
            timeline_enabled = raw_timeline_enabled.strip().lower() in {"1", "true", "yes", "on"}
        else:
            timeline_enabled = bool(raw_timeline_enabled)
        if not timeline_enabled:
            self.logger.info("Timeline 域全局开关为关闭，按任务级策略继续执行说话人检测")

        flush_cfg_raw = runtime_cfg.get("effective", {}).get("flush_policy", {})
        min_support_turns = int(timeline_cfg.get("min_support_turns", 2) or 2)
        min_total_duration = float(timeline_cfg.get("min_total_duration", 2.0) or 2.0)
        merge_similarity = float(
            timeline_cfg.get("merge_similarity_threshold", 0.88) or 0.88
        )
        raw_diarization_enabled = timeline_cfg.get("diarization_enabled", False)
        if isinstance(raw_diarization_enabled, str):
            global_diarization_enabled = raw_diarization_enabled.strip().lower() in {"1", "true", "yes", "on"}
        else:
            global_diarization_enabled = bool(raw_diarization_enabled)
        diarization_model_id = str(timeline_cfg.get("diarization_model_id", "") or "")
        diarization_local_path = str(timeline_cfg.get("diarization_local_path", "") or "")
        diarization_hf_token = timeline_cfg.get("diarization_hf_token")
        diarization_max_speakers_raw = timeline_cfg.get("diarization_max_speakers")
        diarization_min_speakers_raw = timeline_cfg.get("diarization_min_speakers")
        diarization_num_speakers_raw = timeline_cfg.get("diarization_num_speakers")

        def _to_optional_int(value: Any) -> Optional[int]:
            if value is None or value == "":
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                self.logger.warning("timeline diarization 配置非整数，已忽略: {}", value)
                return None
        diarization_num_speakers, diarization_min_speakers, diarization_max_speakers, diarization_enabled = (
            self._resolve_task_diarization_config(
                global_diarization_enabled=global_diarization_enabled,
                global_num_speakers=_to_optional_int(diarization_num_speakers_raw),
                global_min_speakers=_to_optional_int(diarization_min_speakers_raw),
                global_max_speakers=_to_optional_int(diarization_max_speakers_raw),
            )
        )
        boundary_threshold = float(
            timeline_cfg.get("segmentation_boundary_threshold", 0.55) or 0.55
        )
        min_boundary_interval = float(
            timeline_cfg.get("segmentation_min_boundary_interval_sec", 0.2) or 0.2
        )

        self._turn_group_builder = TurnGroupBuilder(
            flush_config=FlushPolicyConfig(
                min_audio_sec=float(flush_cfg_raw.get("min_audio_sec", 6.0) or 6.0),
                min_token_count=int(flush_cfg_raw.get("min_token_count", 40) or 40),
                max_wait_sec=float(flush_cfg_raw.get("max_wait_sec", 5.0) or 5.0),
                tail_idle_sec=float(flush_cfg_raw.get("tail_idle_sec", 1.0) or 1.0),
                long_pause_cut_sec=float(
                    timeline_cfg.get("long_pause_cut_sec", 1.8) or 1.8
                ),
            )
        )

        runtime_checkpoint_service = None
        if job_dir:
            try:
                from app.services.checkpoint import RuntimeCheckpointService

                runtime_checkpoint_service = RuntimeCheckpointService(job_dir=job_dir)
                self._runtime_checkpoint_service = runtime_checkpoint_service
            except Exception as exc:
                self.logger.warning("Timeline 断点服务初始化失败，降级为无断点模式: {}", exc)

        timeline_service = SpeakerTimelineService(
            config=SpeakerTimelineServiceConfig(
                cluster=ClusterManagerConfig(
                    min_support_turns=max(1, min_support_turns),
                    min_total_duration=max(0.0, min_total_duration),
                    merge_similarity_threshold=max(0.0, min(1.0, merge_similarity)),
                ),
                diarization=PyannoteDiarizationConfig(
                    enabled=diarization_enabled,
                    model_id=diarization_model_id,
                    local_path=diarization_local_path,
                    hf_token=diarization_hf_token,
                    prefer_device=str(timeline_cfg.get("device", "auto") or "auto"),
                    max_speakers=diarization_max_speakers,
                    min_speakers=diarization_min_speakers,
                    num_speakers=diarization_num_speakers,
                ),
                turn_builder=TurnBuilderConfig(),
                segmentation=PyannoteSegmentationConfig(
                    boundary_threshold=max(0.0, min(1.0, boundary_threshold)),
                    min_boundary_interval_sec=max(0.0, min_boundary_interval),
                ),
            ),
            runtime_checkpoint_service=runtime_checkpoint_service,
            cancellation_token=self.cancellation_token,
            logger=self.logger,
        )

        chunk_inputs = [
            SpeakerChunkInput(
                chunk_id=getattr(chunk, "chunk_id", None) or f"chunk-{chunk.index}",
                start=float(chunk.start),
                end=float(chunk.end),
                embedding=list(getattr(chunk, "speaker_embedding", None) or []),
                quality_score=float(getattr(chunk, "last_confidence", 1.0) or 1.0),
                is_overlap=False,
            )
            for chunk in audio_chunks
        ]

        try:
            timeline = await asyncio.to_thread(
                timeline_service.build_timeline,
                block_id=f"timeline-{self.job_id}",
                chunks=chunk_inputs,
                audio=timeline_audio_array,
                sample_rate=timeline_audio_sr,
            )
        except Exception as exc:
            self.logger.warning("Timeline 域构建失败，回退旧 speaker 透传: {}", exc)
            return

        if not timeline.turns:
            self.logger.info("Timeline 域未产生 turn，保持旧 speaker 透传")
            return
        self._timeline_turns = list(timeline.turns)

        try:
            self._persist_timeline_turn_links(timeline=timeline)
        except Exception as exc:
            self.logger.warning("Timeline turn 落库失败，继续执行: {}", exc)

        for chunk in audio_chunks:
            best_turn = self._select_best_turn_for_chunk(
                chunk_start=float(chunk.start),
                chunk_end=float(chunk.end),
                turns=timeline.turns,
            )
            if best_turn is None:
                continue
            self._timeline_speaker_by_chunk_index[chunk.index] = best_turn.speaker_id
            self._timeline_turn_by_chunk_index[chunk.index] = best_turn.turn_id

        self.logger.info(
            "Timeline 域构建完成: turns={} speakers={} chunk_bindings={} audio_source={}",
            len(timeline.turns),
            len(timeline.speakers),
            len(self._timeline_speaker_by_chunk_index),
            timeline_audio_source,
        )

    def _resolve_task_diarization_config(
        self,
        *,
        global_diarization_enabled: bool,
        global_num_speakers: Optional[int],
        global_min_speakers: Optional[int],
        global_max_speakers: Optional[int],
    ) -> Tuple[Optional[int], Optional[int], Optional[int], bool]:
        """
        合并全局与任务级说话人数配置。

        分层约定：
        - 全局：模型/设备/阈值默认值；
        - 任务级：是否检测、是否 speaker 介入切分、人数策略（固定人数/自动）。
        """
        task_speaker_count = max(0, int(self._speaker_count or 0))
        task_speaker_min = max(0, int(self._speaker_min_count or 0))
        task_speaker_max = max(0, int(self._speaker_max_count or 0))
        if task_speaker_count > 0:
            return task_speaker_count, None, None, True

        min_speakers = task_speaker_min if task_speaker_min > 0 else global_min_speakers
        max_speakers = task_speaker_max if task_speaker_max > 0 else global_max_speakers
        num_speakers = global_num_speakers
        if min_speakers is not None and max_speakers is not None and max_speakers < min_speakers:
            max_speakers = min_speakers

        is_has_task_range = task_speaker_min > 0 or task_speaker_max > 0
        diarization_enabled = bool(global_diarization_enabled or is_has_task_range)
        return num_speakers, min_speakers, max_speakers, diarization_enabled

    def _resolve_timeline_audio_input(
        self,
        *,
        audio_chunks: Sequence[AudioChunk],
        full_audio_array: Optional[Any],
        full_audio_sr: int,
        job_dir: Optional[Path],
    ) -> Tuple[Optional[np.ndarray], int, str]:
        """为 Timeline 选择音频输入，优先使用分离后音轨。"""
        normalized_full_audio: Optional[np.ndarray] = None
        if full_audio_array is not None:
            normalized_full_audio = np.asarray(full_audio_array, dtype=np.float32).reshape(-1)

        is_has_separated_chunks = any(bool(getattr(chunk, "is_separated", False)) for chunk in audio_chunks)
        if not is_has_separated_chunks:
            return normalized_full_audio, int(full_audio_sr), "original_full_audio"

        if job_dir is not None:
            separated_path = (
                Path(job_dir) / "cache_preprocess" / "separation_full" / "vocals_full.wav"
            )
            if separated_path.exists():
                try:
                    import soundfile as sf

                    loaded_audio, loaded_sr = sf.read(str(separated_path), dtype="float32")
                    separated_audio = np.asarray(loaded_audio, dtype=np.float32)
                    if separated_audio.ndim > 1:
                        separated_audio = separated_audio.mean(axis=1)
                    return separated_audio.reshape(-1), int(loaded_sr), "separation_full_cache"
                except Exception as exc:
                    self.logger.warning("读取分离整轨失败，回退 chunk 重建: {}", exc)

        rebuilt_audio = self._rebuild_timeline_audio_from_chunks(
            audio_chunks=audio_chunks,
            sample_rate=int(full_audio_sr),
            fallback_audio=normalized_full_audio,
        )
        if rebuilt_audio is not None:
            return rebuilt_audio, int(full_audio_sr), "chunk_audio_reconstructed"

        return normalized_full_audio, int(full_audio_sr), "original_full_audio_fallback"

    def _rebuild_timeline_audio_from_chunks(
        self,
        *,
        audio_chunks: Sequence[AudioChunk],
        sample_rate: int,
        fallback_audio: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """基于 chunk.audio 重建时间轴音轨（优先携带分离后人声）。"""
        effective_sr = max(1, int(sample_rate))
        target_length = int(len(fallback_audio)) if fallback_audio is not None else 0

        for chunk in audio_chunks:
            chunk_audio = getattr(chunk, "audio", None)
            if chunk_audio is None:
                continue
            chunk_audio_np = np.asarray(chunk_audio, dtype=np.float32).reshape(-1)
            if chunk_audio_np.size == 0:
                continue
            chunk_start = int(round(float(getattr(chunk, "start", 0.0) or 0.0) * effective_sr))
            if chunk_start < 0:
                chunk_start = 0
            target_length = max(target_length, chunk_start + int(chunk_audio_np.size))

        if target_length <= 0:
            return fallback_audio

        merged_audio = np.zeros(target_length, dtype=np.float32)
        overlap_counter = np.zeros(target_length, dtype=np.float32)
        valid_chunk_count = 0

        for chunk in audio_chunks:
            chunk_audio = getattr(chunk, "audio", None)
            if chunk_audio is None:
                continue
            chunk_audio_np = np.asarray(chunk_audio, dtype=np.float32).reshape(-1)
            if chunk_audio_np.size == 0:
                continue

            chunk_start = int(round(float(getattr(chunk, "start", 0.0) or 0.0) * effective_sr))
            if chunk_start < 0:
                chunk_start = 0
            chunk_end = min(target_length, chunk_start + int(chunk_audio_np.size))
            if chunk_end <= chunk_start:
                continue

            copy_size = chunk_end - chunk_start
            merged_audio[chunk_start:chunk_end] += chunk_audio_np[:copy_size]
            overlap_counter[chunk_start:chunk_end] += 1.0
            valid_chunk_count += 1

        if valid_chunk_count <= 0:
            return fallback_audio

        is_has_overlap = overlap_counter > 1.0
        merged_audio[is_has_overlap] = merged_audio[is_has_overlap] / overlap_counter[is_has_overlap]
        return merged_audio

    @staticmethod
    def _select_best_turn_for_chunk(
        *,
        chunk_start: float,
        chunk_end: float,
        turns: Sequence[Any],
        require_overlap: bool = False,
    ) -> Optional[Any]:
        """按时间重叠优先选取 chunk 对应 turn。"""
        if not turns:
            return None

        best_turn = None
        best_overlap = -1.0
        chunk_mid = (chunk_start + chunk_end) / 2.0
        best_distance = float("inf")

        for turn in turns:
            overlap = max(0.0, min(chunk_end, float(turn.end)) - max(chunk_start, float(turn.start)))
            turn_mid = (float(turn.start) + float(turn.end)) / 2.0
            distance = abs(turn_mid - chunk_mid)
            if overlap > best_overlap or (overlap == best_overlap and distance < best_distance):
                best_turn = turn
                best_overlap = overlap
                best_distance = distance

        if require_overlap and best_overlap <= 0.0:
            return None
        return best_turn

    def _assign_sentence_identity_by_timeline_overlap(
        self,
        sentences: Sequence[SentenceSegment],
        *,
        fallback_chunk: Optional[AudioChunk] = None,
    ) -> None:
        """
        按句级时间窗与 Timeline turn 重叠绑定 speaker/turn。

        Why: Chunk 级绑定在多人交替发言时会误绑定整段，句级重叠可显著提升切换准确率。
        说明：跨 speaker 残留修复已迁移到裁决层（DecisionSegmentationProcessor）内完成。
        """
        if not sentences:
            return

        turns = list(self._timeline_turns or [])
        fallback_speaker = self._resolve_speaker_id_for_chunk(fallback_chunk) if fallback_chunk else "unknown"
        fallback_turn = self._resolve_turn_id_for_chunk(fallback_chunk) if fallback_chunk else None

        for sentence in sentences:
            sentence_start = float(getattr(sentence, "start", 0.0) or 0.0)
            sentence_end = float(getattr(sentence, "end", 0.0) or 0.0)
            if sentence_end <= sentence_start:
                sentence_end = sentence_start + 1e-3

            best_turn = self._select_best_turn_for_chunk(
                chunk_start=sentence_start,
                chunk_end=sentence_end,
                turns=turns,
                require_overlap=True,
            )
            if best_turn is not None:
                sentence.speaker_id = str(getattr(best_turn, "speaker_id", "unknown") or "unknown")
                sentence.turn_id = str(getattr(best_turn, "turn_id", "") or "") or None
                continue

            if sentence.speaker_id is None:
                sentence.speaker_id = fallback_speaker
            if sentence.turn_id is None:
                sentence.turn_id = fallback_turn

    def _repair_cross_speaker_sentences_once(
        self,
        *,
        sentences: List[SentenceSegment],
        turns: Sequence[Any],
    ) -> int:
        """
        单次闭环修复：仅对“句内跨 speaker”句子做一次局部重切。

        Why:
        - 触发点来自 turn 边界，但落点必须是词边界，禁止切断词。
        - 不做二次全局复跑，避免重复计算和结果抖动。
        """
        if not sentences or len(turns) <= 1:
            return 0

        repaired_sentences: List[SentenceSegment] = []
        repaired_split_count = 0
        ordered_turns = sorted(
            list(turns),
            key=lambda item: (
                float(getattr(item, "start", 0.0)),
                float(getattr(item, "end", 0.0)),
            ),
        )
        for sentence in sentences:
            repaired_parts = self._split_sentence_by_turn_boundaries_once(
                sentence=sentence,
                ordered_turns=ordered_turns,
            )
            repaired_sentences.extend(repaired_parts)
            if len(repaired_parts) > 1:
                repaired_split_count += len(repaired_parts) - 1

        if repaired_split_count > 0:
            sentences[:] = repaired_sentences
        return repaired_split_count

    def _split_sentence_by_turn_boundaries_once(
        self,
        *,
        sentence: SentenceSegment,
        ordered_turns: Sequence[Any],
    ) -> List[SentenceSegment]:
        words = list(sentence.words or [])
        if len(words) <= 1:
            return [sentence]

        sentence_start = float(getattr(sentence, "start", 0.0) or 0.0)
        sentence_end = float(getattr(sentence, "end", 0.0) or 0.0)
        if sentence_end <= sentence_start:
            return [sentence]

        overlap_turns = [
            turn
            for turn in ordered_turns
            if min(sentence_end, float(getattr(turn, "end", 0.0))) > max(sentence_start, float(getattr(turn, "start", 0.0)))
        ]
        speaker_set = {
            str(getattr(turn, "speaker_id", "") or "").strip()
            for turn in overlap_turns
            if str(getattr(turn, "speaker_id", "") or "").strip()
        }
        if len(speaker_set) <= 1:
            return [sentence]
        if (
            self._is_enable_soft_cut_overlap_degrade
            and self._is_overlap_turn_window(turns=overlap_turns)
        ):
            # Why: 重叠抢话场景的 diarization 边界噪声高，降级为“不做句级硬切”更稳妥。
            return [sentence]

        boundary_times = self._collect_sentence_turn_change_times(
            words=words,
            overlap_turns=overlap_turns,
            sentence_start=sentence_start,
            sentence_end=sentence_end,
        )
        if not boundary_times:
            return [sentence]

        boundary_by_split_index: Dict[int, Tuple[float, float]] = {}
        for boundary_time in boundary_times:
            split_idx, score = self._select_best_word_boundary_for_turn_change(
                words=words,
                boundary_time=boundary_time,
            )
            if split_idx is None:
                continue
            existing = boundary_by_split_index.get(split_idx)
            if existing is None or score > existing[1]:
                boundary_by_split_index[split_idx] = (boundary_time, score)

        if not boundary_by_split_index:
            return [sentence]

        selected_split_indices = sorted(
            split_idx
            for split_idx in boundary_by_split_index.keys()
            if 0 <= split_idx < len(words) - 1
        )
        if not selected_split_indices:
            return [sentence]

        rebuilt_sentences: List[SentenceSegment] = []
        start_idx = 0
        used_split_indices: List[int] = []
        for split_idx in selected_split_indices:
            is_has_room = split_idx >= start_idx and split_idx < len(words) - 1
            if not is_has_room:
                continue
            rebuilt_sentences.append(
                self._final_splitter._build_sentence(words, start_idx, split_idx)
            )
            used_split_indices.append(split_idx)
            start_idx = split_idx + 1
        if start_idx < len(words):
            rebuilt_sentences.append(
                self._final_splitter._build_sentence(words, start_idx, len(words) - 1)
            )

        if len(rebuilt_sentences) <= 1:
            return [sentence]

        # 用 turn 边界时间吸附相邻句子的 start/end，减少“跨speaker时间窗”残留。
        for idx, split_idx in enumerate(used_split_indices):
            if idx + 1 >= len(rebuilt_sentences):
                break
            boundary_time = float(boundary_by_split_index[split_idx][0])
            left_sentence = rebuilt_sentences[idx]
            right_sentence = rebuilt_sentences[idx + 1]
            left_start = float(getattr(left_sentence, "start", 0.0) or 0.0)
            right_end = float(getattr(right_sentence, "end", 0.0) or 0.0)
            if right_end <= left_start:
                continue
            snapped = max(left_start + 1e-3, min(right_end - 1e-3, boundary_time))
            left_sentence.end = max(left_start + 1e-3, min(float(left_sentence.end), snapped))
            right_sentence.start = min(right_end - 1e-3, max(float(right_sentence.start), snapped))
            if right_sentence.start <= left_sentence.end:
                right_sentence.start = min(right_end - 1e-3, left_sentence.end + 1e-3)
            if right_sentence.end <= right_sentence.start:
                right_sentence.end = right_sentence.start + 1e-3

        for rebuilt in rebuilt_sentences:
            self._copy_sentence_runtime_metadata(
                source=sentence,
                target=rebuilt,
            )
            # 交给后续句级 timeline 绑定重新计算最终 speaker/turn。
            rebuilt.speaker_id = None
            rebuilt.turn_id = None
        return rebuilt_sentences

    def _collect_sentence_turn_change_times(
        self,
        *,
        words: Sequence[WordTimestamp],
        overlap_turns: Sequence[Any],
        sentence_start: float,
        sentence_end: float,
    ) -> List[float]:
        if len(overlap_turns) <= 1:
            return []

        ordered = sorted(
            list(overlap_turns),
            key=lambda item: (
                float(getattr(item, "start", 0.0)),
                float(getattr(item, "end", 0.0)),
            ),
        )
        change_times: List[float] = []
        for idx in range(1, len(ordered)):
            left = ordered[idx - 1]
            right = ordered[idx]
            left_speaker = str(getattr(left, "speaker_id", "") or "").strip()
            right_speaker = str(getattr(right, "speaker_id", "") or "").strip()
            if not left_speaker or not right_speaker or left_speaker == right_speaker:
                continue
            left_start_raw = getattr(left, "start", None)
            left_start = float(left_start_raw) if left_start_raw is not None else 0.0
            left_end_raw = getattr(left, "end", None)
            left_end = float(left_end_raw) if left_end_raw is not None else left_start
            right_start_raw = getattr(right, "start", None)
            right_start = float(right_start_raw) if right_start_raw is not None else 0.0
            right_end_raw = getattr(right, "end", None)
            right_end = float(right_end_raw) if right_end_raw is not None else right_start
            left_duration = max(0.0, left_end - left_start)
            right_duration = max(0.0, right_end - right_start)
            # Why: 极短 turn 常由分割抖动导致，不应直接触发句内重切。
            if min(left_duration, right_duration) < self._SPEAKER_REPAIR_MIN_TURN_DURATION_SEC:
                continue
            if self._is_short_turn_followed_by_same_speaker_continuation(
                turns=ordered,
                index=idx,
                right_speaker=right_speaker,
                right_end=right_end,
                right_duration=right_duration,
            ):
                # Why: 短 turn 若被同 speaker 续段承接，多为 pyannote 抖动；仅在强断句证据时放行。
                if not self._is_repair_boundary_strong_break(
                    words=words,
                    boundary_time=right_start,
                ):
                    continue
            boundary_time = right_start
            # Why: 句首/句尾附近的切点通常由 chunk 边缘时间误差触发，避免把首词单独切出。
            if (
                boundary_time <= sentence_start + self._SPEAKER_REPAIR_EDGE_GUARD_SEC
                or boundary_time >= sentence_end - self._SPEAKER_REPAIR_EDGE_GUARD_SEC
            ):
                continue
            if not (sentence_start < boundary_time < sentence_end):
                continue
            change_times.append(boundary_time)

        deduped: List[float] = []
        for value in sorted(change_times):
            if not deduped or abs(value - deduped[-1]) > 0.08:
                deduped.append(value)
        return deduped

    def _is_short_turn_followed_by_same_speaker_continuation(
        self,
        *,
        turns: Sequence[Any],
        index: int,
        right_speaker: str,
        right_end: float,
        right_duration: float,
    ) -> bool:
        if right_duration >= self._SOFT_CUT_SHORT_TURN_CONTINUATION_SEC:
            return False
        if index + 1 >= len(turns):
            return False

        next_turn = turns[index + 1]
        next_speaker = str(getattr(next_turn, "speaker_id", "") or "").strip()
        if not next_speaker or next_speaker != right_speaker:
            return False
        next_start_raw = getattr(next_turn, "start", None)
        next_start = float(next_start_raw) if next_start_raw is not None else right_end
        continuation_gap = max(0.0, next_start - right_end)
        return continuation_gap <= self._SOFT_CUT_SHORT_TURN_CONTINUATION_GAP_SEC

    def _is_repair_boundary_strong_break(
        self,
        *,
        words: Sequence[WordTimestamp],
        boundary_time: float,
    ) -> bool:
        selection = self._word_boundary_mapper.select_best_boundary(
            words=words,
            event_time=boundary_time,
        )
        if selection is None:
            return False

        split_idx = int(selection.split_idx)
        if split_idx < 0 or split_idx >= len(words) - 1:
            return False

        left_word = words[split_idx]
        right_word = words[split_idx + 1]
        left_text = str(getattr(left_word, "word", "") or "").strip()
        if is_sentence_end_punct(
            left_text,
            str(getattr(right_word, "word", "") or "").strip(),
            sentence_end_chars=tuple(self._SPEAKER_REPAIR_SENTENCE_END_PUNCT),
        ):
            return True

        left_end = float(getattr(left_word, "end", 0.0) or 0.0)
        right_start = float(getattr(right_word, "start", left_end) or left_end)
        pause_duration = max(0.0, right_start - left_end)
        return pause_duration >= self._SPEAKER_REPAIR_STRONG_BREAK_MIN_PAUSE_SEC

    def _select_best_word_boundary_for_turn_change(
        self,
        *,
        words: Sequence[WordTimestamp],
        boundary_time: float,
    ) -> Tuple[Optional[int], float]:
        if len(words) <= 1:
            return None, 0.0

        ranked = self._word_boundary_mapper.rank_boundaries(
            words=words,
            event_time=boundary_time,
        )
        if not ranked:
            return None, 0.0

        best_idx: Optional[int] = None
        best_score = float("-inf")
        for candidate in ranked[:3]:
            local_score = self._score_word_boundary_candidate(
                words=words,
                split_idx=candidate.split_idx,
                boundary_time=boundary_time,
            )
            score = (candidate.score * 1.2) + local_score
            if score > best_score:
                best_score = score
                best_idx = candidate.split_idx

        if best_idx is not None and best_score > -0.8:
            return best_idx, best_score
        return ranked[0].split_idx, ranked[0].score

    def _score_word_boundary_candidate(
        self,
        *,
        words: Sequence[WordTimestamp],
        split_idx: int,
        boundary_time: float,
    ) -> float:
        left_end = float(getattr(words[split_idx], "end", 0.0) or 0.0)
        right_start = float(getattr(words[split_idx + 1], "start", left_end) or left_end)
        gap_left = min(left_end, right_start)
        gap_right = max(left_end, right_start)
        is_in_gap = gap_left <= boundary_time <= gap_right
        if is_in_gap:
            delta = 0.0
        else:
            delta = min(
                abs(left_end - boundary_time),
                abs(right_start - boundary_time),
            )
        distance_score = max(0.0, 1.0 - (delta / 0.8))
        pause_duration = max(0.0, right_start - left_end)
        pause_score = min(1.0, pause_duration / 0.35)
        valley_score = self._score_boundary_vad_valley((left_end + right_start) / 2.0)

        left_ratio, left_speaker = self._resolve_word_speaker_purity(
            words=words,
            start_idx=0,
            end_idx=split_idx,
        )
        right_ratio, right_speaker = self._resolve_word_speaker_purity(
            words=words,
            start_idx=split_idx + 1,
            end_idx=len(words) - 1,
        )
        speaker_purity_score = (left_ratio + right_ratio) / 2.0
        is_has_explicit_change = (
            bool(left_speaker)
            and bool(right_speaker)
            and left_speaker != right_speaker
        )
        speaker_change_bonus = 1.0 if is_has_explicit_change else 0.0

        short_penalty = 0.0
        if split_idx + 1 < 2:
            short_penalty += 0.4
        if len(words) - (split_idx + 1) < 2:
            short_penalty += 0.4

        return (
            distance_score * 1.6
            + pause_score * 1.0
            + valley_score * 0.6
            + speaker_purity_score * 0.8
            + speaker_change_bonus * 1.2
            - short_penalty
        )

    def _score_boundary_vad_valley(self, boundary_time: float) -> float:
        intervals = list(self._vad_intervals or [])
        if not intervals:
            return 0.0
        for start, end in intervals:
            if float(start) <= boundary_time <= float(end):
                return 0.0
        return 1.0

    @staticmethod
    def _resolve_word_speaker_purity(
        *,
        words: Sequence[WordTimestamp],
        start_idx: int,
        end_idx: int,
    ) -> Tuple[float, str]:
        if end_idx < start_idx:
            return 0.0, ""
        counts: Dict[str, int] = {}
        total = 0
        for idx in range(start_idx, end_idx + 1):
            speaker_id = str(getattr(words[idx], "speaker_id", "") or "").strip()
            if not speaker_id:
                continue
            counts[speaker_id] = counts.get(speaker_id, 0) + 1
            total += 1
        if total <= 0 or not counts:
            return 0.0, ""
        best_speaker, best_count = max(counts.items(), key=lambda item: item[1])
        return best_count / total, best_speaker

    @staticmethod
    def _copy_sentence_runtime_metadata(
        *,
        source: SentenceSegment,
        target: SentenceSegment,
    ) -> None:
        target.source = source.source
        target.is_draft = source.is_draft
        target.is_finalized = source.is_finalized
        target.alignment_score = source.alignment_score
        target.matched_ratio = source.matched_ratio
        target.whisper_text = source.whisper_text
        target.sv_original_text = source.sv_original_text
        target.confidence_source = source.confidence_source
        target.warning_type = source.warning_type
        target.group_id = source.group_id
        target.is_soft_break = source.is_soft_break
        target.group_position = source.group_position

    def _bind_word_identity_by_timeline_overlap(
        self,
        *,
        annotated_words: Sequence[AnnotatedWord],
        fallback_speaker_id: Optional[str],
        fallback_turn_id: Optional[str],
    ) -> None:
        """
        按词级时间窗与 Timeline turn 重叠绑定 speaker/turn。

        Why: 兼容历史测试与诊断入口，保持词级身份绑定行为可直接验证。
        """
        if not annotated_words:
            return

        turns = list(self._timeline_turns or [])
        fallback_speaker = str(fallback_speaker_id or "unknown")
        fallback_turn = str(fallback_turn_id or "") or None

        for word in annotated_words:
            word_start = float(getattr(word, "start", 0.0) or 0.0)
            word_end_raw = getattr(word, "end", None)
            word_end = float(word_end_raw if word_end_raw is not None else word_start)
            if word_end <= word_start:
                word_end = word_start + 1e-3

            best_turn = self._select_best_turn_for_chunk(
                chunk_start=word_start,
                chunk_end=word_end,
                turns=turns,
                require_overlap=True,
            )
            if best_turn is not None:
                word.speaker_id = str(getattr(best_turn, "speaker_id", "unknown") or "unknown")
                word.turn_id = str(getattr(best_turn, "turn_id", "") or "") or None
                continue

            if word.speaker_id is None:
                word.speaker_id = fallback_speaker
            if word.turn_id is None:
                word.turn_id = fallback_turn

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

        FastWorker 输出经四层主链定稿并由输出层分发，跳过 Whisper 和对齐。

        v3.1.0: 支持逐 Chunk 中断和检查点保存
        V3.1.0: 集成进度发射器，实时推送 SSE 进度
        """
        return await self._sensevoice_orchestrator_service.run(
            audio_chunks=audio_chunks,
            full_audio_array=full_audio_array,
            full_audio_sr=full_audio_sr,
            job_dir=job_dir,
            processed_indices=processed_indices,
        )

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
        return await self._full_pipeline_orchestrator_service.run(
            audio_chunks=audio_chunks,
            full_audio_array=full_audio_array,
            full_audio_sr=full_audio_sr,
            job_dir=job_dir,
            vad_intervals=vad_intervals,
            processed_indices=processed_indices,
            base_slow_count=base_slow_count,
            base_align_count=base_align_count,
            initial_slow_processed_indices=initial_slow_processed_indices,
            initial_finalized_indices=initial_finalized_indices,
        )

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
            for sentence in sentences:
                sentence.is_finalized = True
                sentence.is_draft = False
            self._emit_output_layer(
                chunk_index=ctx.chunk_index,
                sentence_segments=sentences,
                language=str(ctx.audio_chunk.language or (ctx.sv_result or {}).get("language") or "auto"),
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
                output_traces=None,
                default_trace_reason="draft_segmenter_final",
            )
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

        if self._enable_bridge_batches:
            await self._ingest_bridge_chunks(chunks)
        else:
            self.logger.debug(
                "Chunk %s: 跳过 Bridge 入队（mode=%s）",
                ctx.chunk_index,
                self.transcription_profile,
            )

        total_sentences = 0
        finalized_sentences: List[SentenceSegment] = []
        for chunk in chunks:
            sentences = chunk.sentences
            if is_final_output:
                for sentence in sentences:
                    sentence.is_finalized = True
                    sentence.is_draft = False
                finalized_sentences.extend(sentences)
            else:
                self.subtitle_manager.add_draft_sentences(ctx.chunk_index, sentences)
            total_sentences += len(sentences)

        if is_final_output and finalized_sentences:
            self._emit_output_layer(
                chunk_index=ctx.chunk_index,
                sentence_segments=finalized_sentences,
                language=str(ctx.audio_chunk.language or (ctx.sv_result or {}).get("language") or "auto"),
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
                output_traces=None,
                default_trace_reason="semantic_buffer_final",
            )

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
        for chunk in chunks:
            speaker_id = None
            turn_id = None
            source_indices = self._parse_source_chunk_indices(list(chunk.source_chunks or []))
            if source_indices:
                first_index = source_indices[0]
                source_chunk = self._audio_chunks_by_index.get(first_index)
                if source_chunk is not None:
                    speaker_id = self._resolve_speaker_id_for_chunk(source_chunk)
                    turn_id = self._resolve_turn_id_for_chunk(source_chunk)

            envelopes = self._turn_group_builder.add_chunk(
                chunk,
                speaker_id=speaker_id,
                turn_id=turn_id,
                now=time.time(),
            )
            for envelope in envelopes:
                await self._enqueue_turn_group(envelope)

    async def _flush_bridge_controller(self) -> None:
        """强制刷新 Bridge 控制器缓冲。"""
        envelope = self._turn_group_builder.flush(reason="eof_flush")
        if envelope:
            await self._enqueue_turn_group(envelope)

    async def _enqueue_turn_group(self, envelope: TurnGroupEnvelope) -> None:
        """将 TurnGroup 送入 SlowWorker 队列。"""
        if not envelope:
            return
        await self.queue_inter.put(envelope)

    async def _process_turn_group(
        self,
        envelope: TurnGroupEnvelope,
        *,
        job_dir: Optional[Path],
        total_chunks: int,
        slow_processed_indices: Set[int],
        token: Optional["CancellationToken"],
    ) -> bool:
        """处理 TurnGroup 并推送到对齐阶段。"""
        if not self.slow_worker:
            return False

        group = envelope.group
        chunk_indices = self._parse_source_chunk_indices(group.source_chunks)
        if not chunk_indices:
            self.logger.warning("TurnGroup 缺少 source_chunks: group_id=%s", group.group_id)
            self._record_turn_group_unit(
                group=group,
                status="committed",
                payload={
                    "source_chunks": [],
                    "speaker_id": group.speaker_id,
                    "flush_reason": group.flush_reason,
                    "skipped": True,
                    "skip_reason": "missing_source_chunks",
                },
            )
            return False

        contexts: List[tuple[int, ProcessingContext]] = []
        for idx in chunk_indices:
            ctx = self._context_cache.get(idx)
            if ctx:
                contexts.append((idx, ctx))

        if not contexts:
            self.logger.warning("TurnGroup 缺少上下文缓存: group_id=%s", group.group_id)
            self._record_turn_group_unit(
                group=group,
                status="committed",
                payload={
                    "source_chunks": list(group.source_chunks),
                    "speaker_id": group.speaker_id,
                    "flush_reason": group.flush_reason,
                    "skipped": True,
                    "skip_reason": "missing_context_cache",
                },
            )
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
            self._record_turn_group_unit(
                group=group,
                status="committed",
                payload={
                    "source_chunks": list(group.source_chunks),
                    "speaker_id": group.speaker_id,
                    "flush_reason": group.flush_reason,
                    "skipped": True,
                    "skip_reason": "all_skip_by_policy",
                },
            )
            return await self._push_batch_contexts(
                contexts,
                slow_processed_indices,
                total_chunks=total_chunks,
                job_dir=job_dir,
                token=token,
            )

        if self._full_audio_array is None:
            self.logger.warning("TurnGroup 缺少完整音频，跳过慢流: group_id=%s", group.group_id)
            for _, ctx in contexts:
                ctx.whisper_skipped = True
                ctx.whisper_result = {}
            self._record_turn_group_unit(
                group=group,
                status="committed",
                payload={
                    "source_chunks": list(group.source_chunks),
                    "speaker_id": group.speaker_id,
                    "flush_reason": group.flush_reason,
                    "skipped": True,
                    "skip_reason": "missing_full_audio",
                },
            )
            return await self._push_batch_contexts(
                contexts,
                slow_processed_indices,
                total_chunks=total_chunks,
                job_dir=job_dir,
                token=token,
            )

        group_start = min((seg[0] for seg in group.audio_segments), default=0.0)
        group_end = max((seg[1] for seg in group.audio_segments), default=group_start)
        pause_gap_sec: Optional[float] = None
        if self._last_prompt_audio_end is not None:
            pause_gap_sec = max(0.0, float(group_start) - float(self._last_prompt_audio_end))

        prompt = self._build_whisper_prompt(
            group.prompt_text or None,
            pause_gap_sec=pause_gap_sec,
        )
        whisper_result = await self.slow_worker.process_turn_group(
            group,
            full_audio_array=self._full_audio_array,
            full_audio_sr=self._full_audio_sr,
            prompt_text=prompt,
        )

        self._record_turn_group_unit(
            group=group,
            status="committed",
            payload={
                "source_chunks": list(group.source_chunks),
                "target_turn_ids": list(group.target_turn_ids),
                "speaker_id": group.speaker_id,
                "flush_reason": group.flush_reason,
            },
        )

        self._validate_l0_result(whisper_result, source="slow")
        whisper_text_raw = str(whisper_result.get("raw_text") or "")
        whisper_result["text_raw"] = whisper_text_raw
        whisper_result["prompt"] = prompt
        base_text = whisper_result.get("min_clean_text") or whisper_text_raw
        whisper_result["text"] = self._whisper_sanitizer.sanitize_minimal(
            str(base_text or ""),
            prompt=prompt,
        )

        self._emit_whisper_debug(
            job_dir,
            group_id=group.group_id,
            flush_reason=group.flush_reason,
            whisper_result=whisper_result,
            chunk_indices=chunk_indices,
        )

        if whisper_result and self._hallucination_detector.is_hallucination(whisper_result, prompt):
            self.logger.warning("TurnGroup 检测到 Whisper 幻觉，回退快流: group_id=%s", group.group_id)
            self._reset_prompt_cache(reason="hallucination")
            self._last_prompt_audio_end = float(group_end)
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

        group_language = group.language or whisper_result.get("language") or "auto"
        normalized_whisper = self._text_normalizer.normalize(whisper_result.get("text", ""), group_language)
        whisper_result["text_itn_raw"] = normalized_whisper.text_itn_raw
        whisper_result["text_clean"] = normalized_whisper.text_clean
        whisper_result["text"] = normalized_whisper.text_clean or whisper_result.get("text", "")
        whisper_result["language"] = group_language

        self._update_prompt_cache(
            str(whisper_result.get("text", "")),
            confidence=whisper_result.get("confidence"),
            whisper_result=whisper_result,
        )
        self._last_prompt_audio_end = float(group_end)

        batch_start = min((seg[0] for seg in group.audio_segments), default=0.0)
        chunk_results = self._split_whisper_result_by_chunks(
            whisper_result,
            [idx for idx, _ in contexts],
            batch_start=batch_start,
            language_override=group_language,
        )

        for idx, ctx in contexts:
            setattr(
                ctx,
                "_trace_l0_batch_whisper",
                {
                    "group_id": group.group_id,
                    "chunk_indices": list(chunk_indices),
                    "prompt": str(prompt or ""),
                    "flush_reason": str(group.flush_reason or ""),
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
                    "language": group_language,
                    "raw_result": {"segments": []},
                    "word_time_base": "batch_local",
                    "word_time_offset": float(batch_start),
                },
            )
            chunk_text_raw = str(chunk_result.get("raw_text") or chunk_result.get("text", "") or "")
            chunk_result["raw_text"] = chunk_text_raw
            chunk_result["text_raw"] = chunk_text_raw
            if not chunk_result.get("min_clean_text"):
                chunk_result["min_clean_text"] = self._whisper_sanitizer.sanitize_minimal(
                    chunk_text_raw,
                    prompt=None,
                )
            chunk_result["text"] = str(chunk_result.get("min_clean_text") or "")
            raw_text_for_track = chunk_text_raw
            normalized_chunk = self._text_normalizer.normalize(chunk_result.get("text", ""), group_language)
            chunk_result["text_itn_raw"] = normalized_chunk.text_itn_raw
            chunk_result["text_clean"] = normalized_chunk.text_clean
            chunk_result["text"] = normalized_chunk.text_clean or raw_text_for_track
            chunk_result["language"] = group_language
            ctx.whisper_result = chunk_result
            tracks = self._ensure_text_tracks(ctx)
            tracks.whisper_track = self._build_text_track(
                raw_text_for_track,
                normalized_chunk,
                source="whisper",
                language=group_language,
            )

        return await self._push_batch_contexts(
            contexts,
            slow_processed_indices,
            total_chunks=total_chunks,
            job_dir=job_dir,
            token=token,
        )

    def _record_turn_group_unit(
        self,
        *,
        group: TurnGroup,
        status: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """W-Epoch 单元状态提交（每个 TurnGroup 一次）。"""
        service = self._runtime_checkpoint_service
        if service is None:
            return
        stage = "slow_epoch"
        unit_id = f"turn_group:{group.group_id}"
        try:
            if status == "started":
                service.record_unit_started(stage=stage, unit_id=unit_id, payload=payload)
            elif status == "committed":
                service.record_unit_committed(stage=stage, unit_id=unit_id, payload=payload)
        except Exception as exc:
            self.logger.warning("TurnGroup 单元写入失败（降级继续）: %s", exc)

    def _is_turn_group_committed(self, group_id: str) -> bool:
        """判断 TurnGroup 是否已提交（用于恢复跳过重算）。"""
        service = self._runtime_checkpoint_service
        if service is None:
            return False
        try:
            snapshot = service.load_snapshot()
        except Exception:
            return False
        commits: Dict[str, str]
        if isinstance(snapshot, dict):
            commits = dict(snapshot.get("last_unit_commits", {}))
        else:
            commits = dict(getattr(snapshot, "last_unit_commits", {}) or {})
        committed_unit = commits.get("slow_epoch")
        if not committed_unit:
            return False
        current_unit = f"turn_group:{group_id}"
        if committed_unit == current_unit:
            return True

        def _extract_seq(unit: str) -> Optional[int]:
            if not isinstance(unit, str):
                return None
            if not unit.startswith("turn_group:tg-"):
                return None
            try:
                return int(unit.split(":tg-")[-1])
            except ValueError:
                return None

        committed_seq = _extract_seq(committed_unit)
        current_seq = _extract_seq(current_unit)
        if committed_seq is None or current_seq is None:
            return False
        return current_seq <= committed_seq

    def _record_finalize_batch_unit(
        self,
        *,
        chunk_index: int,
        status: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Finalize 单元状态提交（每个 chunk/finalize_batch 一次）。"""
        service = self._runtime_checkpoint_service
        if service is None:
            return
        stage = "finalize"
        unit_id = f"finalize_batch:{int(chunk_index):06d}"
        try:
            if status == "started":
                service.record_unit_started(stage=stage, unit_id=unit_id, payload=payload)
            elif status == "committed":
                service.record_unit_committed(stage=stage, unit_id=unit_id, payload=payload)
        except Exception as exc:
            self.logger.warning("Finalize 单元写入失败（降级继续）: %s", exc)

    def _is_finalize_batch_committed(self, chunk_index: int) -> bool:
        """判断 finalize_batch 是否已提交（用于恢复跳过重算）。"""
        service = self._runtime_checkpoint_service
        if service is None:
            return False
        try:
            snapshot = service.load_snapshot()
        except Exception:
            return False

        commits: Dict[str, str]
        if isinstance(snapshot, dict):
            commits = dict(snapshot.get("last_unit_commits", {}))
        else:
            commits = dict(getattr(snapshot, "last_unit_commits", {}) or {})
        committed_unit = commits.get("finalize")
        if not committed_unit:
            return False

        current_unit = f"finalize_batch:{int(chunk_index):06d}"
        if committed_unit == current_unit:
            return True

        def _extract_seq(unit: str) -> Optional[int]:
            if not isinstance(unit, str):
                return None
            if not unit.startswith("finalize_batch:"):
                return None
            try:
                return int(unit.split(":")[-1])
            except ValueError:
                return None

        committed_seq = _extract_seq(committed_unit)
        current_seq = _extract_seq(current_unit)
        if committed_seq is None or current_seq is None:
            return False
        return current_seq <= committed_seq

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
            self.logger.warning("规范层缺失 raw_text，跳过规范化")
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
                self.logger.debug("选文层检测到重复: reason={}", reason)

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
            "规范层 slow_raw 标点候选: clean_len={} positions={} avg_conf={:.2f} sample={}",
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
            "_trace_punctuation_pre_input",
            {
                "chosen_source": chosen_source,
                "word_timestamps": list(word_timestamps),
                "sv_punct_source": sv_source,
                "wh_punct_source": wh_source,
                "chosen_text_track": chosen_track,
            },
        )
        output = await self._punctuation_pre_processor.process(
            PunctuationPreInput(
                chosen_text_track=chosen_track,
                sv_punct_source=sv_source,
                wh_punct_source=wh_source,
                word_timestamps=word_timestamps,
            )
        )
        setattr(ctx, "_trace_punctuation_pre_output", output.punct_track)
        return output.punct_track

    def _resolve_gap_ratio_mid(self) -> float:
        gap_ratio_mid = 0.3
        if not self.aligner:
            return gap_ratio_mid
        alignment_service = getattr(self.aligner, "alignment_service", None)
        gap_resolver = getattr(alignment_service, "gap_resolver", None)
        value = getattr(gap_resolver, "_gap_ratio_mid", None)
        if isinstance(value, (int, float)):
            gap_ratio_mid = float(value)
        return gap_ratio_mid

    def _resolve_injection_min_mapping_coverage(self) -> float:
        if not self.aligner:
            return 0.6
        final_splitter = getattr(self.aligner, "final_splitter", None)
        config = getattr(final_splitter, "config", None)
        value = getattr(config, "min_mapping_coverage", None)
        if isinstance(value, (int, float)):
            return float(value)
        return 0.6

    def _record_punct_retry_candidates(self, ctx: ProcessingContext) -> None:
        """集合层/评分层补跑候选仅做埋点，不触发补跑。"""
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
        speaker_id = self._resolve_speaker_id_for_chunk(chunk)
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
        if self._enable_bridge_batches:
            if chunks:
                await self._ingest_bridge_chunks(chunks)
            else:
                self.logger.debug(
                    "SemanticBuffer 尾部刷新无新增语义块，继续执行 Bridge EOF flush",
                )
            await self._flush_bridge_controller()
        if not chunks:
            return
        total_sentences = 0
        finalized_sentences: List[SentenceSegment] = []
        for chunk in chunks:
            sentences = chunk.sentences
            if is_final_output:
                for sentence in sentences:
                    sentence.is_finalized = True
                    sentence.is_draft = False
                finalized_sentences.extend(sentences)
            else:
                self.subtitle_manager.add_draft_sentences(chunk_index, sentences)
            total_sentences += len(sentences)
        if is_final_output and finalized_sentences:
            self._emit_output_layer(
                chunk_index=chunk_index,
                sentence_segments=finalized_sentences,
                language="auto",
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
                output_traces=None,
                default_trace_reason="semantic_buffer_flush_final",
            )
        phase = "定稿" if is_final_output else "草稿"
        self.logger.debug(
            "SemanticBuffer 尾部刷新完成: %s %d 个句子",
            phase,
            total_sentences,
        )

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

    def _build_whisper_prompt(
        self,
        sv_context: Optional[str],
        *,
        pause_gap_sec: Optional[float] = None,
    ) -> Optional[str]:
        """构建 Whisper Prompt（统一策略入口）。"""
        return self._whisper_prompt_policy.build_prompt(
            previous_text=self.previous_whisper_text,
            semantic_text=sv_context,
            pause_gap_sec=pause_gap_sec,
        )

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
    def _normalize_whisper_boundary_token(token: Any) -> str:
        token_str = str(token or "").replace("▁", " ").strip()
        if not token_str:
            return ""
        return token_str.strip("\"'“”‘’()[]{}<>《》【】「」『』，。、！？：；,.!?;:")

    @classmethod
    def _is_cjk_token(cls, token: str) -> bool:
        text = str(token or "")
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

    @classmethod
    def _is_sentence_end_token(cls, token: str) -> bool:
        text = str(token or "").strip()
        if not text:
            return False
        return is_sentence_end_punct(
            text,
            "",
            sentence_end_chars=tuple(cls._SPEAKER_REPAIR_SENTENCE_END_PUNCT),
        )

    @staticmethod
    def _resolve_word_start_end_local(word: Dict[str, Any]) -> Tuple[float, float]:
        start = float(word.get("start", 0.0) or 0.0)
        end = float(word.get("end", word.get("start", 0.0)) or word.get("start", 0.0))
        if end < start:
            end = start
        return start, end

    @classmethod
    def _resolve_whisper_tail_move_count(
        cls,
        left_words: Sequence[Dict[str, Any]],
        right_words: Sequence[Dict[str, Any]],
    ) -> int:
        if not left_words or not right_words:
            return 0

        right_head = cls._normalize_whisper_boundary_token(right_words[0].get("word", ""))
        if not right_head or not cls._is_cjk_token(right_head):
            return 0

        right_start, _ = cls._resolve_word_start_end_local(right_words[0])
        max_tail = min(cls._WHISPER_REBALANCE_MAX_TAIL_WORDS, len(left_words))

        for move_count in range(max_tail, 0, -1):
            tail_words = list(left_words[-move_count:])
            if move_count < len(left_words):
                anchor_word = left_words[-move_count - 1]
                anchor_token = cls._normalize_whisper_boundary_token(anchor_word.get("word", ""))
                if cls._is_sentence_end_token(anchor_token):
                    continue

            normalized_tail_tokens = [
                cls._normalize_whisper_boundary_token(item.get("word", ""))
                for item in tail_words
            ]
            if any(not token for token in normalized_tail_tokens):
                continue
            if any(cls._is_sentence_end_token(token) for token in normalized_tail_tokens):
                continue

            merged_tail = "".join(normalized_tail_tokens)
            if move_count == 1:
                token = normalized_tail_tokens[0]
                is_single_char_carry = len(token) == 1 and token in cls._WHISPER_REBALANCE_SINGLE_CHAR_CARRY
                is_multi_char_carry = token in cls._WHISPER_REBALANCE_MULTI_CHAR_CARRY
                if not (is_single_char_carry or is_multi_char_carry):
                    continue
            else:
                if merged_tail not in cls._WHISPER_REBALANCE_MULTI_CHAR_CARRY:
                    continue
                if any(len(token) != 1 for token in normalized_tail_tokens):
                    continue

            tail_start, tail_end = cls._resolve_word_start_end_local(tail_words[0])
            _, last_tail_end = cls._resolve_word_start_end_local(tail_words[-1])
            tail_end = max(tail_end, last_tail_end)
            if tail_end < tail_start:
                tail_end = tail_start
            if max(0.0, right_start - tail_end) > cls._WHISPER_REBALANCE_MAX_TAIL_GAP_SEC:
                continue
            if (tail_end - tail_start) > cls._WHISPER_REBALANCE_MAX_TAIL_DURATION_SEC:
                continue
            return move_count
        return 0

    @classmethod
    def _rebalance_whisper_words_by_chunk(
        cls,
        words_by_chunk: Dict[int, List[Dict[str, Any]]],
        ordered_indices: Sequence[int],
    ) -> int:
        if len(ordered_indices) <= 1:
            return 0

        moved_word_count = 0
        for left_idx, right_idx in zip(ordered_indices[:-1], ordered_indices[1:]):
            left_words = list(words_by_chunk.get(left_idx) or [])
            right_words = list(words_by_chunk.get(right_idx) or [])
            if not left_words or not right_words:
                continue

            move_count = cls._resolve_whisper_tail_move_count(left_words, right_words)
            if move_count <= 0 or move_count > len(left_words):
                continue

            moved_words = left_words[-move_count:]
            words_by_chunk[left_idx] = left_words[:-move_count]
            words_by_chunk[right_idx] = moved_words + right_words
            moved_word_count += move_count
        return moved_word_count

    @classmethod
    def _resolve_fragment_local_time_range(
        cls,
        words: Sequence[Dict[str, Any]],
        *,
        fallback_start: float,
        fallback_end: float,
    ) -> Tuple[float, float]:
        if not words:
            return fallback_start, max(fallback_end, fallback_start)
        start, _ = cls._resolve_word_start_end_local(words[0])
        _, end = cls._resolve_word_start_end_local(words[-1])
        if end < start:
            end = start
        return start, end

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
            distance = AsyncDualPipelineKernel._distance_to_range(mid, (span_start, span_end))
            if overlap > best_overlap or (overlap == best_overlap and distance < best_distance):
                best_idx = idx
                best_overlap = overlap
                best_distance = distance

        if best_idx is not None:
            return best_idx
        return min(ordered_indices, key=lambda idx: AsyncDualPipelineKernel._distance_to_range(mid, chunk_ranges[idx]))

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
        moved_words_total = 0

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

                moved_words_total += self._rebalance_whisper_words_by_chunk(
                    words_by_chunk=words_by_chunk,
                    ordered_indices=ordered_indices,
                )

                for idx in ordered_indices:
                    words = list(words_by_chunk.get(idx) or [])
                    if not words:
                        continue
                    seg_fragment = dict(seg)
                    seg_fragment["words"] = list(words)
                    fragment_text = self._compose_whisper_text_from_words(words)
                    if fragment_text:
                        seg_fragment["text"] = fragment_text
                    fragment_start, fragment_end = self._resolve_fragment_local_time_range(
                        words,
                        fallback_start=float(seg.get("start", 0.0) or 0.0),
                        fallback_end=float(seg.get("end", 0.0) or 0.0),
                    )
                    seg_fragment["start"] = fragment_start
                    seg_fragment["end"] = fragment_end
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

        if moved_words_total > 0:
            self.logger.debug(
                "Whisper批次回写跨chunk重平衡: moved_words={} chunk_count={}",
                moved_words_total,
                len(ordered_indices),
            )

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
        *,
        group_id: str,
        flush_reason: str,
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
            "group_id": group_id,
            "flush_reason": str(flush_reason or ""),
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
        """输出前置标点域与四层主链诊断到独立文件。"""
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
        split_reason_stats: Dict[str, int] = {}
        split_risk_stats: Dict[str, int] = {}
        for sentence in final_sentences:
            reason = str(getattr(sentence, "split_reason", "") or "")
            risk = str(getattr(sentence, "split_risk", "") or "")
            if reason:
                split_reason_stats[reason] = int(split_reason_stats.get(reason, 0) + 1)
            if risk:
                split_risk_stats[risk] = int(split_risk_stats.get(risk, 0) + 1)

        payload: Dict[str, Any] = {
            "job_id": ctx.job_id,
            "chunk_index": int(ctx.chunk_index),
            "layer": "选文到裁决",
            "chosen_source": ctx.arbitration_result.chosen_source if ctx.arbitration_result else "",
            "arbitration_reason": ctx.arbitration_result.reason if ctx.arbitration_result else "",
            "chosen_clean_len": len(chosen_clean or ""),
            "l2_chosen_text_head": _clip_head(chosen_compact),
            "l2_chosen_text_tail": _clip_tail(chosen_compact),
            "punctuation_pre_positions_total": len(punct_track.positions) if punct_track and punct_track.positions else 0,
            "punctuation_pre_source": punct_track.source if punct_track else "",
            "punctuation_pre_clean_text_match": bool(chosen_clean and punct_ref and chosen_clean == punct_ref),
            "punctuation_pre_clean_ref_head": _clip_head(punct_compact),
            "punctuation_pre_clean_ref_tail": _clip_tail(punct_compact),
            "is_punctuation_pre_match_blocked_by_ref_mismatch": is_ref_mismatch,
            "collection_alignment_score": float(alignment_result.alignment_score),
            "collection_gap_ratio": float(alignment_result.gap_ratio),
            "collection_coverage": float(alignment_result.coverage),
            "collection_gap_positions": list(alignment_result.gap_positions),
            "collection_unmatched_prefix_len": int(unmatched_prefix_len),
            "is_cross_chunk_boundary_suspected": is_cross_chunk_boundary_suspected,
            "scoring_injection_positions_total": int(injection_stats.get("injection_positions_total", 0) or 0),
            "scoring_injection_unmatched_total": int(injection_stats.get("injection_unmatched_total", 0) or 0),
            "scoring_injection_mapping_coverage": float(injection_stats.get("injection_mapping_coverage", 0.0) or 0.0),
            "scoring_injection_blocked": bool(injection_stats.get("injection_blocked", 0.0)),
            "decision_sentence_count": len(final_sentences),
            "decision_split_mapping_coverage": float(split_stats.get("mapping_coverage", 0.0) or 0.0),
            "decision_split_writeback_ratio": float(split_stats.get("writeback_ratio", 0.0) or 0.0),
            "decision_split_writeback_used": bool(split_stats.get("writeback_used", 0.0)),
            "decision_split_writeback_blocked": bool(split_stats.get("writeback_blocked", 0.0)),
            "decision_unknown_pseudo_drop_count": int(split_stats.get("unknown_pseudo_drop_count", 0) or 0),
            "decision_unknown_pseudo_degrade_count": int(
                split_stats.get("unknown_pseudo_degrade_count", 0) or 0
            ),
            "decision_unknown_pseudo_filter_fallback": bool(
                split_stats.get("unknown_pseudo_filter_fallback", False)
            ),
            "decision_split_reason_stats": split_reason_stats,
            "decision_split_risk_stats": split_risk_stats,
            "decision_sentence_texts": [str(sentence.text or "") for sentence in final_sentences],
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
        aligned_facts: AlignedFacts,
        fused_evidence: FusedEvidence,
        words_for_split: Sequence[WordTimestamp],
        injection_stats: Dict[str, Any],
        split_stats: Dict[str, Any],
        final_sentences: Sequence[SentenceSegment],
        output_traces: Sequence[OutputTrace],
    ) -> None:
        """输出前置域与四层主链全量追踪（逐层 + 逐 token）到独立文件。"""

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
                        "split_reason": str(getattr(sentence, "split_reason", "") or ""),
                        "split_risk": str(getattr(sentence, "split_risk", "") or ""),
                        "window_id": str(getattr(sentence, "window_id", "") or ""),
                        "pyannote_frame_time": getattr(sentence, "pyannote_frame_time", None),
                        "mapped_cut_time": getattr(sentence, "mapped_cut_time", None),
                        "mapping_quality": str(getattr(sentence, "mapping_quality", "") or ""),
                        "mapping_reason": str(getattr(sentence, "mapping_reason", "") or ""),
                    }
                )
            return rows

        def _serialize_output_traces(traces: Sequence[OutputTrace]) -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []
            for trace in traces:
                rows.append(
                    {
                        "sentence_index": int(trace.sentence_index),
                        "split_reason": str(trace.split_reason or ""),
                        "split_risk": str(trace.split_risk or ""),
                        "window_id": str(trace.window_id or ""),
                        "pyannote_frame_time": trace.pyannote_frame_time,
                        "mapped_cut_time": trace.mapped_cut_time,
                        "mapping_quality": str(trace.mapping_quality or ""),
                        "mapping_reason": str(trace.mapping_reason or ""),
                        "sentence_start": trace.sentence_start,
                        "sentence_end": trace.sentence_end,
                    }
                )
            return rows

        def _serialize_aligned_facts(facts: AlignedFacts) -> Dict[str, Any]:
            mapping_quality_stats: Dict[str, int] = {}
            for item in list(facts.time_mappings or []):
                quality = str(item.get("mapping_quality", "") or "unknown")
                mapping_quality_stats[quality] = mapping_quality_stats.get(quality, 0) + 1
            return {
                "alignment_score": float(facts.alignment_score),
                "gap_ratio": float(facts.gap_ratio),
                "gap_positions": list(facts.gap_positions or []),
                "annotated_word_count": int(len(facts.annotated_words or [])),
                "speaker_turn_count": int(len(facts.speaker_turns or [])),
                "fast_draft_cut_count": int(len(facts.fast_draft_cuts or [])),
                "time_axis_version": str(facts.time_axis_version or ""),
                "time_mappings": list(facts.time_mappings or []),
                "mapping_quality_stats": mapping_quality_stats,
            }

        def _serialize_fused_evidence(evidence: FusedEvidence) -> Dict[str, Any]:
            return {
                "speaker_change_count": int(len(evidence.speaker_changes or [])),
                "pause_anchor_count": int(len(evidence.pause_anchors or [])),
                "semantic_anchor_count": int(len(evidence.semantic_anchors or [])),
                "punctuation_anchor_count": int(len(evidence.punctuation_anchors or [])),
                "evidence_report": dict(evidence.evidence_report or {}),
                "speaker_changes": list(evidence.speaker_changes or []),
                "pause_anchors": list(evidence.pause_anchors or []),
                "semantic_anchors": list(evidence.semantic_anchors or []),
                "punctuation_anchors": list(evidence.punctuation_anchors or []),
            }

        punctuation_pre_input = getattr(ctx, "_trace_punctuation_pre_input", {}) or {}
        punctuation_pre_output = getattr(ctx, "_trace_punctuation_pre_output", None)
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
            "layer": "全链路追踪",
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
            "punctuation_pre": {
                "input": {
                    "chosen_source": punctuation_pre_input.get("chosen_source"),
                    "word_timestamps": _serialize_word_timestamps(punctuation_pre_input.get("word_timestamps") or []),
                    "sv_punct_source": {
                        "source": str(getattr(punctuation_pre_input.get("sv_punct_source"), "source", "") or ""),
                        "clean_text_ref": str(getattr(punctuation_pre_input.get("sv_punct_source"), "clean_text_ref", "") or ""),
                        "positions": _serialize_punc_positions(getattr(punctuation_pre_input.get("sv_punct_source"), "positions", []) or []),
                    },
                    "wh_punct_source": {
                        "source": str(getattr(punctuation_pre_input.get("wh_punct_source"), "source", "") or ""),
                        "clean_text_ref": str(getattr(punctuation_pre_input.get("wh_punct_source"), "clean_text_ref", "") or ""),
                        "positions": _serialize_punc_positions(getattr(punctuation_pre_input.get("wh_punct_source"), "positions", []) or []),
                    },
                },
                "output": {
                    "source": str(getattr(punctuation_pre_output, "source", "") or ""),
                    "clean_text_ref": str(getattr(punctuation_pre_output, "clean_text_ref", "") or ""),
                    "positions": _serialize_punc_positions(getattr(punctuation_pre_output, "positions", []) or []),
                    "confidence_stats": dict(getattr(punctuation_pre_output, "confidence_stats", {}) or {}),
                },
            },
            "collection": {
                "alignment_score": float(alignment_result.alignment_score),
                "gap_ratio": float(alignment_result.gap_ratio),
                "coverage": float(alignment_result.coverage),
                "gap_positions": list(alignment_result.gap_positions),
                "aligned_words": _serialize_aligned_words(alignment_result.aligned_words),
            },
            "scoring": {
                "injection_stats": dict(injection_stats),
                "words_for_split": _serialize_word_timestamps(words_for_split),
                "aligned_facts": _serialize_aligned_facts(aligned_facts),
                "fused_evidence": _serialize_fused_evidence(fused_evidence),
            },
            "decision": {
                "split_stats": dict(split_stats),
                "final_sentences": _serialize_sentences(final_sentences),
                "output_trace": _serialize_output_traces(output_traces),
            },
            "current_punct_track": {
                "source": str(punct_track.source if punct_track else ""),
                "clean_text_ref": str(punct_track.clean_text_ref if punct_track else ""),
                "positions": _serialize_punc_positions(punct_track.positions if punct_track else []),
            },
        }
        append_debug_layer_trace_line(ctx.job_dir, payload, logger=self.logger)

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

    @staticmethod
    def _estimate_avg_no_speech_prob(whisper_result: Optional[Dict[str, Any]]) -> Optional[float]:
        """估算当前结果的平均 no_speech_prob。"""
        if not isinstance(whisper_result, dict):
            return None
        raw_result = whisper_result.get("raw_result")
        if not isinstance(raw_result, dict):
            return None
        segments = raw_result.get("segments")
        if not isinstance(segments, list) or not segments:
            return None
        values: List[float] = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            value = segment.get("no_speech_prob")
            if isinstance(value, (float, int)):
                values.append(float(value))
        if not values:
            return None
        return float(sum(values) / len(values))

    def _update_prompt_cache(
        self,
        whisper_text: str,
        *,
        confidence: Optional[float] = None,
        whisper_result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """更新 Whisper 上下文缓存（策略层统一裁剪与重置）。"""
        avg_no_speech_prob = self._estimate_avg_no_speech_prob(whisper_result)
        self.previous_whisper_text = self._whisper_prompt_policy.update_history(
            previous_text=self.previous_whisper_text,
            decoded_text=whisper_text,
            confidence=float(confidence) if isinstance(confidence, (float, int)) else None,
            avg_no_speech_prob=avg_no_speech_prob,
            is_hallucination=False,
        )

    def _reset_prompt_cache(self, reason: str) -> None:
        """重置 Whisper 上下文缓存。"""
        self.previous_whisper_text = ""
        self.logger.debug("Whisper 上下文已重置: reason=%s", reason)

    def restore_prompt_cache(self, previous_text: Optional[str]) -> None:
        """恢复 Whisper 上下文缓存（断点续传使用）。"""
        restored = self._whisper_prompt_policy.restore_history(previous_text)
        self.previous_whisper_text = restored
        if restored:
            self.logger.debug(f"[v3.1.0] 已恢复 Whisper 上下文: {len(restored)} 字符")
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
        await self._fast_loop_service.run(
            chunks=chunks,
            full_audio_array=full_audio_array,
            full_audio_sr=full_audio_sr,
            job_dir=job_dir,
            processed_indices=processed_indices,
            total_chunks=total_chunks,
        )

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
        await self._slow_loop_service.run(
            job_dir=job_dir,
            total_chunks=total_chunks,
            base_slow_count=base_slow_count,
            initial_slow_processed_indices=initial_slow_processed_indices,
        )

    # V3.2.0+dev.20260120.03: 对齐阶段下放到流水线
    async def _run_alignment_stage(self, ctx: ProcessingContext) -> None:
        """
        对齐阶段（流水线内执行）

        负责：
        1. 双流对齐（含降级兜底）
        2. 推送定稿
        3. 填充 ctx.final_sentences
        """
        await self._alignment_stage_service.run(ctx)

    def _append_dual_time_compare_debug(
        self,
        job_dir: Optional[Path],
        payload: Dict[str, Any],
    ) -> None:
        """通过实现层调试钩子写入 dual-time 对比记录。"""
        append_debug_dual_time_compare_line(job_dir, payload, logger=self.logger)

    def _append_m2_stage0_debug(
        self,
        job_dir: Optional[Path],
        payload: Dict[str, Any],
    ) -> None:
        """通过实现层调试钩子写入 M2 阶段0观测记录。"""
        append_debug_m2_stage0_line(job_dir, payload, logger=self.logger)

    def _write_dual_time_summary_debug(
        self,
        job_dir: Optional[Path],
        summary: Dict[str, Any],
    ) -> None:
        """通过实现层调试钩子写入 dual-time 汇总。"""
        write_debug_json_payload(
            job_dir,
            "dual_time_eval_summary.json",
            summary,
            logger=self.logger,
        )

    def _run_collection_scoring_decision_once(
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
        turn_id: Optional[str] = None,
        policy_snapshot: Optional[Any] = None,
    ) -> Layer456RunResult:
        """
        四层术语主入口：集合层→评分层→裁决层。

        Why: 编排层统一通过该入口委派门面服务，避免多处分叉调用。
        """
        return self._textflow_facade_service.run_collection_scoring_decision_once(
            tracks=tracks,
            sv_result=sv_result,
            whisper_result=whisper_result,
            sv_words=sv_words,
            punctuation_positions=punctuation_positions,
            punctuation_clean_text=punctuation_clean_text,
            variant=variant,
            speaker_id=speaker_id,
            turn_id=turn_id,
            policy_snapshot=policy_snapshot,
        )

    def _should_record_m2_stage0_sample(self, *, chunk_index: int) -> bool:
        """兼容入口：委派 dual-time 诊断门面服务。"""
        return self._dual_time_diagnostics_service.should_record_m2_stage0_sample(
            chunk_index=chunk_index
        )

    def _build_stage0_run_snapshot(self, run: Layer456RunResult) -> Dict[str, Any]:
        """兼容入口：委派 dual-time 诊断门面服务。"""
        return self._dual_time_diagnostics_service.build_stage0_run_snapshot(run)

    def _maybe_record_m2_stage0_sample(
        self,
        *,
        ctx: ProcessingContext,
        selected_variant: str,
        selected_reason: str,
        legacy_run: Layer456RunResult,
        experiment_run: Optional[Layer456RunResult],
        active_run: Layer456RunResult,
        compare_payload: Optional[Dict[str, Any]],
    ) -> None:
        """兼容入口：委派 dual-time 诊断门面服务。"""
        self._dual_time_diagnostics_service.maybe_record_m2_stage0_sample(
            ctx=ctx,
            selected_variant=selected_variant,
            selected_reason=selected_reason,
            legacy_run=legacy_run,
            experiment_run=experiment_run,
            active_run=active_run,
            compare_payload=compare_payload,
        )

    def _build_dual_time_compare_payload(
        self,
        *,
        ctx: ProcessingContext,
        legacy_run: Layer456RunResult,
        experiment_run: Layer456RunResult,
    ) -> Dict[str, Any]:
        """兼容入口：委派 dual-time 诊断门面服务。"""
        return self._dual_time_diagnostics_service.build_dual_time_compare_payload(
            ctx=ctx,
            legacy_run=legacy_run,
            experiment_run=experiment_run,
        )

    def _select_dual_time_variant(
        self,
        *,
        compare_payload: Dict[str, Any],
        experiment_run: Layer456RunResult,
    ) -> Tuple[str, str]:
        """兼容入口：委派 dual-time 诊断门面服务。"""
        return self._dual_time_diagnostics_service.select_dual_time_variant(
            compare_payload=compare_payload,
            experiment_run=experiment_run,
        )

    def _update_dual_time_summary(self, ctx: ProcessingContext, payload: Dict[str, Any]) -> None:
        """兼容入口：委派 dual-time 诊断门面服务。"""
        self._dual_time_diagnostics_service.update_dual_time_summary(ctx, payload)

    def _write_dual_time_debug_srt(self, job_dir: Optional[Path]) -> None:
        """兼容入口：委派 dual-time 诊断门面服务。"""
        self._dual_time_diagnostics_service.write_dual_time_debug_srt(job_dir)

    @staticmethod
    def _build_srt_from_chunk_map(chunk_map: Dict[int, List[SentenceSegment]]) -> str:
        """兼容入口：委派 dual-time 诊断门面服务。"""
        return DualTimeDiagnosticsService.build_srt_from_chunk_map(chunk_map)

    @staticmethod
    def _extract_sentence_boundaries(sentences: Sequence[SentenceSegment]) -> List[float]:
        """兼容入口：委派 dual-time 诊断门面服务。"""
        return DualTimeDiagnosticsService.extract_sentence_boundaries(sentences)

    @classmethod
    def _compute_boundary_metrics(
        cls,
        legacy_sentences: Sequence[SentenceSegment],
        experiment_sentences: Sequence[SentenceSegment],
        *,
        tolerance_sec: float,
    ) -> Dict[str, float]:
        """兼容入口：委派 dual-time 诊断门面服务。"""
        return DualTimeDiagnosticsService.compute_boundary_metrics(
            legacy_sentences,
            experiment_sentences,
            tolerance_sec=tolerance_sec,
        )

    @staticmethod
    def _compute_word_mae_ms(
        legacy_words: Sequence[WordTimestamp],
        experiment_words: Sequence[WordTimestamp],
    ) -> Dict[str, float]:
        """兼容入口：委派 dual-time 诊断门面服务。"""
        return DualTimeDiagnosticsService.compute_word_mae_ms(
            legacy_words,
            experiment_words,
        )

    @staticmethod
    def _compute_sentence_mae_ms(
        legacy_sentences: Sequence[SentenceSegment],
        experiment_sentences: Sequence[SentenceSegment],
    ) -> Dict[str, float]:
        """兼容入口：委派 dual-time 诊断门面服务。"""
        return DualTimeDiagnosticsService.compute_sentence_mae_ms(
            legacy_sentences,
            experiment_sentences,
        )

    def _finalize_sensevoice_only(self, ctx: ProcessingContext) -> Layer456RunResult:
        """
        Whisper 跳过时的定稿输出（仅使用 SenseVoice 结果）。

        Why: 通过门面服务统一四层路径，保持实现层只负责阶段调度。
        """
        return self._textflow_facade_service.finalize_sensevoice_only(ctx)

    def _build_soft_cut_plan_for_decision(
        self,
        *,
        annotated_words: Sequence[AnnotatedWord],
        stream_id: str,
        block_id: str,
        is_last_chunk: bool,
        aligned_facts: Optional[AlignedFacts] = None,
        fused_evidence: Optional[FusedEvidence] = None,
        policy_snapshot: Optional[Any] = None,
    ) -> Optional[Any]:
        """
        统一 CutPlan 生产入口。

        Why:
        - 主链只依赖 provider 协议，后续 M2 切换只需新增 provider 与配置
        - provider 异常时回退 M1，避免影响主流程可用性
        """
        try:
            plan = self._soft_cut_plan_provider.build_plan(
                annotated_words=annotated_words,
                stream_id=stream_id,
                block_id=block_id,
                is_last_chunk=is_last_chunk,
                aligned_facts=aligned_facts,
                fused_evidence=fused_evidence,
            )
            return self._normalize_soft_cut_plan(
                plan=plan,
                stream_id=stream_id,
                block_id=block_id,
                reason="provider_empty_plan",
            )
        except Exception as exc:
            self.logger.warning(
                "soft-cut provider 执行失败，回退 M1: provider={} error={}",
                self._soft_cut_plan_provider.__class__.__name__,
                exc,
            )
            plan = self._build_soft_cut_plan_for_decision_m1(
                annotated_words=annotated_words,
                stream_id=stream_id,
                block_id=block_id,
                is_last_chunk=is_last_chunk,
                aligned_facts=aligned_facts,
                fused_evidence=fused_evidence,
                policy_snapshot=policy_snapshot,
            )
            return self._normalize_soft_cut_plan(
                plan=plan,
                stream_id=stream_id,
                block_id=block_id,
                reason="provider_exception_fallback",
            )

    def _build_fused_evidence_for_decision(
        self,
        *,
        words: Sequence[AnnotatedWord],
        stream_id: str,
        aligned_facts: Optional[AlignedFacts],
        policy_snapshot: Optional[Any] = None,
    ) -> FusedEvidence:
        """
        构建评分层契约（FusedEvidence）。

        Why:
        - 将 pipeline 内散落的 speaker-change/anchor 拼装逻辑收口为单入口。
        - 该层仅产出证据，不触发裁决，保持与 CutPlan 解耦。
        """
        anchor_candidates = self._build_soft_cut_anchor_candidates(
            words=words,
            policy_snapshot=policy_snapshot,
        )
        pause_anchors = [item for item in anchor_candidates if item.anchor_type == AnchorType.PAUSE_ANCHOR]
        word_anchors = [item for item in anchor_candidates if item.anchor_type == AnchorType.WORD_BOUNDARY]
        semantic_anchors = [item for item in anchor_candidates if item.anchor_type == AnchorType.SEMANTIC_ANCHOR]
        punctuation_anchors = [
            item for item in anchor_candidates if item.anchor_type == AnchorType.PUNCTUATION_ANCHOR
        ]
        base_report: Dict[str, Any] = {
            "enabled": bool(self._is_enable_soft_cut),
            "stream_id": str(stream_id),
            "word_count": int(len(words)),
            "anchor_candidate_count": int(len(anchor_candidates)),
            "priority_profile": str(self._soft_cut_priority_active_profile),
            "time_axis_version": str(
                getattr(aligned_facts, "time_axis_version", "m1_legacy") or "m1_legacy"
            ),
        }
        if not self._is_enable_soft_cut or len(words) <= 1:
            base_report["reason"] = "soft_cut_disabled_or_insufficient_words"
            return self._soft_cut_evidence_fusion.to_fused_evidence(
                speaker_changes=[],
                pause_anchors=self._serialize_soft_cut_anchor_candidates(pause_anchors),
                semantic_anchors=self._serialize_soft_cut_anchor_candidates(semantic_anchors),
                punctuation_anchors=self._serialize_soft_cut_anchor_candidates(punctuation_anchors),
                generation_report=base_report,
            )

        speaker_change_facts = self._build_soft_cut_speaker_change_facts(words=words)

        chunk_start = float(words[0].start if words[0].start is not None else 0.0)
        chunk_end = float(words[-1].end if words[-1].end is not None else chunk_start)
        if chunk_end <= chunk_start:
            chunk_end = chunk_start + 1e-3
        chunk_id = f"{stream_id}:{int(chunk_start * 1000)}"
        builder_result = self._soft_cut_evidence_builder.build(
            chunk_id=chunk_id,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            speaker_change_facts=speaker_change_facts,
            anchor_candidates=anchor_candidates,
        )
        time_axis_version = str(
            getattr(aligned_facts, "time_axis_version", "m1_legacy") or "m1_legacy"
        )
        word_time_confidence = self._collect_word_time_confidence_for_fusion(words=words)
        fusion_result = self._soft_cut_evidence_fusion.fuse(
            cut_windows=builder_result.cut_windows,
            pause_anchors=pause_anchors,
            word_anchors=word_anchors,
            semantic_anchors=semantic_anchors,
            punctuation_anchors=punctuation_anchors,
            time_axis_version=time_axis_version,
            word_time_confidence=word_time_confidence,
        )
        report = {
            **base_report,
            "speaker_change_fact_count": int(len(speaker_change_facts)),
            "word_time_confidence_count": int(len(word_time_confidence or [])),
            "builder": dict(builder_result.generation_report or {}),
            "fusion": dict(fusion_result.generation_report or {}),
            "fused_windows": self._serialize_soft_cut_windows(fusion_result.fused_windows),
        }
        return self._soft_cut_evidence_fusion.to_fused_evidence(
            speaker_changes=self._serialize_soft_cut_speaker_changes(builder_result.speaker_changes),
            pause_anchors=self._serialize_soft_cut_anchor_candidates(pause_anchors),
            semantic_anchors=self._serialize_soft_cut_anchor_candidates(semantic_anchors),
            punctuation_anchors=self._serialize_soft_cut_anchor_candidates(punctuation_anchors),
            generation_report=report,
        )

    @staticmethod
    def _serialize_soft_cut_speaker_changes(items: Sequence[Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in items or []:
            rows.append(
                {
                    "time": float(getattr(item, "time", 0.0) or 0.0),
                    "from_speaker": str(getattr(item, "from_speaker", "") or ""),
                    "to_speaker": str(getattr(item, "to_speaker", "") or ""),
                    "pyannote_confidence": float(getattr(item, "pyannote_confidence", 0.0) or 0.0),
                    "pause_duration": float(getattr(item, "pause_duration", 0.0) or 0.0),
                    "embedding_distance": float(getattr(item, "embedding_distance", 0.0) or 0.0),
                    "embedding_threshold": float(getattr(item, "embedding_threshold", 0.0) or 0.0),
                    "is_abrupt_energy_shift": bool(getattr(item, "is_abrupt_energy_shift", False)),
                    "level": str(getattr(getattr(item, "level", None), "value", "") or ""),
                    "tags": sorted(
                        [
                            str(getattr(tag, "value", str(tag)))
                            for tag in list(getattr(item, "tags", set()) or set())
                        ]
                    ),
                }
            )
        return rows

    @staticmethod
    def _serialize_soft_cut_anchor_candidates(items: Sequence[AnchorCandidate]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in items or []:
            rows.append(
                {
                    "anchor_type": str(getattr(item.anchor_type, "value", item.anchor_type)),
                    "anchor_time": float(item.anchor_time),
                    "source": str(item.source),
                    "confidence": (
                        float(item.confidence) if item.confidence is not None else None
                    ),
                    "base_score_override": (
                        float(item.base_score_override)
                        if item.base_score_override is not None
                        else None
                    ),
                }
            )
        return rows

    @staticmethod
    def _serialize_soft_cut_windows(windows: Sequence[Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for window in windows or []:
            candidate_anchors = list(getattr(window, "candidate_anchors", []) or [])
            top_anchor = candidate_anchors[0] if candidate_anchors else None
            rows.append(
                {
                    "window_id": str(getattr(window, "window_id", "") or ""),
                    "trigger_time": float(getattr(window, "trigger_time", 0.0) or 0.0),
                    "trigger_level": str(getattr(getattr(window, "trigger_level", None), "value", "") or ""),
                    "trigger_source": str(
                        getattr(getattr(window, "trigger_source", None), "value", "") or ""
                    ),
                    "trigger_score": float(getattr(window, "trigger_score", 0.0) or 0.0),
                    "start_time": float(getattr(window, "start_time", 0.0) or 0.0),
                    "end_time": float(getattr(window, "end_time", 0.0) or 0.0),
                    "candidate_anchor_count": int(len(candidate_anchors)),
                    "top_anchor_type": (
                        str(getattr(getattr(top_anchor, "anchor_type", None), "value", "") or "")
                        if top_anchor is not None
                        else ""
                    ),
                    "top_anchor_time": (
                        float(getattr(top_anchor, "anchor_time", 0.0) or 0.0)
                        if top_anchor is not None
                        else None
                    ),
                    "top_anchor_score": (
                        float(getattr(top_anchor, "final_score", 0.0) or 0.0)
                        if top_anchor is not None
                        else None
                    ),
                    "top_anchor_source": (
                        str(getattr(getattr(top_anchor, "evidence_source", None), "value", "") or "")
                        if top_anchor is not None
                        else ""
                    ),
                }
            )
        return rows

    @staticmethod
    def _collect_word_time_confidence_for_fusion(
        *,
        words: Sequence[AnnotatedWord],
    ) -> Optional[List[float]]:
        values: List[float] = []
        for item in words or []:
            confidence = getattr(item, "confidence", None)
            if confidence is None:
                continue
            try:
                values.append(max(0.0, min(1.0, float(confidence))))
            except (TypeError, ValueError):
                continue
        return values or None

    def _build_soft_cut_plan_for_decision_m1(
        self,
        *,
        annotated_words: Sequence[AnnotatedWord],
        stream_id: str,
        block_id: str,
        is_last_chunk: bool,
        aligned_facts: Optional[AlignedFacts] = None,
        fused_evidence: Optional[FusedEvidence] = None,
        policy_snapshot: Optional[Any] = None,
    ) -> Optional[Any]:
        """
        在评分层 -> 裁决层之间生成 CutPlan。

        说明：
        - 关闭开关或无有效说话人变化时返回 None，保持旧路径；
        - 仅执行当前 chunk 的即时裁决，不维护 deferred 跨 chunk 状态。
        """
        if not self._is_enable_soft_cut:
            return self._build_empty_soft_cut_plan(
                stream_id=stream_id,
                block_id=block_id,
                reason="soft_cut_disabled",
            )
        words = list(annotated_words or [])
        if len(words) <= 1:
            return self._build_empty_soft_cut_plan(
                stream_id=stream_id,
                block_id=block_id,
                reason="insufficient_words",
            )

        chunk_start = float(words[0].start if words[0].start is not None else 0.0)
        chunk_end = float(words[-1].end if words[-1].end is not None else chunk_start)
        if chunk_end <= chunk_start:
            chunk_end = chunk_start + 1e-3

        speaker_change_facts = self._build_soft_cut_speaker_change_facts(words=words)
        anchor_candidates = self._build_soft_cut_anchor_candidates(
            words=words,
            policy_snapshot=policy_snapshot,
        )
        pause_anchors = [item for item in anchor_candidates if item.anchor_type == AnchorType.PAUSE_ANCHOR]
        word_anchors = [item for item in anchor_candidates if item.anchor_type == AnchorType.WORD_BOUNDARY]
        semantic_anchors = [item for item in anchor_candidates if item.anchor_type == AnchorType.SEMANTIC_ANCHOR]
        punctuation_anchors = [
            item for item in anchor_candidates if item.anchor_type == AnchorType.PUNCTUATION_ANCHOR
        ]

        builder_window_count = 0
        builder_high_count = 0
        builder_mid_count = 0
        builder_low_count = 0
        builder_result = self._soft_cut_evidence_builder.build(
            chunk_id=f"{stream_id}:{int(chunk_start * 1000)}",
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            speaker_change_facts=speaker_change_facts,
            anchor_candidates=anchor_candidates,
        )
        cut_windows: List[CutWindow] = list(builder_result.cut_windows or [])
        builder_window_count = int(builder_result.generation_report.get("window_count", 0))
        builder_high_count = int(builder_result.generation_report.get("high_count", 0))
        builder_mid_count = int(builder_result.generation_report.get("mid_count", 0))
        builder_low_count = int(builder_result.generation_report.get("low_count", 0))

        time_axis_version = str(getattr(aligned_facts, "time_axis_version", "m1_legacy") or "m1_legacy")
        fusion_result = self._soft_cut_evidence_fusion.fuse(
            cut_windows=cut_windows,
            pause_anchors=pause_anchors,
            word_anchors=word_anchors,
            semantic_anchors=semantic_anchors,
            punctuation_anchors=punctuation_anchors,
            time_axis_version=time_axis_version,
        )
        all_windows = list(fusion_result.fused_windows or [])
        if not all_windows:
            return self._build_empty_soft_cut_plan(
                stream_id=stream_id,
                block_id=block_id,
                reason="no_cut_windows",
            )

        window_contexts = self._resolve_soft_cut_window_contexts(
            words=words,
            cut_windows=all_windows,
        )

        plan = self._soft_cut_decision_engine.decide(
            block_id=block_id,
            cut_windows=all_windows,
            window_contexts=window_contexts,
            current_time=chunk_end,
        )
        plan.generation_report.update(
            {
                "builder_window_count": builder_window_count,
                "builder_high_count": builder_high_count,
                "builder_mid_count": builder_mid_count,
                "builder_low_count": builder_low_count,
                "priority_profile": str(self._soft_cut_priority_active_profile),
                "fusion_output_window_count": int(fusion_result.generation_report.get("output_window_count", 0)),
                "recompute_window_count": 0,
                "affected_window_count": 0,
                "affected_window_ids": [],
                "affected_reason_by_window": {},
                "affected_resolved_count": 0,
                "affected_expired_count": 0,
                "expired_on_last_chunk_count": 0,
                "deferred_state_stats": {
                    "pending": 0,
                    "resolved": 0,
                    "forced": 0,
                    "expired": 0,
                },
            }
        )
        return plan

    def _normalize_soft_cut_plan(
        self,
        *,
        plan: Optional[CutPlan],
        stream_id: str,
        block_id: str,
        reason: str,
    ) -> CutPlan:
        if plan is not None:
            return plan
        return self._build_empty_soft_cut_plan(
            stream_id=stream_id,
            block_id=block_id,
            reason=reason,
        )

    @staticmethod
    def _build_empty_soft_cut_plan(
        *,
        stream_id: str,
        block_id: str,
        reason: str,
    ) -> CutPlan:
        return CutPlan(
            plan_id=f"{block_id}-empty-plan",
            block_id=block_id,
            decisions=[],
            deferred_cuts=[],
            generation_report={
                "fallback_reason": str(reason or "unknown"),
                "stream_id": str(stream_id),
            },
        )

    def _build_soft_cut_speaker_change_facts(
        self,
        *,
        words: Sequence[AnnotatedWord],
    ) -> List[SpeakerChangeFact]:
        if not self._is_enable_speaker_guided_split:
            return []
        chunk_start = float(words[0].start if words and words[0].start is not None else 0.0)
        turn_duration_by_turn_id = self._build_turn_duration_by_turn_id()
        facts: List[SpeakerChangeFact] = []
        for index in range(1, len(words)):
            left = words[index - 1]
            right = words[index]
            left_speaker = str(left.speaker_id or "").strip()
            right_speaker = str(right.speaker_id or "").strip()
            is_valid_pair = bool(left_speaker and right_speaker)
            if not is_valid_pair or left_speaker == right_speaker:
                continue

            left_end = float(left.end if left.end is not None else 0.0)
            right_start = float(right.start if right.start is not None else left_end)
            pause_duration = max(0.0, right_start - left_end)
            left_duration = max(
                0.0,
                float(left.end if left.end is not None else left_end)
                - float(left.start if left.start is not None else left_end),
            )
            right_duration = max(
                0.0,
                float(right.end if right.end is not None else right_start)
                - float(right.start if right.start is not None else right_start),
            )
            # Why: 词级 speaker 跳变若仅发生在亚词级短词，通常是时间轴抖动，不应直接入窗。
            if (
                min(left_duration, right_duration) <= self._SOFT_CUT_SHORT_WORD_DURATION_SEC
                and pause_duration < self._SOFT_CUT_SHORT_WORD_MIN_PAUSE_SEC
            ):
                continue
            left_start = float(left.start if left.start is not None else chunk_start)
            is_leading_boundary = index == 1
            if is_leading_boundary:
                is_near_chunk_start = abs(left_start - chunk_start) <= self._SOFT_CUT_LEADING_WORD_MAX_OFFSET_SEC
                is_short_leading_word = left_duration <= self._SOFT_CUT_LEADING_WORD_MAX_DURATION_SEC
                is_gap_not_strong = pause_duration <= self._SOFT_CUT_LEADING_WORD_MAX_GAP_SEC
                # Why: chunk 开头短首词常由跨 chunk 边界的 speaker 变化映射导致，优先避免孤词切分。
                if is_near_chunk_start and is_short_leading_word and is_gap_not_strong:
                    continue
            left_turn_id = str(getattr(left, "turn_id", "") or "").strip()
            right_turn_id = str(getattr(right, "turn_id", "") or "").strip()
            known_turn_durations = []
            if left_turn_id:
                left_turn_duration = turn_duration_by_turn_id.get(left_turn_id)
                if left_turn_duration is not None:
                    known_turn_durations.append(left_turn_duration)
            else:
                inferred_left_turn_duration = self._infer_turn_duration_for_word(word=left)
                if inferred_left_turn_duration is not None:
                    known_turn_durations.append(inferred_left_turn_duration)
            if right_turn_id:
                right_turn_duration = turn_duration_by_turn_id.get(right_turn_id)
                if right_turn_duration is not None:
                    known_turn_durations.append(right_turn_duration)
            else:
                inferred_right_turn_duration = self._infer_turn_duration_for_word(word=right)
                if inferred_right_turn_duration is not None:
                    known_turn_durations.append(inferred_right_turn_duration)
            # Why: turn 时长过短时，词级 speaker 跳变大概率是 pyannote 回弹抖动，不进入 soft-cut。
            if known_turn_durations and min(known_turn_durations) < self._SOFT_CUT_MIN_TURN_DURATION_SEC:
                continue
            facts.append(
                SpeakerChangeFact(
                    time=right_start,
                    from_speaker=left_speaker,
                    to_speaker=right_speaker,
                    pyannote_confidence=0.85,
                    pause_duration=pause_duration,
                    embedding_distance=0.65,
                    embedding_threshold=0.50,
                    is_abrupt_energy_shift=pause_duration <= 0.05,
                )
            )
        facts.extend(self._build_soft_cut_timeline_change_facts(words=words))
        return self._dedupe_soft_cut_speaker_change_facts(facts=facts)

    def _build_turn_duration_by_turn_id(self) -> Dict[str, float]:
        durations: Dict[str, float] = {}
        for turn in list(self._timeline_turns or []):
            turn_id = str(getattr(turn, "turn_id", "") or "").strip()
            if not turn_id:
                continue
            turn_start_raw = getattr(turn, "start", None)
            turn_end_raw = getattr(turn, "end", None)
            turn_start = float(turn_start_raw) if turn_start_raw is not None else 0.0
            turn_end = float(turn_end_raw) if turn_end_raw is not None else turn_start
            durations[turn_id] = max(0.0, turn_end - turn_start)
        return durations

    def _build_soft_cut_timeline_change_facts(
        self,
        *,
        words: Sequence[AnnotatedWord],
    ) -> List[SpeakerChangeFact]:
        turns = sorted(
            list(self._timeline_turns or []),
            key=lambda item: (
                float(getattr(item, "start", 0.0)),
                float(getattr(item, "end", 0.0)),
            ),
        )
        if len(turns) <= 1 or not words:
            return []

        chunk_start = float(words[0].start if words[0].start is not None else 0.0)
        chunk_end = float(words[-1].end if words[-1].end is not None else chunk_start)
        if chunk_end <= chunk_start:
            chunk_end = chunk_start + 1e-3

        facts: List[SpeakerChangeFact] = []
        for index in range(1, len(turns)):
            left_turn = turns[index - 1]
            right_turn = turns[index]
            left_speaker = str(getattr(left_turn, "speaker_id", "") or "").strip()
            right_speaker = str(getattr(right_turn, "speaker_id", "") or "").strip()
            if not left_speaker or not right_speaker or left_speaker == right_speaker:
                continue

            right_start_raw = getattr(right_turn, "start", None)
            right_start = float(right_start_raw) if right_start_raw is not None else 0.0
            left_end_raw = getattr(left_turn, "end", None)
            left_end = float(left_end_raw) if left_end_raw is not None else right_start
            left_start_raw = getattr(left_turn, "start", None)
            left_start = float(left_start_raw) if left_start_raw is not None else left_end
            right_end_raw = getattr(right_turn, "end", None)
            right_end = float(right_end_raw) if right_end_raw is not None else right_start
            left_duration = max(0.0, left_end - left_start)
            right_duration = max(0.0, right_end - right_start)
            # Why: 短 turn 更可能是 pyannote 抖动回弹，直接触发 soft-cut 会放大为单词碎切。
            if min(left_duration, right_duration) < self._SOFT_CUT_MIN_TURN_DURATION_SEC:
                continue
            if self._is_short_turn_continuation_inside_word_body(
                words=words,
                turns=turns,
                index=index,
                right_speaker=right_speaker,
                right_start=right_start,
                right_end=right_end,
                right_duration=right_duration,
            ):
                continue
            # Why: chunk 边缘附近的 turn-change 更容易把首/尾词切成孤词，交给邻近 chunk 处理更稳妥。
            if (
                right_start <= chunk_start + self._SOFT_CUT_CHUNK_EDGE_GUARD_SEC
                or right_start >= chunk_end - self._SOFT_CUT_CHUNK_EDGE_GUARD_SEC
            ):
                continue
            if right_start < chunk_start or right_start > chunk_end:
                continue
            if not self._is_timeline_change_inside_word_span(words=words, trigger_time=right_start):
                continue

            boundary_confidence = float(getattr(right_turn, "boundary_confidence", 0.0) or 0.0)
            if boundary_confidence <= 0.0:
                boundary_confidence = float(getattr(left_turn, "boundary_confidence", 0.0) or 0.0)
            if boundary_confidence <= 0.0:
                boundary_confidence = 0.75

            pause_duration = max(0.0, right_start - left_end)
            facts.append(
                SpeakerChangeFact(
                    time=right_start,
                    from_speaker=left_speaker,
                    to_speaker=right_speaker,
                    pyannote_confidence=max(0.5, min(0.99, boundary_confidence)),
                    pause_duration=pause_duration,
                    embedding_distance=0.60,
                    embedding_threshold=0.50,
                    is_abrupt_energy_shift=pause_duration <= 0.05,
                    tags={SpeakerChangeTag.SUSPECTED_MISSED},
                )
            )
        return facts

    def _infer_turn_duration_for_word(
        self,
        *,
        word: AnnotatedWord,
    ) -> Optional[float]:
        """
        在 turn_id 缺失时按重叠回推词所属 turn 时长。

        Why:
        - 少数词级绑定会出现 turn_id 缺失，若直接放过会绕开短 turn 过滤并误触发切分。
        """
        word_start = float(word.start if word.start is not None else 0.0)
        word_end = float(word.end if word.end is not None else word_start)
        if word_end <= word_start:
            word_end = word_start + 1e-3

        best_overlap = 0.0
        best_duration: Optional[float] = None
        for turn in list(self._timeline_turns or []):
            turn_start_raw = getattr(turn, "start", None)
            turn_end_raw = getattr(turn, "end", None)
            turn_start = float(turn_start_raw) if turn_start_raw is not None else 0.0
            turn_end = float(turn_end_raw) if turn_end_raw is not None else turn_start
            overlap = max(0.0, min(word_end, turn_end) - max(word_start, turn_start))
            if overlap <= best_overlap:
                continue
            best_overlap = overlap
            best_duration = max(0.0, turn_end - turn_start)
        return best_duration

    def _is_short_turn_continuation_inside_word_body(
        self,
        *,
        words: Sequence[AnnotatedWord],
        turns: Sequence[Any],
        index: int,
        right_speaker: str,
        right_start: float,
        right_end: float,
        right_duration: float,
    ) -> bool:
        """
        过滤“短 turn + 同 speaker 续段 + 词内触发”的高风险切点。

        Why:
        - 该形态常见于笑声/呼吸触发的 pyannote 假边界
        - 若直接入窗，最终会稳定落在前一词末，形成“延后一词切”
        """
        if right_duration >= self._SOFT_CUT_SHORT_TURN_CONTINUATION_SEC:
            return False
        if index + 1 >= len(turns):
            return False

        next_turn = turns[index + 1]
        next_speaker = str(getattr(next_turn, "speaker_id", "") or "").strip()
        if not next_speaker or next_speaker != right_speaker:
            return False
        next_start_raw = getattr(next_turn, "start", None)
        next_start = float(next_start_raw) if next_start_raw is not None else right_end
        continuation_gap = max(0.0, next_start - right_end)
        if continuation_gap > self._SOFT_CUT_SHORT_TURN_CONTINUATION_GAP_SEC:
            return False
        return self._is_time_inside_word_body(words=words, trigger_time=right_start)

    def _is_time_inside_word_body(
        self,
        *,
        words: Sequence[AnnotatedWord],
        trigger_time: float,
    ) -> bool:
        for word in words:
            word_start = float(word.start if word.start is not None else 0.0)
            word_end = float(word.end if word.end is not None else word_start)
            if word_end <= word_start:
                continue
            if trigger_time <= word_start + self._SOFT_CUT_IN_WORD_BODY_MARGIN_SEC:
                continue
            if trigger_time >= word_end - self._SOFT_CUT_IN_WORD_BODY_MARGIN_SEC:
                continue
            return True
        return False

    @staticmethod
    def _is_overlap_turn_window(
        *,
        turns: Sequence[Any],
    ) -> bool:
        if len(turns) <= 1:
            return False
        ordered = sorted(
            list(turns),
            key=lambda item: (
                float(getattr(item, "start", 0.0)),
                float(getattr(item, "end", 0.0)),
            ),
        )
        for idx in range(1, len(ordered)):
            left = ordered[idx - 1]
            right = ordered[idx]
            left_end = float(getattr(left, "end", 0.0))
            right_start = float(getattr(right, "start", 0.0))
            if right_start < left_end - 1e-3:
                return True
        return False

    @staticmethod
    def _is_timeline_change_inside_word_span(
        *,
        words: Sequence[AnnotatedWord],
        trigger_time: float,
    ) -> bool:
        is_has_left_word = False
        is_has_right_word = False
        for word in words:
            word_start = float(word.start if word.start is not None else 0.0)
            word_end = float(word.end if word.end is not None else word_start)
            if word_end <= trigger_time + 1e-3:
                is_has_left_word = True
            if word_start >= trigger_time - 1e-3:
                is_has_right_word = True
            if is_has_left_word and is_has_right_word:
                return True
        return False

    @staticmethod
    def _dedupe_soft_cut_speaker_change_facts(
        *,
        facts: Sequence[SpeakerChangeFact],
    ) -> List[SpeakerChangeFact]:
        if not facts:
            return []

        ordered = sorted(
            list(facts),
            key=lambda item: (
                float(item.time),
                str(item.from_speaker),
                str(item.to_speaker),
            ),
        )
        merged: List[SpeakerChangeFact] = []
        for fact in ordered:
            if not merged:
                merged.append(fact)
                continue

            last = merged[-1]
            is_same_pair = (
                str(last.from_speaker) == str(fact.from_speaker)
                and str(last.to_speaker) == str(fact.to_speaker)
            )
            is_close_time = abs(float(last.time) - float(fact.time)) <= 0.08
            if not is_same_pair or not is_close_time:
                merged.append(fact)
                continue

            merged[-1] = SpeakerChangeFact(
                time=float(last.time) if float(last.pyannote_confidence) >= float(fact.pyannote_confidence) else float(fact.time),
                from_speaker=str(last.from_speaker),
                to_speaker=str(last.to_speaker),
                pyannote_confidence=max(float(last.pyannote_confidence), float(fact.pyannote_confidence)),
                pause_duration=max(float(last.pause_duration), float(fact.pause_duration)),
                embedding_distance=max(float(last.embedding_distance), float(fact.embedding_distance)),
                embedding_threshold=min(float(last.embedding_threshold), float(fact.embedding_threshold)),
                is_abrupt_energy_shift=bool(last.is_abrupt_energy_shift or fact.is_abrupt_energy_shift),
                tags=set(last.tags).union(set(fact.tags)),
            )
        return merged

    def _build_soft_cut_anchor_candidates(
        self,
        *,
        words: Sequence[AnnotatedWord],
        policy_snapshot: Optional[Any] = None,
    ) -> List[AnchorCandidate]:
        anchors: List[AnchorCandidate] = []
        last_pause_anchor_index: Optional[int] = None
        semantic_anchor_words = self._resolve_soft_cut_semantic_anchor_words(policy_snapshot)
        for index in range(1, len(words)):
            left = words[index - 1]
            right = words[index]
            left_end = float(left.end if left.end is not None else 0.0)
            right_start = float(right.start if right.start is not None else left_end)
            boundary_time = right_start
            pause_duration = max(0.0, right_start - left_end)

            anchors.append(
                AnchorCandidate(
                    anchor_type=AnchorType.WORD_BOUNDARY,
                    anchor_time=boundary_time,
                    source="word_boundary",
                    confidence=1.0,
                )
            )
            if pause_duration >= self._SOFT_CUT_PAUSE_ANCHOR_TRIGGER_SEC:
                pause_anchor = AnchorCandidate(
                    anchor_type=AnchorType.PAUSE_ANCHOR,
                    anchor_time=boundary_time,
                    source="pause_gap",
                    confidence=min(1.0, 0.5 + pause_duration),
                )
                # Why: 相邻短间隔 pause 会在同一语义片段内生成过密窗口，需抑制重复弱锚点。
                if last_pause_anchor_index is not None:
                    previous_pause = anchors[last_pause_anchor_index]
                    anchor_gap = max(0.0, float(pause_anchor.anchor_time) - float(previous_pause.anchor_time))
                    if anchor_gap < self._SOFT_CUT_PAUSE_ANCHOR_MIN_GAP_SEC:
                        previous_confidence = float(previous_pause.confidence or 0.0)
                        current_confidence = float(pause_anchor.confidence or 0.0)
                        if current_confidence >= previous_confidence:
                            anchors[last_pause_anchor_index] = pause_anchor
                        continue
                anchors.append(pause_anchor)
                last_pause_anchor_index = len(anchors) - 1

            trailing_punct = str(left.trailing_punct or "").strip()
            if trailing_punct and trailing_punct[-1] in {"。", "！", "？", ".", "!", "?"}:
                punct_char = trailing_punct[-1]
                punct_confidence = (
                    self._SOFT_CUT_PUNCT_ANCHOR_CONFIDENCE_STRONG
                    if punct_char in {"！", "？", "!", "?"}
                    else self._SOFT_CUT_PUNCT_ANCHOR_CONFIDENCE_WEAK
                )
                anchors.append(
                    AnchorCandidate(
                        anchor_type=AnchorType.PUNCTUATION_ANCHOR,
                        anchor_time=boundary_time,
                        source=f"punct_proxy:{punct_char}",
                        confidence=punct_confidence,
                    )
                )

            right_word = str(right.word or "").strip().lower()
            if right_word in semantic_anchor_words:
                anchors.append(
                    AnchorCandidate(
                        anchor_type=AnchorType.SEMANTIC_ANCHOR,
                        anchor_time=boundary_time,
                        source="conjunction_rule",
                        confidence=0.3,
                    )
                )
        return anchors

    def _resolve_soft_cut_semantic_anchor_words(
        self,
        policy_snapshot: Optional[Any],
    ) -> set[str]:
        if policy_snapshot is not None:
            snapshot_words = getattr(policy_snapshot, "semantic_anchor_words", None)
            if snapshot_words:
                normalized = {
                    str(item or "").strip().lower()
                    for item in snapshot_words
                    if str(item or "").strip()
                }
                if normalized:
                    return normalized
        return set(self._soft_cut_semantic_conjunctions)

    def _resolve_soft_cut_window_contexts(
        self,
        *,
        words: Sequence[AnnotatedWord],
        cut_windows: Sequence[Any],
    ) -> Dict[str, WindowDecisionContext]:
        contexts: Dict[str, WindowDecisionContext] = {}
        for window in cut_windows:
            trigger_time = float(getattr(window, "trigger_time", 0.0))
            window_start = float(getattr(window, "start_time", trigger_time))
            window_end = float(getattr(window, "end_time", trigger_time))
            covered_words = 0
            has_fast_source = False
            has_non_fast_source = False
            for word in words:
                source = str(getattr(word, "confidence_source", "") or "").strip().lower()
                word_end = float(word.end if word.end is not None else 0.0)
                if word_end <= trigger_time:
                    covered_words += 1
                word_start = float(word.start if word.start is not None else word_end)
                if word_end <= word_start:
                    word_end = word_start + 1e-3
                if word_end < window_start - 1e-6 or word_start > window_end + 1e-6:
                    continue
                if not source:
                    continue
                if self._is_fast_confidence_source(source):
                    has_fast_source = True
                else:
                    has_non_fast_source = True
            window_id = str(getattr(window, "window_id", "") or "")
            if not window_id:
                continue
            contexts[window_id] = WindowDecisionContext(
                current_sentence_word_count=covered_words,
                waiting_word_count=max(0, len(words) - covered_words),
                depends_on_fast_draft=bool(has_fast_source and not has_non_fast_source),
            )
        return contexts

    @staticmethod
    def _is_fast_confidence_source(source: str) -> bool:
        normalized = str(source or "").strip().lower()
        return normalized in {"fast", "sensevoice", "sv", "draft", "fast_draft", "m1_fast"}

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

    def _emit_output_layer(
        self,
        *,
        chunk_index: int,
        sentence_segments: Sequence[SentenceSegment],
        language: str,
        injection_report: Dict[str, Any],
        segmentation_report: Dict[str, Any],
        output_traces: Optional[Sequence[OutputTrace]],
        default_trace_reason: str,
    ) -> Any:
        """
        输出层唯一分发方法（内部主入口）。

        Why:
        - 输出层要求单入口，所有路径统一经过 OutputLayerProcessor；
        - trace 在入口先对齐句段数量，保障 transport payload 稳定可追溯。
        """
        normalized_traces = self._normalize_output_traces_for_sentences(
            final_sentences=sentence_segments,
            output_traces=output_traces or [],
            default_reason=default_trace_reason,
        )
        return self._output_processor.process(
            OutputLayerInput(
                chunk_index=chunk_index,
                sentence_segments=list(sentence_segments),
                language=str(language or "auto"),
                injection_report=dict(injection_report),
                segmentation_report=dict(segmentation_report),
                output_traces=normalized_traces,
            )
        )

    @staticmethod
    def _normalize_output_traces_for_sentences(
        *,
        final_sentences: Sequence[SentenceSegment],
        output_traces: Sequence[OutputTrace],
        default_reason: str,
    ) -> List[OutputTrace]:
        if len(final_sentences) == len(output_traces):
            return list(output_traces)
        normalized: List[OutputTrace] = []
        for sentence_index, sentence in enumerate(final_sentences):
            normalized.append(
                OutputTrace(
                    sentence_index=sentence_index,
                    split_reason=str(getattr(sentence, "split_reason", "") or default_reason),
                    split_risk=str(getattr(sentence, "split_risk", "") or ""),
                    window_id=str(getattr(sentence, "window_id", "") or ""),
                    pyannote_frame_time=getattr(sentence, "pyannote_frame_time", None),
                    mapped_cut_time=getattr(sentence, "mapped_cut_time", None),
                    mapping_quality=str(getattr(sentence, "mapping_quality", "") or "default"),
                    mapping_reason=str(getattr(sentence, "mapping_reason", "") or default_reason),
                    sentence_start=float(sentence.start),
                    sentence_end=float(sentence.end),
                )
            )
        return normalized

    @staticmethod
    def _build_final_fallback_sentence(words_for_split: Sequence[WordTimestamp]) -> Optional[SentenceSegment]:
        """基于裁决层输入词流构建定稿单句兜底。"""
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
        await self._align_loop_service.run(
            results=results,
            job_dir=job_dir,
            total_chunks=total_chunks,
            base_align_count=base_align_count,
            initial_finalized_indices=initial_finalized_indices,
        )

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
def get_async_dual_pipeline_kernel(
    job_id: str,
    queue_maxsize: int = 5,
    logger: Optional[logging.Logger] = None,
    cancellation_token: Optional["CancellationToken"] = None  # v3.1.0: 新增
) -> AsyncDualPipelineKernel:
    """
    获取异步双流流水线实例

    Args:
        job_id: 任务 ID
        queue_maxsize: 队列最大长度
        logger: 日志记录器
        cancellation_token: 取消令牌（可选，v3.1.0）

    Returns:
        AsyncDualPipelineKernel 实例
    """
    return AsyncDualPipelineKernel(
        job_id=job_id,
        queue_maxsize=queue_maxsize,
        logger=logger,
        cancellation_token=cancellation_token  # v3.1.0
    )















