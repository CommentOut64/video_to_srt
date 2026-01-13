"""
转录处理服务
整合了processor.py和原transcription_service.py的所有功能
"""
import os, subprocess, uuid, threading, json, math, gc, logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import OrderedDict  # 新增导入
from pydub import AudioSegment, silence
from app.services.whisper_service import get_whisper_service, load_audio as whisper_load_audio
from app.services.config_adapter import ConfigAdapter
# 从新架构导入 VAD 配置（2025-12-17 统一配置定义）
from app.services.audio.vad_service import VADConfig, VADMethod, get_vad_service
from app.services.audio.chunk_engine import ChunkEngine, AudioChunk
import torch
import shutil
import psutil
import numpy as np


class ProcessingMode(Enum):
    """
    处理模式枚举
    用于智能决策使用内存模式还是硬盘模式进行音频处理

    注意：DISK 模式（硬盘分段）已废弃，仅保留 MEMORY 模式
    """
    MEMORY = "memory"  # 内存模式（默认，高性能）
    DISK = "disk"      # 硬盘模式（已废弃，仅保留用于向后兼容）


class BreakToGlobalSeparation(Exception):
    """熔断异常：触发时需要升级为全局人声分离模式"""
    pass


@dataclass
class CircuitBreakerState:
    """
    熔断器状态（支持模型升级）

    用于监控转录质量，当大量段落需要重试时：
    1. 优先尝试升级模型（如果允许且未达上限）
    2. 无法升级时才触发熔断
    """
    consecutive_retries: int = 0        # 连续重试计数
    total_retries: int = 0              # 总重试次数
    total_segments: int = 0             # 总段落数
    processed_segments: int = 0         # 已处理段落数

    # === Phase 3: 升级跟踪 ===
    escalation_count: int = 0                           # 已升级次数
    current_model: Optional[str] = None                 # 当前使用的模型
    escalation_history: List[str] = field(default_factory=list)  # 升级历史

    def record_retry(self):
        """记录一次重试"""
        self.consecutive_retries += 1
        self.total_retries += 1

    def record_success(self):
        """记录一次成功（重置连续计数）"""
        self.consecutive_retries = 0
        self.processed_segments += 1

    def record_escalation(self, new_model: str):
        """
        记录一次模型升级

        Args:
            new_model: 升级后的模型名称
        """
        if self.current_model:
            self.escalation_history.append(f"{self.current_model} -> {new_model}")
        self.current_model = new_model
        self.escalation_count += 1
        # 升级后重置连续重试计数，给新模型机会
        self.consecutive_retries = 0

    def should_escalate(self, demucs_settings) -> bool:
        """
        判断是否应该升级模型（优先于熔断）

        升级条件：
        1. 允许自动升级 (auto_escalation=True)
        2. 未达到最大升级次数
        3. 满足熔断条件（连续重试或比例过高）

        Args:
            demucs_settings: Demucs配置对象

        Returns:
            bool: True表示应该升级模型
        """
        if not demucs_settings.auto_escalation:
            return False

        if self.escalation_count >= demucs_settings.max_escalations:
            return False

        # 满足熔断条件时，优先升级
        return self._check_break_condition(demucs_settings)

    def should_break(self, demucs_settings) -> bool:
        """
        判断是否应该触发熔断

        注意：只有在无法升级时才触发熔断

        Args:
            demucs_settings: Demucs配置对象

        Returns:
            bool: True表示应该触发熔断
        """
        if not demucs_settings.circuit_breaker_enabled:
            return False

        # 如果还能升级，不触发熔断
        if self.should_escalate(demucs_settings):
            return False

        return self._check_break_condition(demucs_settings)

    def _check_break_condition(self, demucs_settings) -> bool:
        """
        检查是否满足熔断/升级条件

        熔断条件（满足任一即触发）：
        1. 连续 N 个 segment 都触发重试（默认N=3）
        2. 总重试比例超过阈值（默认20%）

        Args:
            demucs_settings: Demucs配置对象

        Returns:
            bool: True表示满足熔断/升级条件
        """
        # 条件1：连续重试次数
        if self.consecutive_retries >= demucs_settings.consecutive_threshold:
            return True

        # 条件2：总重试比例（至少处理5个segment后才检查）
        if self.processed_segments >= 5:
            retry_ratio = self.total_retries / self.processed_segments
            if retry_ratio >= demucs_settings.ratio_threshold:
                return True

        return False

    def get_stats(self) -> Dict:
        """获取统计信息（扩展）"""
        return {
            "consecutive_retries": self.consecutive_retries,
            "total_retries": self.total_retries,
            "total_segments": self.total_segments,
            "processed_segments": self.processed_segments,
            "retry_ratio": self.total_retries / max(1, self.processed_segments),
            # Phase 3 新增
            "escalation_count": self.escalation_count,
            "current_model": self.current_model,
            "escalation_history": self.escalation_history,
        }


class CircuitBreakAction(Enum):
    """熔断后的处理动作"""
    CONTINUE = "continue"           # 继续处理，标记问题段落
    FALLBACK_ORIGINAL = "fallback"  # 降级使用原始音频
    FAIL = "fail"                   # 任务失败
    PAUSE = "pause"                 # 暂停等待人工介入


class CircuitBreakHandler:
    """
    熔断异常处理器

    负责在熔断触发时执行用户配置的处理策略
    """

    def __init__(self, job: "JobState", settings):
        """
        初始化熔断处理器

        Args:
            job: 任务状态对象
            settings: DemucsSettings 配置对象
        """
        self.job = job
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.problem_segments: List[int] = []  # 记录问题段落索引

    def handle(
        self,
        breaker_state: CircuitBreakerState,
        current_segment_idx: int,
        sse_manager = None
    ) -> CircuitBreakAction:
        """
        处理熔断异常

        Args:
            breaker_state: 熔断器状态
            current_segment_idx: 当前段落索引
            sse_manager: SSE管理器（用于推送事件）

        Returns:
            CircuitBreakAction: 处理动作
        """
        action_str = self.settings.on_break

        # 解析处理动作
        try:
            action = CircuitBreakAction(action_str)
        except ValueError:
            # 如果配置值无效，默认使用 CONTINUE
            self.logger.warning(f"无效的熔断处理策略: {action_str}，使用默认值 continue")
            action = CircuitBreakAction.CONTINUE

        # 记录问题段落
        self.problem_segments.append(current_segment_idx)

        # 推送 SSE 事件
        if sse_manager:
            self._push_circuit_break_event(breaker_state, action, sse_manager)

        # 根据策略执行操作
        if action == CircuitBreakAction.FAIL:
            self.logger.error(
                f"熔断触发，任务终止。问题段落: {self.problem_segments}"
            )
            raise BreakToGlobalSeparation(
                f"熔断触发，任务终止。问题段落: {self.problem_segments}"
            )

        elif action == CircuitBreakAction.PAUSE:
            self.logger.warning(
                f"熔断触发，等待人工介入。问题段落: {self.problem_segments}"
            )
            self.job.paused = True
            self.job.status = "paused"
            self.job.message = f"熔断触发，等待人工介入。问题段落: {self.problem_segments}"
            raise BreakToGlobalSeparation(self.job.message)

        else:  # CONTINUE 或 FALLBACK_ORIGINAL
            self.logger.warning(
                f"熔断触发，采用 {action.value} 策略继续处理。"
                f"问题段落: {self.problem_segments}"
            )

        return action

    def get_problem_report(self) -> Dict:
        """
        获取问题报告

        Returns:
            包含问题统计和建议的字典
        """
        return {
            "total_problem_segments": len(self.problem_segments),
            "problem_indices": self.problem_segments,
            "suggestion": self._get_suggestion()
        }

    def _get_suggestion(self) -> str:
        """
        根据问题段落数量给出建议

        Returns:
            建议文本
        """
        count = len(self.problem_segments)
        if count == 0:
            return "所有段落处理正常"
        elif count <= 3:
            return "少量段落可能需要手动调整时间轴"
        elif count <= 10:
            return "建议检查这些段落的字幕准确性"
        else:
            return "大量段落有问题，建议使用更高质量的模型重新处理"

    def _push_circuit_break_event(
        self,
        state: CircuitBreakerState,
        action: CircuitBreakAction,
        sse_manager
    ):
        """
        推送熔断处理事件

        Args:
            state: 熔断器状态
            action: 处理动作
            sse_manager: SSE管理器
        """
        try:
            sse_manager.push_event(
                self.job.job_id,
                "circuit_breaker_handled",
                {
                    "action": action.value,
                    "problem_segments": self.problem_segments,
                    "stats": state.get_stats(),
                    "suggestion": self._get_suggestion()
                }
            )
        except Exception as e:
            self.logger.debug(f"SSE推送失败（非致命）: {e}")


from app.models.job_models import JobSettings, JobState
from app.models.hardware_models import HardwareInfo, OptimizationConfig
from app.services.hardware_service import get_hardware_detector, get_hardware_optimizer
from app.services.cpu_affinity_service import CPUAffinityManager, CPUAffinityConfig
from app.services.job_index_service import get_job_index_service
from app.core.config import config  # 导入统一配置

# 全局模型缓存 (按 (model, compute_type, device) 键)
_model_cache: Dict[Tuple[str, str, str], object] = {}


_model_lock = threading.Lock()


class TranscriptionService:
    """
    转录处理服务
    整合了所有转录相关功能
    """

    def __init__(self, jobs_root: str):
        """
        初始化转录服务

        Args:
            jobs_root: 任务工作目录根路径
        """
        self.jobs_root = Path(jobs_root)
        self.jobs_root.mkdir(parents=True, exist_ok=True)

        self.jobs: Dict[str, JobState] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

        # 集成CPU亲和性管理器
        self.cpu_manager = CPUAffinityManager()

        # 集成硬件检测
        self.hardware_detector = get_hardware_detector()
        self.hardware_optimizer = get_hardware_optimizer()
        self._hardware_info: Optional[HardwareInfo] = None
        self._optimization_config: Optional[OptimizationConfig] = None

        # 集成任务索引服务
        self.job_index = get_job_index_service(jobs_root)
        # 启动时清理无效映射
        self.job_index.cleanup_invalid_mappings()

        # 集成SSE管理器（用于实时进度推送）
        from app.services.sse_service import get_sse_manager
        self.sse_manager = get_sse_manager()
        self.logger.info("SSE管理器已集成")

        # 初始化新架构 Pipeline（2025-12-17 架构改造）
        from app.pipelines.audio_processing_pipeline import AudioProcessingPipeline
        from app.pipelines.async_dual_pipeline import AsyncDualPipeline

        self.audio_pipeline = None  # 延迟初始化，等硬件检测完成
        self.transcription_pipeline = None  # 延迟初始化，等硬件检测完成
        self.logger.info("Pipeline 架构已准备")

        # 记录CPU信息
        sys_info = self.cpu_manager.get_system_info()
        if sys_info.get('supported', False):
            self.logger.info(
                f" CPU信息: {sys_info['logical_cores']}个逻辑核心, "
                f"{sys_info.get('physical_cores', '?')}个物理核心, "
                f"平台: {sys_info.get('platform', '?')}"
            )
        else:
            self.logger.warning("CPU亲和性功能不可用")

        # 执行硬件检测
        self._detect_hardware()

        # 启动时扫描并加载所有任务（修复重启后无法打开旧任务的问题）
        self._load_all_jobs_from_disk()

    def _detect_hardware(self):
        """执行硬件检测并生成优化配置"""
        try:
            self.logger.info("开始硬件检测...")
            self._hardware_info = self.hardware_detector.detect()
            self._optimization_config = self.hardware_optimizer.get_optimization_config(self._hardware_info)

            # 记录检测结果
            hw = self._hardware_info
            opt = self._optimization_config
            self.logger.info(f"硬件检测完成GPU: {'' if hw.cuda_available else ''}, "
                           f"CPU: {hw.cpu_cores}核/{hw.cpu_threads}线程, "
                           f"内存: {hw.memory_total_mb}MB, "
                           f"优化配置: batch={opt.batch_size}, device={opt.recommended_device}")

            # 硬件检测完成后，初始化 Pipeline
            self._initialize_pipelines()
        except Exception as e:
            self.logger.error(f"硬件检测失败: {e}")

    def _initialize_pipelines(self):
        """
        初始化新架构 Pipeline（2025-12-17 架构改造）

        职责：组装 - 根据硬件配置初始化 AudioProcessingPipeline 和 AsyncDualPipeline

        注意：PreprocessingPipeline 不在这里初始化，因为它需要动态配置（每个任务可能不同）
        """
        try:
            from app.pipelines.audio_processing_pipeline import (
                AudioProcessingPipeline,
                AudioProcessingConfig
            )
            from app.pipelines.async_dual_pipeline import AsyncDualPipeline

            # 初始化音频处理流水线（旧架构，保持向后兼容）
            audio_config = AudioProcessingConfig(
                vad_config=VADConfig(),  # 使用默认 VAD 配置
                enable_demucs=True,
                auto_strategy=True
            )
            self.audio_pipeline = AudioProcessingPipeline(
                logger=self.logger
            )

            # V3.5: 转录流水线在每次任务执行时动态创建（根据 transcription_profile 配置）
            # 不再在这里初始化 self.transcription_pipeline
            self.transcription_pipeline = None

            self.logger.info("Pipeline 初始化完成")
        except Exception as e:
            self.logger.error(f"Pipeline 初始化失败: {e}")
            self.audio_pipeline = None
            self.transcription_pipeline = None

    async def _run_pipeline_v2(self, job: JobState):
        """
        新架构 Pipeline 入口方法（2025-12-17 架构改造）

        职责：
        1. 触发：接收任务，传给 Pipeline
        2. 收尾：拿到结果，生成 SRT 文件

        注意：此方法是简化版，暂不支持断点续传、任务状态管理等复杂功能
        这些功能将在后续阶段逐步集成

        支持两种预处理模式：
        - 新架构：PreprocessingPipeline（Stage模式，支持频谱分诊、按需分离、熔断回溯）
        - 旧架构：AudioProcessingPipeline（整轨分离）

        Args:
            job: 任务状态对象
        """
        try:
            from app.pipelines.async_dual_pipeline import AsyncDualPipeline
            from app.services.config_adapter import ConfigAdapter

            job_dir = Path(job.dir)
            input_path = job_dir / job.filename

            # ==========================================
            # 阶段 1: 音频前处理
            # ==========================================
            self._update_progress(job, 'audio_processing', 0, '音频处理中...')

            # 判断使用哪种预处理架构
            use_new_preprocessing = self._should_use_new_preprocessing(job.settings)

            if use_new_preprocessing:
                # 使用新架构：PreprocessingPipeline（Stage模式）
                self.logger.info("使用新架构预处理流水线（Stage模式）")
                chunks = await self._run_new_preprocessing(job, input_path)
            else:
                # 使用旧架构：AudioProcessingPipeline（整轨分离）
                self.logger.info("使用旧架构预处理流水线（整轨分离）")
                if not self.audio_pipeline:
                    raise RuntimeError("旧架构 Pipeline 未初始化")

                audio_result = await self.audio_pipeline.process(
                    video_path=str(input_path),
                    progress_callback=lambda p: self._update_progress(
                        job, 'audio_processing', p, f'音频处理中 {int(p*100)}%'
                    )
                )
                chunks = audio_result.chunks

            job.total = len(chunks)
            self.logger.info(f"音频处理完成: {len(chunks)} 个 Chunk")

            # ==========================================
            # 阶段 2: 异步双流转录（AsyncDualPipeline）
            # V3.5: 根据 transcription_profile 动态创建流水线
            # ==========================================
            self._update_progress(job, 'transcription', 0, '转录中...')

            # 获取转录模式配置
            transcription_profile = ConfigAdapter.get_transcription_profile(job.settings)
            self.logger.info(f"转录模式: {transcription_profile}")

            # 动态创建转录流水线
            self.transcription_pipeline = AsyncDualPipeline(
                job_id=job.job_id,
                transcription_profile=transcription_profile,
                logger=self.logger
            )

            # 调用转录流水线
            results = await self.transcription_pipeline.run(
                audio_chunks=chunks
            )

            # 从 ProcessingContext 中提取句子
            final_sentences = []
            for ctx in results:
                if hasattr(ctx, 'sv_result') and ctx.sv_result:
                    # 从 streaming_subtitle 获取句子
                    from app.services.streaming_subtitle import get_streaming_subtitle_manager
                    subtitle_manager = get_streaming_subtitle_manager(job.job_id)
                    chunk_indices = subtitle_manager.chunk_sentences.get(ctx.chunk_index, [])
                    for idx in chunk_indices:
                        if idx in subtitle_manager.sentences:
                            final_sentences.append(subtitle_manager.sentences[idx])

            self.logger.info(f"转录完成: {len(final_sentences)} 个句子")

            # ==========================================
            # 阶段 3: 生成 SRT 文件（收尾）
            # ==========================================
            self._update_progress(job, 'finalize', 0, '生成字幕文件...')

            srt_path = job_dir / f"{Path(job.filename).stem}.srt"
            self._generate_srt_from_sentences(final_sentences, srt_path)

            job.srt_path = str(srt_path)
            job.status = 'completed'
            job.message = '转录完成'
            job.progress = 100

            self.logger.info(f"任务完成: {job.job_id}")

        except Exception as e:
            self.logger.error(f"Pipeline 执行失败: {e}", exc_info=True)
            job.status = 'failed'
            job.error = str(e)
            job.message = f'失败: {str(e)}'
            raise

    def _generate_srt_from_sentences(self, sentences: List, output_path: Path):
        """
        从句子列表生成 SRT 文件
        V3.1.1+dev.20260106.03: 生成前自动修复时间戳重叠

        Args:
            sentences: 句子列表
            output_path: 输出路径
        """
        # V3.1.1+dev.20260106.03: 转换为字典列表以便修复重叠
        from app.utils.text_utils import repair_timestamp_overlaps, detect_timestamp_overlaps

        segments = []
        for sentence in sentences:
            segments.append({
                'start': sentence.start,
                'end': sentence.end,
                'text': sentence.text
            })

        # 检测并修复重叠
        overlaps = detect_timestamp_overlaps(segments)
        if overlaps:
            self.logger.warning(f"检测到 {len(overlaps)} 处时间戳重叠，自动修复中...")
            segments = repair_timestamp_overlaps(segments, gap_ms=1.0)
            self.logger.info(f"已修复 {len(overlaps)} 处时间戳重叠")

        # 生成 SRT 内容
        srt_content = []
        for i, seg in enumerate(segments, 1):
            start = self._format_srt_timestamp(seg['start'])
            end = self._format_srt_timestamp(seg['end'])
            text = seg['text']

            srt_content.append(f"{i}\n{start} --> {end}\n{text}\n")

        output_path.write_text('\n'.join(srt_content), encoding='utf-8')
        self.logger.info(f"SRT 文件已生成: {output_path}")

    def _format_srt_timestamp(self, seconds: float) -> str:
        """
        格式化 SRT 时间戳

        Args:
            seconds: 秒数

        Returns:
            SRT 格式时间戳 (HH:MM:SS,mmm)
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _should_use_new_preprocessing(self, settings: JobSettings) -> bool:
        """
        判断是否使用新架构预处理流水线

        判断逻辑：
        - 如果启用了频谱分诊（enable_spectral_triage=True）
        - 或者启用了熔断回溯（enable_fuse_breaker=True）
        - 或者使用按需分离模式（separation_mode='on_demand'）
        则使用新架构

        Args:
            settings: 任务设置

        Returns:
            bool: True表示使用新架构，False表示使用旧架构
        """
        preprocessing = settings.preprocessing

        # 调试日志：输出配置值
        self.logger.info(
            f"预处理配置检查: "
            f"enable_spectral_triage={preprocessing.enable_spectral_triage}, "
            f"enable_fuse_breaker={preprocessing.enable_fuse_breaker}, "
            f"separation_mode={preprocessing.separation_mode}"
        )

        # 检查是否启用了新功能
        use_new = (
            preprocessing.enable_spectral_triage or
            preprocessing.enable_fuse_breaker or
            preprocessing.separation_mode == 'on_demand'
        )

        self.logger.info(f"使用{'新' if use_new else '旧'}架构预处理流水线")

        return use_new

    async def _run_new_preprocessing(self, job: JobState, input_path: Path) -> List:
        """
        使用新架构预处理流水线处理音频

        Args:
            job: 任务状态对象
            input_path: 输入文件路径

        Returns:
            List[AudioChunk]: 预处理完成的 Chunk 列表
        """
        from app.pipelines.preprocessing_pipeline import PreprocessingPipeline
        from app.services.audio.chunk_engine import ChunkEngine
        from app.services.audio.vad_service import VADConfig

        # V3.9: 根据引擎类型选择 VAD 配置
        # 英语使用 Whisper，需要合并 VAD（避免幻觉）
        # 其他语言使用 SenseVoice，需要保留停顿信息
        language = getattr(job.settings, 'language', 'auto')
        is_english = language in {'en', 'english'}

        if is_english:
            # Whisper 模式：合并 VAD，避免幻觉
            vad_config = VADConfig(
                merge_max_gap=1.0,
                merge_max_duration=12.0,
                smart_target_duration=12.0
            )
            self.logger.info("使用 Whisper VAD 配置（合并模式）")
        else:
            # SenseVoice 模式：保留停顿信息
            vad_config = VADConfig(
                merge_max_gap=0.3,           # 只合并极短停顿
                merge_max_duration=8.0,      # 更短的 chunk
                smart_target_duration=8.0    # 软上限降低
            )
            self.logger.info("使用 SenseVoice VAD 配置（保留停顿）")

        # 创建自定义 ChunkEngine
        chunk_engine = ChunkEngine(
            vad_config=vad_config,
            logger=self.logger
        )

        # 创建 PreprocessingPipeline 实例
        preprocessing_pipeline = PreprocessingPipeline(
            config=job.settings.preprocessing,
            chunk_engine=chunk_engine,
            logger=self.logger
        )

        # 执行预处理
        chunks = await preprocessing_pipeline.process(
            video_path=str(input_path),
            job_state=job
        )

        # 记录统计信息
        stats = preprocessing_pipeline.get_statistics(chunks)
        self.logger.info(
            f"预处理统计: "
            f"总chunk数={stats['total_chunks']}, "
            f"需要分离={stats['need_separation']}, "
            f"已分离={stats['separated']}, "
            f"分离比例={stats['separation_ratio']:.2%}, "
            f"熔断重试总次数={stats['fuse_retry_total']}, "
            f"最大重试次数={stats['fuse_retry_max']}"
        )

        return chunks

    def _load_all_jobs_from_disk(self):
        """
        启动时扫描并加载所有任务到内存（修复重启后无法打开旧任务的问题）

        这个方法会扫描 jobs 目录中的所有任务，并加载到内存中，
        避免重启后因为内存为空导致无法访问旧任务
        """
        try:
            loaded_count = 0
            for job_dir in self.jobs_root.iterdir():
                if not job_dir.is_dir():
                    continue

                job_id = job_dir.name
                if job_id in self.jobs:
                    # 已加载，跳过
                    continue

                # 尝试加载任务
                job = self.get_job(job_id)
                if job:
                    loaded_count += 1

            if loaded_count > 0:
                self.logger.info(f"启动时已加载 {loaded_count} 个历史任务到内存")
        except Exception as e:
            self.logger.error(f"加载历史任务失败: {e}")
    
    def get_hardware_info(self) -> Optional[HardwareInfo]:
        """获取硬件信息"""
        return self._hardware_info
    
    def get_optimization_config(self) -> Optional[OptimizationConfig]:
        """获取优化配置"""  
        return self._optimization_config
    
    def get_optimized_job_settings(self, base_settings: Optional[JobSettings] = None) -> JobSettings:
        """获取基于硬件优化的任务设置"""
        # 使用硬件优化配置作为默认值
        if self._optimization_config:
            optimized = JobSettings(
                model=base_settings.model if base_settings else "medium",
                compute_type=base_settings.compute_type if base_settings else "auto",
                device=self._optimization_config.recommended_device,
                batch_size=self._optimization_config.batch_size,
                word_timestamps=base_settings.word_timestamps if base_settings else False
            )
            return optimized
        
        # 如果没有硬件信息，使用传入的设置或默认设置
        return base_settings or JobSettings()

    def create_job(
        self,
        filename: str,
        src_path: str,
        settings: JobSettings,
        job_id: Optional[str] = None
    ) -> JobState:
        """
        创建转录任务

        Args:
            filename: 文件名
            src_path: 源文件路径
            settings: 任务设置
            job_id: 任务ID（可选，不提供则自动生成）

        Returns:
            JobState: 创建的任务状态对象
        """
        job_id = job_id or uuid.uuid4().hex
        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        dest_path = job_dir / filename

        # V3.1.1+dev.20260106.01: 使用硬链接替代复制，节省磁盘空间
        # 硬链接让 input/video.mp4 和 jobs/{job_id}/video.mp4 指向同一数据块
        # 支持多个任务指向同一个视频文件，删除任务时不影响原始文件
        if os.path.abspath(src_path) != os.path.abspath(dest_path):
            try:
                # 优先使用硬链接
                os.link(src_path, dest_path)
                self.logger.debug(f"硬链接创建成功: {src_path} -> {dest_path}")
            except (OSError, NotImplementedError) as e:
                # 硬链接失败时降级到复制（跨文件系统、网络挂载等场景）
                self.logger.warning(f"硬链接创建失败，回退到复制: {e}")
                try:
                    shutil.copyfile(src_path, dest_path)
                    self.logger.debug(f"文件已复制: {src_path} -> {dest_path}")
                except Exception as copy_err:
                    self.logger.warning(f"文件复制失败: {copy_err}")

        # 创建任务状态对象
        job = JobState(
            job_id=job_id,
            filename=filename,
            dir=str(job_dir),
            input_path=src_path,
            settings=settings,
            status="uploaded",
            phase="pending",
            message="文件已上传"
        )

        with self.lock:
            self.jobs[job_id] = job

        # 添加文件路径到任务ID的映射
        self.job_index.add_mapping(src_path, job_id)

        # 持久化任务元信息（重启后可恢复）
        self.save_job_meta(job)

        # 立即异步提取音频，确保前端能加载波形图（不阻塞响应）
        audio_path = job_dir / 'audio.wav'
        if not audio_path.exists():
            self.logger.info(f"[{job_id}] 启动后台音频提取...")
            import threading

            def extract_audio_for_waveform():
                """后台提取音频供波形图使用"""
                try:
                    import warnings
                    import librosa
                    import soundfile as sf
                    # 抑制 librosa 的 PySoundFile/audioread 警告
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="PySoundFile failed")
                        warnings.filterwarnings("ignore", message="audioread")
                        audio_array, sr = librosa.load(str(dest_path), sr=16000, mono=True)
                    sf.write(str(audio_path), audio_array, sr)
                    self.logger.info(f"[{job_id}] 音频提取完成: {audio_path}")
                except Exception as e:
                    self.logger.error(f"[{job_id}] 音频提取失败: {e}")

            threading.Thread(
                target=extract_audio_for_waveform,
                daemon=True,
                name=f"AudioExtract-{job_id[:8]}"
            ).start()
        else:
            self.logger.debug(f"[{job_id}] 音频文件已存在，跳过提取")

        self.logger.info(f"任务已创建: {job_id} - {filename}")
        return job

    def save_job_meta(self, job: JobState) -> bool:
        """
        保存任务元信息到 job_meta.json（用于重启后恢复）

        使用原子写入确保断电安全：先写临时文件，再rename替换

        Args:
            job: 任务状态对象

        Returns:
            bool: 是否成功保存
        """
        job_dir = Path(job.dir)
        meta_file = job_dir / "job_meta.json"

        try:
            job_dir.mkdir(parents=True, exist_ok=True)

            # 原子写入：先写临时文件，再rename替换
            temp_file = meta_file.with_suffix(".tmp")
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(job.to_meta_dict(), f, indent=2, ensure_ascii=False)

            # 原子替换
            temp_file.replace(meta_file)
            self.logger.debug(f"任务元信息已保存: {job.job_id}")
            return True
        except Exception as e:
            self.logger.error(f"保存任务元信息失败 {job.job_id}: {e}")
            return False

    def load_job_meta(self, job_id: str) -> Optional[JobState]:
        """
        从 job_meta.json 加载任务元信息

        Args:
            job_id: 任务ID

        Returns:
            Optional[JobState]: 恢复的任务状态对象
        """
        job_dir = self.jobs_root / job_id
        meta_file = job_dir / "job_meta.json"

        if not meta_file.exists():
            return None

        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            job = JobState.from_meta_dict(data)

            # 确保 dir 路径正确（可能因为项目迁移而改变）
            job.dir = str(job_dir)

            self.logger.debug(f"从 job_meta.json 加载任务: {job_id}")
            return job
        except Exception as e:
            self.logger.error(f"加载任务元信息失败 {job_id}: {e}")
            return None

    def get_job(self, job_id: str) -> Optional[JobState]:
        """
        获取任务状态

        Args:
            job_id: 任务ID

        Returns:
            Optional[JobState]: 任务状态对象，不存在则返回None
        """
        with self.lock:
            # 首先从内存中查找
            if job_id in self.jobs:
                return self.jobs[job_id]

        # 如果内存中没有，尝试从 jobs 目录读取（修复重启后无法打开旧任务的问题）
        job_dir = self.jobs_root / job_id
        if not job_dir.exists():
            return None

        try:
            # 优先从 job_meta.json 加载（包含完整的任务状态）
            job = self.load_job_meta(job_id)
            if job:
                # 缓存到内存
                with self.lock:
                    self.jobs[job_id] = job
                self.logger.info(f"从 job_meta.json 恢复任务: {job_id}")
                return job

            # 降级：从目录文件推断状态（兼容旧版本）
            # 尝试找到原始文件
            filename = "未知文件"
            input_path = None

            # 从目录中查找视频/音频文件
            for ext in ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.mp3', '.wav', '.m4a']:
                matches = list(job_dir.glob(f"*{ext}"))
                if matches:
                    filename = matches[0].name
                    input_path = str(matches[0])
                    break

            # 检查是否有 SRT 文件（表示已完成）
            srt_files = list(job_dir.glob("*.srt"))
            is_finished = len(srt_files) > 0

            # 创建 JobState 对象
            job = JobState(
                job_id=job_id,
                filename=filename,
                dir=str(job_dir),
                input_path=input_path,
                status='finished' if is_finished else 'processing',
                phase='editing' if is_finished else 'transcribing',
                progress=100 if is_finished else 0,
                message='已完成' if is_finished else '处理中',
                srt_path=str(srt_files[0]) if srt_files else None
            )

            # 尝试从 checkpoint 获取更详细的信息
            checkpoint_path = job_dir / "checkpoint.json"
            if checkpoint_path.exists():
                try:
                    with open(checkpoint_path, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                        total_segments = checkpoint_data.get('total_segments', 0)
                        processed_indices = checkpoint_data.get('processed_indices', [])
                        if total_segments > 0:
                            job.progress = min((len(processed_indices) / total_segments) * 100, 100)
                        job.phase = checkpoint_data.get('phase', 'transcribing')
                        job.language = checkpoint_data.get('language')
                        # 从checkpoint恢复segments
                        if 'unaligned_results' in checkpoint_data:
                            job.segments = checkpoint_data['unaligned_results']
                except Exception as e:
                    self.logger.warning(f"读取checkpoint失败 {checkpoint_path}: {e}")

            # 缓存到内存
            with self.lock:
                self.jobs[job_id] = job

            # 同时保存 job_meta.json 以便下次直接加载
            self.save_job_meta(job)

            self.logger.info(f"从磁盘恢复任务（旧版兼容）: {job_id}")
            return job

        except Exception as e:
            self.logger.error(f"从磁盘读取任务失败 {job_id}: {e}")
            return None

    def scan_incomplete_jobs(self) -> List[Dict]:
        """
        扫描所有未完成的任务（有checkpoint.json的任务）

        Returns:
            List[Dict]: 未完成任务列表
        """
        incomplete_jobs = []

        try:
            # 遍历所有任务目录
            for job_dir in self.jobs_root.iterdir():
                if not job_dir.is_dir():
                    continue

                checkpoint_path = job_dir / "checkpoint.json"
                if not checkpoint_path.exists():
                    continue

                try:
                    # 加载检查点数据
                    with open(checkpoint_path, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)

                    job_id = checkpoint_data.get('job_id') or job_dir.name
                    total_segments = checkpoint_data.get('total_segments', 0)
                    processed_indices = checkpoint_data.get('processed_indices', [])
                    processed_count = len(processed_indices)

                    # 计算进度
                    if total_segments > 0:
                        progress = (processed_count / total_segments) * 100
                    else:
                        progress = 0

                    # 从索引中查找文件名
                    file_path = self.job_index.get_file_path(job_id)
                    filename = os.path.basename(file_path) if file_path else "未知文件"

                    incomplete_jobs.append({
                        'job_id': job_id,
                        'filename': filename,
                        'file_path': file_path,  # 添加文件路径
                        'progress': round(progress, 2),
                        'processed_segments': processed_count,
                        'total_segments': total_segments,
                        'phase': checkpoint_data.get('phase', 'unknown'),
                        'dir': str(job_dir)
                    })

                except Exception as e:
                    self.logger.warning(f"读取检查点失败 {checkpoint_path}: {e}")
                    continue

            self.logger.info(f"扫描到 {len(incomplete_jobs)} 个未完成任务")
            return incomplete_jobs

        except Exception as e:
            self.logger.error(f"扫描未完成任务失败: {e}")
            return []

    def restore_job_from_checkpoint(self, job_id: str) -> Optional[JobState]:
        """
        从检查点恢复任务状态（无 checkpoint 时从头开始）

        Args:
            job_id: 任务ID

        Returns:
            Optional[JobState]: 恢复的任务状态对象
        """
        job_dir = self.jobs_root / job_id
        if not job_dir.exists():
            return None

        # 尝试加载 checkpoint（可能不存在）
        checkpoint = self._load_checkpoint(job_dir)

        try:
            # 查找原文件
            filename = "unknown"
            input_path = None

            # 从目录中查找视频/音频文件
            for ext in ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.mp3', '.wav', '.m4a']:
                matches = list(job_dir.glob(f"*{ext}"))
                if matches:
                    filename = matches[0].name
                    input_path = str(matches[0])
                    break

            if not input_path:
                self.logger.warning(f"无法找到任务 {job_id} 的输入文件")
                return None

            # 创建默认的CPU亲和性配置
            from app.services.cpu_affinity_service import CPUAffinityConfig
            default_cpu_config = CPUAffinityConfig(
                enabled=True,
                strategy="auto",
                custom_cores=None,
                exclude_cores=None
            )

            # 根据是否有 checkpoint 决定恢复状态
            if checkpoint:
                # 有 checkpoint，从断点恢复
                phase = checkpoint.get('phase', 'pending')
                total_segments = checkpoint.get('total_segments', 0)
                processed_indices = checkpoint.get('processed_indices', [])
                processed = len(processed_indices)
                progress = round((processed / max(1, total_segments)) * 100, 2)
                message = f"已暂停 ({processed}/{total_segments}段)"
                self.logger.info(f"从检查点恢复任务: {job_id}")
            else:
                # 无 checkpoint，从头开始
                phase = 'pending'
                total_segments = 0
                processed = 0
                progress = 0
                message = "程序重启，任务将从头开始"
                self.logger.info(f"无检查点，任务将从头开始: {job_id}")

            # 创建任务状态对象
            job = JobState(
                job_id=job_id,
                filename=filename,
                dir=str(job_dir),
                input_path=input_path,
                settings=JobSettings(cpu_affinity=default_cpu_config),
                status="paused",
                phase=phase,
                message=message,
                total=total_segments,
                processed=processed,
                progress=progress
            )

            with self.lock:
                self.jobs[job_id] = job

            return job

        except Exception as e:
            self.logger.error(f"从检查点恢复任务失败: {e}")
            return None

    def check_file_checkpoint(self, file_path: str) -> Optional[Dict]:
        """
        检查文件是否有可用的断点

        Args:
            file_path: 文件路径

        Returns:
            Optional[Dict]: 断点信息，无断点则返回None
        """
        # 从索引中查找任务ID
        job_id = self.job_index.get_job_id(file_path)
        if not job_id:
            return None

        # 检查任务目录和checkpoint是否存在
        job_dir = self.jobs_root / job_id
        if not job_dir.exists():
            # 清理无效映射
            self.job_index.remove_mapping(file_path)
            return None

        checkpoint = self._load_checkpoint(job_dir)
        if not checkpoint:
            return None

        # 返回断点信息
        total_segments = checkpoint.get('total_segments', 0)
        processed_indices = checkpoint.get('processed_indices', [])
        processed_count = len(processed_indices)

        if total_segments > 0:
            progress = (processed_count / total_segments) * 100
        else:
            progress = 0

        return {
            'job_id': job_id,
            'progress': round(progress, 2),
            'processed_segments': processed_count,
            'total_segments': total_segments,
            'phase': checkpoint.get('phase', 'unknown'),
            'can_resume': True
        }

    def start_job(self, job_id: str):
        """
        启动转录任务（V2.2: 废弃，由队列服务调用_run_pipeline_v2）

        注意: 此方法保留是为了向后兼容，不再自动创建线程

        Args:
            job_id: 任务ID
        """
        # 关键改动: 不再自动创建线程，由队列服务统一管理
        # 新逻辑: 只更新状态，实际执行由队列服务控制
        job = self.get_job(job_id)
        if not job:
            self.logger.warning(f"任务未找到: {job_id}")
            return

        if job.status not in ("uploaded", "failed", "paused", "created"):
            self.logger.warning(f"任务无法启动: {job_id}, 状态: {job.status}")
            return

        job.canceled = False
        job.paused = False
        job.error = None
        # 状态由队列服务设置，这里不改

        self.logger.warning(f"start_job已废弃，请使用队列服务: {job_id}")

    def pause_job(self, job_id: str) -> bool:
        """
        暂停转录任务（保存断点）

        Args:
            job_id: 任务ID

        Returns:
            bool: 是否成功设置暂停标志
        """
        job = self.get_job(job_id)
        if not job:
            return False

        job.paused = True
        job.message = "暂停中..."
        self.logger.info(f"⏸️ 任务暂停请求: {job_id}")
        return True

    def _force_remove_directory(self, directory: Path, job_id: str, max_retries: int = 3):
        """
        V3.1.0: 强制删除目录，处理 Windows 文件占用问题

        策略：
        1. 先触发垃圾回收，释放可能的文件句柄
        2. 尝试直接删除目录（最快）
        3. 如果失败，等待后重试（处理延迟释放）
        4. 如果仍失败，逐个删除文件，跳过无法删除的

        Args:
            directory: 要删除的目录路径
            job_id: 任务ID（用于日志）
            max_retries: 最大重试次数
        """
        import time
        import stat

        # 步骤1: 触发垃圾回收，释放可能的文件句柄
        gc.collect()
        time.sleep(0.1)  # 给系统一点时间释放资源

        # 步骤2: 尝试直接删除（最快路径）
        for attempt in range(max_retries):
            try:
                shutil.rmtree(directory)
                self.logger.info(f"[强制删除] 成功删除目录: {job_id}, 尝试次数: {attempt + 1}")
                return
            except PermissionError as e:
                if attempt < max_retries - 1:
                    self.logger.warning(
                        f"[强制删除] 删除失败 (尝试 {attempt + 1}/{max_retries}): {e}, "
                        f"等待 {0.5 * (attempt + 1)}s 后重试"
                    )
                    time.sleep(0.5 * (attempt + 1))  # 指数退避
                else:
                    self.logger.warning(f"[强制删除] 直接删除失败，尝试逐个删除文件: {job_id}")

        # 步骤3: 逐个删除文件（降级策略）
        failed_files = []
        for root, dirs, files in os.walk(directory, topdown=False):
            # 删除文件
            for name in files:
                file_path = Path(root) / name
                try:
                    # 尝试修改文件权限（Windows 只读文件）
                    os.chmod(file_path, stat.S_IWRITE)
                    file_path.unlink()
                except Exception as e:
                    self.logger.warning(f"[强制删除] 无法删除文件: {file_path.name}, {e}")
                    failed_files.append(str(file_path))

            # 删除空目录
            for name in dirs:
                dir_path = Path(root) / name
                try:
                    dir_path.rmdir()
                except Exception as e:
                    self.logger.debug(f"[强制删除] 无法删除目录: {dir_path.name}, {e}")

        # 尝试删除根目录
        try:
            directory.rmdir()
            self.logger.info(f"[强制删除] 逐个删除完成: {job_id}")
        except Exception as e:
            if failed_files:
                self.logger.error(
                    f"[强制删除] 部分文件无法删除: {job_id}, "
                    f"失败文件数: {len(failed_files)}, 错误: {e}"
                )
            else:
                self.logger.warning(f"[强制删除] 根目录删除失败: {job_id}, {e}")

    def cancel_job(self, job_id: str, delete_data: bool = False) -> bool:
        """
        取消转录任务

        Args:
            job_id: 任务ID
            delete_data: 是否删除任务数据

        Returns:
            bool: 是否成功设置取消标志
        """
        job = self.get_job(job_id)
        if not job:
            return False

        job.canceled = True
        job.message = "取消中..."
        self.logger.info(f"🛑 任务取消请求: {job_id}, 删除数据: {delete_data}")

        # 如果需要删除数据
        if delete_data:
            try:
                job_dir = Path(job.dir)

                # V3.1.2+dev.20260112.01: 先取消 MediaPrep 的转码任务，释放文件句柄
                # 避免 WinError 32 文件占用问题
                try:
                    from app.services.media_prep_service import get_media_prep_service
                    media_prep = get_media_prep_service()
                    killed = media_prep.cancel_tasks(job_id)
                    if killed > 0:
                        self.logger.info(f"[删除任务] 已终止 {killed} 个 MediaPrep 子进程: {job_id}")
                        # 给系统一点时间释放文件句柄
                        import time
                        time.sleep(0.5)
                except Exception as e:
                    self.logger.warning(f"[删除任务] 取消 MediaPrep 任务失败: {e}")

                # 先从内存中移除任务，避免删除失败时仍显示"未知文件"
                with self.lock:
                    if job_id in self.jobs:
                        del self.jobs[job_id]
                        self.logger.info(f"已从内存移除任务: {job_id}")

                # 移除文件路径映射
                if job.input_path:
                    self.job_index.remove_mapping(job.input_path)

                # 最后删除任务目录
                if job_dir.exists():
                    # V3.1.0: 使用强制删除逻辑，处理 Windows 文件占用问题
                    self._force_remove_directory(job_dir, job_id)
                    self.logger.info(f"已删除任务数据: {job_id}")
            except Exception as e:
                self.logger.error(f"删除任务数据失败: {e}")

        return True

    def _update_progress(
        self,
        job: JobState,
        phase: str,
        phase_ratio: float,
        message: str = ""
    ):
        """
        更新任务进度

        Args:
            job: 任务状态对象
            phase: 当前阶段 (extract/split/transcribe/srt)
            phase_ratio: 当前阶段完成比例 (0.0-1.0)
            message: 进度消息
        """
        job.phase = phase

        # 计算阶段内进度（0-100，保留1位小数）
        job.phase_percent = round(max(0.0, min(1.0, phase_ratio)) * 100, 1)

        # 使用配置中的进度权重
        phase_weights = config.PHASE_WEIGHTS
        total_weight = config.TOTAL_WEIGHT

        # 计算累计进度
        done_weight = 0
        for p, w in phase_weights.items():
            if p == phase:
                break
            done_weight += w

        current_weight = phase_weights.get(phase, 0) * max(0.0, min(1.0, phase_ratio))
        # 改为1位小数
        job.progress = round((done_weight + current_weight) / total_weight * 100, 1)

        if message:
            job.message = message

        # 推送SSE进度更新（线程安全）
        self._push_sse_progress(job)

    def _push_sse_progress(self, job: JobState):
        """
        推送SSE进度更新（线程安全）
        同时推送到单任务频道和全局频道，确保 TaskMonitor 实时更新

        Args:
            job: 任务状态对象
        """
        try:
            # 动态获取SSE管理器（确保获取到已设置loop的实例）
            from app.services.sse_service import get_sse_manager
            sse_manager = get_sse_manager()

            # 1. 推送到单任务频道（EditorView 使用）
            channel_id = f"job:{job.job_id}"
            progress_data = {
                "job_id": job.job_id,
                "phase": job.phase,
                "percent": job.progress,
                "phase_percent": job.phase_percent,  # 新增：阶段内进度
                "message": job.message,
                "status": job.status,
                "processed": job.processed,
                "total": job.total,
                "language": job.language or ""
            }
            sse_manager.broadcast_sync(channel_id, "progress.overall", progress_data)

            # 2. 推送到全局频道（TaskMonitor 使用）
            global_progress_data = {
                "id": job.job_id,  # 全局频道使用 "id"
                "percent": job.progress,
                "phase_percent": job.phase_percent,  # 新增：阶段内进度
                "message": job.message,
                "status": job.status,
                "phase": job.phase,
                "processed": job.processed,
                "total": job.total
            }
            sse_manager.broadcast_sync("global", "job_progress", global_progress_data)

        except Exception as e:
            # SSE推送失败不应影响转录流程
            self.logger.debug(f"SSE推送失败: {e}")

    def _push_sse_signal(self, job: JobState, signal_code: str, message: str = ""):
        """
        推送SSE信号事件（用于关键节点通知）

        Args:
            job: 任务状态对象
            signal_code: 信号代码（如 "job_complete", "job_failed", "job_canceled"）
            message: 附加消息
        """
        try:
            # 动态获取SSE管理器（确保获取到已设置loop的实例）
            from app.services.sse_service import get_sse_manager
            sse_manager = get_sse_manager()

            channel_id = f"job:{job.job_id}"
            sse_manager.broadcast_sync(
                channel_id,
                f"signal.{signal_code}",
                {
                    "job_id": job.job_id,
                    "signal": signal_code,
                    "message": message or job.message,
                    "status": job.status,
                    "percent": job.progress
                }
            )
        except Exception as e:
            self.logger.debug(f"SSE信号推送失败（非致命）: {e}")

    def _trigger_media_post_process(self, job_id: str):
        """
        异步触发媒体预处理（转录完成后调用）
        生成波形峰值、视频缩略图、Proxy视频等，为编辑器做准备

        Args:
            job_id: 任务ID
        """
        try:
            import asyncio
            import aiohttp

            async def do_post_process():
                """异步执行预处理请求"""
                try:
                    # 调用媒体预处理接口
                    async with aiohttp.ClientSession() as session:
                        url = f"http://127.0.0.1:8000/api/media/{job_id}/post-process"
                        async with session.post(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                            if resp.status == 200:
                                result = await resp.json()
                                self.logger.info(f"媒体预处理完成: peaks={result.get('peaks')}, thumbnails={result.get('thumbnails')}, proxy={result.get('proxy')}")
                            else:
                                self.logger.warning(f"媒体预处理请求失败: {resp.status}")
                except asyncio.TimeoutError:
                    self.logger.warning(f"媒体预处理超时: {job_id}")
                except Exception as e:
                    self.logger.warning(f"媒体预处理异常: {e}")

            # 尝试在现有事件循环中执行
            try:
                loop = asyncio.get_running_loop()
                asyncio.ensure_future(do_post_process(), loop=loop)
            except RuntimeError:
                # 没有运行中的事件循环，创建新线程执行
                import threading
                def run_in_thread():
                    asyncio.run(do_post_process())
                thread = threading.Thread(target=run_in_thread, daemon=True)
                thread.start()

            self.logger.info(f"已触发媒体预处理任务: {job_id}")

            # 触发 720p 高清转码（队列空闲时执行）
            self._trigger_720p_transcode_if_idle(job_id)

        except Exception as e:
            # 预处理失败不影响转录结果
            self.logger.warning(f"触发媒体预处理失败（非致命）: {e}")

    def _trigger_720p_transcode_if_idle(self, job_id: str):
        """
        在转录完成后触发 720p 高清转码（仅在队列空闲时）

        策略:
        1. 检查视频是否需要转码（H.265 等）
        2. 检查转录队列是否空闲
        3. 如果空闲，立即启动 720p 转码
        4. 如果队列有任务，延迟触发（使用低优先级）
        """
        try:
            from pathlib import Path
            from app.core.config import config
            from app.services.media_prep_service import get_media_prep_service

            self.logger.info(f"[720p] 开始检查是否需要触发720p转码: {job_id}")

            job_dir = config.JOBS_DIR / job_id
            if not job_dir.exists():
                self.logger.info(f"[720p] 任务目录不存在，跳过: {job_id}")
                return

            # 查找视频文件
            video_file = None
            video_exts = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.webm', '.flv', '.m4v']
            for file in job_dir.iterdir():
                if file.is_file() and file.suffix.lower() in video_exts:
                    video_file = file
                    break

            if not video_file:
                self.logger.info(f"[720p] 未找到视频文件，跳过: {job_id}")
                return

            self.logger.info(f"[720p] 找到视频文件: {video_file.name}")

            # 使用正确的文件名：proxy_720p.mp4
            proxy_720p = job_dir / "proxy_720p.mp4"

            # 720p 已存在则跳过
            if proxy_720p.exists():
                self.logger.info(f"[720p] 已存在，跳过: {job_id}")
                return

            media_prep = get_media_prep_service()

            # 检查 720p 是否已在队列中
            proxy_status = media_prep.get_proxy_status(job_id)
            if proxy_status and proxy_status.get("status") in ["queued", "processing", "completed"]:
                self.logger.info(f"[720p] 任务已存在或完成（状态={proxy_status.get('status')}），跳过: {job_id}")
                return

            # 检查 360p 预览是否已完成（必须先有360p才能启动720p）
            # 修复：优先检查文件是否存在，因为内存状态可能在服务重启后丢失
            preview_360p = job_dir / "preview_360p.mp4"

            # 文件存在即认为360p已完成（内存状态可能丢失）
            if preview_360p.exists():
                preview_completed = True
                self.logger.debug(f"[720p] 360p文件存在，认为已完成: {job_id}")
            else:
                # 文件不存在，检查内存状态（可能正在转码中）
                preview_status = media_prep.get_preview_status(job_id)
                preview_completed = preview_status and preview_status.get("status") == "completed"

            if not preview_completed:
                self.logger.info(f"[720p] 360p预览未完成，跳过720p转码: {job_id}")
                return

            # 检查转录队列是否空闲
            queue_idle = self._is_transcription_queue_idle()
            self.logger.info(f"[720p] 队列空闲状态: {queue_idle}")

            if queue_idle:
                # 队列空闲，立即启动（正常优先级）
                self.logger.info(f"[720p] 队列空闲，立即启动转码: {job_id}")
                success = media_prep.enqueue_proxy(job_id, video_file, proxy_720p, priority=10)
                self.logger.info(f"[720p] 入队结果: {success}")
            else:
                # 队列繁忙，使用低优先级（独立进程模式）
                self.logger.info(f"[720p] 队列繁忙，低优先级排队: {job_id}")
                success = media_prep.enqueue_proxy(job_id, video_file, proxy_720p, priority=1)
                self.logger.info(f"[720p] 入队结果: {success}")

        except Exception as e:
            self.logger.warning(f"[720p] 触发转码失败（非致命）: {e}")

    def _is_transcription_queue_idle(self) -> bool:
        """检查转录任务队列是否空闲"""
        try:
            from app.services.job_queue_service import get_job_queue

            job_queue = get_job_queue()
            if not job_queue:
                return True

            # 检查是否有正在处理或等待的任务
            active_jobs = [
                j for j in job_queue.jobs.values()
                if j.status in ['pending', 'processing', 'queued']
            ]

            return len(active_jobs) == 0

        except Exception as e:
            self.logger.debug(f"检查队列状态失败: {e}")
            return True  # 默认认为空闲

    def _push_sse_segment(self, job: JobState, segment_result: dict, processed: int, total: int):
        """
        推送单个segment的转录结果（流式输出）

        Args:
            job: 任务状态对象
            segment_result: 单个segment的转录结果（未对齐）
            processed: 已处理的segment数量
            total: 总segment数量
        """
        try:
            # 动态获取SSE管理器
            from app.services.sse_service import get_sse_manager
            sse_manager = get_sse_manager()

            channel_id = f"job:{job.job_id}"
            sse_manager.broadcast_sync(
                channel_id,
                "subtitle.segment",
                {
                    "segment_index": segment_result.get('segment_index', 0),
                    "segments": segment_result.get('segments', []),
                    "language": segment_result.get('language', job.language),
                    "progress": {
                        "processed": processed,
                        "total": total,
                        "percentage": round(processed / max(1, total) * 100, 2)
                    }
                }
            )
            self.logger.debug(f"推送segment #{segment_result.get('segment_index', 0)} 转录结果")
        except Exception as e:
            # SSE推送失败不应影响转录流程
            self.logger.debug(f"SSE segment推送失败（非致命）: {e}")

    def _push_sse_aligned(self, job: JobState, aligned_results: List[Dict]):
        """
        推送对齐完成事件（流式输出）

        Args:
            job: 任务状态对象
            aligned_results: 对齐后的结果列表
        """
        try:
            # 动态获取SSE管理器
            from app.services.sse_service import get_sse_manager
            sse_manager = get_sse_manager()

            channel_id = f"job:{job.job_id}"

            # 提取对齐后的segments
            segments = []
            word_segments = []
            if aligned_results and len(aligned_results) > 0:
                segments = aligned_results[0].get('segments', [])
                word_segments = aligned_results[0].get('word_segments', [])

            sse_manager.broadcast_sync(
                channel_id,
                "subtitle.aligned",
                {
                    "segments": segments,
                    "word_segments": word_segments,
                    "message": "对齐完成"
                }
            )
            self.logger.info(f"推送对齐完成事件，共 {len(segments)} 条字幕")
        except Exception as e:
            # SSE推送失败不应影响转录流程
            self.logger.debug(f"SSE aligned推送失败（非致命）: {e}")

    def _save_checkpoint(self, job_dir: Path, data: dict, job: JobState):
        """
        原子性保存检查点
        使用"写临时文件 -> 重命名"策略，确保文件要么完整写入，要么保持原样

        Args:
            job_dir: 任务目录
            data: 检查点数据
            job: 任务状态对象（用于获取settings）
        """
        # 添加原始设置到checkpoint（用于校验参数兼容性）
        # 使用 ConfigAdapter 统一新旧配置
        demucs_strategy = ConfigAdapter.get_demucs_strategy(job.settings)
        data["original_settings"] = {
            "model": job.settings.model,
            "device": job.settings.device,
            "word_timestamps": job.settings.word_timestamps,
            "compute_type": job.settings.compute_type,
            "batch_size": job.settings.batch_size,
            "demucs": {
                "enabled": ConfigAdapter.is_demucs_enabled(job.settings),
                "mode": demucs_strategy,
            }
        }

        # 确保 demucs 字段存在（向后兼容）
        if "demucs" not in data:
            data["demucs"] = {
                "enabled": ConfigAdapter.is_demucs_enabled(job.settings),
                "mode": demucs_strategy,
                "bgm_level": "none",
                "bgm_ratios": [],
                "global_separation_done": False,
                "vocals_path": None,
                "circuit_breaker": None,
                "retry_triggered": False
            }

        checkpoint_path = job_dir / "checkpoint.json"
        temp_path = checkpoint_path.with_suffix(".tmp")

        try:
            # 1. 写入临时文件
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # 2. 原子替换（Windows/Linux/macOS 均支持）
            # 如果程序在这里崩溃，checkpoint.json 依然是旧版本，不会损坏
            os.replace(temp_path, checkpoint_path)

            # 3. 同步保存任务元信息（用于重启后恢复任务状态）
            # 这样每次保存检查点时，任务的进度和状态都会被持久化
            self.save_job_meta(job)

        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}")
            # 保存失败不应中断主流程，仅记录日志

    def _load_checkpoint(self, job_dir: Path) -> Optional[dict]:
        """
        加载检查点，如果文件损坏则返回 None

        Args:
            job_dir: 任务目录

        Returns:
            Optional[dict]: 检查点数据，不存在或损坏则返回 None
        """
        checkpoint_path = job_dir / "checkpoint.json"
        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            self.logger.warning(f"检查点文件损坏，将重新开始任务: {checkpoint_path} - {e}")
            return None

    def _flush_checkpoint_after_split(
        self,
        job_dir: Path,
        job: JobState,
        segments: List[Dict],
        processing_mode: ProcessingMode,
        demucs_state: Dict = None
    ):
        """
        分段完成后强制刷新checkpoint（确保断点续传一致性）

        这是断点续传的关键节点！
        只有分段元数据被持久化后，后续的转录索引才有意义。

        Args:
            job_dir: 任务目录
            job: 任务状态对象
            segments: 分段元数据列表
            processing_mode: 当前处理模式
            demucs_state: Demucs状态数据（可选）
        """
        import time

        checkpoint_data = {
            "job_id": job.job_id,
            "phase": "split_complete",  # 明确标记分段完成
            "processing_mode": processing_mode.value,  # 记录模式
            "total_segments": len(segments),
            "processed_indices": [],
            "segments": segments,
            "unaligned_results": [],
            "timestamp": time.time()  # 时间戳用于调试
        }

        # 添加 demucs 状态（如果提供）
        if demucs_state:
            checkpoint_data["demucs"] = demucs_state

        # 强制同步写入（确保数据落盘）
        self._save_checkpoint(job_dir, checkpoint_data, job)

        # 验证写入成功
        saved_checkpoint = self._load_checkpoint(job_dir)
        if saved_checkpoint is None:
            raise RuntimeError("checkpoint write verification failed: file not readable")

        if saved_checkpoint.get('phase') != 'split_complete':
            raise RuntimeError("checkpoint write verification failed: phase mismatch")

        if len(saved_checkpoint.get('segments', [])) != len(segments):
            raise RuntimeError("checkpoint write verification failed: segments count mismatch")

        self.logger.info(f"checkpoint flushed and verified after split (mode: {processing_mode.value}, segments: {len(segments)})")


    # ========== 核心处理方法 ==========

    def _get_audio_duration(self, audio_path: str) -> float:
        """
        获取音频时长（秒）

        Args:
            audio_path: 音频文件路径

        Returns:
            float: 音频时长（秒）
        """
        try:
            # 方法1: 使用pydub（精确但较慢）
            audio = AudioSegment.from_wav(audio_path)
            duration = len(audio) / 1000.0
            self.logger.debug(f"音频时长（pydub）: {duration:.1f}秒")
            return duration
        except Exception as e:
            self.logger.warning(f"pydub获取时长失败，使用文件大小估算: {e}")
            # 方法2: 根据文件大小估算（16kHz, 16bit, mono ≈ 32KB/秒）
            try:
                file_size = os.path.getsize(audio_path)
                duration = file_size / 32000
                self.logger.debug(f"音频时长（估算）: {duration:.1f}秒")
                return duration
            except Exception as e2:
                self.logger.error(f"获取音频时长失败: {e2}")
                return 0.0

    def _decide_processing_mode(self, audio_path: str, job: JobState) -> ProcessingMode:
        """
        智能决策处理模式（内存模式 vs 硬盘模式）

        决策逻辑：
        1. 估算音频内存需求
        2. 检测系统可用内存
        3. 预留安全余量（模型、转录中间变量等）
        4. 决定使用哪种模式

        Args:
            audio_path: 音频文件路径
            job: 任务状态对象

        Returns:
            ProcessingMode: 处理模式
        """
        # 获取音频时长（秒）
        audio_duration_sec = self._get_audio_duration(audio_path)

        # 估算音频内存需求 (16kHz, float32)
        # 公式: duration * 16000 * 4 bytes
        estimated_audio_mb = (audio_duration_sec * 16000 * 4) / (1024 * 1024)

        # 预留额外内存（模型加载、VAD处理、转录中间变量等）
        # 保守估计：音频内存的2倍 + 500MB基础开销
        total_estimated_mb = estimated_audio_mb * 2 + 500

        # 获取系统可用内存
        mem_info = psutil.virtual_memory()
        available_mb = mem_info.available / (1024 * 1024)
        total_mb = mem_info.total / (1024 * 1024)

        # 安全阈值：动态计算，综合考虑多种因素
        # 1. 基础保留：2GB（保证系统基本运行）
        # 2. 动态保留：可用内存的10%（而非总内存的20%，避免过度保守）
        # 3. 最大保留上限：4GB（避免在大内存系统上过度保留）
        base_reserve_mb = 2048
        dynamic_reserve_mb = available_mb * 0.1
        safety_reserve_mb = min(base_reserve_mb + dynamic_reserve_mb, 4096)
        usable_mb = available_mb - safety_reserve_mb

        self.logger.info(f"内存评估:")
        self.logger.info(f"音频时长: {audio_duration_sec/60:.1f}分钟")
        self.logger.info(f"预估需求: {total_estimated_mb:.0f}MB")
        self.logger.info(f"可用内存: {available_mb:.0f}MB")
        self.logger.info(f"安全余量: {safety_reserve_mb:.0f}MB")
        self.logger.info(f"可用于处理: {usable_mb:.0f}MB")

        # 决策
        if usable_mb >= total_estimated_mb:
            self.logger.info("选择【内存模式】- 内存充足，使用高性能模式")
            job.message = "内存充足，使用高性能模式"
            return ProcessingMode.MEMORY
        else:
            self.logger.warning(f"选择【硬盘模式】- 内存不足（需要{total_estimated_mb:.0f}MB，可用{usable_mb:.0f}MB）")
            job.message = "内存受限，使用稳定模式"
            return ProcessingMode.DISK

    def _safe_load_audio(self, audio_path: str, job: JobState) -> np.ndarray:
        """
        安全加载音频到内存（带异常处理）

        用于内存模式下将完整音频一次性加载到内存中。
        包含加载验证和详细的异常处理，加载失败时抛出RuntimeError触发降级。

        Args:
            audio_path: 音频文件路径
            job: 任务状态对象（用于更新状态消息）

        Returns:
            np.ndarray: 音频数组（float32, 16kHz采样率）

        Raises:
            RuntimeError: 音频加载失败时抛出，调用方可据此触发硬盘模式降级
        """
        try:
            self.logger.info(f"加载音频到内存: {audio_path}")
            # 使用 whisper_service 提供的 load_audio 函数加载音频
            audio_array = whisper_load_audio(audio_path)

            # 验证加载结果
            if audio_array is None or len(audio_array) == 0:
                raise ValueError("音频数组为空")

            # 记录加载信息
            duration_sec = len(audio_array) / 16000
            memory_mb = audio_array.nbytes / (1024 * 1024)
            # Audio loaded: {duration_sec/60:.1f}分钟")
            self.logger.info(f"内存占用: {memory_mb:.1f}MB")
            self.logger.info(f"采样点数: {len(audio_array):,}")

            return audio_array

        except MemoryError as e:
            self.logger.error(f"内存不足，无法加载音频: {e}")
            job.message = "内存不足，自动切换到硬盘模式"
            raise RuntimeError(f"内存不足: {e}")

        except Exception as e:
            self.logger.error(f"音频加载失败: {e}")
            job.message = f"音频加载失败: {e}"
            raise RuntimeError(f"音频加载失败（可能文件损坏）: {e}")

    # ==========================================
    # Demucs 人声分离相关方法
    # ==========================================

    def _detect_bgm(self, audio_path: str, job: JobState):
        """
        执行BGM检测，更新进度
        
        【v2.1 重构】使用频谱分诊替代旧的分位数采样 + Demucs 检测
        新方法：纯频谱特征分析，无需运行 Demucs，速度更快且更准确

        Args:
            audio_path: 音频文件路径
            job: 任务状态对象

        Returns:
            Tuple[BGMLevel, List[float]]: (BGM强度级别, 各采样点的音乐得分列表)
        """
        from app.services.demucs_service import BGMLevel
        from app.services.audio_spectrum_classifier import get_spectrum_classifier
        import librosa

        self._update_progress(job, 'bgm_detect', 0, 'BGM检测中（频谱分诊）...')

        try:
            # 加载音频
            audio_array, sr = librosa.load(audio_path, sr=16000)
            
            # 使用频谱分诊器进行快速全局预判
            spectrum_classifier = get_spectrum_classifier()
            level_str, avg_score = spectrum_classifier.quick_global_diagnosis(
                audio_array, sr=sr, sample_duration=10.0
            )
            
            # 转换为 BGMLevel 枚举
            level_map = {
                "none": BGMLevel.NONE,
                "light": BGMLevel.LIGHT,
                "heavy": BGMLevel.HEAVY,
                "unknown": BGMLevel.LIGHT  # unknown 保守处理为 light
            }
            level = level_map.get(level_str, BGMLevel.LIGHT)
            
            # 构造兼容的 ratios 列表（用于日志和 SSE）
            # 频谱分诊返回的是音乐得分，这里用 avg_score 填充
            ratios = [avg_score, avg_score, avg_score]

            self._update_progress(job, 'bgm_detect', 1, f'BGM检测完成: {level.value}')

            # 推送SSE事件
            self._push_sse_bgm_detected(job, level, ratios)

            self.logger.info(
                f"BGM检测结果（频谱分诊）: {level.value}, "
                f"音乐得分={avg_score:.2f}"
            )

            return level, ratios

        except Exception as e:
            self.logger.warning(f"BGM检测失败，将跳过Demucs: {e}")
            # 失败时返回 NONE 级别，不影响主流程
            from app.services.demucs_service import BGMLevel
            return BGMLevel.NONE, []
    
    def _detect_bgm_legacy(self, audio_path: str, job: JobState):
        """
        [已废弃] 执行BGM检测（旧版：分位数采样 + Demucs 检测）
        
        ⚠️ 此方法已废弃，保留供参考和回退使用
        问题：Demucs 分离残差约 1-3%，导致纯人声也被误判为 light
        
        新方法请使用 _detect_bgm()（频谱分诊版本）

        Args:
            audio_path: 音频文件路径
            job: 任务状态对象

        Returns:
            Tuple[BGMLevel, List[float]]: (BGM强度级别, 各采样点的BGM比例列表)
        """
        from app.services.demucs_service import get_demucs_service, BGMLevel

        self._update_progress(job, 'bgm_detect', 0, 'BGM检测中...')

        try:
            demucs = get_demucs_service()

            # 执行BGM检测（旧版：分位数采样）
            level, ratios = demucs.detect_background_music_level(audio_path)

            self._update_progress(job, 'bgm_detect', 1, f'BGM检测完成: {level.value}')

            # 推送SSE事件
            self._push_sse_bgm_detected(job, level, ratios)

            self.logger.info(
                f"BGM检测结果: {level.value}, "
                f"比例={ratios}, 最大={max(ratios) if ratios else 0:.2f}"
            )

            return level, ratios

        except Exception as e:
            self.logger.warning(f"BGM检测失败，将跳过Demucs: {e}")
            # 失败时返回 NONE 级别，不影响主流程
            from app.services.demucs_service import BGMLevel
            return BGMLevel.NONE, []

    def _separate_vocals_global(self, audio_path: str, job: JobState) -> str:
        """
        执行全局人声分离，更新进度

        Args:
            audio_path: 原始音频路径
            job: 任务状态对象

        Returns:
            str: 分离后的人声文件路径
        """
        from app.services.demucs_service import get_demucs_service

        self._update_progress(job, 'demucs_global', 0, '人声分离中...')

        try:
            demucs = get_demucs_service()

            def progress_callback(progress: float, message: str):
                """进度回调"""
                self._update_progress(job, 'demucs_global', progress, message)

            # 执行人声分离
            vocals_path = demucs.separate_vocals(
                audio_path,
                progress_callback=progress_callback
            )

            self._update_progress(job, 'demucs_global', 1, '人声分离完成')
            self.logger.info(f"全局人声分离完成: {vocals_path}")

            return vocals_path

        except Exception as e:
            self.logger.error(f"人声分离失败: {e}")
            # 分离失败时返回原始音频路径（降级处理）
            self.logger.warning("人声分离失败，将使用原始音频继续处理")
            return audio_path

    def _push_sse_bgm_detected(self, job: JobState, level, ratios):
        """
        推送BGM检测结果事件

        Args:
            job: 任务状态对象
            level: BGM强度级别
            ratios: 各采样点的BGM比例列表
        """
        try:
            from app.services.sse_service import get_sse_manager

            sse_manager = get_sse_manager()
            channel_id = f"job:{job.job_id}"

            # 将numpy类型转换为Python原生类型，避免JSON序列化错误
            native_ratios = [float(r) for r in ratios] if ratios else []
            max_ratio = float(max(ratios)) if ratios else 0.0

            # 构造事件数据
            event_data = {
                "level": level.value,
                "ratios": native_ratios,
                "max_ratio": max_ratio,
                "recommendation": self._get_demucs_recommendation(level)
            }

            # 广播事件
            sse_manager.broadcast_sync(channel_id, "signal.bgm_detected", event_data)

        except Exception as e:
            # SSE推送失败不应影响主流程
            self.logger.debug(f"SSE推送失败（非致命）: {e}")

    def _push_sse_separation_strategy(self, job: JobState, strategy):
        """
        推送分离策略决策事件

        Args:
            job: 任务状态对象
            strategy: SeparationStrategy 对象
        """
        try:
            from app.services.sse_service import get_sse_manager

            sse_manager = get_sse_manager()
            channel_id = f"job:{job.job_id}"

            # 使用 strategy.to_dict() 获取事件数据
            event_data = strategy.to_dict()

            # 广播事件
            sse_manager.broadcast_sync(channel_id, "signal.separation_strategy", event_data)

            self.logger.debug(f"分离策略事件已推送: {strategy.reason}")

        except Exception as e:
            # SSE推送失败不应影响主流程
            self.logger.debug(f"SSE推送失败（非致命）: {e}")

    def _push_sse_model_escalated(
        self,
        job: JobState,
        from_model: str,
        to_model: str,
        reason: str,
        breaker_state: CircuitBreakerState
    ):
        """
        推送模型升级事件

        Args:
            job: 任务状态对象
            from_model: 原模型名称
            to_model: 新模型名称
            reason: 升级原因
            breaker_state: 熔断器状态
        """
        try:
            from app.services.sse_service import get_sse_manager

            sse_manager = get_sse_manager()
            channel_id = f"job:{job.job_id}"

            # 构造事件数据 (使用 ConfigAdapter 兼容新旧配置)
            event_data = {
                "from_model": from_model,
                "to_model": to_model,
                "reason": reason,
                "escalation_count": breaker_state.escalation_count,
                "max_escalations": ConfigAdapter.get_max_escalations(job.settings),
                "stats": breaker_state.get_stats()
            }

            # 广播事件
            sse_manager.broadcast_sync(channel_id, "signal.model_upgrade", event_data)

            self.logger.info(f"模型升级事件已推送: {from_model} -> {to_model}")

        except Exception as e:
            # SSE推送失败不应影响主流程
            self.logger.debug(f"SSE推送失败（非致命）: {e}")

    def _push_sse_circuit_breaker_triggered(
        self,
        job: JobState,
        circuit_breaker: Optional[CircuitBreakerState]
    ):
        """
        推送熔断触发事件（Phase 4: 扩展包含升级信息）

        Args:
            job: 任务状态对象
            circuit_breaker: 熔断器状态对象
        """
        try:
            from app.services.sse_service import get_sse_manager

            sse_manager = get_sse_manager()
            channel_id = f"job:{job.job_id}"

            # 构造事件数据（扩展包含升级历史）
            stats = circuit_breaker.get_stats() if circuit_breaker else {}
            event_data = {
                "triggered": True,
                "reason": self._get_circuit_break_reason(circuit_breaker),
                "stats": stats,
                "action": "升级为全局人声分离模式"
            }

            # 广播事件
            sse_manager.broadcast_sync(channel_id, "signal.circuit_breaker", event_data)

            self.logger.info("熔断事件已推送到前端")

        except Exception as e:
            # SSE推送失败不应影响主流程
            self.logger.debug(f"SSE推送失败（非致命）: {e}")

    def _get_circuit_break_reason(self, circuit_breaker: Optional[CircuitBreakerState]) -> str:
        """
        生成熔断原因描述

        Args:
            circuit_breaker: 熔断器状态

        Returns:
            熔断原因描述字符串
        """
        if not circuit_breaker:
            return "转录质量低，触发熔断升级"

        stats = circuit_breaker.get_stats()

        # 如果有升级历史，说明已经尝试过升级
        if circuit_breaker.escalation_count > 0:
            return (
                f"已升级 {circuit_breaker.escalation_count} 次模型仍未改善，触发熔断。"
                f"升级历史: {', '.join(circuit_breaker.escalation_history)}"
            )
        else:
            return (
                f"连续 {stats['consecutive_retries']} 个段落重试失败，"
                f"总重试率 {stats['retry_ratio']:.1%}，触发熔断"
            )

    def _get_demucs_recommendation(self, level) -> str:
        """
        根据BGM级别返回建议的处理模式

        Args:
            level: BGM强度级别

        Returns:
            str: 建议的处理模式描述
        """
        from app.services.demucs_service import BGMLevel

        if level == BGMLevel.HEAVY:
            return "全局分离"
        elif level == BGMLevel.LIGHT:
            return "按需分离"
        else:
            return "无需分离"

    # ==========================================
    # 按需分离与熔断机制相关方法
    # ==========================================

    def _check_transcription_confidence(
        self,
        result: Dict,
        logprob_threshold: float,
        no_speech_threshold: float
    ) -> bool:
        """
        检查转录结果的置信度

        Args:
            result: 转录结果字典
            logprob_threshold: logprob阈值（低于此值需要重试）
            no_speech_threshold: no_speech_prob阈值（高于此值需要重试）

        Returns:
            bool: True表示置信度低，需要重试
        """
        segments = result.get('segments', [])

        if not segments:
            return True  # 没有识别出内容，需要重试

        # 计算平均置信度
        total_logprob = 0
        total_no_speech = 0
        count = 0

        for seg in segments:
            if 'avg_logprob' in seg:
                total_logprob += seg['avg_logprob']
                count += 1
            if 'no_speech_prob' in seg:
                total_no_speech += seg['no_speech_prob']

        if count == 0:
            return False  # 没有置信度信息，不重试

        avg_logprob = total_logprob / count
        avg_no_speech = total_no_speech / count if count > 0 else 0

        # 判断是否需要重试
        if avg_logprob < logprob_threshold:
            self.logger.debug(f"avg_logprob={avg_logprob:.2f} < {logprob_threshold}, 需要重试")
            return True

        if avg_no_speech > no_speech_threshold:
            self.logger.debug(f"no_speech_prob={avg_no_speech:.2f} > {no_speech_threshold}, 需要重试")
            return True

        return False

    def _is_better_result(self, new_result: Dict, old_result: Dict) -> bool:
        """
        比较两个转录结果，判断新结果是否更好

        Args:
            new_result: 新的转录结果
            old_result: 旧的转录结果

        Returns:
            bool: True表示新结果更好
        """
        new_segments = new_result.get('segments', [])
        old_segments = old_result.get('segments', [])

        # 如果新结果没有内容，旧的更好
        if not new_segments:
            return False

        # 如果旧结果没有内容，新的更好
        if not old_segments:
            return True

        # 比较平均logprob
        def get_avg_logprob(segments):
            logprobs = [s.get('avg_logprob', -1) for s in segments if 'avg_logprob' in s]
            return np.mean(logprobs) if logprobs else -1

        new_logprob = get_avg_logprob(new_segments)
        old_logprob = get_avg_logprob(old_segments)

        # 新结果的logprob更高（更接近0）则更好
        return new_logprob > old_logprob

    def _transcribe_segment_with_retry(
        self,
        seg_meta: Dict,
        model,
        job: JobState,
        audio_array: Optional[np.ndarray] = None,
        circuit_breaker: Optional[CircuitBreakerState] = None
    ) -> Optional[Dict]:
        """
        带重试的转录方法（支持Demucs人声分离重试 + 动态熔断）

        流程：
        1. 首次转录（使用原始音频）
        2. 检查置信度
        3. 如果置信度低，使用Demucs分离人声后重试
        4. 更新熔断器状态
        5. 检查是否触发熔断
        6. 返回置信度更高的结果

        Args:
            seg_meta: 段落元数据
            model: Whisper模型
            job: 任务状态对象
            audio_array: 完整音频数组（内存模式）
            circuit_breaker: 熔断器状态对象

        Returns:
            Optional[Dict]: 转录结果

        Raises:
            BreakToGlobalSeparation: 当触发熔断条件时抛出
        """
        demucs_settings = job.settings.demucs

        # 首次转录
        result = self._transcribe_segment(seg_meta, model, job, audio_array)

        if not result or not demucs_settings.enabled:
            if circuit_breaker:
                circuit_breaker.record_success()
            return result

        # 检查是否需要重试
        needs_retry = self._check_transcription_confidence(
            result,
            demucs_settings.retry_threshold_logprob,
            demucs_settings.retry_threshold_no_speech
        )

        if not needs_retry:
            # 不需要重试，记录成功
            if circuit_breaker:
                circuit_breaker.record_success()
            return result

        # ========== 需要重试的逻辑 ==========
        self.logger.info(f"段落 {seg_meta['index']} 置信度低，尝试人声分离重试")

        # 更新熔断器状态
        if circuit_breaker:
            circuit_breaker.record_retry()

            # 检查是否触发熔断
            if circuit_breaker.should_break(demucs_settings):
                stats = circuit_breaker.get_stats()
                self.logger.warning(
                    f"触发熔断！连续重试={stats['consecutive_retries']}, "
                    f"总重试比例={stats['retry_ratio']:.1%}"
                )
                raise BreakToGlobalSeparation(
                    f"连续{stats['consecutive_retries']}段需要Demucs重试，"
                    f"建议升级为全局人声分离模式"
                )

        # 尝试按需分离
        try:
            from app.services.demucs_service import get_demucs_service
            demucs = get_demucs_service()

            start_sec = seg_meta['start']
            end_sec = seg_meta['end']

            if audio_array is not None:
                # 内存模式：分离人声
                vocals = demucs.separate_vocals_segment(
                    audio_array, sr=16000,
                    start_sec=start_sec, end_sec=end_sec
                )

                # 构造临时seg_meta
                retry_seg = seg_meta.copy()
                retry_seg['start'] = 0  # 因为vocals已经是切片
                retry_seg['end'] = len(vocals) / 16000

                # 重新转录
                retry_result = self._transcribe_segment_in_memory(
                    vocals,
                    retry_seg,
                    model,
                    job,
                    is_vocals=True  # 标记是人声
                )
            else:
                # 硬盘模式：暂不支持
                self.logger.warning("硬盘模式暂不支持Demucs重试")
                return result

            if retry_result:
                # 校正时间偏移（恢复到原始时间轴）
                original_start = seg_meta['start']
                for seg in retry_result.get('segments', []):
                    seg['start'] += original_start
                    seg['end'] += original_start

                # 比较两次结果，返回更好的
                if self._is_better_result(retry_result, result):
                    self.logger.info(f"段落 {seg_meta['index']} 重试成功，使用分离后的结果")
                    retry_result['used_demucs'] = True
                    return retry_result

        except Exception as e:
            self.logger.warning(f"Demucs重试失败: {e}")

        return result

    def _get_model(self, settings: JobSettings, job: Optional[JobState] = None):
        """
        获取 Faster-Whisper 模型（带缓存）

        优先使用模型管理服务检查并下载模型，否则使用简单缓存

        Args:
            settings: 任务设置
            job: 任务状态对象(可选,用于更新下载进度)

        Returns:
            模型对象
        """
        # 尝试使用模型管理服务检查并下载模型
        try:
            from app.services.model_manager_service import get_model_manager
            model_mgr = get_model_manager()
            whisper_model_info = model_mgr.whisper_models.get(settings.model)

            if whisper_model_info:
                # 检查模型状态
                if whisper_model_info.status == "not_downloaded" or whisper_model_info.status == "incomplete":
                    self.logger.warning(f"Whisper模型未下载或不完整: {settings.model}")

                    # 获取模型大小信息
                    model_size_mb = whisper_model_info.size_mb

                    # 如果模型大小>=1GB,给出特殊提示
                    download_msg = ""
                    if model_size_mb >= 1024:
                        size_gb = model_size_mb / 1024
                        download_msg = f"当前下载模型大于1GB ({size_gb:.1f}GB),请耐心等待"
                        self.logger.info(f"{download_msg}")
                    else:
                        download_msg = f"开始下载模型 {settings.model} ({model_size_mb}MB)"

                    # 更新任务状态
                    if job:
                        job.message = download_msg

                    self.logger.info(f"自动触发下载Whisper模型: {settings.model} ({model_size_mb}MB)")

                    # 触发下载
                    success = model_mgr.download_whisper_model(settings.model)
                    if not success:
                        self.logger.warning(f"模型管理器下载失败或已在下载中,使用备用方式")
                        raise RuntimeError("模型管理器下载失败")

                    # 等待下载完成（最多等待10分钟）
                    import time
                    max_wait_time = 600  # 10分钟
                    wait_interval = 5  # 每5秒检查一次
                    elapsed = 0

                    while elapsed < max_wait_time:
                        time.sleep(wait_interval)
                        elapsed += wait_interval

                        current_status = model_mgr.whisper_models[settings.model].status
                        progress = model_mgr.whisper_models[settings.model].download_progress

                        if current_status == "ready":
                            self.logger.info(f"Whisper模型下载完成: {settings.model}")
                            if job:
                                job.message = f"模型下载完成,准备加载"
                            break
                        elif current_status == "error":
                            self.logger.error(f"模型管理器下载失败,使用备用方式")
                            raise RuntimeError(f"Whisper模型下载失败: {settings.model}")
                        else:
                            # 如果模型大小>=1GB,定期提醒用户耐心等待
                            if model_size_mb >= 1024 and elapsed % 30 == 0:  # 每30秒提醒一次
                                wait_msg = f"当前下载模型大于1GB,请耐心等待... {progress:.1f}% ({elapsed}s/{max_wait_time}s)"
                                self.logger.info(f"{wait_msg}")
                                if job:
                                    job.message = wait_msg
                            else:
                                wait_msg = f"等待模型下载... {progress:.1f}%"
                                self.logger.info(f"{wait_msg} ({elapsed}s/{max_wait_time}s)")
                                # 更新任务状态(每次都更新,这样用户可以看到进度变化)
                                if job:
                                    job.message = wait_msg

                    if elapsed >= max_wait_time:
                        self.logger.error(f"模型下载超时,使用备用方式")
                        raise TimeoutError(f"Whisper模型下载超时: {settings.model}")

        except Exception as e:
            self.logger.warning(f"模型管理服务检查失败,使用备用方式: {e}")

        # 尝试使用模型预加载管理器
        try:
            from app.services.model_preload_manager import get_model_manager as get_preload_manager
            model_manager = get_preload_manager()
            if model_manager:
                self.logger.debug("使用模型预加载管理器获取模型")
                if job:
                    job.message = "加载模型中"
                return model_manager.get_model(settings)
        except Exception as e:
            self.logger.debug(f"无法使用模型预加载管理器，回退到本地缓存: {e}")
            pass

        # 回退到简单缓存机制
        key = (settings.model, settings.compute_type, settings.device)
        with _model_lock:
            if key in _model_cache:
                self.logger.debug(f"命中模型缓存: {key}")
                if job:
                    job.message = "使用缓存的模型"
                return _model_cache[key]

            self.logger.info(f"加载模型: {key}")
            if job:
                job.message = f"加载模型 {settings.model}"

            # 处理 auto 模式：解析为具体的计算类型
            compute_type_resolved = settings.compute_type
            if compute_type_resolved == "auto":
                from app.services.whisper_service import get_auto_compute_type
                compute_type_resolved = get_auto_compute_type(settings.device)
                self.logger.info(f"auto模式已解析为: {compute_type_resolved}")

            # 首先尝试仅使用本地文件 (使用 Faster-Whisper)
            try:
                from app.core.config import config
                from faster_whisper import WhisperModel
                m = WhisperModel(
                    settings.model,
                    device=settings.device,
                    compute_type=compute_type_resolved,  # 使用解析后的计算类型
                    download_root=str(config.HF_CACHE_DIR),
                    local_files_only=True  # 禁止自动下载，只使用本地文件
                )
                _model_cache[key] = m
                if job:
                    job.message = "模型加载完成"
                return m
            except Exception as e:
                self.logger.warning(f"本地加载失败,允许下载: {e}")
                if job:
                    job.message = "本地模型不存在,正在下载"
                # 如果本地加载失败,允许下载
                m = WhisperModel(
                    settings.model,
                    device=settings.device,
                    compute_type=compute_type_resolved,  # 使用解析后的计算类型
                    download_root=str(config.HF_CACHE_DIR),
                    local_files_only=False  # 允许下载
                )
                _model_cache[key] = m
                if job:
                    job.message = "模型下载并加载完成"
                return m

  
    def _transcribe_segment_unaligned(
        self,
        seg: Dict,
        model,
        job: JobState
    ) -> Optional[Dict]:
        """
        转录单个音频段（仅转录，不对齐）

        Args:
            seg: 段信息 {file, start_ms, duration_ms, index}
            model: Faster-Whisper 模型
            job: 任务状态

        Returns:
            Dict: 未对齐的转录结果
            {
                "segment_index": 0,
                "language": "zh",
                "segments": [{"id": 0, "start": 10.5, "end": 15.2, "text": "..."}]
            }
        """
        # 使用 whisper_service 提供的 load_audio 函数
        audio = whisper_load_audio(seg['file'])

        try:
            # 使用 Faster-Whisper 转录
            segments_gen, info = model.transcribe(
                audio,
                language=job.language,
                beam_size=5,
                vad_filter=True
            )

            # 转换生成器为列表
            segments_list = list(segments_gen)

            if not segments_list:
                return None

            # 检测语言（首次）
            if not job.language and info.language:
                job.language = info.language
                self.logger.info(f"检测到语言: {job.language}")

            # 时间偏移校正（针对粗略时间戳）
            start_offset = seg['start_ms'] / 1000.0
            adjusted_segments = []

            for idx, s in enumerate(segments_list):
                adjusted_segments.append({
                    'id': idx,
                    'start': s.start + start_offset,
                    'end': s.end + start_offset,
                    'text': s.text.strip()
                })

            return {
                'segment_index': seg.get('index', 0),
                'language': info.language or job.language,
                'segments': adjusted_segments
            }

        finally:
            del audio
            gc.collect()

    def _transcribe_segment_in_memory(
        self,
        audio_array: np.ndarray,
        seg_meta: Dict,
        model,
        job: JobState,
        is_vocals: bool = False
    ) -> Optional[Dict]:
        """
        从内存切片转录（Zero-copy，高性能）

        内存模式下使用，直接从完整音频数组中切片，无需磁盘IO。

        Args:
            audio_array: 完整音频数组
            seg_meta: 分段元数据 {"index": 0, "start": 0.0, "end": 30.5, "mode": "memory"}
            model: Faster-Whisper 模型
            job: 任务状态
            is_vocals: 是否是Demucs分离后的人声（用于日志）

        Returns:
            Dict: 未对齐的转录结果
        """
        sr = 16000
        start_sample = int(seg_meta['start'] * sr)
        end_sample = int(seg_meta['end'] * sr)

        # Zero-copy切片（numpy view，不复制数据）
        audio_slice = audio_array[start_sample:end_sample]

        try:
            # 使用 Faster-Whisper 转录
            segments_gen, info = model.transcribe(
                audio_slice,
                language=job.language,
                beam_size=5,
                vad_filter=False  # 已经是切片，不需要再做 VAD
            )

            # 转换生成器为列表
            segments_list = list(segments_gen)

            if not segments_list:
                return None

            # 检测语言（首次）
            if not job.language and info.language:
                job.language = info.language
                self.logger.info(f"detected language: {job.language}")

            # 时间偏移校正
            start_offset = seg_meta['start']
            adjusted_segments = []

            for idx, s in enumerate(segments_list):
                adjusted_segments.append({
                    'id': idx,
                    'start': s.start + start_offset,
                    'end': s.end + start_offset,
                    'text': s.text.strip()
                })

            return {
                'segment_index': seg_meta['index'],
                'language': info.language or job.language,
                'segments': adjusted_segments
            }

        finally:
            # 注意：audio_slice是view，不需要单独释放
            gc.collect()

    def _transcribe_segment_from_disk(
        self,
        seg: Dict,
        model,
        job: JobState
    ) -> Optional[Dict]:
        """
        从文件加载转录（硬盘模式）

        硬盘模式下使用，从segment文件加载音频进行转录。

        Args:
            seg: 分段信息 {"index": 0, "file": "segment_0.wav", "start": 0.0, "end": 30.0, "mode": "disk"}
            model: Faster-Whisper 模型
            job: 任务状态

        Returns:
            Dict: 未对齐的转录结果
        """
        # 使用 whisper_service 提供的 load_audio 函数
        audio = whisper_load_audio(seg['file'])

        try:
            # 使用 Faster-Whisper 转录
            segments_gen, info = model.transcribe(
                audio,
                language=job.language,
                beam_size=5,
                vad_filter=True
            )

            # 转换生成器为列表
            segments_list = list(segments_gen)

            if not segments_list:
                return None

            # 检测语言（首次）
            if not job.language and info.language:
                job.language = info.language
                self.logger.info(f"detected language: {job.language}")

            # 时间偏移校正（使用start字段，秒为单位）
            start_offset = seg.get('start', seg.get('start_ms', 0) / 1000.0)
            adjusted_segments = []

            for idx, s in enumerate(segments_list):
                adjusted_segments.append({
                    'id': idx,
                    'start': s.start + start_offset,
                    'end': s.end + start_offset,
                    'text': s.text.strip()
                })

            return {
                'segment_index': seg['index'],
                'language': info.language or job.language,
                'segments': adjusted_segments
            }

        finally:
            del audio
            gc.collect()

    def _transcribe_segment(
        self,
        seg_meta: Dict,
        model,
        job: JobState,
        audio_array: Optional[np.ndarray] = None
    ) -> Optional[Dict]:
        """
        统一转录入口（根据模式自动选择）

        Args:
            seg_meta: 分段元数据
            model: Whisper模型
            job: 任务状态
            audio_array: 音频数组（内存模式时必须提供）

        Returns:
            Dict: 未对齐的转录结果
        """
        mode = seg_meta.get('mode', 'disk')

        if mode == 'memory':
            if audio_array is None:
                raise ValueError("memory mode requires audio_array parameter")
            return self._transcribe_segment_in_memory(audio_array, seg_meta, model, job)
        else:
            return self._transcribe_segment_from_disk(seg_meta, model, job)

    def _check_memory_during_transcription(self, job: JobState) -> bool:
        """
        转录过程中检查内存状态

        如果内存严重不足，暂停任务并警告用户。

        Args:
            job: 任务状态对象

        Returns:
            bool: True=继续处理，False=需要暂停
        """
        mem_info = psutil.virtual_memory()
        available_mb = mem_info.available / (1024 * 1024)
        percent_used = mem_info.percent

        # 危险阈值：可用内存<500MB 或 使用率>95%
        if available_mb < 500 or percent_used > 95:
            self.logger.error(f"memory critically low! available: {available_mb:.0f}MB, usage: {percent_used}%")
            job.status = 'paused'
            job.message = f"memory insufficient (available {available_mb:.0f}MB), please close other programs"
            job.paused = True

            # 推送警告SSE
            self._push_sse_signal(job, "memory_warning",
                f"memory critically low (available {available_mb:.0f}MB), task paused")

            return False

        # 警告阈值：可用内存<1GB 或 使用率>90%
        if available_mb < 1024 or percent_used > 90:
            self.logger.warning(f"memory tight: available {available_mb:.0f}MB, usage {percent_used}%")
            # 不暂停，但记录警告

        return True

    def _align_all_results(
        self,
        unaligned_results: List[Dict],
        job: JobState,
        audio_path: str
    ) -> List[Dict]:
        """
        合并转录结果的分段

        此方法合并所有 segments 并返回，不执行对齐操作。
        """
        self.logger.info(f"合并 {len(unaligned_results)} 个分段的转录结果（跳过强制对齐）")

        # 合并所有segments
        all_segments = []
        for result in unaligned_results:
            all_segments.extend(result['segments'])

        if not all_segments:
            self.logger.warning("没有可处理的内容")
            return []

        # 直接返回合并后的结果（Faster-Whisper 已提供时间戳）
        return [{
            'segments': all_segments,
            'word_segments': []  # 新架构使用伪对齐生成字级时间戳
        }]

    def _push_sse_align_progress(
        self,
        job: JobState,
        current_batch: int,
        total_batches: int,
        aligned_count: int,
        total_count: int
    ):
        """
        推送对齐进度SSE事件（前端进度条实时更新）

        事件类型: "align_progress"

        Args:
            job: 任务状态对象
            current_batch: 当前批次号（1-based）
            total_batches: 总批次数
            aligned_count: 已对齐的segment数量
            total_count: 总segment数量
        """
        try:
            from app.services.sse_service import get_sse_manager
            sse_manager = get_sse_manager()

            channel_id = f"job:{job.job_id}"

            # 计算百分比
            batch_progress = (current_batch / total_batches) * 100 if total_batches > 0 else 0
            segment_progress = (aligned_count / total_count) * 100 if total_count > 0 else 0

            sse_manager.broadcast_sync(
                channel_id,
                "progress.align",
                {
                    "job_id": job.job_id,
                    "phase": "align",
                    "batch": {
                        "current": current_batch,
                        "total": total_batches,
                        "progress": round(batch_progress, 2)
                    },
                    "segments": {
                        "aligned": aligned_count,
                        "total": total_count,
                        "progress": round(segment_progress, 2)
                    },
                    "message": f"aligning batch {current_batch}/{total_batches} ({aligned_count}/{total_count} segments)"
                }
            )

        except Exception as e:
            self.logger.debug(f"SSE align progress push failed (non-fatal): {e}")

    def _align_all_results_batched(
        self,
        unaligned_results: List[Dict],
        job: JobState,
        audio_source,  # Union[np.ndarray, str]
        processing_mode: ProcessingMode
    ) -> List[Dict]:
        """
        批量处理转录结果的分段

        此方法合并所有 segments，应用时间校验和微调，然后返回。
        """
        self.logger.info(f"处理 {len(unaligned_results)} 个分段的转录结果（跳过强制对齐）")

        # 1. 合并所有segments
        all_segments = []
        for result in unaligned_results:
            all_segments.extend(result['segments'])

        if not all_segments:
            self.logger.warning("没有可处理的内容")
            return []

        # 2. 对结果进行边界校验，过滤异常结果
        valid_segments = []
        for seg in all_segments:
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            text = seg.get('text', '').strip()

            # 校验1：时间戳必须有效
            if start is None or end is None or start < 0 or end <= start:
                self.logger.warning(f"过滤无效时间戳: start={start}, end={end}, text={text[:20] if text else ''}...")
                continue

            # 校验2：字幕时长不能过长（超过30秒可能是异常）
            duration = end - start
            if duration > 30:
                self.logger.warning(f"过滤过长字幕({duration:.1f}s): {text[:30] if text else ''}...")
                continue

            # 校验3：字幕时长不能过短（小于0.1秒可能是噪音）
            if duration < 0.1 and len(text) > 0:
                self.logger.warning(f"过滤过短字幕({duration:.2f}s): {text}")
                continue

            valid_segments.append(seg)

        # 3. 字幕时间微调 - 修正"抢先出现"问题
        valid_segments = self._adjust_subtitle_timing(valid_segments)

        self.logger.info(f"处理完成: {len(valid_segments)}/{len(all_segments)} 个有效字幕段")

        return [{
            'segments': valid_segments,
            'word_segments': []  # 新架构使用伪对齐生成字级时间戳
        }]

    def _adjust_subtitle_timing(
        self,
        segments: List[Dict],
        start_delay_ms: int = 25,
        end_padding_ms: int = 25
    ) -> List[Dict]:
        """
        字幕时间微调 - 修正对齐偏差

        问题背景：
        转录后的字幕可能出现时间偏差，需要适当调整。

        解决方案：
        1. 将字幕开始时间延后 start_delay_ms（默认25ms）
        2. 将字幕结束时间延后 end_padding_ms（默认25ms，给一点余量）
        3. 确保相邻字幕不重叠

        Args:
            segments: 对齐后的字幕列表
            start_delay_ms: 开始时间延迟（毫秒），推荐20-50ms
            end_padding_ms: 结束时间延长（毫秒），推荐20-50ms

        Returns:
            调整后的字幕列表
        """
        if not segments:
            return segments

        start_delay = start_delay_ms / 1000.0
        end_padding = end_padding_ms / 1000.0

        adjusted = []
        for i, seg in enumerate(segments):
            new_seg = seg.copy()
            old_start = seg.get('start', 0)
            old_end = seg.get('end', 0)

            # 延迟开始时间
            new_start = old_start + start_delay

            # 延长结束时间
            new_end = old_end + end_padding

            # 确保开始时间不超过结束时间
            if new_start >= new_end:
                new_start = old_start  # 回退到原始开始时间

            # 确保不与下一条字幕重叠
            if i < len(segments) - 1:
                next_start = segments[i + 1].get('start', float('inf'))
                # 下一条也会被延迟，所以比较时要考虑
                next_adjusted_start = next_start + start_delay
                if new_end > next_adjusted_start:
                    new_end = next_adjusted_start - 0.05  # 留50ms间隔

            # 确保结束时间仍然有效
            if new_end <= new_start:
                new_end = old_end  # 回退到原始结束时间

            new_seg['start'] = round(new_start, 3)
            new_seg['end'] = round(new_end, 3)
            adjusted.append(new_seg)

        self.logger.info(f"字幕时间微调: 延迟开始25ms, 延长结束25ms")
        return adjusted

    def _format_ts(self, sec: float) -> str:
        """
        格式化时间戳为SRT格式

        Args:
            sec: 秒数

        Returns:
            str: SRT时间戳 (HH:MM:SS,mmm)
        """
        if sec < 0:
            sec = 0

        ms = int(round(sec * 1000))
        h = ms // 3600000
        ms %= 3600000
        m = ms // 60000
        ms %= 60000
        s = ms // 1000
        ms %= 1000

        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    def _generate_srt(self, results: List[Dict], path: str, word_level: bool):
        """
        生成SRT字幕文件
        V3.1.1+dev.20260106.03: 生成前自动修复时间戳重叠

        Args:
            results: 转录结果列表
            path: 输出文件路径
            word_level: 是否使用词级时间戳
        """
        # V3.1.1+dev.20260106.03: 导入重叠修复函数
        from app.utils.text_utils import repair_timestamp_overlaps, detect_timestamp_overlaps

        all_entries = []

        for r in results:
            if not r:
                continue

            # 词级时间戳模式
            if word_level and r.get('word_segments'):
                for w in r['word_segments']:
                    if w.get('start') is not None and w.get('end') is not None:
                        txt = (w.get('word') or '').strip()
                        if txt:
                            all_entries.append({
                                'start': w['start'],
                                'end': w['end'],
                                'text': txt
                            })

            # 句子级时间戳模式（默认）
            elif r.get('segments'):
                for s in r['segments']:
                    if s.get('start') is not None and s.get('end') is not None:
                        txt = (s.get('text') or '').strip()
                        if txt:
                            all_entries.append({
                                'start': s['start'],
                                'end': s['end'],
                                'text': txt
                            })

        # 过滤无效时间戳
        all_entries = [e for e in all_entries if e['end'] > e['start']]

        # V3.1.1+dev.20260106.03: 检测并修复重叠
        if all_entries:
            overlaps = detect_timestamp_overlaps(all_entries)
            if overlaps:
                self.logger.warning(f"检测到 {len(overlaps)} 处时间戳重叠，自动修复中...")
                all_entries = repair_timestamp_overlaps(all_entries, gap_ms=1.0)
                self.logger.info(f"已修复 {len(overlaps)} 处时间戳重叠")

        # 写入SRT格式
        lines = []
        for n, e in enumerate(all_entries, 1):
            lines.append(str(n))  # 序号
            lines.append(
                f"{self._format_ts(e['start'])} --> {self._format_ts(e['end'])}"
            )  # 时间戳
            lines.append(e['text'])  # 字幕文本
            lines.append("")  # 空行

        # 写入文件
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        self.logger.info(f"SRT文件已生成: {path}, 共{len(all_entries)}条字幕")

    def clear_model_cache(self):
        """
        清空模型缓存（供队列服务调用）

        注意: 新架构已移除对齐模型，仅清理 Whisper 模型
        """
        global _model_cache

        with _model_lock:
            for key in list(_model_cache.keys()):
                try:
                    del _model_cache[key]
                except:
                    pass
            _model_cache.clear()
            self.logger.info("Whisper模型缓存已清空")

    # ==========================================
    # SenseVoice 集成方法（Phase 3）
    # ==========================================

    def _sensevoice_transcribe(
        self,
        audio_array: np.ndarray,
        job: 'JobState',
        sample_rate: int = 16000
    ) -> 'SenseVoiceResult':
        """
        调用 SenseVoice 服务进行转录（返回带真实字级时间戳的结果）

        Args:
            audio_array: 音频数组
            job: 任务状态对象
            sample_rate: 采样率

        Returns:
            SenseVoiceResult: 转录结果，包含:
                - text: 原始文本（带标签）
                - text_clean: 清洗后文本
                - words: 字级时间戳列表（真实时间戳，非伪对齐）
                - confidence: 平均置信度
                - language: 检测到的语言
                - emotion: 情感标签
                - event: 事件标签
        """
        from app.services.sensevoice_onnx_service import get_sensevoice_service
        from app.models.sensevoice_models import SenseVoiceResult, WordTimestamp

        self.logger.debug("调用 SenseVoice 转录服务")

        try:
            service = get_sensevoice_service()

            # 确保模型已加载
            if not service.is_loaded:
                self.logger.info("加载 SenseVoice 模型...")
                service.load_model()

            # 调用转录（返回字典）
            # 注: SenseVoice 的 language 参数实际上是自动检测，这里传入 "auto" 即可
            result_dict = service.transcribe_audio_array(
                audio_array=audio_array,
                sample_rate=sample_rate,
                language="auto"
            )

            # 转换为 SenseVoiceResult 对象
            words = [
                WordTimestamp(**w) if isinstance(w, dict) else w
                for w in result_dict.get('words', [])
            ]

            result = SenseVoiceResult(
                text=result_dict.get('text', ''),
                text_clean=result_dict.get('text_clean', ''),
                confidence=result_dict.get('confidence', 1.0),
                words=words,
                start=0.0,  # Chunk 级别的起始时间，由调用者设置
                end=len(audio_array) / sample_rate,
                language=result_dict.get('language'),
                emotion=result_dict.get('emotion'),
                event=result_dict.get('event'),
                raw_result=result_dict
            )

            self.logger.debug(f"SenseVoice 转录完成: {len(result.text_clean)} 字符, {len(words)} 个字")
            return result

        except Exception as e:
            self.logger.error(f"SenseVoice 转录失败: {e}")
            raise

    def _split_sentences(
        self,
        sv_result: 'SenseVoiceResult',
        chunk_start_time: float = 0.0,
        split_config: Optional['SplitConfig'] = None,  # Layer 1: 允许传入配置
        enable_grouping: bool = True                    # Layer 2: 是否启用语义分组 (预留)
    ) -> List['SentenceSegment']:
        """
        将 SenseVoice 结果切分为句子（基于真实字级时间戳）

        Args:
            sv_result: SenseVoice 转录结果（包含真实字级时间戳）
            chunk_start_time: Chunk 在完整音频中的起始时间（用于时间偏移）
            split_config: 分句配置（Layer 1 新增，可选）
            enable_grouping: 是否启用语义分组（Layer 2 预留，可选）

        Returns:
            List[SentenceSegment]: 句子列表
        """
        from app.models.sensevoice_models import SentenceSegment, TextSource
        from app.services.sentence_splitter import SentenceSplitter, SplitConfig

        self.logger.debug(f"开始句子切分: {len(sv_result.words)} 个字")

        if not sv_result.words:
            return []

        # Layer 1: 使用传入的配置或默认配置
        config = split_config or SplitConfig()
        splitter = SentenceSplitter(config)
        sentences = splitter.split(sv_result.words, sv_result.text_clean)

        # Layer 2: 语义分组 (可选，已实现)
        if enable_grouping:
            from app.services.semantic_grouper import SemanticGrouper, GroupConfig
            grouper = SemanticGrouper(GroupConfig(language=config.language))
            sentences = grouper.group(sentences)

        # 【阶段三】VAD 边缘吸附 (Head Snap)
        # 逻辑：如果是 Chunk 的第一句话，且 CTC 延迟在合理范围内（<0.6s），
        # 强制将其 start 对齐到 Chunk 的物理起始点 (0.0 相对时间)
        HEAD_SNAP_THRESHOLD = 0.6  # 最大允许吸附的延迟（秒）

        if sentences:
            first_sent = sentences[0]
            # 检查第一个词的相对开始时间（相对于 chunk）
            # 如果 > 0 且 < 阈值，说明 CTC 有延迟，需要吸附到 VAD 边缘
            if 0 < first_sent.start < HEAD_SNAP_THRESHOLD:
                self.logger.debug(
                    f"Head Snap: '{first_sent.text[:15]}...' 延迟修正 "
                    f"{first_sent.start:.3f}s -> 0.0s (相对VAD)"
                )
                # 修正句子开始时间
                first_sent.start = 0.0
                # 同时修正第一个单词的开始时间（保持一致性）
                if first_sent.words:
                    first_sent.words[0].start = 0.0

        # 调整时间偏移（将 Chunk 内的相对时间转换为绝对时间）
        for sentence in sentences:
            sentence.start += chunk_start_time
            sentence.end += chunk_start_time
            sentence.source = TextSource.SENSEVOICE
            # V3.1.2+dev.20260111.01: 使用 update_confidence 确保 display_confidence 同步更新
            sentence.update_confidence(sv_result.confidence, source="sensevoice")

            # 调整字级时间戳的偏移
            for word in sentence.words:
                word.start += chunk_start_time
                word.end += chunk_start_time

        self.logger.debug(f"句子切分完成: {len(sentences)} 句")
        return sentences

    def _split_text_by_punctuation(self, text: str) -> List[str]:
        """
        基于标点符号切分文本

        Args:
            text: 原始文本

        Returns:
            List[str]: 句子列表
        """
        import re

        # 句末标点
        sentence_end_pattern = r'([。？！.?!])'

        # 使用正则切分，保留标点
        parts = re.split(sentence_end_pattern, text)

        # 合并标点到前一个句子
        sentences = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and re.match(sentence_end_pattern, parts[i + 1]):
                sentences.append(parts[i] + parts[i + 1])
                i += 2
            else:
                if parts[i].strip():
                    sentences.append(parts[i])
                i += 1

        return [s.strip() for s in sentences if s.strip()]

    def _generate_pseudo_word_timestamps(
        self,
        text: str,
        start: float,
        end: float,
        confidence: float = 1.0
    ) -> List['WordTimestamp']:
        """
        生成伪字级时间戳（均匀分布）

        Args:
            text: 句子文本
            start: 句子开始时间
            end: 句子结束时间
            confidence: 置信度

        Returns:
            List[WordTimestamp]: 字级时间戳列表
        """
        from app.models.sensevoice_models import WordTimestamp

        if not text:
            return []

        duration = end - start
        char_duration = duration / len(text)

        words = []
        for i, char in enumerate(text):
            word_start = start + i * char_duration
            word_end = start + (i + 1) * char_duration

            words.append(WordTimestamp(
                word=char,
                start=word_start,
                end=word_end,
                confidence=confidence,
                is_pseudo=True  # 标记为伪对齐
            ))

        return words

    def _generate_subtitle_from_sentences(
        self,
        sentences: List['SentenceSegment'],
        output_path: str,
        include_translation: bool = False
    ) -> str:
        """
        从句子列表生成 SRT 字幕文件

        Args:
            sentences: 句子列表
            output_path: 输出文件路径
            include_translation: 是否包含翻译（双语字幕）

        Returns:
            str: 生成的SRT文件路径
        """
        self.logger.info(f"生成SRT字幕: {len(sentences)} 句 -> {output_path}")

        lines = []
        for idx, sentence in enumerate(sentences, 1):
            # 序号
            lines.append(str(idx))

            # 时间戳
            start_ts = self._format_ts(sentence.start)
            end_ts = self._format_ts(sentence.end)
            lines.append(f"{start_ts} --> {end_ts}")

            # 字幕文本（使用清洗后的文本）
            if include_translation and sentence.translation:
                # 双语字幕：原文 + 翻译
                lines.append(sentence.text_clean or sentence.text)
                lines.append(sentence.translation)
            else:
                lines.append(sentence.text_clean or sentence.text)

            # 空行分隔
            lines.append("")

        # 写入文件
        # 确保父目录存在
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        self.logger.info(f"SRT字幕生成完成: {output_path}")
        return output_path

    def _save_raw_transcription(
        self,
        job: 'JobState',
        sentences: List['SentenceSegment']
    ):
        """
        保存原始转录数据（未经 LLM 处理的 SenseVoice 原始输出）

        Args:
            job: 任务状态
            sentences: 句子列表
        """
        import json
        from pathlib import Path

        try:
            # 准备保存数据
            raw_data = {
                "job_id": job.job_id,
                "filename": job.filename,
                "timestamp": self._get_current_timestamp(),
                "total_sentences": len(sentences),
                "sentences": []
            }

            # 收集所有句子的原始数据
            for idx, sentence in enumerate(sentences):
                sentence_data = {
                    "index": idx,
                    "start": sentence.start,
                    "end": sentence.end,
                    "text": sentence.text,  # 包含原始标签的文本
                    "confidence": sentence.confidence,
                    "source": sentence.source.value if hasattr(sentence.source, 'value') else str(sentence.source),
                    "words": []
                }

                # 添加字级时间戳
                if hasattr(sentence, 'words') and sentence.words:
                    for word in sentence.words:
                        word_data = {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "confidence": word.confidence,
                            "is_pseudo": word.is_pseudo
                        }
                        sentence_data["words"].append(word_data)

                raw_data["sentences"].append(sentence_data)

            # 保存到文件
            output_path = Path(job.dir) / f"{job.job_id}_raw_transcription.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"原始转录数据已保存: {output_path}")

        except Exception as e:
            self.logger.error(f"保存原始转录数据失败: {e}", exc_info=True)

    def _get_current_timestamp(self) -> str:
        """获取当前时间戳字符串"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _init_chunk_states(
        self,
        vad_segments: List[dict],
        audio_array: np.ndarray,
        sr: int = 16000
    ) -> List['ChunkProcessState']:
        """
        初始化所有 Chunk 的处理状态

        关键：保存原始音频引用，用于熔断回溯

        Args:
            vad_segments: VAD 切分结果
            audio_array: 完整音频数组
            sr: 采样率

        Returns:
            List[ChunkProcessState]: Chunk 状态列表
        """
        from app.models.circuit_breaker_models import ChunkProcessState, SeparationLevel

        self.logger.info(f"初始化 {len(vad_segments)} 个 Chunk 状态...")

        states = []
        for seg in vad_segments:
            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)
            chunk_audio = audio_array[start_sample:end_sample]

            state = ChunkProcessState(
                chunk_index=seg['index'],
                start_time=seg['start'],
                end_time=seg['end'],
                original_audio=chunk_audio.copy(),  # 关键：保存原始音频副本
                current_audio=chunk_audio,           # 当前使用的音频（可能被分离后替换）
                sample_rate=sr,
                separation_level=SeparationLevel.NONE
            )
            states.append(state)

        self.logger.info(f"Chunk 状态初始化完成")
        return states

    def _convert_audio_chunks_to_states(
        self,
        audio_chunks: List['AudioChunk']
    ) -> List['ChunkProcessState']:
        """
        将 AudioChunk 列表转换为 ChunkProcessState 列表

        用于将新架构 PreprocessingPipeline 的输出转换为旧架构的数据结构，
        以便复用现有的 _transcribe_chunk_with_fusing 方法。

        Args:
            audio_chunks: PreprocessingPipeline 输出的 AudioChunk 列表

        Returns:
            List[ChunkProcessState]: 转换后的 Chunk 状态列表
        """
        from app.models.circuit_breaker_models import ChunkProcessState
        from app.services.audio.chunk_engine import AudioChunk

        self.logger.info(f"转换 {len(audio_chunks)} 个 AudioChunk 到 ChunkProcessState...")

        states = []
        for chunk in audio_chunks:
            state = ChunkProcessState(
                chunk_index=chunk.index,
                start_time=chunk.start,
                end_time=chunk.end,
                # 音频引用：original_audio 用于熔断回溯
                original_audio=chunk.original_audio if chunk.original_audio is not None else chunk.audio.copy(),
                current_audio=chunk.audio,
                sample_rate=chunk.sample_rate,
                # 分离状态
                separation_level=chunk.separation_level,
                separation_model_used=chunk.separation_model,
                # 熔断状态
                fuse_retry_count=chunk.fuse_retry_count,
            )
            states.append(state)

        self.logger.info(f"AudioChunk 转换完成: {len(states)} 个 ChunkProcessState")
        return states

    async def _transcribe_chunk_with_fusing(
        self,
        chunk_state: 'ChunkProcessState',
        job: 'JobState',
        subtitle_manager: 'StreamingSubtitleManager',
        demucs_service
    ) -> List['SentenceSegment']:
        """
        单个 Chunk 的转录流程（含熔断回溯）

        流程：
        1. 使用 current_audio 进行 SenseVoice 转录
        2. 评估置信度和事件标签
        3. 熔断决策
        4. 如需熔断：回溯到 original_audio，升级分离，重新转录
        5. 止损点：max_retry=1

        Args:
            chunk_state: Chunk 处理状态
            job: 任务状态对象
            subtitle_manager: 流式字幕管理器
            demucs_service: Demucs 服务实例

        Returns:
            List[SentenceSegment]: 句子列表
        """
        from app.services.fuse_breaker import get_fuse_breaker, execute_fuse_upgrade, FuseAction

        fuse_breaker = get_fuse_breaker()

        while True:
            # 1. SenseVoice 转录
            sv_result = self._sensevoice_transcribe(chunk_state.current_audio, job, chunk_state.sample_rate)

            # 2. 分句
            sentences = self._split_sentences(sv_result, chunk_state.start_time)

            # 3. 计算置信度和事件标签
            avg_confidence = sum(s.confidence for s in sentences) / len(sentences) if sentences else 0.0
            event_tag = sv_result.event  # SenseVoice 检测到的事件（BGM/Noise等）

            # 4. 熔断决策
            decision = fuse_breaker.should_fuse(
                chunk_state=chunk_state,
                confidence=avg_confidence,
                event_tag=event_tag
            )

            self.logger.debug(
                f"Chunk {chunk_state.chunk_index} 熔断决策: {decision.action.value}, "
                f"置信度={avg_confidence:.2f}, 事件={event_tag}"
            )

            # 5. 处理决策
            if decision.action == FuseAction.ACCEPT:
                # 接受结果，推送草稿事件（Phase 5: 双模态架构）
                # 使用 add_draft_sentences 批量推送，触发 subtitle.draft 事件
                subtitle_manager.add_draft_sentences(chunk_state.chunk_index, sentences)
                return sentences

            elif decision.action == FuseAction.UPGRADE_SEPARATION:
                # 熔断回溯：使用原始音频重新分离
                self.logger.info(
                    f"Chunk {chunk_state.chunk_index} 触发熔断，"
                    f"升级分离: {chunk_state.separation_level.value} → {decision.next_separation_level.value}"
                )

                chunk_state = execute_fuse_upgrade(
                    chunk_state=chunk_state,
                    next_level=decision.next_separation_level,
                    demucs_service=demucs_service
                )

                # 继续循环，使用升级后的音频重新转录
                continue

            else:
                # 未知动作，接受当前结果
                subtitle_manager.add_draft_sentences(chunk_state.chunk_index, sentences)
                return sentences

    async def _whisper_text_patch_with_arbitration(
        self,
        sentence: 'SentenceSegment',
        sentence_index: int,
        audio_array: np.ndarray,
        job: 'JobState',
        subtitle_manager: 'StreamingSubtitleManager',
        is_trash_suspect: bool = False
    ) -> 'SentenceSegment':
        """
        Whisper 补刀 + 仲裁判决（二次安检机制）

        核心逻辑：
        - 如果是垃圾嫌疑样本，根据 Whisper 反馈判决去留
        - 如果是常规补刀，直接采纳 Whisper 结果

        Args:
            sentence: 原始句子
            sentence_index: 句子索引
            audio_array: 完整音频数组
            job: 任务状态
            subtitle_manager: 流式字幕管理器
            is_trash_suspect: 是否是垃圾嫌疑样本

        Returns:
            SentenceSegment: 更新后的句子（或标记删除）
        """
        from app.services.whisper_service import get_whisper_service
        from app.models.sensevoice_models import TextSource

        whisper_service = get_whisper_service()

        # 确保 Whisper 模型已加载（首次运行会自动下载）
        if not whisper_service.is_loaded:
            self.logger.info("加载 Whisper 模型...")
            whisper_service.load_model()  # 自动下载 CTranslate2 格式模型

        # 提取对应时间段的音频（增加前向重叠，为 Whisper 提供上下文预热）
        sr = 16000
        WHISPER_OVERLAP_SEC = 0.5  # Whisper 上下文重叠时长（秒）

        # 计算重叠后的起始位置（不小于0）
        overlap_start = max(0.0, sentence.start - WHISPER_OVERLAP_SEC)
        start_sample = int(overlap_start * sr)
        end_sample = int(sentence.end * sr)
        audio_segment = audio_array[start_sample:end_sample]

        # 记录日志（调试用）
        if overlap_start < sentence.start:
            self.logger.debug(
                f"Whisper 补刀添加 {sentence.start - overlap_start:.2f}s 前向重叠: "
                f"[{overlap_start:.2f}s, {sentence.end:.2f}s]"
            )

        # 获取上下文提示（现在使用清洗后的文本，避免下划线等原始 token）
        context = subtitle_manager.get_context_window(sentence_index)

        # Whisper 转录
        result = whisper_service.transcribe(
            audio=audio_segment,
            initial_prompt=context,
            language=getattr(job.settings, 'language', 'auto'),
            word_timestamps=False,
            condition_on_previous_text=False  # 禁用前文条件化，避免与 initial_prompt 形成双重提示词增益
        )

        whisper_text = result.get('text', '').strip()
        whisper_conf = self._estimate_whisper_confidence(result)
        segments = result.get('segments', [])

        # ========== 第三道防线: 双流校验与置信度门控 ==========
        from app.services.text_normalizer import TextNormalizer

        # A. avg_logprob 检查（模型在瞎猜）
        avg_logprob = -0.5  # 默认值
        avg_no_speech = 0.0  # 默认值
        if segments:
            avg_logprob = sum(s.get('avg_logprob', -0.5) for s in segments) / len(segments)
            avg_no_speech = sum(s.get('no_speech_prob', 0.0) for s in segments) / len(segments)

            # 熔断条件1: avg_logprob 过低（模型不确信）
            if avg_logprob < -1.0:
                self.logger.warning(
                    f"Whisper 熔断(avg_logprob={avg_logprob:.2f} < -1.0) {sentence_index}: "
                    f"'{whisper_text[:50]}...', 回退到 SenseVoice"
                )
                return sentence

            # 熔断条件2: no_speech_prob 高但仍输出文本（静音段被强行翻译）
            if avg_no_speech > 0.6 and whisper_text:
                self.logger.warning(
                    f"Whisper 熔断(no_speech={avg_no_speech:.2f} > 0.6) {sentence_index}: "
                    f"'{whisper_text[:50]}...', 回退到 SenseVoice"
                )
                return sentence

        # B. 幻觉正则检测（使用升级后的 TextNormalizer）
        if TextNormalizer.is_whisper_hallucination(whisper_text):
            self.logger.warning(
                f"Whisper 熔断(幻觉检测) {sentence_index}: '{whisper_text[:50]}...', 回退到 SenseVoice"
            )
            return sentence

        # C. 双流长度/内容对比（关键创新）
        sensevoice_text = sentence.text_clean or sentence.text or ""

        # 特殊情况: SenseVoice 为空，但 Whisper 有输出
        if not sensevoice_text and whisper_text:
            # 如果 Whisper 置信度极高，给予特权放行
            if avg_logprob > -0.5:
                self.logger.info(
                    f"SenseVoice 为空，Whisper 高置信度(logprob={avg_logprob:.2f})特权放行 {sentence_index}: "
                    f"'{whisper_text[:50]}...'"
                )
                # 继续正常流程
            else:
                # Whisper 置信度不够高，保守起见不采纳
                self.logger.warning(
                    f"SenseVoice 为空，Whisper 置信度不足(logprob={avg_logprob:.2f}) {sentence_index}: "
                    f"'{whisper_text[:50]}...', 保持 SenseVoice 空结果"
                )
                return sentence

        # 长度暴涨检测: Whisper 输出远超 SenseVoice
        if sensevoice_text:
            len_sv = len(sensevoice_text)
            len_w = len(whisper_text)

            # 公式: len(whisper) > 3 * len(sensevoice) + 10
            if len_w > 3 * len_sv + 10:
                # 特权放行: 如果 Whisper 置信度极高，可能是 SenseVoice 漏识别
                if avg_logprob > -0.5:
                    self.logger.info(
                        f"Whisper 长度暴涨但置信度高(logprob={avg_logprob:.2f}) {sentence_index}, 特权放行"
                    )
                else:
                    self.logger.warning(
                        f"Whisper 熔断(长度暴涨 {len_w} > 3*{len_sv}+10) {sentence_index}: "
                        f"'{whisper_text[:50]}...', 回退到 SenseVoice"
                    )
                    return sentence

        # D. 原有检测逻辑（保留兼容）
        if context and whisper_text:
            # 检测是否包含大量下划线（幻觉标志）
            underscore_ratio = whisper_text.count('_') / max(len(whisper_text), 1)
            if underscore_ratio > 0.3:  # 超过 30% 是下划线
                self.logger.warning(
                    f"Whisper 补刀检测到下划线幻觉 {sentence_index}: "
                    f"'{whisper_text[:50]}...' (下划线占比 {underscore_ratio:.1%}), 回退到 SenseVoice"
                )
                return sentence

            # 检测是否重复了提示词内容（相似度过高）
            context_words = set(context.split())
            whisper_words = set(whisper_text.split())
            if len(context_words) > 0:
                overlap_ratio = len(context_words & whisper_words) / len(context_words)
                # 如果 Whisper 输出与 context 重叠度超过 80%，且长度相近，可能是照抄
                if overlap_ratio > 0.8 and abs(len(whisper_text) - len(context)) < len(context) * 0.3:
                    self.logger.warning(
                        f"Whisper 补刀检测到提示词重复 {sentence_index}: "
                        f"与 context 重叠度 {overlap_ratio:.1%}, 回退到 SenseVoice"
                    )
                    return sentence

        # === 判决时刻 ===
        if is_trash_suspect:
            # 仲裁阈值
            WHISPER_ARBITRATION_CONF = 0.5  # Whisper 也必须有一定确信度
            MIN_TEXT_LENGTH = 1  # 最少文本长度

            # V3.8 修复: Whisper 空输出时，保留 SenseVoice 而非删除
            # 有音频却没有 Whisper 输出，说明 Whisper 幻觉，应回退到 SenseVoice
            if len(whisper_text) < MIN_TEXT_LENGTH:
                self.logger.warning(
                    f"仲裁结果: Whisper 空输出，保留 SenseVoice 结果 {sentence_index} - "
                    f"SenseVoice({sentence.confidence:.2f}): '{sensevoice_text[:50]}...'"
                )
                # 不删除，直接返回原 SenseVoice 句子
                return sentence

            # 判死刑条件：Whisper 有输出但置信度极低（两者都不可靠）
            if whisper_conf < WHISPER_ARBITRATION_CONF:
                # V3.8: 如果 SenseVoice 有实质内容，优先保留而非删除
                if sensevoice_text and len(sensevoice_text) >= 2:
                    self.logger.warning(
                        f"仲裁结果: Whisper 低置信度({whisper_conf:.2f})，保留 SenseVoice {sentence_index}: "
                        f"'{sensevoice_text[:50]}...'"
                    )
                    return sentence
                else:
                    # SenseVoice 也为空或单字符，且 Whisper 低置信度，才删除
                    self.logger.info(
                        f"仲裁结果: 删除垃圾片段 {sentence_index} - "
                        f"SenseVoice({sentence.confidence:.2f}, '{sensevoice_text}') + Whisper({whisper_conf:.2f}, '{whisper_text}')"
                    )
                    subtitle_manager.mark_for_deletion(
                        index=sentence_index,
                        reason=f"whisper_arbitration_failed_conf_{whisper_conf:.2f}"
                    )
                    return sentence
            else:
                # 挽救成功！
                self.logger.info(
                    f"仲裁结果: 挽救片段 {sentence_index} - "
                    f"'{whisper_text}' (Whisper Conf {whisper_conf:.2f})"
                )

        # 常规补刀 OR 垃圾样本通过仲裁 => 采纳 Whisper 结果
        if not whisper_text:
            self.logger.warning(f"Whisper 补刀返回空文本，保留原结果")
            return sentence

        # 保存 Whisper 备选文本
        sentence.whisper_alternative = whisper_text

        # 使用伪对齐更新句子
        subtitle_manager.update_sentence(
            index=sentence_index,
            new_text=whisper_text,
            source=TextSource.WHISPER_PATCH,
            confidence=whisper_conf
        )

        return subtitle_manager.sentences[sentence_index]

    async def _whisper_text_patch(
        self,
        sentence: 'SentenceSegment',
        sentence_index: int,
        audio_array: np.ndarray,
        job: 'JobState',
        subtitle_manager: 'StreamingSubtitleManager'
    ) -> 'SentenceSegment':
        """
        Whisper 补刀（时空解耦版：仅取文本）

        核心原则：
        - SenseVoice 确定的时间轴（start/end）不可变
        - 仅使用 Whisper 的文本结果
        - 新文本使用伪对齐生成字级时间戳

        Args:
            sentence: 原始句子（由 SenseVoice 生成）
            sentence_index: 句子索引
            audio_array: 完整音频数组
            job: 任务状态
            subtitle_manager: 流式字幕管理器

        Returns:
            SentenceSegment: 更新后的句子
        """
        from app.services.whisper_service import get_whisper_service
        from app.models.sensevoice_models import TextSource

        whisper_service = get_whisper_service()

        # 确保 Whisper 模型已加载（首次运行会自动下载）
        if not whisper_service.is_loaded:
            self.logger.info("加载 Whisper 模型...")
            whisper_service.load_model()  # 自动下载 CTranslate2 格式模型

        # 提取对应时间段的音频（增加前向重叠，为 Whisper 提供上下文预热）
        sr = 16000
        WHISPER_OVERLAP_SEC = 0.5  # Whisper 上下文重叠时长（秒）

        # 计算重叠后的起始位置（不小于0）
        overlap_start = max(0.0, sentence.start - WHISPER_OVERLAP_SEC)
        start_sample = int(overlap_start * sr)
        end_sample = int(sentence.end * sr)
        audio_segment = audio_array[start_sample:end_sample]

        # 记录日志（调试用）
        if overlap_start < sentence.start:
            self.logger.debug(
                f"Whisper 补刀添加 {sentence.start - overlap_start:.2f}s 前向重叠: "
                f"[{overlap_start:.2f}s, {sentence.end:.2f}s]"
            )

        # 获取上下文提示（现在使用清洗后的文本，避免下划线等原始 token）
        context = subtitle_manager.get_context_window(sentence_index)

        # Whisper 转录（仅取文本，弃用时间戳）
        result = whisper_service.transcribe(
            audio=audio_segment,
            initial_prompt=context,
            language=getattr(job.settings, 'language', 'auto'),
            word_timestamps=False,  # 不需要字级时间戳，使用伪对齐
            condition_on_previous_text=False  # 禁用前文条件化，避免与 initial_prompt 形成双重提示词增益
        )

        whisper_text = result.get('text', '').strip()
        segments = result.get('segments', [])

        # ========== 第三道防线: 双流校验与置信度门控 ==========
        from app.services.text_normalizer import TextNormalizer

        # A. avg_logprob 检查（模型在瞎猜）
        avg_logprob = -0.5  # 默认值
        avg_no_speech = 0.0  # 默认值
        if segments:
            avg_logprob = sum(s.get('avg_logprob', -0.5) for s in segments) / len(segments)
            avg_no_speech = sum(s.get('no_speech_prob', 0.0) for s in segments) / len(segments)

            # 熔断条件1: avg_logprob 过低（模型不确信）
            if avg_logprob < -1.0:
                self.logger.warning(
                    f"Whisper 熔断(avg_logprob={avg_logprob:.2f} < -1.0) {sentence_index}: "
                    f"'{whisper_text[:50]}...', 回退到 SenseVoice"
                )
                return sentence

            # 熔断条件2: no_speech_prob 高但仍输出文本（静音段被强行翻译）
            if avg_no_speech > 0.6 and whisper_text:
                self.logger.warning(
                    f"Whisper 熔断(no_speech={avg_no_speech:.2f} > 0.6) {sentence_index}: "
                    f"'{whisper_text[:50]}...', 回退到 SenseVoice"
                )
                return sentence

        # B. 幻觉正则检测（使用升级后的 TextNormalizer）
        if TextNormalizer.is_whisper_hallucination(whisper_text):
            self.logger.warning(
                f"Whisper 熔断(幻觉检测) {sentence_index}: '{whisper_text[:50]}...', 回退到 SenseVoice"
            )
            return sentence

        # C. 双流长度对比
        sensevoice_text = sentence.text_clean or sentence.text or ""
        if sensevoice_text:
            len_sv = len(sensevoice_text)
            len_w = len(whisper_text)

            # 长度暴涨检测
            if len_w > 3 * len_sv + 10:
                if avg_logprob > -0.5:
                    self.logger.info(
                        f"Whisper 长度暴涨但置信度高(logprob={avg_logprob:.2f}) {sentence_index}, 特权放行"
                    )
                else:
                    self.logger.warning(
                        f"Whisper 熔断(长度暴涨 {len_w} > 3*{len_sv}+10) {sentence_index}: "
                        f"'{whisper_text[:50]}...', 回退到 SenseVoice"
                    )
                    return sentence

        # D. 原有检测逻辑（保留兼容）
        if context and whisper_text:
            # 检测是否包含大量下划线（幻觉标志）
            underscore_ratio = whisper_text.count('_') / max(len(whisper_text), 1)
            if underscore_ratio > 0.3:  # 超过 30% 是下划线
                self.logger.warning(
                    f"Whisper 补刀检测到下划线幻觉 {sentence_index}: "
                    f"'{whisper_text[:50]}...' (下划线占比 {underscore_ratio:.1%}), 回退到 SenseVoice"
                )
                return sentence

            # 检测是否重复了提示词内容（相似度过高）
            context_words = set(context.split())
            whisper_words = set(whisper_text.split())
            if len(context_words) > 0:
                overlap_ratio = len(context_words & whisper_words) / len(context_words)
                # 如果 Whisper 输出与 context 重叠度超过 80%，且长度相近，可能是照抄
                if overlap_ratio > 0.8 and abs(len(whisper_text) - len(context)) < len(context) * 0.3:
                    self.logger.warning(
                        f"Whisper 补刀检测到提示词重复 {sentence_index}: "
                        f"与 context 重叠度 {overlap_ratio:.1%}, 回退到 SenseVoice"
                    )
                    return sentence

        if not whisper_text:
            self.logger.warning(f"Whisper 补刀返回空文本，保留原结果")
            return sentence

        # 保存 Whisper 备选文本
        sentence.whisper_alternative = whisper_text

        # 使用伪对齐更新句子
        subtitle_manager.update_sentence(
            index=sentence_index,
            new_text=whisper_text,
            source=TextSource.WHISPER_PATCH,
            confidence=self._estimate_whisper_confidence(result)
        )

        return subtitle_manager.sentences[sentence_index]

    def _estimate_whisper_confidence(self, result: dict) -> float:
        """估算 Whisper 结果置信度"""
        segments = result.get('segments', [])
        if not segments:
            # 无可靠片段时不返回置信度，让前端隐藏徽章
            return None

        # 基于 avg_logprob 和 no_speech_prob 计算
        total_logprob = sum(s.get('avg_logprob', -0.5) for s in segments)
        avg_logprob = total_logprob / len(segments)

        avg_no_speech = sum(s.get('no_speech_prob', 0.1) for s in segments) / len(segments)

        # 转换为 0-1 置信度
        confidence = min(1.0, max(0.0, 1.0 + avg_logprob))  # logprob 越接近 0 越好
        confidence *= (1.0 - avg_no_speech)  # no_speech 越低越好

        return round(confidence, 3)

    async def _whisper_buffer_pool_process(
        self,
        patch_queue: List[Dict],
        audio_array: np.ndarray,
        job: 'JobState',
        subtitle_manager: 'StreamingSubtitleManager',
        progress_tracker
    ):
        """
        使用 Whisper 缓冲池批量处理句子（DEEP_LISTEN 模式）

        优势:
        - 累积多个短 Chunk，拼接后一次性推理
        - 利用 Whisper 长上下文能力，从根源消除短音频幻觉
        - 长文本回填对齐到原始时间戳

        Args:
            patch_queue: 需要补刀的句子队列
            audio_array: 完整音频数组
            job: 任务状态
            subtitle_manager: 流式字幕管理器
            progress_tracker: 进度追踪器
        """
        from app.services.whisper_service import get_whisper_service
        from app.services.whisper_buffer_pool import WhisperBufferService, WhisperBufferConfig
        from app.services.text_normalizer import TextNormalizer
        from app.models.sensevoice_models import TextSource
        from app.services.progress_tracker import ProcessPhase

        whisper_service = get_whisper_service()

        # 确保 Whisper 模型已加载
        if not whisper_service.is_loaded:
            self.logger.info("加载 Whisper 模型...")
            whisper_service.load_model()

        # 初始化缓冲池服务
        buffer_config = WhisperBufferConfig(
            min_duration_sec=5.0,       # 累积 5 秒后触发
            max_chunk_count=3,          # 或累积 3 个 Chunk 后触发
            silence_trigger_sec=1.0,    # 长静音也触发
            max_duration_sec=30.0,      # 最大 30 秒
        )
        buffer_service = WhisperBufferService(buffer_config)

        sr = 16000
        processed_count = 0
        total_count = len(patch_queue)

        self.logger.info(f"Whisper 缓冲池模式: 处理 {total_count} 个句子")

        # 按顺序处理句子，累积到缓冲池
        for idx, item in enumerate(patch_queue):
            sent_idx = item["index"]
            sentence = item["sentence"]

            # 提取音频片段
            start_sample = int(sentence.start * sr)
            end_sample = int(sentence.end * sr)
            audio_segment = audio_array[start_sample:end_sample]

            # 获取 SenseVoice 文本
            sv_text = getattr(sentence, 'text_clean', None) or sentence.text or ""
            sv_conf = sentence.confidence

            # 添加到缓冲池
            buffer_service.add_chunk(
                index=sent_idx,
                start=sentence.start,
                end=sentence.end,
                audio=audio_segment,
                sensevoice_text=sv_text,
                sensevoice_confidence=sv_conf
            )

            # 检测是否与下一个句子有长静音间隔
            has_long_silence = False
            if idx < len(patch_queue) - 1:
                next_sentence = patch_queue[idx + 1]["sentence"]
                gap = next_sentence.start - sentence.end
                if gap > buffer_config.silence_trigger_sec:
                    has_long_silence = True

            is_last = (idx == len(patch_queue) - 1)

            # 检查是否触发缓冲池处理
            if buffer_service.should_trigger(has_long_silence=has_long_silence, is_eof=is_last):
                # 获取上下文提示
                context = subtitle_manager.get_context_window(sent_idx)

                # 处理缓冲池
                aligned_results = buffer_service.process_buffer(
                    whisper_service=whisper_service,
                    language=getattr(job.settings, 'language', 'auto'),
                    initial_prompt=context
                )

                # 应用对齐结果
                for result in aligned_results:
                    chunk_idx = result["chunk_index"]
                    whisper_text = result.get("whisper_text", "").strip()
                    whisper_conf = result.get("whisper_confidence", 0.5)
                    sv_text_orig = result.get("sensevoice_text", "")

                    # 跳过空结果
                    if not whisper_text:
                        self.logger.debug(f"缓冲池对齐: Chunk {chunk_idx} Whisper 输出为空，保留 SenseVoice")
                        processed_count += 1
                        continue

                    # 幻觉检测
                    if TextNormalizer.is_whisper_hallucination(whisper_text):
                        self.logger.warning(
                            f"缓冲池对齐: Chunk {chunk_idx} Whisper 幻觉 '{whisper_text[:30]}...', 保留 SenseVoice"
                        )
                        processed_count += 1
                        continue

                    # 双流长度对比
                    if sv_text_orig:
                        len_sv = len(sv_text_orig)
                        len_w = len(whisper_text)
                        if len_w > 3 * len_sv + 10 and whisper_conf < 0.7:
                            self.logger.warning(
                                f"缓冲池对齐: Chunk {chunk_idx} 长度暴涨 ({len_w} > 3*{len_sv}+10), 保留 SenseVoice"
                            )
                            processed_count += 1
                            continue

                    # 更新句子
                    if chunk_idx in subtitle_manager.sentences:
                        subtitle_manager.update_sentence(
                            index=chunk_idx,
                            new_text=whisper_text,
                            source=TextSource.WHISPER_PATCH,
                            confidence=whisper_conf
                        )
                        self.logger.debug(
                            f"缓冲池对齐: Chunk {chunk_idx} 更新为 '{whisper_text[:50]}...'"
                        )

                    processed_count += 1

                # 更新进度
                progress_tracker.update_phase(ProcessPhase.WHISPER_PATCH, increment=len(aligned_results))

        # 处理缓冲池中的剩余内容
        if not buffer_service.is_empty:
            context = subtitle_manager.get_context_window(patch_queue[-1]["index"]) if patch_queue else ""
            remaining_results = buffer_service.flush_remaining(
                whisper_service=whisper_service,
                language=getattr(job.settings, 'language', 'auto'),
                initial_prompt=context
            )

            for result in remaining_results:
                chunk_idx = result["chunk_index"]
                whisper_text = result.get("whisper_text", "").strip()
                whisper_conf = result.get("whisper_confidence", 0.5)

                if whisper_text and not TextNormalizer.is_whisper_hallucination(whisper_text):
                    if chunk_idx in subtitle_manager.sentences:
                        subtitle_manager.update_sentence(
                            index=chunk_idx,
                            new_text=whisper_text,
                            source=TextSource.WHISPER_PATCH,
                            confidence=whisper_conf
                        )
                processed_count += 1

            progress_tracker.update_phase(ProcessPhase.WHISPER_PATCH, increment=len(remaining_results))

        self.logger.info(f"Whisper 缓冲池处理完成: {processed_count}/{total_count} 个句子")

    async def _post_process_enhancement(
        self,
        sentences: List['SentenceSegment'],
        audio_array: np.ndarray,
        job: 'JobState',
        subtitle_manager: 'StreamingSubtitleManager',
        solution_config: 'SolutionConfig'
    ) -> List['SentenceSegment']:
        """
        后处理增强层（所有 Chunk 转录完成后执行）

        根据用户配置执行：
        1. 低置信度句子 → Whisper 补刀（仅文本 + 伪对齐）
        2. [可选] LLM 校对
        3. [可选] LLM 翻译

        注意：这不是熔断，熔断在转录阶段已经处理完成

        Args:
            sentences: 所有句子列表
            audio_array: 完整音频数组
            job: 任务状态
            subtitle_manager: 流式字幕管理器
            solution_config: 方案配置

        Returns:
            List[SentenceSegment]: 增强后的句子列表
        """
        from app.services.progress_tracker import get_progress_tracker, ProcessPhase
        from app.services.solution_matrix import EnhancementMode, ProofreadMode, TranslateMode
        from app.core.thresholds import needs_whisper_patch, is_critical_patch_needed

        progress_tracker = get_progress_tracker(job.job_id, solution_config.preset_id)

        # 检查任务是否已取消
        if job.canceled:
            self.logger.info(f"任务已取消，跳过后处理增强: {job.job_id}")
            return sentences

        # 调试日志：确认方法被调用
        self.logger.debug(f"开始后处理增强: {len(sentences)} 句, enhancement={solution_config.enhancement.value}")

        # V3.1.0: 极速模式（sensevoice_only）完全跳过 Whisper 补刀
        # 极速模式的设计目标是纯 SenseVoice 输出，不加载 Whisper 模型
        if solution_config.enhancement == EnhancementMode.OFF:
            self.logger.info("极速模式: 跳过所有 Whisper 补刀和仲裁")
            # 仍然执行 LLM 校对/翻译（如果配置了）
            if solution_config.proofread != ProofreadMode.OFF:
                self.logger.info("LLM 校对功能待实现")
            if solution_config.translate != TranslateMode.OFF:
                self.logger.info("LLM 翻译功能待实现")
            return sentences

        # 1. 收集需要 Whisper 补刀的句子（含强制补刀、常规补刀、垃圾核查）
        patch_queue = []
        # 阈值配置
        GARBAGE_CONFIDENCE_THRESHOLD = 0.4  # 低于此值触发 Whisper 仲裁

        for i, sentence in enumerate(sentences):
            should_patch = False
            is_critical = False
            is_trash_suspect = False  # 是否是"垃圾嫌疑"需要 Whisper 仲裁

            # === DEEP_LISTEN 快速路径 ===
            # DEEP_LISTEN 模式下所有句子都要补刀，直接入队，跳过后续冗余判断
            if solution_config.enhancement == EnhancementMode.DEEP_LISTEN:
                patch_queue.append({
                    "index": i,
                    "sentence": sentence,
                    "is_critical": False,
                    "is_trash_suspect": False
                })
                self.logger.debug(f"[DEEP_LISTEN] 句子{i} 直接加入双流队列")
                continue

            # 计算片段时长和清洗后文本长度
            duration = sentence.end - sentence.start
            text_clean = getattr(sentence, 'text_clean', None) or sentence.text
            clean_text = text_clean.strip() if text_clean else ""
            text_length = len(clean_text)

            # 【阶段四】强制关键补刀条件（无论用户设置如何，必须修）
            if is_critical_patch_needed(clean_text, duration, sentence.confidence):
                should_patch = True
                is_critical = True
                self.logger.warning(
                    f"触发强制补刀: '{clean_text}' "
                    f"(conf={sentence.confidence:.2f}, dur={duration:.2f}s)"
                )

            # 【新增】低置信度"二次安检"逻辑
            # 如果整句置信度极低（可能是幻觉/噪音），用 Whisper 仲裁
            elif sentence.confidence < GARBAGE_CONFIDENCE_THRESHOLD:
                should_patch = True
                is_trash_suspect = True
                self.logger.info(
                    f"触发低置信度核查: '{clean_text[:30]}...' "
                    f"(conf={sentence.confidence:.2f})"
                )

            # 【新增】字级强制补刀（独立检查，不受 enhancement 配置影响）
            # 条件1: 单字符实词且置信度 < 0.9
            # 条件2: 任意实词置信度极低 (< 0.35)，几乎肯定是识别错误
            if not should_patch and sentence.words:
                SINGLE_CHAR_CONF_THRESHOLD = 0.9  # 单字符词的高置信度要求
                CRITICAL_WORD_CONF_THRESHOLD = 0.35  # 极低置信度阈值（低于此值几乎肯定是错误）
                LOW_CONF_WORD_COUNT_THRESHOLD = 3  # 多个低置信度词的数量阈值
                LOW_WORD_CONF_THRESHOLD = 0.5  # 低置信度词的阈值

                low_conf_word_count = 0  # 统计低置信度词数量

                for w in sentence.words:
                    # 去除空白和 SentencePiece 边界标记 (U+2581)
                    word_text = w.word.strip().lstrip('\u2581').lower()
                    word_conf = w.confidence

                    # 跳过空字符、标点
                    if not word_text or (len(word_text) == 1 and not word_text.isalnum()):
                        continue

                    # 条件1: 单字符实词 + 置信度 < 0.9
                    if len(word_text) == 1 and word_text.isalnum() and word_conf < SINGLE_CHAR_CONF_THRESHOLD:
                        should_patch = True
                        is_critical = True
                        self.logger.warning(
                            f"触发字级单字符强制补刀: Sentence {i} 含单字符词 '{word_text}' "
                            f"(conf={word_conf:.2f})"
                        )
                        break

                    # 条件2: 任意实词置信度极低 (< 0.35)，几乎肯定是识别错误
                    if len(word_text) >= 2 and word_conf < CRITICAL_WORD_CONF_THRESHOLD:
                        should_patch = True
                        is_critical = True
                        self.logger.warning(
                            f"触发字级极低置信度强制补刀: Sentence {i} 含极低置信度词 '{word_text}' "
                            f"(conf={word_conf:.2f})"
                        )
                        break

                    # 统计低置信度词数量
                    if len(word_text) >= 2 and word_conf < LOW_WORD_CONF_THRESHOLD:
                        low_conf_word_count += 1

                # 条件3: 多个低置信度词（>= 3个），整句可能有问题
                if not should_patch and low_conf_word_count >= LOW_CONF_WORD_COUNT_THRESHOLD:
                    should_patch = True
                    is_critical = True
                    self.logger.warning(
                        f"触发字级多低置信度词强制补刀: Sentence {i} 含 {low_conf_word_count} 个低置信度词"
                    )

            # 常规补刀条件（遵循用户设置，仅 SMART_PATCH 模式）
            # 注: DEEP_LISTEN 模式已在循环开头通过快速路径处理
            if not should_patch and solution_config.enhancement == EnhancementMode.SMART_PATCH:
                # SMART_PATCH 模式: 仅低置信度句子触发补刀
                # 【阶段五】构建字级时间戳列表
                words_data = [{"word": w.word, "confidence": w.confidence} for w in sentence.words]

                # 增强版补刀判断：置信度、短片段、单字符、字级触发
                if needs_whisper_patch(
                    sentence.confidence,
                    duration=duration,
                    text_length=text_length,
                    words=words_data  # 【阶段五】传入字级数据
                ):
                    should_patch = True

            if should_patch:
                patch_queue.append({
                    "index": i,
                    "sentence": sentence,
                    "is_critical": is_critical,
                    "is_trash_suspect": is_trash_suspect
                })

        # 调试日志：输出补刀队列统计
        self.logger.debug(f"补刀队列构建完成: {len(patch_queue)} 个句子需要补刀")

        # 2. Whisper 补刀阶段（含仲裁判决）
        if patch_queue:
            progress_tracker.start_phase(ProcessPhase.WHISPER_PATCH, len(patch_queue), "Whisper 补刀中...")

            # === DEEP_LISTEN 模式: 使用 Whisper 缓冲池批量处理 ===
            if solution_config.enhancement == EnhancementMode.DEEP_LISTEN:
                await self._whisper_buffer_pool_process(
                    patch_queue=patch_queue,
                    audio_array=audio_array,
                    job=job,
                    subtitle_manager=subtitle_manager,
                    progress_tracker=progress_tracker
                )
            else:
                # === SMART_PATCH 模式: 逐句处理 ===
                for idx, item in enumerate(patch_queue):
                    # 检查任务是否已取消
                    if job.canceled:
                        self.logger.info(f"任务已取消，停止 Whisper 补刀: {job.job_id}")
                        break

                    sent_idx = item["index"]
                    sentence = item["sentence"]
                    is_trash_suspect = item["is_trash_suspect"]

                    # 执行 Whisper 补刀并获取仲裁结果
                    await self._whisper_text_patch_with_arbitration(
                        sentence=sentence,
                        sentence_index=sent_idx,
                        audio_array=audio_array,
                        job=job,
                        subtitle_manager=subtitle_manager,
                        is_trash_suspect=is_trash_suspect
                    )
                    progress_tracker.update_phase(ProcessPhase.WHISPER_PATCH, increment=1)

            progress_tracker.complete_phase(ProcessPhase.WHISPER_PATCH)

        # 3. 清理被标记为垃圾的句子
        deleted_count = subtitle_manager.remove_marked_sentences()
        if deleted_count > 0:
            self.logger.info(f"Whisper 仲裁：删除了 {deleted_count} 个确认为垃圾的片段")

        # 4. [可选] LLM 校对
        if solution_config.proofread != ProofreadMode.OFF:
            # TODO: 实现 LLM 校对
            self.logger.info("LLM 校对功能待实现")

        # 4. [可选] LLM 翻译
        if solution_config.translate != TranslateMode.OFF:
            # TODO: 实现 LLM 翻译
            self.logger.info("LLM 翻译功能待实现")

        return subtitle_manager.get_all_sentences()

    async def _process_video_sensevoice(self, job: 'JobState'):
        """
        SenseVoice 主处理流程（v2.1 概念澄清版）

        流程说明：
        1-4: 准备阶段（音频提取、VAD、频谱分诊、按需分离）
        5: 转录阶段（逐Chunk转录 + 熔断回溯）
        6-8: 后处理增强阶段（Whisper补刀、LLM校对/翻译）
        9: 输出阶段（生成字幕）
        """
        from app.services.streaming_subtitle import get_streaming_subtitle_manager, remove_streaming_subtitle_manager
        from app.services.progress_tracker import get_progress_tracker, remove_progress_tracker, ProcessPhase
        from app.services.solution_matrix import SolutionConfig, TranslateMode
        from app.services.demucs_service import get_demucs_service
        from app.services.sse_service import get_sse_manager
        from pathlib import Path

        # 检查任务是否已取消
        if job.canceled:
            self.logger.info(f"任务已取消，停止执行: {job.job_id}")
            return

        def push_signal_event(sse_manager, job_id: str, signal_code: str, message: str = ""):
            """推送信号事件（使用统一命名空间格式）"""
            sse_manager.broadcast_sync(
                f"job:{job_id}",
                f"signal.{signal_code}",
                {"signal": signal_code, "message": message}
            )

        # 获取方案配置 - v3.5: 优先使用新版配置，兼容旧版
        # 检查是否有 v3.5 新版配置
        if hasattr(job.settings, 'transcription') and hasattr(job.settings.transcription, 'transcription_profile'):
            # v3.5 新版配置: 从 job.settings 创建 SolutionConfig
            solution_config = SolutionConfig.from_job_settings(job.settings)
            self.logger.info(f"使用 v3.5 配置: profile={job.settings.transcription.transcription_profile}, "
                           f"enhancement={solution_config.enhancement.value}, "
                           f"proofread={solution_config.proofread.value}")
        else:
            # 旧版配置: 从 sensevoice.preset_id 创建
            preset_id = getattr(job.settings.sensevoice, 'preset_id', 'default')
            solution_config = SolutionConfig.from_preset(preset_id)
            self.logger.info(f"使用旧版预设: {preset_id}")

        # 初始化管理器
        subtitle_manager = get_streaming_subtitle_manager(job.job_id)
        progress_tracker = get_progress_tracker(job.job_id, solution_config.preset_id)

        try:
            # ========== 新架构：使用 PreprocessingPipeline ==========
            # 统一执行：音频提取 + VAD切分 + 频谱分诊 + 按需分离
            from app.pipelines.preprocessing_pipeline import PreprocessingPipeline
            from app.services.audio.vad_service import VADConfig
            import soundfile as sf

            self.logger.info("使用新架构 PreprocessingPipeline（Stage模式）")

            # V3.1.0: 根据语言选择 VAD 配置
            # 英语使用 Whisper，需要合并 VAD（避免幻觉）
            # 其他语言使用 SenseVoice，需要保留停顿信息以获得更好的断句
            language = getattr(job.settings, 'language', 'auto')
            is_english = language in {'en', 'english'}

            if is_english:
                # Whisper 模式：合并 VAD，避免幻觉
                vad_config = VADConfig(
                    merge_max_gap=1.0,
                    merge_max_duration=12.0,
                    smart_target_duration=12.0
                )
                self.logger.info(f"VAD配置: Whisper模式（合并），language={language}")
            else:
                # SenseVoice 模式：保留停顿信息，获得更自然的断句
                vad_config = VADConfig(
                    merge_max_gap=0.3,           # 只合并极短停顿（保留自然停顿）
                    merge_max_duration=8.0,      # 更短的 chunk（避免强制切分）
                    smart_target_duration=8.0    # 软上限降低
                )
                self.logger.info(f"VAD配置: SenseVoice模式（保留停顿），language={language}")

            # 创建预处理流水线（传入 VAD 配置）
            preprocessing_pipeline = PreprocessingPipeline(
                config=job.settings.preprocessing,
                vad_config=vad_config,
                logger=self.logger
            )

            # 执行预处理（包含：音频提取、VAD、频谱分诊、按需分离）
            progress_tracker.start_phase(ProcessPhase.EXTRACT, 1, "预处理流水线...")
            audio_chunks = await preprocessing_pipeline.process(
                video_path=job.input_path,
                job_state=job
            )
            progress_tracker.complete_phase(ProcessPhase.EXTRACT)

            # 获取预处理统计信息
            stats = preprocessing_pipeline.get_statistics(audio_chunks)
            self.logger.info(
                f"PreprocessingPipeline 完成: "
                f"总chunk数={stats['total_chunks']}, "
                f"需要分离={stats['need_separation']}, "
                f"已分离={stats['separated']}, "
                f"分离比例={stats['separation_ratio']:.2%}"
            )

            # 保存音频文件供波形图使用（从第一个chunk获取采样率）
            if audio_chunks:
                sr = audio_chunks[0].sample_rate
                # 重新加载完整音频用于保存（PreprocessingPipeline内部已处理）
                import librosa
                audio_array, _ = librosa.load(job.input_path, sr=sr, mono=True)
                audio_path = Path(job.dir) / "audio.wav"
                sf.write(str(audio_path), audio_array, sr)
                self.logger.info(f"音频文件已保存: {audio_path}")

            # 转换 AudioChunk 到 ChunkProcessState（兼容现有转录流程）
            chunk_states = self._convert_audio_chunks_to_states(audio_chunks)

            # 获取 demucs_service 引用（用于熔断回溯）
            demucs_service = get_demucs_service()

            # 5. 逐Chunk转录 + 熔断回溯（转录层核心）
            progress_tracker.start_phase(ProcessPhase.SENSEVOICE, len(chunk_states), "SenseVoice 转录...")
            all_sentences = []

            for chunk_state in chunk_states:
                # 检查任务是否已取消
                if job.canceled:
                    self.logger.info(f"任务已取消，停止转录: {job.job_id}")
                    break

                # 单个 Chunk 转录（含熔断回溯循环）
                sentences = await self._transcribe_chunk_with_fusing(
                    chunk_state=chunk_state,
                    job=job,
                    subtitle_manager=subtitle_manager,
                    demucs_service=demucs_service
                )
                all_sentences.extend(sentences)
                progress_tracker.update_phase(ProcessPhase.SENSEVOICE, increment=1)

            progress_tracker.complete_phase(ProcessPhase.SENSEVOICE)

            # 保存原始转录数据（未分句的完整数据）
            self._save_raw_transcription(job, all_sentences)

            # 6. 后处理增强（Whisper补刀、LLM校对/翻译）
            final_results = await self._post_process_enhancement(
                all_sentences, audio_array, job, subtitle_manager, solution_config
            )

            # 7. 生成字幕
            progress_tracker.start_phase(ProcessPhase.SRT, 1, "生成字幕...")
            output_path = str(Path(job.dir) / f"{job.job_id}.srt")
            self._generate_subtitle_from_sentences(
                final_results,
                output_path,
                include_translation=(solution_config.translate != TranslateMode.OFF)
            )
            progress_tracker.complete_phase(ProcessPhase.SRT)

            # 8. 完成
            job.status = 'completed'
            push_signal_event(get_sse_manager(), job.job_id, "job_complete", "处理完成")

        except Exception as e:
            self.logger.error(f"SenseVoice 处理失败: {e}", exc_info=True)
            job.status = 'failed'
            job.error = str(e)
            push_signal_event(get_sse_manager(), job.job_id, "job_failed", str(e))
            raise

        finally:
            # 清理资源
            remove_streaming_subtitle_manager(job.job_id)
            remove_progress_tracker(job.job_id)


# 单例处理器
_service_instance: Optional[TranscriptionService] = None


def get_transcription_service(root: str) -> TranscriptionService:
    """获取转录服务实例（单例模式）"""
    global _service_instance
    if _service_instance is None:
        _service_instance = TranscriptionService(root)
    return _service_instance
