"""
PipelineOrchestrator - 流水线编排器 V3.2.0+dev.20260125.07

统一管理预处理、转录、收尾流程，从 TranscriptionService 解耦。

职责：
1. 流水线完整执行（run_pipeline）
2. 预处理阶段编排（run_preprocessing）
3. Profile 选择逻辑（resolve_profiles）
4. 恢复上下文构建（build_resume_context）

V3.2.0+dev.20260125.07: 支持运行时依赖注入（方案 B）
- 核心依赖：构造函数注入（job_lifecycle, sse_manager, hardware_profile_provider）
- 运行时依赖：run_pipeline() 参数（cancellation_token, progress_emitter 等）
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from app.core.config import config
from app.models.job_models import JobState
from app.schemas.profile_config import ProfileConfig
from app.schemas.resume_context import ResumeContext
from app.services.checkpoint import RuntimeCheckpointService

if TYPE_CHECKING:
    from app.services.hardware_profile_service import HardwareProfileProvider
    from app.services.job_lifecycle_service import JobLifecycleService
    from app.services.sse_service import SSEManager
    from app.services.job.checkpoint_manager import CheckpointManagerV37
    from app.services.progress_emitter import ProgressEventEmitter
    from app.services.progress_tracker import ProgressTracker
    from app.services.streaming_subtitle import StreamingSubtitleManager
    from app.utils.cancellation_token import CancellationToken

# V3.2.0+dev.20260125.11: 导入 SubtitleOutputService 用于 SRT 生成
from app.services.subtitle_output_service import get_subtitle_output_service
# V3.2.0+dev.20260125.11: 使用 SSEPublisher 统一推送进度事件
from app.services.sse_publisher import get_sse_publisher


class PipelineOrchestrator:
    """流水线编排器 - 统一管理预处理、转录、收尾流程"""

    def __init__(
        self,
        job_lifecycle: "JobLifecycleService",
        sse_manager: "SSEManager",
        hardware_profile_provider: "HardwareProfileProvider",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.job_lifecycle = job_lifecycle
        self.sse_manager = sse_manager
        self.hardware_profile_provider = hardware_profile_provider
        self.logger = logger or logging.getLogger(__name__)
        self._preprocess_vad_intervals: Optional[List[Tuple[float, float]]] = None

    async def run_pipeline(
        self,
        job: JobState,
        *,
        cancellation_token: Optional["CancellationToken"] = None,
        progress_emitter: Optional["ProgressEventEmitter"] = None,
        progress_tracker: Optional["ProgressTracker"] = None,
        subtitle_manager: Optional["StreamingSubtitleManager"] = None,
        checkpoint_manager: Optional["CheckpointManagerV37"] = None,
        job_dir: Optional[Path] = None,
        full_audio_array: Optional[np.ndarray] = None,
        full_audio_sr: int = 16000,
    ) -> None:
        """
        完整流水线执行入口

        V3.2.0+dev.20260125.07: 支持运行时依赖注入

        Args:
            job: 任务状态对象
            cancellation_token: 取消令牌（可选）
            progress_emitter: 进度发射器（可选，用于细粒度进度推送）
            progress_tracker: 进度追踪器（可选）
            subtitle_manager: 字幕管理器（可选，外部传入避免重复创建）
            checkpoint_manager: 检查点管理器（可选）
            job_dir: 任务目录（可选，默认从 job.dir 获取）
            full_audio_array: 完整音频数组（可选，用于 Audio Overlap）
            full_audio_sr: 音频采样率（默认 16000）
        """
        try:
            from app.pipelines.async_dual_pipeline import AsyncDualPipeline
            from app.services.streaming_subtitle import get_streaming_subtitle_manager

            # V3.2.0+dev.20260125.07: 使用传入的 job_dir，否则从 job.dir 获取
            _job_dir = job_dir if job_dir else Path(job.dir)
            input_path = _job_dir / job.filename

            # 构建恢复上下文和 Profile 配置
            # V3.2.0+dev.20260125.07: 支持外部传入 checkpoint_manager
            resume_context = self.build_resume_context(job, checkpoint_manager, _job_dir)
            profile_config = self.resolve_profiles(job)

            # 阶段 1: 预处理
            if progress_emitter:
                progress_emitter.update_preprocess(0, "extract", "音频前处理...")
            self._update_progress(job, "audio_processing", 0, "音频处理中...")
            self.logger.info("使用新架构预处理流水线（Stage模式）")
            chunks = await self.run_preprocessing(
                job,
                input_path,
                profile_config,
                resume_context,
                cancellation_token=cancellation_token,
                progress_emitter=progress_emitter,
            )
            vad_intervals = self._preprocess_vad_intervals

            job.total = len(chunks)
            self.logger.info(f"音频处理完成: {len(chunks)} 个 Chunk")

            # V3.2.0+dev.20260125.07: 预处理完成
            if progress_emitter:
                progress_emitter.update_preprocess(100, "completed", "预处理完成")

            # 阶段 2: 转录
            if progress_emitter:
                progress_emitter.update_fast(0, len(chunks), force_push=True)
            self._update_progress(job, "transcription", 0, "转录中...")
            self.logger.info(f"转录模式: {profile_config.transcription_profile}")

            draft_name = (
                profile_config.draft_engine.get_engine_name()
                if profile_config.draft_engine
                else "none"
            )
            patch_name = (
                profile_config.patch_engine.get_engine_name()
                if profile_config.patch_engine
                else "none"
            )
            self.logger.info(
                "新 ASR 引擎已启用: draft=%s, patch=%s, profile=%s",
                draft_name,
                patch_name,
                profile_config.transcription_profile,
            )

            # V3.2.0+dev.20260125.07: 使用传入的 subtitle_manager，否则获取
            _subtitle_manager = subtitle_manager if subtitle_manager else get_streaming_subtitle_manager(job.job_id)

            # 恢复字幕状态
            if resume_context.is_resuming and resume_context.sentences_snapshot:
                subtitle_checkpoint_data = {
                    "sentences_snapshot": resume_context.sentences_snapshot,
                    "sentence_count": resume_context.sentence_count,
                    "chunk_sentences_map": resume_context.chunk_sentences_map,
                }
                if _subtitle_manager.restore_from_checkpoint(subtitle_checkpoint_data):
                    self.logger.info(
                        "字幕状态已恢复: %s 个句子",
                        len(resume_context.sentences_snapshot),
                    )
                    _subtitle_manager.push_restored_subtitles_to_frontend()
                else:
                    self.logger.warning("字幕恢复失败，将从头生成字幕")

            # 创建转录流水线
            from app.services.punctuation.debug_utils import is_debug_punctuation_enabled
            debug_config = getattr(job.settings, "debug", None)
            debug_punctuation = is_debug_punctuation_enabled(
                env_value=os.getenv("DEBUG_PUNCTUATION"),
                config_value=bool(getattr(debug_config, "punctuation_output", False)),
            )
            preprocessing = getattr(job.settings, "preprocessing", None)
            is_enable_speaker_detection = bool(
                getattr(preprocessing, "is_enable_speaker_detection", True)
            )
            is_enable_speaker_guided_split = bool(
                getattr(preprocessing, "is_enable_speaker_guided_split", True)
            )
            speaker_count = max(0, int(getattr(preprocessing, "speaker_count", 0) or 0))
            speaker_min_count = max(
                0,
                int(getattr(preprocessing, "speaker_min_count", 0) or 0),
            )
            speaker_max_count = max(
                0,
                int(getattr(preprocessing, "speaker_max_count", 0) or 0),
            )
            transcription_pipeline = AsyncDualPipeline(
                job_id=job.job_id,
                transcription_profile=profile_config.transcription_profile,
                draft_engine=profile_config.draft_engine,
                patch_engine=profile_config.patch_engine,
                patching_threshold=profile_config.patching_threshold,
                debug_punctuation=debug_punctuation,
                is_enable_speaker_detection=is_enable_speaker_detection,
                is_enable_speaker_guided_split=is_enable_speaker_guided_split,
                speaker_count=speaker_count,
                speaker_min_count=speaker_min_count,
                speaker_max_count=speaker_max_count,
                logger=self.logger,
                cancellation_token=cancellation_token,
                progress_emitter=progress_emitter,
            )

            # 恢复 Whisper 上下文
            if resume_context.previous_whisper_text:
                transcription_pipeline.restore_prompt_cache(
                    resume_context.previous_whisper_text
                )
                self.logger.info(
                    "[v3.1.0] 已恢复 Whisper 上下文: %s 字符",
                    len(resume_context.previous_whisper_text),
                )

            # 执行转录
            contexts = await transcription_pipeline.run(
                audio_chunks=chunks,
                full_audio_array=full_audio_array,
                full_audio_sr=full_audio_sr,
                job_dir=_job_dir,
                vad_intervals=vad_intervals,
                processed_indices=(
                    resume_context.safe_processed_indices
                    if resume_context.is_resuming
                    else None
                ),
                initial_slow_processed_indices=(
                    resume_context.slow_processed_indices
                    if resume_context.is_resuming
                    else None
                ),
                initial_finalized_indices=(
                    resume_context.finalized_indices
                    if resume_context.is_resuming
                    else None
                ),
            )

            # V3.2.0+dev.20260125.07: 转录完成，推送细粒度进度
            total_chunks = len(chunks)
            if progress_emitter:
                progress_emitter.update_fast(total_chunks, total_chunks, force_push=True)
                progress_emitter.update_slow(total_chunks, total_chunks, force_push=True)
                progress_emitter.update_align(total_chunks, total_chunks, force_push=True)

            # 收集最终句子
            pipeline_sentences = []
            for ctx in contexts:
                if hasattr(ctx, "sv_result") and ctx.sv_result:
                    chunk_indices = _subtitle_manager.chunk_sentences.get(
                        ctx.chunk_index, []
                    )
                    for idx in chunk_indices:
                        if idx in _subtitle_manager.sentences:
                            pipeline_sentences.append(_subtitle_manager.sentences[idx])

            final_sentences = pipeline_sentences
            subtitle_snapshot = _subtitle_manager.get_all_sentences()
            if subtitle_snapshot:
                final_sentences = subtitle_snapshot

            self.logger.info(f"转录完成: {len(final_sentences)} 个句子")

            # 阶段 3: 生成 SRT 文件
            # V3.2.0+dev.20260125.11: 使用 SubtitleOutputService 替代内部方法
            self._update_progress(job, "finalize", 0, "生成字幕文件...")
            srt_path = _job_dir / f"{Path(job.filename).stem}.srt"
            subtitle_output = get_subtitle_output_service()
            from app.services.user_config_service import get_user_config_service
            offset = get_user_config_service().resolve_subtitle_time_offset(
                getattr(job, "subtitle_time_offset", None)
            )
            segments = subtitle_output.build_segments(
                final_sentences,
                apply_offset=True,
                offset_override=offset
            )
            subtitle_output.write_srt(segments, srt_path)

            job.srt_path = str(srt_path)
            job.status = "completed"
            job.message = "转录完成"
            job.progress = 100

            # V3.2.0+dev.20260125.07: 使用 progress_emitter 标记完成
            if progress_emitter:
                progress_emitter.complete("处理完成")

            self.logger.info(f"任务完成: {job.job_id}")

        except Exception as exc:
            self.logger.error(f"Pipeline 执行失败: {exc}", exc_info=True)
            job.status = "failed"
            job.error = str(exc)
            job.message = f"失败: {str(exc)}"
            raise

    async def run_preprocessing(
        self,
        job: JobState,
        input_path: Path,
        profile_config: ProfileConfig,
        resume_context: Optional[ResumeContext] = None,
        *,
        cancellation_token: Optional["CancellationToken"] = None,
        progress_emitter: Optional["ProgressEventEmitter"] = None,
    ) -> List:
        """预处理阶段

        V3.2.0+dev.20260125.07: 支持 cancellation_token 和 progress_emitter
        """
        self._preprocess_vad_intervals = None
        from app.pipelines.preprocessing_pipeline import PreprocessingPipeline
        from app.services.audio.chunk_engine import ChunkEngine

        if profile_config.vad_profile == "whisper":
            vad_label = "Whisper"
        else:
            vad_label = "SenseVoice"
        self.logger.info("使用 %s VAD 配置", vad_label)

        chunk_engine = ChunkEngine(logger=self.logger)

        # V3.2.0+dev.20260131: 创建诊断服务
        from app.services.segmentation_diagnostic_service import get_diagnostic_service
        diagnostic_service = get_diagnostic_service(
            job_id=job.job_id,
            job_dir=Path(job.dir) if job.dir else None,
            enabled=None  # 默认启用，可通过环境变量控制
        )

        preprocessing_pipeline = PreprocessingPipeline(
            config=job.settings.preprocessing,
            chunk_engine=chunk_engine,
            vad_config=profile_config.vad_config,
            logger=self.logger,
            cancellation_token=cancellation_token,
            progress_emitter=progress_emitter,
            diagnostic_service=diagnostic_service,  # V3.2.0+dev.20260131: 传递诊断服务
        )

        checkpoint_data = (
            resume_context.checkpoint if resume_context and resume_context.checkpoint else None
        )
        chunks = await preprocessing_pipeline.process(
            video_path=str(input_path),
            job_state=job,
            job_dir=Path(job.dir),
            checkpoint=checkpoint_data,
        )

        stats = preprocessing_pipeline.get_statistics(chunks)
        self.logger.info(
            "预处理统计: "
            "总chunk数=%s, "
            "需要分离=%s, "
            "已分离=%s, "
            "分离比例=%.2f%%, "
            "熔断重试总次数=%s, "
            "最大重试次数=%s",
            stats["total_chunks"],
            stats["need_separation"],
            stats["separated"],
            stats["separation_ratio"] * 100,
            stats["fuse_retry_total"],
            stats["fuse_retry_max"],
        )

        # V3.2.0+dev.20260131: 导出诊断文件
        if diagnostic_service:
            try:
                diagnostic_file = diagnostic_service.export_to_file()
                if diagnostic_file:
                    self.logger.info(f"断句诊断文件已保存: {diagnostic_file}")
            except Exception as e:
                self.logger.warning(f"导出诊断文件失败: {e}", exc_info=True)

        self._preprocess_vad_intervals = preprocessing_pipeline.get_vad_intervals()
        return chunks

    def resolve_profiles(self, job: JobState) -> ProfileConfig:
        """Profile 选择逻辑"""
        from app.core.asr.engine_resolver import EngineResolver
        from app.core.thresholds import ThresholdConfig
        from app.services.runtime_param_resolver import build_vad_config_for_profile

        transcription = getattr(job.settings, "transcription", None)
        transcription_profile = (
            transcription.transcription_profile
            if transcription
            else "sensevoice_only"
        )

        # VAD Profile 选择：英语用 Whisper，其他用 SenseVoice
        language = getattr(job.settings, "language", "auto")
        is_english = language in {"en", "english"}
        vad_profile = "whisper" if is_english else "sensevoice"
        vad_config = build_vad_config_for_profile(vad_profile)

        # 构建 ASR 引擎
        optimization_config = None
        try:
            hardware_info = self.hardware_profile_provider.get_hardware_info(
                is_force_refresh=False
            )
            optimization_config = self.hardware_profile_provider.get_optimization_config(
                hardware_info
            )
        except Exception as exc:
            self.logger.warning("获取硬件优化配置失败: %s", exc)

        resolver = EngineResolver(
            hardware_profile_provider=self.hardware_profile_provider,
            logger=self.logger,
        )
        draft_engine, patch_engine = resolver.resolve_profile_engines(
            transcription_profile,
            job=job,
            optimization_config=optimization_config,
        )

        # 补刀阈值
        patching_threshold_value = getattr(
            transcription,
            "patching_threshold",
            0.60,
        )
        patching_threshold = ThresholdConfig(
            whisper_patch_trigger_confidence=patching_threshold_value
        )

        return ProfileConfig(
            transcription_profile=transcription_profile,
            vad_profile=vad_profile,
            vad_config=vad_config,
            draft_engine=draft_engine,
            patch_engine=patch_engine,
            patching_threshold=patching_threshold,
        )

    def build_resume_context(
        self,
        job: JobState,
        checkpoint_manager: Optional["CheckpointManagerV37"] = None,
        job_dir: Optional[Path] = None,
    ) -> ResumeContext:
        """构建恢复上下文

        V3.2.0+dev.20260125.07: 支持外部传入 checkpoint_manager
        """
        from app.services.job.checkpoint_manager import CheckpointManagerV37

        _job_dir = job_dir if job_dir else Path(job.dir)
        runtime_state_service = RuntimeCheckpointService(job_dir=_job_dir)

        # Phase 1: runtime_state.db 优先作为恢复入口。
        if runtime_state_service.has_runtime_state():
            snapshot = runtime_state_service.load_snapshot()
            preprocess_commit = snapshot.last_unit_commits.get("preprocess")
            runtime_checkpoint = {
                "runtime_state": {
                    "source": "runtime_state.db",
                    "last_unit_commits": snapshot.last_unit_commits,
                },
                "preprocessing": {
                    # 仅当预处理单元已提交，才允许跳过到已完成状态。
                    "vad_completed": preprocess_commit in {
                        "vad_chunk",
                        "triage_chunk",
                        "separation_chunk",
                        "langid_chunk",
                        "speaker_chunk",
                    },
                    "spectral_triage_completed": preprocess_commit in {
                        "triage_chunk",
                        "separation_chunk",
                        "langid_chunk",
                        "speaker_chunk",
                    },
                    "separation_completed": preprocess_commit in {
                        "separation_chunk",
                        "langid_chunk",
                        "speaker_chunk",
                    },
                },
            }
            return ResumeContext.from_checkpoint(runtime_checkpoint)

        _checkpoint_manager = checkpoint_manager if checkpoint_manager else CheckpointManagerV37(_job_dir, logger=self.logger)
        checkpoint = _checkpoint_manager.load_checkpoint()
        if not checkpoint:
            return ResumeContext(checkpoint=None)

        checkpoint_dict = (
            checkpoint.to_dict() if hasattr(checkpoint, "to_dict") else checkpoint
        )
        return ResumeContext.from_checkpoint(checkpoint_dict)

    def _update_progress(
        self,
        job: JobState,
        phase: str,
        phase_ratio: float,
        message: str = "",
    ) -> None:
        """更新进度"""
        job.phase = phase
        job.phase_percent = round(max(0.0, min(1.0, phase_ratio)) * 100, 1)

        phase_weights = config.PHASE_WEIGHTS
        total_weight = config.TOTAL_WEIGHT

        done_weight = 0
        for phase_name, weight in phase_weights.items():
            if phase_name == phase:
                break
            done_weight += weight

        current_weight = phase_weights.get(phase, 0) * max(
            0.0, min(1.0, phase_ratio)
        )
        job.progress = round((done_weight + current_weight) / total_weight * 100, 1)

        if message:
            job.message = message

        self._push_sse_progress(job)

    def _push_sse_progress(self, job: JobState) -> None:
        """推送 SSE 进度事件"""
        if not self.sse_manager:
            return
        try:
            publisher = get_sse_publisher(job.job_id, self.sse_manager, job=job)
            publisher.publish_job_progress(job)
        except Exception as exc:
            self.logger.debug("SSE推送失败: %s", exc)

    # V3.2.0+dev.20260125.11: 移除 _generate_srt_from_sentences 和 _format_srt_timestamp
    # 已迁移至 SubtitleOutputService
