"""
PreprocessingPipeline - 预处理流水线（新架构）

负责统一管理所有预处理步骤：
1. 音频提取 + VAD切分
2. 音频预检（可选）
3. 人声分离（可选，支持全局/按需模式）

与旧的 AudioProcessingPipeline（已归档）的区别：
- 旧版：整轨分离 + VAD切分
- 新版：VAD切分 + 音频预检 + 按需分离（Stage模式）

v3.1.0 更新：
- 集成 CancellationToken 支持暂停/取消
- 支持断点续传检查点保存
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from pathlib import Path

from app.services.audio.chunk_engine import ChunkEngine, AudioChunk
from app.services.audio.vad_service import VADConfig
from app.pipelines.stages.spectral_triage_stage import SpectralTriageStage
from app.pipelines.stages.separation_stage import SeparationStage
from app.models.job_models import PreprocessingConfig, JobState
from app.models.preprocess_artifacts import PreprocessArtifacts
from app.services.demucs_service import get_demucs_service
from app.services.preprocess_cache_service import PreprocessCacheService
from app.services.checkpoint import PauseBarrier, RuntimeCheckpointService
from app.services.asr_risk_guard_service import ASRRiskGuardService, ASRRiskGuardResult

# v3.1.0: 导入取消令牌
if TYPE_CHECKING:
    from app.utils.cancellation_token import CancellationToken
    from app.services.progress_emitter import ProgressEventEmitter


class PreprocessingPipeline:
    """
    预处理流水线 - 统一管理所有预处理步骤

    采用 Stage 模式，支持灵活的预处理流程配置。

    v3.1.0: 支持 CancellationToken 实现暂停/取消/断点续传
    """

    PREPROCESS_STAGE_WEIGHTS = {
        "vad": 7.0,
        "spectrum_analysis": 6.0,
        "demucs": 7.0
    }
    PREPROCESS_TOTAL_WEIGHT = 20.0

    def __init__(
        self,
        config: PreprocessingConfig,
        chunk_engine: Optional[ChunkEngine] = None,
        vad_config: Optional[VADConfig] = None,  # V3.1.0: 新增 VAD 配置参数
        logger: Optional[logging.Logger] = None,
        cancellation_token: Optional["CancellationToken"] = None,  # v3.1.0: 新增
        progress_emitter: Optional["ProgressEventEmitter"] = None,
        diagnostic_service: Optional[Any] = None  # V3.2.0+dev.20260131: 诊断服务
    ):
        """
        初始化预处理流水线

        Args:
            config: 预处理配置
            chunk_engine: 音频切分引擎（可选）
            vad_config: VAD 配置（可选，V3.1.0 新增，用于语言特定的 VAD 策略）
            logger: 日志记录器（可选）
            cancellation_token: 取消令牌（可选，v3.1.0）
            progress_emitter: 进度发射器（可选）
            diagnostic_service: 诊断服务实例（可选，V3.2.0+dev.20260131）
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.cancellation_token = cancellation_token  # v3.1.0
        self.progress_emitter = progress_emitter
        self.diagnostic_service = diagnostic_service  # V3.2.0+dev.20260131
        self.is_asr_risk_guard_enabled = bool(getattr(config, "enable_asr_risk_guard", True))
        self.asr_risk_guard_service = ASRRiskGuardService(logger=self.logger)
        self._vad_intervals: Optional[List[Tuple[float, float]]] = None
        self._last_preprocess_artifacts: Optional[PreprocessArtifacts] = None
        if vad_config is None:
            from app.services.runtime_param_resolver import build_vad_config

            self.vad_config = build_vad_config()
        else:
            self.vad_config = vad_config  # V3.1.0: 保存 VAD 配置

        # 初始化 ChunkEngine（用于音频提取和VAD切分）
        self.chunk_engine = chunk_engine or ChunkEngine(logger=self.logger)

        # 初始化音频预检阶段（如果启用）
        if config.enable_spectral_triage:
            self.spectral_triage_stage = SpectralTriageStage(
                threshold=config.spectrum_threshold,
                logger=self.logger,
                cancellation_token=cancellation_token,  # v3.1.0: 传递令牌
                use_dnsmos_triage=config.use_dnsmos_triage,
                use_smart_probe=config.use_smart_probe,
            )
            self.logger.info(
                "音频预检已启用: threshold=%s, smart_probe=%s",
                config.spectrum_threshold,
                config.use_smart_probe,
            )
        else:
            self.spectral_triage_stage = None
            self.logger.info("音频预检已禁用")

        # 初始化人声分离阶段（如果启用）
        enable_demucs = config.demucs_strategy != "off"
        if enable_demucs:
            demucs_service = get_demucs_service()
            try:
                demucs_service.set_model(config.demucs_model)
            except Exception as e:
                self.logger.warning(
                    f"设置 Demucs 模型 {config.demucs_model} 失败，使用默认模型: {e}"
                )
            self.separation_stage = SeparationStage(
                mode=config.separation_mode,
                demucs_service=demucs_service,
                logger=self.logger,
                cancellation_token=cancellation_token  # v3.1.0: 传递令牌
            )
            self.logger.info(
                f"人声分离已启用: mode={config.separation_mode}, "
                f"model={config.demucs_model}"
            )
        else:
            self.separation_stage = None
            self.logger.info("人声分离已禁用")

    def _report_preprocess_progress(self, stage: str, ratio: float, message: str = "") -> None:
        if not self.progress_emitter:
            return
        if stage not in self.PREPROCESS_STAGE_WEIGHTS:
            return
        ratio = max(0.0, min(1.0, ratio))
        completed = 0.0
        for key in ("vad", "spectrum_analysis", "demucs"):
            weight = self.PREPROCESS_STAGE_WEIGHTS[key]
            if key == stage:
                completed += weight * ratio
                break
            completed += weight
        percent = (completed / self.PREPROCESS_TOTAL_WEIGHT) * 100
        self.progress_emitter.update_preprocess(percent, stage, message)

    def _run_asr_risk_guard(
        self,
        chunks: List[AudioChunk],
        job_dir: Optional[Path],
    ) -> Optional[ASRRiskGuardResult]:
        """执行 ASR 风险前置检测，失败时回退原流程。"""
        if not self.is_asr_risk_guard_enabled:
            return None
        if not chunks:
            return None
        try:
            result = self.asr_risk_guard_service.evaluate_chunks(chunks)
            if job_dir:
                try:
                    self.asr_risk_guard_service.save_report(job_dir, result)
                except OSError as exc:
                    self.logger.warning("保存 ASR 风险检测报告失败: %s", exc)
            return result
        except (ValueError, RuntimeError) as exc:
            self.logger.warning("ASR 风险检测失败，回退原流程: %s", exc)
            return None

    async def process(
        self,
        video_path: str,
        job_state: Optional[JobState] = None,
        job_dir: Optional[Path] = None,  # v3.1.0: 用于保存检查点
        checkpoint: Optional[dict] = None  # V3.1.0: 用于跳过已完成的步骤
    ) -> List[AudioChunk]:
        """
        执行完整的预处理流程

        流程：
        1. 音频提取 + VAD切分 -> List[AudioChunk]
        2. 音频预检（可选）-> 标记 needs_separation
        3. 人声分离（可选）-> 分离标记的chunk

        Args:
            video_path: 视频/音频文件路径
            job_state: 任务状态（可选，用于进度回调）
            job_dir: 任务目录（可选，v3.1.0 用于保存检查点）
            checkpoint: 检查点数据（可选，V3.1.0 用于跳过已完成的步骤）

        Returns:
            List[AudioChunk]: 预处理完成的 Chunk 列表
        """
        self.logger.info(f"开始预处理流程: {video_path}")
        token = self.cancellation_token  # v3.1.0: 简化引用
        job_id = job_state.job_id if job_state else None
        channel_identifier = (
            str(getattr(job_state, "project_id", "") or "").strip()
            if job_state is not None
            else ""
        ) or job_id
        self._vad_intervals = None
        runtime_checkpoint_service: Optional[RuntimeCheckpointService] = None
        pause_barrier: Optional[PauseBarrier] = None
        if job_dir:
            runtime_checkpoint_service = RuntimeCheckpointService(job_dir=job_dir)
            pause_barrier = PauseBarrier(
                token=token,
                runtime_checkpoint_service=runtime_checkpoint_service,
                logger=self.logger,
            )
        sse_manager = None
        if job_id:
            from app.services.sse_service import get_sse_manager

            sse_manager = get_sse_manager()

        def broadcast_preprocess_event(event: str, payload: Dict[str, Any]) -> None:
            if not sse_manager or not channel_identifier:
                return
            sse_manager.broadcast_sync(f"project:{channel_identifier}", event, payload)

        cache_service: Optional[PreprocessCacheService] = None
        if job_dir:
            cache_service = PreprocessCacheService(job_dir=job_dir, logger=self.logger)
            try:
                cache_service.ensure_dirs()
            except Exception as e:
                self.logger.warning("[V3.2.0+dev.20260122.03] 初始化预处理缓存失败: %s", e)
                cache_service = None
            if cache_service:
                try:
                    cache_service.run_gc(job_dir.parent, self.config, exclude_job_id=job_dir.name)
                except Exception as e:
                    self.logger.warning("[V3.2.0+dev.20260122.03] 预处理缓存 GC 失败: %s", e)
                try:
                    cache_service.maybe_warn_cache_pressure(job_dir.name, self.config)
                except Exception as e:
                    self.logger.warning("[V3.2.0+dev.20260122.03] 预处理缓存告警失败: %s", e)

        # V3.2.0+dev.20260123.03: 优先尝试分离缓存（统一输出层优先）
        runtime_separation_mode = self.config.separation_mode if self.separation_stage else "off"
        chunks = None
        skip_vad = False
        skip_triage = False
        skip_triage_reason = ""
        skip_separation = False
        used_vad_cache = False
        used_separation_cache = False
        is_force_global_by_asr_risk = False
        if cache_service:
            separation_chunks = cache_service.load_separation_chunks(expected_mode=runtime_separation_mode)
            if separation_chunks:
                chunks = separation_chunks
                skip_vad = True
                skip_triage = True
                skip_triage_reason = "separation_cache"
                skip_separation = True
                used_separation_cache = True
                self.logger.info("[V3.2.0+dev.20260123.03] 分离缓存命中，跳过 VAD/分诊/分离")
                self._report_preprocess_progress("vad", 1.0, "分离缓存命中")
                self._report_preprocess_progress("spectrum_analysis", 1.0, "分离缓存命中")
                self._report_preprocess_progress("demucs", 1.0, "分离缓存命中")

        # V3.2.0+dev.20260122.03: 优先尝试预处理缓存
        if cache_service and not skip_vad:
            cached_chunks = cache_service.load_vad_chunks()
            if cached_chunks:
                chunks = cached_chunks
                skip_vad = True
                used_vad_cache = True
                self.logger.info("[V3.2.0+dev.20260122.03] VAD 缓存命中，跳过 VAD")
                self._report_preprocess_progress("vad", 1.0, "VAD 缓存命中")

        # V3.1.0: 检查是否可以从 checkpoint 恢复 chunks
        if not skip_vad and checkpoint and isinstance(checkpoint, dict):
            preprocessing = checkpoint.get("preprocessing", {})
            if isinstance(preprocessing, dict):
                chunks_metadata = preprocessing.get("chunks_metadata", [])
                if chunks_metadata and preprocessing.get("vad_completed", False):
                    chunks = await self._restore_chunks_from_metadata(
                        video_path, chunks_metadata
                    )
                    if chunks:
                        skip_vad = True
                        self.logger.info(f"[V3.1.0] 从 checkpoint 恢复 {len(chunks)} 个chunk，跳过 VAD")
                        self._report_preprocess_progress("vad", 1.0, "VAD 检查点恢复")
                        if cache_service:
                            try:
                                cache_service.save_vad_chunks(chunks)
                            except Exception as e:
                                self.logger.warning("[V3.2.0+dev.20260122.03] 保存 VAD 缓存失败: %s", e)

        if not skip_vad:
            # V3.1.0: Stage 1 拆分为两个原子区域
            # Stage 1a: 音频提取（FFmpeg，耗时 1-5 秒）
            self.logger.info("Stage 1a: 音频提取")

            if token:
                token.enter_atomic_region("ffmpeg_extract")

            try:
                # 音频提取（包含 FFmpeg 转码和降采样）
                chunks = await self._extract_and_vad(video_path, job_state)
                self.logger.info(f"音频提取和 VAD 切分完成: {len(chunks)} 个chunk")
            finally:
                if token:
                    has_pending = token.exit_atomic_region()
                    if has_pending:
                        self.logger.info("[V3.1.0] FFmpeg/VAD 完成后检测到待处理请求")

            # V3.1.0: Stage 1 检查点（音频提取 + VAD 完成，包含 chunks_metadata）
            if token and job_dir and chunks:
                if pause_barrier:
                    pause_barrier.on_unit_start(
                        stage="preprocess",
                        unit_id="vad_chunk",
                        payload={"total_chunks": len(chunks)},
                    )
                # 保存 chunks 的元数据（不包含音频数据，用于恢复时重建 chunks）
                # 注意：AudioChunk 使用 start/end 而非 start_time/end_time
                chunks_metadata = [
                    {
                        "index": c.index,
                        "start_time": c.start,  # AudioChunk.start
                        "end_time": c.end,      # AudioChunk.end
                        "duration": c.end - c.start,
                        "sample_rate": c.sample_rate,
                    }
                    for c in chunks
                ]
                # V3.1.0: 保存 chunks_metadata 到检查点（用于恢复时跳过 VAD）
                # 注意：checkpoint_data 需要包装在 "preprocessing" 键下
                checkpoint_data = {
                    "preprocessing": {
                        "audio_extracted": True,
                        "vad_completed": True,
                        "total_chunks": len(chunks),
                        "chunks_metadata": chunks_metadata
                    }
                }
                token.check_and_save(checkpoint_data, job_dir)
                if pause_barrier:
                    decision = pause_barrier.on_unit_end(
                        stage="preprocess",
                        unit_id="vad_chunk",
                        payload={"total_chunks": len(chunks)},
                    )
                    if decision.should_stop:
                        if decision.stop_reason == "canceled":
                            token.raise_if_canceled()
                        if decision.stop_reason == "paused":
                            token.raise_if_paused()
            if cache_service and chunks and not used_vad_cache and not used_separation_cache:
                try:
                    cache_service.save_vad_chunks(chunks)
                except Exception as e:
                    self.logger.warning("[V3.2.0+dev.20260122.03] 保存 VAD 缓存失败: %s", e)
        else:
            self.logger.info("[V3.1.0] 跳过 Stage 1 (音频提取 + VAD)")
            if self._vad_intervals is None:
                self.logger.info("VAD 区间缺失（使用缓存/检查点恢复），对齐阶段将跳过 VAD 锚点约束")

        # Stage 1.5: ASR 风险前置检测（命中后强制全局分离并旁路音频预检）
        if chunks and self.separation_stage and self.is_asr_risk_guard_enabled:
            self.logger.info("Stage 1.5: ASR 风险前置检测")
            asr_risk_result = self._run_asr_risk_guard(chunks=chunks, job_dir=job_dir)
            if asr_risk_result is not None:
                self.logger.info(
                    "ASR 风险检测完成: risk=%s, ratio=%.2f%%, chunks=%s",
                    asr_risk_result.risk_chunk_count,
                    asr_risk_result.risk_chunk_ratio * 100.0,
                    asr_risk_result.risk_chunk_indices[:12],
                )
                if asr_risk_result.is_risk_detected:
                    is_force_global_by_asr_risk = True
                    runtime_separation_mode = "global"
                    skip_triage = True
                    skip_triage_reason = "asr_risk_guard"

                    if used_separation_cache:
                        # 旧 on_demand 缓存无法保证规避 ASR 风险，命中后必须重跑全局分离。
                        used_separation_cache = False
                        skip_separation = False
                        self.logger.info("ASR 风险命中，忽略已有分离缓存，改为全局分离")

                    if cache_service:
                        global_cached_chunks = cache_service.load_separation_chunks(expected_mode="global")
                        if global_cached_chunks:
                            chunks = global_cached_chunks
                            skip_vad = True
                            skip_separation = True
                            used_separation_cache = True
                            self.logger.info("ASR 风险命中且全局分离缓存命中，跳过分离")
                    self.logger.warning(
                        "ASR 风险命中，旁路音频预检并强制全局分离: %s",
                        asr_risk_result.reason,
                    )

        # Stage 2: 音频预检（如果启用，逐chunk可中断）
        if self.spectral_triage_stage:
            self.logger.info("Stage 2: 音频预检")
            used_triage_cache = False
            if skip_triage:
                used_triage_cache = True
                if skip_triage_reason == "asr_risk_guard":
                    self.logger.info("ASR 风险命中，跳过 DNSMOS 音频预检")
                    self._report_preprocess_progress("spectrum_analysis", 1.0, "ASR 风险命中，跳过预检")
                else:
                    self.logger.info("[V3.2.0+dev.20260123.03] 分离缓存命中，跳过音频预检")
                    self._report_preprocess_progress("spectrum_analysis", 1.0, "分离缓存命中")
            elif cache_service:
                triage_cache = cache_service.load_triage_results(
                    total_chunks=len(chunks),
                    use_dnsmos_triage=self.spectral_triage_stage.use_dnsmos_triage,
                    threshold=self.spectral_triage_stage.threshold,
                    use_smart_probe=self.spectral_triage_stage.use_smart_probe,
                    smart_probe_params=self.spectral_triage_stage.get_smart_probe_params(),
                    triage_version=self.spectral_triage_stage.triage_version,
                    dnsmos_model_hash=self.spectral_triage_stage.get_dnsmos_model_hash(),
                    dnsmos_threshold_profile_hash=self.spectral_triage_stage.get_dnsmos_threshold_profile_hash(),
                )
                if triage_cache:
                    cache_service.apply_triage_results(chunks, triage_cache)
                    used_triage_cache = True
                    self.logger.info("[V3.2.0+dev.20260122.03] 音频预检缓存命中，跳过预检")
                    self._report_preprocess_progress("spectrum_analysis", 1.0, "音频预检缓存命中")

            if not used_triage_cache:
                self._report_preprocess_progress("spectrum_analysis", 0.0, "音频预检中...")
                chunks = await self.spectral_triage_stage.process(chunks, job_dir=job_dir)
                self._report_preprocess_progress("spectrum_analysis", 1.0, "音频预检完成")
                if cache_service:
                    snapshot = self.spectral_triage_stage.get_cache_snapshot()
                    try:
                        cache_service.save_triage_results(
                            chunks=chunks,
                            triage_log=snapshot.get("triage_log"),
                            probe_state=snapshot.get("probe_state"),
                            use_dnsmos_triage=self.spectral_triage_stage.use_dnsmos_triage,
                            threshold=self.spectral_triage_stage.threshold,
                            use_smart_probe=self.spectral_triage_stage.use_smart_probe,
                            triage_version=self.spectral_triage_stage.triage_version,
                            dnsmos_model_hash=self.spectral_triage_stage.get_dnsmos_model_hash(),
                            dnsmos_threshold_profile_hash=self.spectral_triage_stage.get_dnsmos_threshold_profile_hash(),
                        )
                    except Exception as e:
                        self.logger.warning("[V3.2.0+dev.20260122.03] 保存音频预检缓存失败: %s", e)

            # 统计音频预检结果
            stats = self.spectral_triage_stage.get_statistics(chunks)
            self.logger.info(
                f"音频预检完成: {stats['need_separation']}/{stats['total_chunks']} "
                f"个chunk需要分离 (比例: {stats['separation_ratio']:.2%})"
            )

            # v3.1.0: 音频预检完成后检查点
            if token and job_dir:
                if pause_barrier:
                    pause_barrier.on_unit_start(
                        stage="preprocess",
                        unit_id="triage_chunk",
                        payload={"need_separation_count": stats['need_separation']},
                    )
                checkpoint_data = {
                    "spectral_triage": {
                        "completed": True,
                        "need_separation_count": stats['need_separation']
                    }
                }
                token.check_and_save(checkpoint_data, job_dir)
                if pause_barrier:
                    decision = pause_barrier.on_unit_end(
                        stage="preprocess",
                        unit_id="triage_chunk",
                        payload={"need_separation_count": stats['need_separation']},
                    )
                    if decision.should_stop:
                        if decision.stop_reason == "canceled":
                            token.raise_if_canceled()
                        if decision.stop_reason == "paused":
                            token.raise_if_paused()
        else:
            self.logger.info("Stage 2: 音频预检已跳过")
            self._report_preprocess_progress("spectrum_analysis", 1.0, "音频预检跳过")

        # Stage 3: 人声分离（如果启用）
        cached_separation_indices = set()
        cache_separation_completed = False
        if cache_service and chunks and not used_separation_cache:
            try:
                cache_service.begin_separation(runtime_separation_mode, len(chunks))
                cached_separation_indices, cache_separation_completed = cache_service.load_separation_cache(
                    chunks=chunks,
                    expected_mode=runtime_separation_mode,
                )
                if cache_separation_completed and len(cached_separation_indices) < len(chunks):
                    self.logger.warning("[V3.2.0+dev.20260122.03] 分离缓存不完整，回退重做")
                    cache_separation_completed = False
                cache_service.save_passthrough_chunks(
                    chunks=chunks,
                    only_unseparated=self.separation_stage is not None,
                )
            except Exception as e:
                self.logger.warning("[V3.2.0+dev.20260122.03] 准备分离缓存失败: %s", e)
                cached_separation_indices = set()
                cache_separation_completed = False

        if self.separation_stage:
            self.logger.info("Stage 3: 人声分离")
            if skip_separation:
                cache_separation_completed = True
                self.logger.info("[V3.2.0+dev.20260123.03] 分离缓存命中，跳过分离")
                self._report_preprocess_progress("demucs", 1.0, "分离缓存命中")
            elif cache_separation_completed:
                self.logger.info("[V3.2.0+dev.20260122.03] 人声分离缓存命中，跳过分离")
                self._report_preprocess_progress("demucs", 1.0, "人声分离缓存命中")
            else:
                self._report_preprocess_progress("demucs", 0.0, "人声分离中...")
                previous_stage_mode = self.separation_stage.mode
                self.separation_stage.mode = runtime_separation_mode
                try:
                    if runtime_separation_mode == "global":
                        chunks = await self.separation_stage.process(
                            chunks=chunks,
                            audio_path=video_path,
                            job_dir=job_dir,
                            separated_indices=cached_separation_indices,
                            cache_service=cache_service
                        )
                    else:
                        chunks = await self.separation_stage.process(
                            chunks=chunks,
                            job_dir=job_dir,
                            separated_indices=cached_separation_indices,
                            cache_service=cache_service
                        )
                finally:
                    self.separation_stage.mode = previous_stage_mode
                self._report_preprocess_progress("demucs", 1.0, "人声分离完成")

            # 统计分离结果
            stats = self.separation_stage.get_statistics(chunks)
            self.logger.info(
                f"人声分离完成: {stats['separated']}/{stats['total_chunks']} "
                f"个chunk已分离 (比例: {stats['separation_ratio']:.2%})"
            )

            if cache_service and not used_separation_cache:
                try:
                    cache_service.finalize_separation(len(chunks))
                except Exception as e:
                    self.logger.warning("[V3.2.0+dev.20260122.03] 完成分离缓存失败: %s", e)

            # v3.1.0: 人声分离完成后检查点
            if token and job_dir:
                if pause_barrier:
                    pause_barrier.on_unit_start(
                        stage="preprocess",
                        unit_id="separation_chunk",
                        payload={
                            "mode": runtime_separation_mode,
                            "separated_count": stats['separated'],
                            "is_force_global_by_asr_risk": is_force_global_by_asr_risk,
                        },
                    )
                checkpoint_data = {
                    "separation": {
                        "completed": True,
                        "mode": runtime_separation_mode,
                        "separated_count": stats['separated']
                    }
                }
                token.check_and_save(checkpoint_data, job_dir)
                if pause_barrier:
                    decision = pause_barrier.on_unit_end(
                        stage="preprocess",
                        unit_id="separation_chunk",
                        payload={
                            "mode": runtime_separation_mode,
                            "separated_count": stats['separated'],
                            "is_force_global_by_asr_risk": is_force_global_by_asr_risk,
                        },
                    )
                    if decision.should_stop:
                        if decision.stop_reason == "canceled":
                            token.raise_if_canceled()
                        if decision.stop_reason == "paused":
                            token.raise_if_paused()
        else:
            self.logger.info("Stage 3: 人声分离已跳过")
            self._report_preprocess_progress("demucs", 1.0, "人声分离跳过")
            if cache_service and chunks and not used_separation_cache:
                try:
                    cache_service.finalize_separation(len(chunks))
                except Exception as e:
                    self.logger.warning("[V3.2.0+dev.20260122.03] 完成分离缓存失败: %s", e)

        # Stage 4: LangID 语言检测（失败不阻断）
        if chunks:
            self.logger.info("Stage 4: 语言检测")
            try:
                from app.services.speech_analysis_service import get_speech_analysis_service
                from app.utils.cancellation_token import CancelledException, PausedException

                langid_service = get_speech_analysis_service(logger=self.logger)
                langid_metadata: Optional[Dict[str, Any]] = None
                lang_map: Dict[int, Dict[str, Any]] = {}
                missing_indices = [chunk.index for chunk in chunks]
                cache_complete = False
                langid_started_at = time.time()
                total_chunks = len(chunks)
                resolved_mode = self.config.language_detection_mode
                resolved_device = self.config.language_detection_device

                if cache_service:
                    try:
                        langid_metadata = langid_service.build_langid_cache_metadata(
                            mode=self.config.language_detection_mode,
                            device=self.config.language_detection_device,
                            whitelist=self.config.langid_whitelist,
                            logit_bias_score=self.config.langid_logit_bias_score,
                        )
                        recovered = cache_service.recover_langid_cache(
                            chunks=chunks,
                            expected_metadata=langid_metadata,
                        )
                        if recovered:
                            lang_map, missing_indices, cache_complete = recovered
                            if not missing_indices:
                                self.logger.info(
                                    "[V3.2.0+dev.20260127.05] LangID 缓存命中: %d/%d",
                                    len(lang_map),
                                    len(chunks),
                                )
                        else:
                            lang_map = {}
                            missing_indices = [chunk.index for chunk in chunks]
                    except Exception as exc:
                        self.logger.warning("[V3.2.0+dev.20260127.05] LangID 缓存恢复失败: %s", exc)
                        lang_map = {}
                        missing_indices = [chunk.index for chunk in chunks]
                        cache_complete = False

                if langid_metadata:
                    resolved_mode = langid_metadata.get("mode", resolved_mode)
                    resolved_device = langid_metadata.get("resolved_device", resolved_device)

                broadcast_preprocess_event(
                    "preprocessing.langid.started",
                    {
                        "schema_version": "1.0",
                        "job_id": job_id,
                        "mode": resolved_mode,
                        "total_chunks": total_chunks,
                        "resolved_device": resolved_device,
                        "model_id": langid_metadata.get("model_id") if langid_metadata else None,
                        "language_code_standard": "ISO-639-3",
                    },
                )

                if missing_indices:
                    missing_set = set(missing_indices)
                    target_chunks = [chunk for chunk in chunks if chunk.index in missing_set]
                    processed_indices: Set[int] = set(lang_map.keys())

                    def on_predictions(batch_chunks, predictions):
                        for chunk, prediction in zip(batch_chunks, predictions):
                            lang_map[chunk.index] = {
                                "language": prediction.language,
                                "confidence": prediction.confidence,
                                "raw_label": prediction.raw_label,
                                "raw_language": prediction.raw_language,
                                "raw_confidence": prediction.raw_confidence,
                            }
                            processed_indices.add(chunk.index)
                        broadcast_preprocess_event(
                            "preprocessing.langid.progress",
                            {
                                "schema_version": "1.0",
                                "job_id": job_id,
                                "processed": len(processed_indices),
                                "total": total_chunks,
                                "percentage": round(len(processed_indices) / max(1, total_chunks) * 100, 2),
                            },
                        )
                        if cache_service and langid_metadata:
                            try:
                                progress = cache_service.build_langid_progress(
                                    processed_indices,
                                    total_chunks,
                                    complete=False,
                                )
                                cache_service.save_langid_cache(lang_map, langid_metadata, progress)
                            except Exception as cache_exc:
                                self.logger.warning("[V3.2.0+dev.20260127.05] LangID 缓存写入失败: %s", cache_exc)
                        if token:
                            if job_dir and (token.is_canceled or token.is_paused):
                                token.check_and_save({"preprocessing": {"total_chunks": total_chunks}}, job_dir)
                            else:
                                token.raise_if_canceled()
                                token.raise_if_paused()

                    lang_predictions = langid_service.detect_languages(
                        chunks=target_chunks,
                        mode=resolved_mode,
                        device=resolved_device,
                        whitelist=self.config.langid_whitelist,
                        logit_bias_score=self.config.langid_logit_bias_score,
                        on_predictions=on_predictions,
                    )
                    for chunk_index, prediction in lang_predictions.items():
                        lang_map[chunk_index] = {
                            "language": prediction.language,
                            "confidence": prediction.confidence,
                            # V3.2.2+dev.20260201.01: 添加 Top-2 语言置信度
                            "top2_confidence": prediction.top2_confidence,
                            "raw_label": prediction.raw_label,
                            "raw_language": prediction.raw_language,
                            "raw_confidence": prediction.raw_confidence,
                        }

                    if cache_service and langid_metadata:
                        processed_indices.update(lang_map.keys())
                        try:
                            progress = cache_service.build_langid_progress(
                                processed_indices,
                                total_chunks,
                                complete=len(processed_indices) >= total_chunks,
                            )
                            cache_service.save_langid_cache(lang_map, langid_metadata, progress)
                        except Exception as cache_exc:
                            self.logger.warning("[V3.2.0+dev.20260127.05] LangID 缓存写入失败: %s", cache_exc)

                if token and job_dir and pause_barrier:
                    pause_barrier.on_unit_start(
                        stage="preprocess",
                        unit_id="langid_chunk",
                        payload={"processed": len(lang_map), "total": len(chunks)},
                    )
                    token.check_and_save(
                        {"preprocessing": {"total_chunks": len(chunks)}},
                        job_dir,
                    )
                    decision = pause_barrier.on_unit_end(
                        stage="preprocess",
                        unit_id="langid_chunk",
                        payload={"processed": len(lang_map), "total": len(chunks)},
                    )
                    if decision.should_stop:
                        if decision.stop_reason == "canceled":
                            token.raise_if_canceled()
                        if decision.stop_reason == "paused":
                            token.raise_if_paused()
                elif cache_service and langid_metadata and lang_map and not cache_complete:
                    try:
                        progress = cache_service.build_langid_progress(
                            lang_map.keys(),
                            len(chunks),
                            complete=True,
                        )
                        cache_service.save_langid_cache(lang_map, langid_metadata, progress)
                    except Exception as cache_exc:
                        self.logger.warning("[V3.2.0+dev.20260127.05] LangID 缓存写入失败: %s", cache_exc)

                for chunk in chunks:
                    cached = lang_map.get(chunk.index)
                    if not cached:
                        chunk.language = "auto"
                        chunk.language_confidence = {}
                        continue

                    # V3.2.2+dev.20260201.01: 使用 Top-2 置信度字典
                    top2_conf = cached.get("top2_confidence")
                    if top2_conf and isinstance(top2_conf, dict):
                        chunk.language_confidence = top2_conf
                    else:
                        # 回退：使用单一置信度
                        lang = cached.get("language", "auto")
                        conf = float(cached.get("confidence", 0.0))
                        chunk.language_confidence = {lang: conf} if lang != "auto" and conf > 0 else {}

                    chunk.language = cached.get("language", "auto")
                    # 获取 Top-1 置信度用于阈值判断
                    top1_confidence = max(chunk.language_confidence.values()) if chunk.language_confidence else 0.0
                    if top1_confidence < self.config.langid_confidence_threshold:
                        chunk.language = "auto"

                languages = [chunk.language for chunk in chunks if chunk.language]
                if languages:
                    from collections import Counter

                    dominant = Counter(languages).most_common(1)[0][0]
                    self.logger.info("LangID 完成: dominant=%s, chunks=%d", dominant, len(chunks))
                    language_distribution = dict(Counter(languages))
                else:
                    dominant = None
                    language_distribution = {}

                if cache_service and langid_metadata:
                    try:
                        cache_service.save_langid_report(
                            chunks=chunks,
                            language_map=lang_map,
                            metadata=langid_metadata,
                            confidence_threshold=self.config.langid_confidence_threshold,
                        )
                    except Exception as report_exc:
                        self.logger.warning("[V3.2.0+dev.20260127.10] LangID 报告写入失败: %s", report_exc)

                broadcast_preprocess_event(
                    "preprocessing.langid.completed",
                    {
                        "schema_version": "1.0",
                        "job_id": job_id,
                        "total_chunks": len(chunks),
                        "duration_ms": int((time.time() - langid_started_at) * 1000),
                        "language_distribution": language_distribution,
                        "dominant_language": dominant,
                        "coverage_ratio": round(len(lang_map) / max(1, len(chunks)), 4),
                        "cache_saved": bool(cache_service and langid_metadata),
                    },
                )
            except (CancelledException, PausedException):
                raise
            except Exception as e:
                self.logger.error("[V3.2.0+dev.20260127.05] LangID 失败，回退为 auto: %s", e)
                broadcast_preprocess_event(
                    "preprocessing.langid.error",
                    {
                        "schema_version": "1.0",
                        "job_id": job_id,
                        "error_type": "langid_failed",
                        "message": str(e),
                        "fallback_strategy": "all_chunks_set_to_auto",
                    },
                )
                for chunk in chunks:
                    chunk.language = "auto"
                    chunk.language_confidence = {}

        # Phase 4: 旧 Speaker 声纹提取服务已下线（由 Timeline 域统一接管）。
        if False and chunks and self.config.enable_speaker_embedding:
            self.logger.info("Stage 5: 旧 Speaker 声纹提取（已下线）")
            try:
                from app.utils.cancellation_token import CancelledException, PausedException

                speaker_service = None
                speaker_metadata: Optional[Dict[str, Any]] = None
                embedding_map: Dict[int, List[float]] = {}
                missing_indices = [chunk.index for chunk in chunks]
                cache_complete = False
                speaker_started_at = time.time()
                total_chunks = len(chunks)
                resolved_device = self.config.language_detection_device

                if cache_service:
                    try:
                        speaker_metadata = speaker_service.build_speaker_cache_metadata(
                            mode=self.config.language_detection_mode,
                            device=self.config.language_detection_device,
                        )
                        recovered = cache_service.recover_speaker_cache(
                            chunks=chunks,
                            expected_metadata=speaker_metadata,
                        )
                        if recovered:
                            embedding_map, missing_indices, cache_complete = recovered
                            if not missing_indices:
                                self.logger.info(
                                    "[V3.2.0+dev.20260127.06] Speaker 缓存命中: %d/%d",
                                    len(embedding_map),
                                    len(chunks),
                                )
                        else:
                            embedding_map = {}
                            missing_indices = [chunk.index for chunk in chunks]
                    except Exception as exc:
                        self.logger.warning("[V3.2.0+dev.20260127.06] Speaker 缓存恢复失败: %s", exc)
                        embedding_map = {}
                        missing_indices = [chunk.index for chunk in chunks]
                        cache_complete = False

                if speaker_metadata:
                    resolved_device = speaker_metadata.get("resolved_device", resolved_device)

                broadcast_preprocess_event(
                    "preprocessing.speaker.started",
                    {
                        "schema_version": "1.0",
                        "job_id": job_id,
                        "total_chunks": total_chunks,
                        "embedding_dim": speaker_metadata.get("embedding_dim") if speaker_metadata else 192,
                        "model_id": speaker_metadata.get("model_id") if speaker_metadata else None,
                        "resolved_device": resolved_device,
                    },
                )

                if missing_indices:
                    missing_set = set(missing_indices)
                    target_chunks = [chunk for chunk in chunks if chunk.index in missing_set]
                    processed_indices: Set[int] = set(embedding_map.keys())

                    def on_embeddings(batch_chunks, embeddings):
                        for chunk, embedding in zip(batch_chunks, embeddings):
                            embedding_map[chunk.index] = embedding.embedding
                            processed_indices.add(chunk.index)
                        broadcast_preprocess_event(
                            "preprocessing.speaker.progress",
                            {
                                "schema_version": "1.0",
                                "job_id": job_id,
                                "processed": len(processed_indices),
                                "total": total_chunks,
                                "percentage": round(len(processed_indices) / max(1, total_chunks) * 100, 2),
                            },
                        )
                        if cache_service and speaker_metadata:
                            try:
                                progress = cache_service.build_speaker_progress(
                                    processed_indices,
                                    total_chunks,
                                    complete=False,
                                )
                                cache_service.save_speaker_cache(embedding_map, speaker_metadata, progress)
                            except Exception as cache_exc:
                                self.logger.warning(
                                    "[V3.2.0+dev.20260127.06] Speaker 缓存写入失败: %s",
                                    cache_exc,
                                )
                        if token:
                            if job_dir and (token.is_canceled or token.is_paused):
                                token.check_and_save({"preprocessing": {"total_chunks": total_chunks}}, job_dir)
                            else:
                                token.raise_if_canceled()
                                token.raise_if_paused()

                    speaker_results = speaker_service.extract_embeddings(
                        chunks=target_chunks,
                        device=resolved_device,
                        mode=self.config.language_detection_mode,
                        on_embeddings=on_embeddings,
                    )
                    for chunk_index, embedding in speaker_results.items():
                        embedding_map[chunk_index] = embedding.embedding

                    if cache_service and speaker_metadata:
                        processed_indices.update(embedding_map.keys())
                        try:
                            progress = cache_service.build_speaker_progress(
                                processed_indices,
                                total_chunks,
                                complete=len(processed_indices) >= total_chunks,
                            )
                            cache_service.save_speaker_cache(embedding_map, speaker_metadata, progress)
                        except Exception as cache_exc:
                            self.logger.warning(
                                "[V3.2.0+dev.20260127.06] Speaker 缓存写入失败: %s",
                                cache_exc,
                            )

                if token and job_dir and pause_barrier:
                    pause_barrier.on_unit_start(
                        stage="preprocess",
                        unit_id="speaker_chunk",
                        payload={"processed": len(embedding_map), "total": len(chunks)},
                    )
                    token.check_and_save(
                        {"preprocessing": {"total_chunks": len(chunks)}},
                        job_dir,
                    )
                    decision = pause_barrier.on_unit_end(
                        stage="preprocess",
                        unit_id="speaker_chunk",
                        payload={"processed": len(embedding_map), "total": len(chunks)},
                    )
                    if decision.should_stop:
                        if decision.stop_reason == "canceled":
                            token.raise_if_canceled()
                        if decision.stop_reason == "paused":
                            token.raise_if_paused()
                elif cache_service and speaker_metadata and embedding_map and not cache_complete:
                    try:
                        progress = cache_service.build_speaker_progress(
                            embedding_map.keys(),
                            len(chunks),
                            complete=True,
                        )
                        cache_service.save_speaker_cache(embedding_map, speaker_metadata, progress)
                    except Exception as cache_exc:
                        self.logger.warning(
                            "[V3.2.0+dev.20260127.06] Speaker 缓存写入失败: %s",
                            cache_exc,
                        )

                for chunk in chunks:
                    chunk.speaker_embedding = embedding_map.get(chunk.index)

                broadcast_preprocess_event(
                    "preprocessing.speaker.completed",
                    {
                        "schema_version": "1.0",
                        "job_id": job_id,
                        "total_embeddings": len(embedding_map),
                        "duration_ms": int((time.time() - speaker_started_at) * 1000),
                        "cache_saved": bool(cache_service and speaker_metadata),
                    },
                )
            except (CancelledException, PausedException):
                raise
            except Exception as e:
                self.logger.error("[V3.2.0+dev.20260127.06] Speaker 失败，已跳过: %s", e)
                broadcast_preprocess_event(
                    "preprocessing.speaker.error",
                    {
                        "schema_version": "1.0",
                        "job_id": job_id,
                        "error_type": "speaker_failed",
                        "message": str(e),
                    },
                )
                for chunk in chunks:
                    chunk.speaker_embedding = None

        # Phase 4: 旧 Speaker 在线聚类服务已下线（由 Timeline 域统一接管）。
        has_embeddings = any(getattr(chunk, "speaker_embedding", None) for chunk in chunks)
        if False and chunks and self.config.enable_speaker_embedding and has_embeddings:
            self.logger.info("Stage 6: 旧 Speaker 在线聚类（已下线）")
            try:
                _cluster_factory = None

                class SpeakerClusterConfig:  # type: ignore[no-redef]
                    pass

                cluster_config = SpeakerClusterConfig(
                    similarity_threshold=0.50,  # 折中阈值
                    debounce_count=2,
                    min_turn_duration=1.5,
                    min_turn_chunks=3,
                    enabled=True,
                )
                cluster_service = None
                assert _cluster_factory is not None
                cluster_service = _cluster_factory(config=cluster_config, logger=self.logger)
                cluster_service.reset()

                speaker_changes = 0
                for chunk in chunks:
                    speaker_embedding = getattr(chunk, "speaker_embedding", None)
                    if not speaker_embedding:
                        continue
                    result = cluster_service.cluster(
                        embedding=speaker_embedding,
                        chunk_start=chunk.start,
                        chunk_end=chunk.end,
                    )
                    chunk.primary_speaker_id = result.speaker_id
                    if result.is_speaker_changed:
                        speaker_changes += 1

                self.logger.info(
                    "说话人聚类完成: %d 人, %d 次切换",
                    cluster_service.speaker_count,
                    speaker_changes,
                )
            except Exception as e:
                self.logger.warning("说话人聚类失败，已跳过: %s", e)

        self.logger.info(f"预处理流程完成: {len(chunks)} 个chunk准备就绪")

        self._last_preprocess_artifacts = PreprocessArtifacts(
            chunks=chunks,
            vad_intervals=self.get_vad_intervals() or [],
            full_audio=None,
            sample_rate=chunks[0].sample_rate if chunks else 16000,
            language_map={},
        )

        return chunks

    async def _extract_and_vad(
        self,
        video_path: str,
        job_state: Optional[JobState] = None
    ) -> List[AudioChunk]:
        """
        音频提取 + VAD切分

        使用 ChunkEngine 完成音频提取和VAD切分，但不执行Demucs分离。

        Args:
            video_path: 视频/音频文件路径
            job_state: 任务状态（可选）

        Returns:
            List[AudioChunk]: VAD切分后的 Chunk 列表
        """
        # V3.1.0: 使用传入的 VAD 配置，如果没有则使用默认配置
        if self.vad_config is None:
            from app.services.runtime_param_resolver import build_vad_config

            vad_config = build_vad_config()
        else:
            vad_config = self.vad_config
        self.logger.info(f"VAD配置: merge_max_gap={vad_config.merge_max_gap}s, merge_max_duration={vad_config.merge_max_duration}s")

        # 定义进度回调
        def progress_callback(progress: float, message: str):
            if job_state:
                job_state.phase_percent = progress * 100
                job_state.message = message
                self.logger.debug(f"进度: {progress:.1%} - {message}")
            self._report_preprocess_progress("vad", progress, message)

        # 使用 ChunkEngine 处理音频（不启用Demucs）
        chunks, full_audio, sr = self.chunk_engine.process_audio(
            audio_path=video_path,
            enable_demucs=False,  # 关键：不在这里执行Demucs
            vad_config=vad_config,
            progress_callback=progress_callback,
            diagnostic_service=self.diagnostic_service  # V3.2.0+dev.20260131: 传递诊断服务
        )
        self._vad_intervals = self._build_vad_intervals_from_segments(
            self.chunk_engine.get_last_vad_segments()
        )
        if not self._vad_intervals:
            self.logger.info("未获取到真实 VAD 区间，对齐阶段将跳过 VAD 锚点约束")

        self.logger.info(
            f"VAD切分完成: {len(chunks)} 个chunk, "
            f"采样率: {sr}Hz, "
            f"总时长: {len(full_audio) / sr:.2f}s"
        )

        return chunks

    async def _restore_chunks_from_metadata(
        self,
        video_path: str,
        chunks_metadata: List[dict]
    ) -> Optional[List[AudioChunk]]:
        """
        V3.1.0: 从 checkpoint 元数据恢复 AudioChunk 列表

        与完整 VAD 不同，这里直接加载音频并根据已知的时间戳切分，
        避免重新执行 VAD 检测。

        Args:
            video_path: 视频/音频文件路径
            chunks_metadata: chunk 元数据列表（从 checkpoint 加载）

        Returns:
            Optional[List[AudioChunk]]: 恢复的 chunks，失败返回 None
        """
        try:
            import librosa
            import numpy as np

            if not chunks_metadata:
                self.logger.warning("[V3.1.0] chunks_metadata 为空，无法恢复")
                return None

            # 获取目标采样率（从第一个 chunk 的元数据获取）
            target_sr = chunks_metadata[0].get("sample_rate", 16000)

            # 加载完整音频
            self.logger.info(f"[V3.1.0] 加载音频用于 chunk 恢复: {video_path}")
            full_audio, sr = librosa.load(video_path, sr=target_sr, mono=True)
            self.logger.info(f"[V3.1.0] 音频加载完成: 时长 {len(full_audio) / sr:.2f}s, 采样率 {sr}Hz")

            # 根据元数据切分 chunks
            chunks = []
            for meta in chunks_metadata:
                index = meta.get("index", len(chunks))
                start_time = meta.get("start_time", 0.0)
                end_time = meta.get("end_time", 0.0)
                duration = meta.get("duration", end_time - start_time)

                # 计算采样点范围
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)

                # 边界检查
                start_sample = max(0, start_sample)
                end_sample = min(len(full_audio), end_sample)

                if end_sample <= start_sample:
                    self.logger.warning(f"[V3.1.0] 跳过无效 chunk {index}: start={start_time:.2f}s, end={end_time:.2f}s")
                    continue

                # 提取音频片段
                chunk_audio = full_audio[start_sample:end_sample]

                # 创建 AudioChunk（使用 start/end 而非 start_time/end_time）
                chunk = AudioChunk(
                    index=index,
                    audio=chunk_audio,
                    start=start_time,   # AudioChunk 使用 start
                    end=end_time,       # AudioChunk 使用 end
                    sample_rate=sr
                )
                chunks.append(chunk)

            self.logger.info(f"[V3.1.0] 从 checkpoint 恢复了 {len(chunks)} 个 chunk")
            return chunks

        except Exception as e:
            self.logger.error(f"[V3.1.0] 从 checkpoint 恢复 chunks 失败: {e}")
            return None

    def get_vad_intervals(self) -> Optional[List[Tuple[float, float]]]:
        """返回最近一次 VAD 语音区间（秒级）。"""
        if not self._vad_intervals:
            return None
        return list(self._vad_intervals)

    def get_last_preprocess_artifacts(self) -> Optional[PreprocessArtifacts]:
        """返回最近一次预处理的结构化产物快照。"""
        return self._last_preprocess_artifacts

    @staticmethod
    def _build_vad_intervals_from_segments(
        segments: Optional[List[Dict[str, Any]]],
    ) -> Optional[List[Tuple[float, float]]]:
        """从 VAD 段列表提取区间，过滤无效边界。"""
        if not segments:
            return None
        intervals: List[Tuple[float, float]] = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            start = seg.get("start")
            end = seg.get("end")
            if start is None or end is None:
                continue
            try:
                start_val = float(start)
                end_val = float(end)
            except (TypeError, ValueError):
                continue
            if end_val <= start_val:
                continue
            intervals.append((start_val, end_val))
        return intervals or None

    def get_statistics(self, chunks: List[AudioChunk]) -> dict:
        """
        获取预处理统计信息

        Args:
            chunks: 已处理的 AudioChunk 列表

        Returns:
            dict: 统计信息
        """
        total = len(chunks)

        # 音频预检统计
        need_separation = sum(1 for c in chunks if c.needs_separation)

        # 人声分离统计
        separated = sum(1 for c in chunks if c.is_separated)

        # 分离级别统计
        from app.models.circuit_breaker_models import SeparationLevel
        htdemucs_count = sum(
            1 for c in chunks
            if c.is_separated and c.separation_level == SeparationLevel.HTDEMUCS
        )
        mdx_extra_count = sum(
            1 for c in chunks
            if c.is_separated and c.separation_level == SeparationLevel.MDX_EXTRA
        )

        # 熔断回溯统计
        fuse_retry_count = sum(c.fuse_retry_count for c in chunks)
        max_retry = max((c.fuse_retry_count for c in chunks), default=0)

        return {
            "total_chunks": total,
            "need_separation": need_separation,
            "separated": separated,
            "not_separated": total - separated,
            "htdemucs_count": htdemucs_count,
            "mdx_extra_count": mdx_extra_count,
            "separation_ratio": separated / total if total > 0 else 0.0,
            "fuse_retry_total": fuse_retry_count,
            "fuse_retry_max": max_retry,
        }


# 便捷函数
def get_preprocessing_pipeline(
    config: PreprocessingConfig,
    chunk_engine: Optional[ChunkEngine] = None,
    logger: Optional[logging.Logger] = None,
    cancellation_token: Optional["CancellationToken"] = None,  # v3.1.0: 新增
    progress_emitter: Optional["ProgressEventEmitter"] = None
) -> PreprocessingPipeline:
    """
    获取预处理流水线实例

    Args:
        config: 预处理配置
        chunk_engine: 音频切分引擎（可选）
        logger: 日志记录器（可选）
        cancellation_token: 取消令牌（可选，v3.1.0）
        progress_emitter: 进度发射器（可选）

    Returns:
        PreprocessingPipeline 实例
    """
    return PreprocessingPipeline(
        config=config,
        chunk_engine=chunk_engine,
        logger=logger,
        cancellation_token=cancellation_token,  # v3.1.0
        progress_emitter=progress_emitter
    )
