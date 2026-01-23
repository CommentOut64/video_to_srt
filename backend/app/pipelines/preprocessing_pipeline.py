"""
PreprocessingPipeline - 预处理流水线（新架构）

负责统一管理所有预处理步骤：
1. 音频提取 + VAD切分
2. 频谱分诊（可选）
3. 人声分离（可选，支持全局/按需模式）

与旧的 AudioProcessingPipeline（已归档）的区别：
- 旧版：整轨分离 + VAD切分
- 新版：VAD切分 + 频谱分诊 + 按需分离（Stage模式）

v3.1.0 更新：
- 集成 CancellationToken 支持暂停/取消
- 支持断点续传检查点保存
"""

import logging
from typing import List, Optional, TYPE_CHECKING
from pathlib import Path

from app.services.audio.chunk_engine import ChunkEngine, AudioChunk
from app.services.audio.vad_service import VADConfig
from app.pipelines.stages.spectral_triage_stage import SpectralTriageStage
from app.pipelines.stages.separation_stage import SeparationStage
from app.models.job_models import PreprocessingConfig, JobState
from app.services.demucs_service import get_demucs_service
from app.services.preprocess_cache_service import PreprocessCacheService

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
        progress_emitter: Optional["ProgressEventEmitter"] = None
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
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.cancellation_token = cancellation_token  # v3.1.0
        self.progress_emitter = progress_emitter
        if vad_config is None:
            from app.services.runtime_param_resolver import build_vad_config

            self.vad_config = build_vad_config()
        else:
            self.vad_config = vad_config  # V3.1.0: 保存 VAD 配置

        # 初始化 ChunkEngine（用于音频提取和VAD切分）
        self.chunk_engine = chunk_engine or ChunkEngine(logger=self.logger)

        # 初始化频谱分诊阶段（如果启用）
        if config.enable_spectral_triage:
            self.spectral_triage_stage = SpectralTriageStage(
                threshold=config.spectrum_threshold,
                logger=self.logger,
                cancellation_token=cancellation_token,  # v3.1.0: 传递令牌
                use_snr_triage=config.use_snr_triage
            )
            self.logger.info(
                f"频谱分诊已启用: threshold={config.spectrum_threshold}"
            )
        else:
            self.spectral_triage_stage = None
            self.logger.info("频谱分诊已禁用")

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
        2. 频谱分诊（可选）-> 标记 needs_separation
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

        # V3.2.0+dev.20260122.03: 优先尝试预处理缓存
        chunks = None
        skip_vad = False
        used_vad_cache = False
        if cache_service:
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
            if cache_service and chunks and not used_vad_cache:
                try:
                    cache_service.save_vad_chunks(chunks)
                except Exception as e:
                    self.logger.warning("[V3.2.0+dev.20260122.03] 保存 VAD 缓存失败: %s", e)
        else:
            self.logger.info("[V3.1.0] 跳过 Stage 1 (音频提取 + VAD)")

        # Stage 2: 频谱分诊（如果启用，逐chunk可中断）
        if self.spectral_triage_stage:
            self.logger.info("Stage 2: 频谱分诊")
            used_triage_cache = False
            if cache_service:
                triage_cache = cache_service.load_triage_results(
                    total_chunks=len(chunks),
                    use_snr_triage=self.spectral_triage_stage.use_snr_triage,
                    threshold=self.spectral_triage_stage.threshold,
                    use_smart_probe=self.spectral_triage_stage.use_smart_probe,
                    smart_probe_params=self.spectral_triage_stage.get_smart_probe_params(),
                )
                if triage_cache:
                    cache_service.apply_triage_results(chunks, triage_cache)
                    used_triage_cache = True
                    self.logger.info("[V3.2.0+dev.20260122.03] 频谱分诊缓存命中，跳过分诊")
                    self._report_preprocess_progress("spectrum_analysis", 1.0, "频谱分诊缓存命中")

            if not used_triage_cache:
                self._report_preprocess_progress("spectrum_analysis", 0.0, "频谱分诊中...")
                chunks = await self.spectral_triage_stage.process(chunks, job_dir=job_dir)
                self._report_preprocess_progress("spectrum_analysis", 1.0, "频谱分诊完成")
                if cache_service:
                    snapshot = self.spectral_triage_stage.get_cache_snapshot()
                    try:
                        cache_service.save_triage_results(
                            chunks=chunks,
                            triage_log=snapshot.get("triage_log"),
                            probe_state=snapshot.get("probe_state"),
                            use_snr_triage=self.spectral_triage_stage.use_snr_triage,
                            threshold=self.spectral_triage_stage.threshold,
                            use_smart_probe=self.spectral_triage_stage.use_smart_probe,
                        )
                    except Exception as e:
                        self.logger.warning("[V3.2.0+dev.20260122.03] 保存分诊缓存失败: %s", e)

            # 统计分诊结果
            stats = self.spectral_triage_stage.get_statistics(chunks)
            self.logger.info(
                f"频谱分诊完成: {stats['need_separation']}/{stats['total_chunks']} "
                f"个chunk需要分离 (比例: {stats['separation_ratio']:.2%})"
            )

            # v3.1.0: 频谱分诊完成后检查点
            if token and job_dir:
                checkpoint_data = {
                    "spectral_triage": {
                        "completed": True,
                        "need_separation_count": stats['need_separation']
                    }
                }
                token.check_and_save(checkpoint_data, job_dir)
        else:
            self.logger.info("Stage 2: 频谱分诊已跳过")
            self._report_preprocess_progress("spectrum_analysis", 1.0, "频谱分诊跳过")

        # Stage 3: 人声分离（如果启用）
        separation_mode = self.config.separation_mode if self.separation_stage else "off"
        cached_separation_indices = set()
        cache_separation_completed = False
        if cache_service and chunks:
            try:
                cache_service.begin_separation(separation_mode, len(chunks))
                cached_separation_indices, cache_separation_completed = cache_service.load_separation_cache(
                    chunks=chunks,
                    expected_mode=separation_mode,
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
            if cache_separation_completed:
                self.logger.info("[V3.2.0+dev.20260122.03] 人声分离缓存命中，跳过分离")
                self._report_preprocess_progress("demucs", 1.0, "人声分离缓存命中")
            else:
                self._report_preprocess_progress("demucs", 0.0, "人声分离中...")
                if self.config.separation_mode == "global":
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
                self._report_preprocess_progress("demucs", 1.0, "人声分离完成")

            # 统计分离结果
            stats = self.separation_stage.get_statistics(chunks)
            self.logger.info(
                f"人声分离完成: {stats['separated']}/{stats['total_chunks']} "
                f"个chunk已分离 (比例: {stats['separation_ratio']:.2%})"
            )

            if cache_service:
                try:
                    cache_service.finalize_separation(len(chunks))
                except Exception as e:
                    self.logger.warning("[V3.2.0+dev.20260122.03] 完成分离缓存失败: %s", e)

            # v3.1.0: 人声分离完成后检查点
            if token and job_dir:
                checkpoint_data = {
                    "separation": {
                        "completed": True,
                        "mode": self.config.separation_mode,
                        "separated_count": stats['separated']
                    }
                }
                token.check_and_save(checkpoint_data, job_dir)
        else:
            self.logger.info("Stage 3: 人声分离已跳过")
            self._report_preprocess_progress("demucs", 1.0, "人声分离跳过")
            if cache_service and chunks:
                try:
                    cache_service.finalize_separation(len(chunks))
                except Exception as e:
                    self.logger.warning("[V3.2.0+dev.20260122.03] 完成分离缓存失败: %s", e)

        self.logger.info(f"预处理流程完成: {len(chunks)} 个chunk准备就绪")

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
            progress_callback=progress_callback
        )

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

    def get_statistics(self, chunks: List[AudioChunk]) -> dict:
        """
        获取预处理统计信息

        Args:
            chunks: 已处理的 AudioChunk 列表

        Returns:
            dict: 统计信息
        """
        total = len(chunks)

        # 频谱分诊统计
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
