"""
SeparationStage - 人声分离阶段

负责根据分离模式和频谱分诊结果，执行人声分离。
支持全局分离和按需分离两种模式。

v3.1.0 更新：
- 集成 CancellationToken 支持暂停/取消
- 全局模式：整轨分离为原子操作
- 按需模式：逐 Chunk 可中断

V3.1.2+dev.20260109.01 更新：
- 添加命令行进度条支持（tqdm）
"""

import asyncio
import logging
from typing import List, Optional, TYPE_CHECKING, Tuple
from pathlib import Path

import numpy as np
import librosa

from app.services.audio.chunk_engine import AudioChunk
from app.services.demucs_service import DemucsService, get_demucs_service
from app.models.circuit_breaker_models import SeparationLevel

# v3.1.0: 导入取消令牌
if TYPE_CHECKING:
    from app.utils.cancellation_token import CancellationToken
    from app.services.preprocess_cache_service import PreprocessCacheService

# V3.1.2+dev.20260109.01: 导入 tqdm 进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class SeparationStage:
    """
    人声分离阶段

    职责：
    - 根据分离模式执行人声分离
    - 全局分离模式：整轨分离
    - 按需分离模式：仅分离标记的chunk
    - 保存原始音频用于熔断回溯

    v3.1.0: 支持 CancellationToken 实现暂停/取消/断点续传
    原子单位：
    - 全局模式：整轨分离为原子操作
    - 按需模式：单个 Chunk 分离，可在每个 Chunk 之间中断
    """

    def __init__(
        self,
        mode: str = 'on_demand',
        demucs_service: Optional[DemucsService] = None,
        logger: Optional[logging.Logger] = None,
        cancellation_token: Optional["CancellationToken"] = None,  # v3.1.0: 新增
        show_progress: bool = True  # V3.1.2+dev.20260109.01: 新增，显示命令行进度条
    ):
        """
        初始化人声分离阶段

        Args:
            mode: 分离模式，'global' 或 'on_demand'
            demucs_service: Demucs服务实例，如果为None则使用全局单例
            logger: 日志记录器，如果为None则创建新的
            cancellation_token: 取消令牌（可选，v3.1.0）
            show_progress: 是否显示命令行进度条（默认 True，V3.1.2+dev.20260109.01）
        """
        self.mode = mode
        self.demucs_service = demucs_service or get_demucs_service()
        self.logger = logger or logging.getLogger(__name__)
        self.cancellation_token = cancellation_token  # v3.1.0
        self.show_progress = show_progress  # V3.1.2+dev.20260109.01

        if mode not in ['global', 'on_demand']:
            raise ValueError(f"不支持的分离模式: {mode}，仅支持 'global' 或 'on_demand'")

    async def process(
        self,
        chunks: List[AudioChunk],
        audio_path: Optional[str] = None,
        job_dir: Optional[Path] = None,  # v3.1.0: 用于保存检查点
        separated_indices: Optional[set] = None,  # v3.1.0: 已分离的索引（用于恢复）
        cache_service: Optional["PreprocessCacheService"] = None
    ) -> List[AudioChunk]:
        """
        执行人声分离

        Args:
            chunks: 待处理的AudioChunk列表
            audio_path: 原始音频文件路径（全局分离模式需要）
            job_dir: 任务目录（可选，v3.1.0 用于保存检查点）
            separated_indices: 已分离的chunk索引集合（可选，v3.1.0 用于恢复）

        Returns:
            处理后的AudioChunk列表
        """
        if not chunks:
            self.logger.warning("收到空的chunk列表，跳过人声分离")
            return chunks

        if cache_service:
            cache_service.begin_separation(self.mode, len(chunks))

        if self.mode == 'global':
            return await self._process_global(chunks, audio_path, job_dir, cache_service)
        else:
            return await self._process_on_demand(chunks, job_dir, separated_indices, cache_service)

    async def _process_global(
        self,
        chunks: List[AudioChunk],
        audio_path: str,
        job_dir: Optional[Path] = None,  # v3.1.0
        cache_service: Optional["PreprocessCacheService"] = None
    ) -> List[AudioChunk]:
        """
        全局分离模式

        将 Chunk 音频按组拼接后分离，再切片回填到各 Chunk。

        v3.1.0: 整轨分离为原子操作，不可中断

        Args:
            chunks: AudioChunk列表
            audio_path: 原始音频文件路径
            job_dir: 任务目录（可选，v3.1.0）

        Returns:
            更新后的AudioChunk列表
        """
        if not audio_path:
            raise ValueError("全局分离模式需要提供audio_path参数")

        self.logger.info(f"执行全局人声分离: {audio_path}")

        token = self.cancellation_token  # v3.1.0

        # v3.1.0: 进入原子区域（整轨分离不可中断）
        if token:
            token.enter_atomic_region("demucs_global_separation")

        try:
            runtime_model = self.demucs_service.config.model_name
            try:
                from app.services.runtime_param_resolver import get_demucs_runtime_params

                runtime_demucs = get_demucs_runtime_params()
                runtime_model = str(runtime_demucs.get("model_name") or runtime_model)
            except Exception:
                runtime_model = self.demucs_service.config.model_name

            group_duration_sec = self._resolve_global_group_duration_sec()
            groups = self._build_global_groups(
                chunks=chunks,
                max_group_duration_sec=group_duration_sec,
            )
            self.logger.info(
                "全局分离分组执行: groups=%s group_duration_sec=%.1f",
                len(groups),
                group_duration_sec,
            )

            for group_index, group_chunks in enumerate(groups):
                concatenated_audio, chunk_spans = self._concat_group_audio(group_chunks)
                if concatenated_audio.size == 0 or not chunk_spans:
                    continue
                sample_rate = int(group_chunks[0].sample_rate) if group_chunks else 16000
                separated_audio = await asyncio.to_thread(
                    self.demucs_service.separate_chunk,
                    concatenated_audio,
                    runtime_model,
                    sample_rate,
                )
                self._apply_group_separation_result(
                    chunk_spans=chunk_spans,
                    separated_audio=separated_audio,
                    separation_model=runtime_model,
                )
                self.logger.debug(
                    "全局分离分组完成: group_index=%s chunk_count=%s audio_sec=%.2f",
                    group_index,
                    len(group_chunks),
                    float(len(concatenated_audio)) / float(max(sample_rate, 1)),
                )
            if cache_service:
                model_name = self.demucs_service.get_loaded_model_name() or self.demucs_service.config.model_name
                try:
                    cache_service.save_global_separation(
                        chunks=chunks,
                        separated_path="",
                        separation_model=model_name
                    )
                except Exception as e:
                    self.logger.warning("[V3.2.0+dev.20260122.03] 保存全局分离缓存失败: %s", e)

        finally:
            # v3.1.0: 退出原子区域
            if token:
                has_pending = token.exit_atomic_region()
                if has_pending:
                    self.logger.info("[v3.1.0] 检测到待处理的暂停/取消请求")

        if token and cache_service and token.is_paused:
            try:
                cache_service.reset_separation_cache()
                self.logger.info("[V3.2.0+dev.20260122.03] 全局分离暂停，已清理分离缓存")
            except Exception as e:
                self.logger.warning("[V3.2.0+dev.20260122.03] 清理分离缓存失败: %s", e)

        # v3.1.0: 原子区域结束后检查暂停/取消
        if token and job_dir:
            checkpoint_data = {
                "separation": {
                    "mode": "global",
                    "global_separation_done": True,
                    "separated_indices": list(range(len(chunks)))
                }
            }
            token.check_and_save(checkpoint_data, job_dir)

        return chunks

    def _resolve_global_group_duration_sec(self) -> float:
        """读取全局分组时长配置。"""
        default_duration = 1800.0
        try:
            from app.services.runtime_param_resolver import get_demucs_runtime_params

            runtime_demucs = get_demucs_runtime_params()
            raw_value = runtime_demucs.get("global_group_duration_sec", default_duration)
            duration = float(raw_value)
            return max(60.0, duration)
        except Exception:
            return default_duration

    @staticmethod
    def _build_global_groups(
        chunks: List[AudioChunk],
        max_group_duration_sec: float,
    ) -> List[List[AudioChunk]]:
        """按累计语音时长分组，避免整轨一次性分离导致显存峰值过高。"""
        if not chunks:
            return []

        groups: List[List[AudioChunk]] = []
        current_group: List[AudioChunk] = []
        current_duration = 0.0

        for chunk in sorted(chunks, key=lambda item: item.index):
            chunk_duration = max(0.0, float(chunk.end - chunk.start))
            chunk_sample_rate = int(getattr(chunk, "sample_rate", 16000) or 16000)
            if (
                current_group
                and (
                    current_duration + chunk_duration > max_group_duration_sec
                    or int(current_group[-1].sample_rate) != chunk_sample_rate
                )
            ):
                groups.append(current_group)
                current_group = []
                current_duration = 0.0

            current_group.append(chunk)
            current_duration += chunk_duration

        if current_group:
            groups.append(current_group)

        return groups

    @staticmethod
    def _concat_group_audio(
        chunks: List[AudioChunk],
    ) -> Tuple[np.ndarray, List[Tuple[AudioChunk, int, int]]]:
        """将分组内 Chunk 音频拼接，并返回每个 Chunk 的切片范围。"""
        parts: List[np.ndarray] = []
        spans: List[Tuple[AudioChunk, int, int]] = []
        offset = 0
        for chunk in chunks:
            chunk_audio = np.asarray(chunk.audio, dtype=np.float32).reshape(-1)
            if chunk_audio.size == 0:
                continue
            start = offset
            end = start + int(chunk_audio.size)
            spans.append((chunk, start, end))
            parts.append(chunk_audio)
            offset = end
        if not parts:
            return np.zeros(0, dtype=np.float32), []
        return np.concatenate(parts), spans

    def _apply_group_separation_result(
        self,
        *,
        chunk_spans: List[Tuple[AudioChunk, int, int]],
        separated_audio: np.ndarray,
        separation_model: str,
    ) -> None:
        """将分组分离结果回填到 Chunk，并执行能量回退保护。"""
        if not chunk_spans:
            return

        expected_size = chunk_spans[-1][2]
        merged_audio = np.asarray(separated_audio, dtype=np.float32).reshape(-1)
        if merged_audio.size < expected_size:
            merged_audio = np.pad(merged_audio, (0, expected_size - merged_audio.size))
        elif merged_audio.size > expected_size:
            merged_audio = merged_audio[:expected_size]

        for chunk, start, end in chunk_spans:
            if chunk.original_audio is None:
                chunk.original_audio = np.asarray(chunk.audio, dtype=np.float32).copy()

            separated_chunk_audio = merged_audio[start:end]
            original_chunk_audio = np.asarray(chunk.original_audio, dtype=np.float32)
            sep_rms = float(np.sqrt(np.mean(separated_chunk_audio ** 2))) if separated_chunk_audio.size else 0.0
            orig_rms = float(np.sqrt(np.mean(original_chunk_audio ** 2))) if original_chunk_audio.size else 0.0

            if sep_rms < orig_rms * 0.2 and orig_rms > 1e-3:
                chunk.audio = original_chunk_audio
                chunk.is_separated = False
                chunk.separation_level = SeparationLevel.NONE
                chunk.separation_model = None
                continue

            chunk.audio = separated_chunk_audio
            chunk.is_separated = True
            chunk.separation_level = (
                SeparationLevel.MDX_EXTRA
                if separation_model == SeparationLevel.MDX_EXTRA.value
                else SeparationLevel.HTDEMUCS
            )
            chunk.separation_model = separation_model

    async def _process_on_demand(
        self,
        chunks: List[AudioChunk],
        job_dir: Optional[Path] = None,  # v3.1.0
        separated_indices: Optional[set] = None,  # v3.1.0
        cache_service: Optional["PreprocessCacheService"] = None
    ) -> List[AudioChunk]:
        """
        按需分离模式

        仅对标记为needs_separation=True的chunk进行分离

        v3.1.0: 逐 Chunk 可中断，每个 Chunk 分离后保存检查点
        V3.1.2+dev.20260109.01: 添加命令行进度条支持

        Args:
            chunks: AudioChunk列表
            job_dir: 任务目录（可选，v3.1.0）
            separated_indices: 已分离的chunk索引集合（可选，v3.1.0）

        Returns:
            更新后的AudioChunk列表
        """
        need_sep_chunks = [c for c in chunks if c.needs_separation]
        self.logger.info(
            f"按需分离: {len(need_sep_chunks)}/{len(chunks)} 个chunk需要分离"
        )

        if not need_sep_chunks:
            self.logger.info("无需分离的chunk，跳过分离阶段")
            return chunks

        token = self.cancellation_token  # v3.1.0
        separated_indices = separated_indices or set()
        runtime_model = None
        try:
            from app.services.runtime_param_resolver import get_demucs_runtime_params

            runtime_demucs = get_demucs_runtime_params()
            runtime_model = runtime_demucs.get("model_name")
        except Exception:
            runtime_model = None

        # V3.1.2+dev.20260109.01: 创建进度条
        iterator = need_sep_chunks
        if self.show_progress and TQDM_AVAILABLE:
            iterator = tqdm(
                need_sep_chunks,
                desc="人声分离",
                unit="chunk",
                ncols=80,
                leave=False
            )

        # 逐个分离需要分离的chunk
        for chunk in iterator:
            # v3.1.0: 跳过已分离的 chunk（用于恢复）
            if chunk.index in separated_indices:
                self.logger.debug(f"跳过已分离的 chunk {chunk.index}")
                continue

            # v3.1.0: 进入原子区域（单个 Chunk 分离）
            if token:
                token.enter_atomic_region(f"demucs_chunk_{chunk.index}")

            try:
                # 保存原始音频（用于熔断回溯）
                if chunk.original_audio is None:
                    chunk.original_audio = chunk.audio.copy()

                # 选择分离模型
                model = chunk.recommended_model or runtime_model or 'htdemucs'

                self.logger.debug(
                    f"分离 Chunk {chunk.index}: 使用模型 {model}"
                )

                # 执行分离
                separated_audio = await self._separate_chunk(
                    chunk.audio,
                    chunk.sample_rate,
                    model
                )

                # 更新chunk
                chunk.audio = separated_audio
                chunk.is_separated = True
                chunk.separation_level = (
                    SeparationLevel.MDX_EXTRA
                    if model == SeparationLevel.MDX_EXTRA.value
                    else SeparationLevel.HTDEMUCS
                )
                chunk.separation_model = model
                if cache_service:
                    try:
                        cache_service.save_separation_chunk(chunk, is_separated=True)
                    except Exception as e:
                        self.logger.warning("[V3.2.0+dev.20260122.03] 保存分离缓存失败: %s", e)

            except Exception as e:
                self.logger.error(
                    f"分离 Chunk {chunk.index} 失败: {e}，保持原始音频"
                )
                chunk.is_separated = False
                chunk.separation_level = SeparationLevel.NONE
                chunk.separation_model = None
                if cache_service:
                    try:
                        cache_service.save_separation_chunk(chunk, is_separated=False)
                    except Exception as e:
                        self.logger.warning("[V3.2.0+dev.20260122.03] 保存分离缓存失败: %s", e)
                continue

            finally:
                # v3.1.0: 退出原子区域
                if token:
                    has_pending = token.exit_atomic_region()
                    if has_pending:
                        self.logger.info(f"[v3.1.0] Chunk {chunk.index} 分离完成后检测到待处理请求")

            # v3.1.0: 每个 Chunk 分离完成后检查暂停/取消并保存检查点
            if token and job_dir:
                separated_indices.add(chunk.index)
                checkpoint_data = {
                    "separation": {
                        "mode": "on_demand",
                        "separated_indices": list(separated_indices),
                        "separated_count": len(separated_indices)
                    }
                }
                token.check_and_save(checkpoint_data, job_dir)

        # 统计
        separated_count = sum(1 for c in chunks if c.is_separated)
        self.logger.info(f"按需分离完成: {separated_count} 个chunk已分离")

        return chunks

    async def _separate_chunk(
        self,
        audio: np.ndarray,
        sr: int,
        model: str
    ) -> np.ndarray:
        """
        分离单个chunk的人声

        Args:
            audio: 音频数组
            sr: 采样率
            model: 分离模型名称

        Returns:
            分离后的人声音频数组
        """
        # 调用DemucsService的chunk级别分离方法
        separated_audio = self.demucs_service.separate_chunk(
            audio=audio,
            model=model,
            sr=sr
        )

        self.logger.debug(f"Chunk分离完成，模型={model}")
        return separated_audio

    def get_statistics(self, chunks: List[AudioChunk]) -> dict:
        """
        获取分离统计信息

        Args:
            chunks: 已处理的AudioChunk列表

        Returns:
            统计信息字典
        """
        total = len(chunks)
        separated = sum(1 for c in chunks if c.is_separated)
        htdemucs = sum(
            1 for c in chunks
            if c.is_separated and c.separation_level == SeparationLevel.HTDEMUCS
        )
        mdx_extra = sum(
            1 for c in chunks
            if c.is_separated and c.separation_level == SeparationLevel.MDX_EXTRA
        )

        return {
            "total_chunks": total,
            "separated": separated,
            "not_separated": total - separated,
            "htdemucs_count": htdemucs,
            "mdx_extra_count": mdx_extra,
            "separation_ratio": separated / total if total > 0 else 0.0
        }

    async def _load_separated_audio_to_chunks(
        self,
        chunks: List[AudioChunk],
        separated_path: str
    ) -> None:
        """
        V3.1.1+dev.20260107.06: 加载分离后的音频并更新所有chunk

        根据每个chunk的时间戳，从分离后的音频文件中提取对应片段，
        并应用能量检测自动回退机制。

        Args:
            chunks: AudioChunk列表
            separated_path: 分离后的音频文件路径
        """
        # 加载分离后的音频（在线程中执行，避免阻塞）
        separated_audio, sep_sr = await asyncio.to_thread(
            librosa.load,
            separated_path,
            sr=None,  # 保持原始采样率
            mono=True
        )

        self.logger.info(
            f"加载分离后音频: {separated_path}, "
            f"采样率={sep_sr}, 时长={len(separated_audio)/sep_sr:.2f}s"
        )

        # 统计
        fallback_count = 0
        success_count = 0

        for chunk in chunks:
            # 保存原始音频（用于能量检测和熔断回溯）
            if chunk.original_audio is None:
                chunk.original_audio = chunk.audio.copy()

            # 根据时间戳提取对应片段
            # 注意：chunk的时间戳是基于原始音频的，需要转换到分离后音频的采样率
            start_sample = int(chunk.start * sep_sr)
            end_sample = int(chunk.end * sep_sr)

            # 边界检查
            start_sample = max(0, start_sample)
            end_sample = min(len(separated_audio), end_sample)

            if start_sample >= end_sample:
                self.logger.warning(
                    f"Chunk {chunk.index}: 时间戳越界，跳过 "
                    f"(start={chunk.start:.2f}s, end={chunk.end:.2f}s)"
                )
                continue

            # 提取分离后的音频片段
            sep_chunk_audio = separated_audio[start_sample:end_sample]

            # 重采样到chunk的采样率（如果不同）
            if sep_sr != chunk.sample_rate:
                sep_chunk_audio = librosa.resample(
                    sep_chunk_audio,
                    orig_sr=sep_sr,
                    target_sr=chunk.sample_rate
                )

            # 能量检测：如果分离后能量显著低于原始，自动回退
            sep_rms = np.sqrt(np.mean(sep_chunk_audio**2))
            orig_rms = np.sqrt(np.mean(chunk.original_audio**2))

            # 阈值：分离后 RMS < 原始 RMS * 20% 且原始 RMS > 1e-3
            if sep_rms < orig_rms * 0.2 and orig_rms > 1e-3:
                self.logger.warning(
                    f"Chunk {chunk.index}: Demucs 分离失败，能量过低 "
                    f"(sep_rms={sep_rms:.6f}, orig_rms={orig_rms:.6f}, "
                    f"ratio={sep_rms/orig_rms*100:.1f}%)，回退到原始音频"
                )
                # 回退到原始音频，但仍标记为已处理
                chunk.is_separated = False
                fallback_count += 1
            else:
                # 更新chunk音频为分离后的音频
                chunk.audio = sep_chunk_audio
                chunk.is_separated = True
                chunk.separation_level = SeparationLevel.HTDEMUCS
                chunk.separation_model = 'htdemucs'
                success_count += 1

                self.logger.debug(
                    f"Chunk {chunk.index}: 分离成功 "
                    f"(sep_rms={sep_rms:.6f}, orig_rms={orig_rms:.6f})"
                )

        self.logger.info(
            f"全局分离音频加载完成: 成功={success_count}, "
            f"回退={fallback_count}, 总计={len(chunks)}"
        )

