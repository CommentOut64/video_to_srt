"""
SpectralTriageStage - 频谱分诊阶段

负责为每个AudioChunk进行频谱分析，判断是否需要人声分离，并推荐合适的模型。

v3.1.0 更新：
- 集成 CancellationToken 支持暂停/取消
- 支持逐 Chunk 中断和检查点保存

V3.1.1+dev.20260108.02 更新：
- 支持 use_snr_triage 配置参数
- 默认启用 SNR+C50 三层决策策略

V3.1.1+dev.20260108.06 更新：
- 新增分诊详细日志导出功能
- 保存每个 Chunk 的 SNR/C50/决策层级等信息到任务目录

V3.1.2+dev.20260109.01 更新：
- 新增智能探针模式（SmartProbeService）
- 添加命令行进度条支持（tqdm）
"""

import logging
import json
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING, Dict, Any
from pathlib import Path

from app.services.audio.chunk_engine import AudioChunk
from app.services.audio_spectrum_classifier import AudioSpectrumClassifier, get_spectrum_classifier

# v3.1.0: 导入取消令牌
if TYPE_CHECKING:
    from app.utils.cancellation_token import CancellationToken

# V3.1.2+dev.20260109.01: 导入 tqdm 进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class SpectralTriageStage:
    """
    频谱分诊阶段

    职责：
    - 为每个AudioChunk进行频谱分析
    - 判断是否需要人声分离
    - 推荐合适的分离模型（htdemucs/mdx_extra）

    v3.1.0: 支持 CancellationToken 实现暂停/取消/断点续传
    原子单位：单个 Chunk 分诊，可在每个 Chunk 之间中断

    V3.1.1+dev.20260108.02: 支持 SNR+C50 三层决策策略配置
    """

    def __init__(
        self,
        classifier: Optional[AudioSpectrumClassifier] = None,
        threshold: float = 0.35,
        logger: Optional[logging.Logger] = None,
        cancellation_token: Optional["CancellationToken"] = None,  # v3.1.0: 新增
        use_snr_triage: bool = True,  # V3.1.1+dev.20260108.02: 新增，默认启用
        use_smart_probe: bool = True,  # V3.1.2+dev.20260109.01: 新增，智能探针模式（默认启用）
        show_progress: bool = True  # V3.1.2+dev.20260109.01: 新增，显示命令行进度条
    ):
        """
        初始化频谱分诊阶段

        Args:
            classifier: 频谱分类器实例，如果为None则使用全局单例
            threshold: 分诊阈值，默认0.35
            logger: 日志记录器，如果为None则创建新的
            cancellation_token: 取消令牌（可选，v3.1.0）
            use_snr_triage: 是否启用 SNR+C50 三层决策策略（默认 True，V3.1.1+dev.20260108.02）
            use_smart_probe: 是否启用智能探针模式（默认 True，V3.1.2+dev.20260109.01）
            show_progress: 是否显示命令行进度条（默认 True，V3.1.2+dev.20260109.01）
        """
        # V3.1.1+dev.20260108.02: 根据配置决定是否启用 SNR 策略
        self.classifier = classifier or get_spectrum_classifier(use_snr_strategy=use_snr_triage)
        self.threshold = threshold
        self.logger = logger or logging.getLogger(__name__)
        self.cancellation_token = cancellation_token  # v3.1.0
        self.use_snr_triage = use_snr_triage  # V3.1.1+dev.20260108.02
        self.use_smart_probe = use_smart_probe  # V3.1.2+dev.20260109.01
        self.show_progress = show_progress  # V3.1.2+dev.20260109.01
        self._smart_probe = None  # 懒加载
        self._last_triage_log: List[Dict[str, Any]] = []
        self._last_probe_state: Optional[Dict[str, Any]] = None

    def _get_smart_probe(self):
        """
        获取智能探针服务实例（懒加载）

        V3.1.2+dev.20260109.01: 新增
        """
        if self._smart_probe is None and self.use_smart_probe:
            try:
                from app.services.smart_probe_service import get_smart_probe_service
                self._smart_probe = get_smart_probe_service()
                self.logger.info("智能探针服务已加载")
            except Exception as e:
                self.logger.warning(f"智能探针服务加载失败: {e}")
                self._smart_probe = None
        return self._smart_probe

    def get_smart_probe_params(self) -> Optional[Dict[str, Any]]:
        """获取智能探针当前参数"""
        smart_probe = self._get_smart_probe()
        if smart_probe is None:
            return None
        return {
            "snr_threshold": smart_probe.threshold,
            "max_step_chunks": smart_probe.max_step,
        }

    def get_cache_snapshot(self) -> Dict[str, Any]:
        """获取最近一次分诊缓存快照"""
        return {
            "triage_log": self._last_triage_log,
            "probe_state": self._last_probe_state,
        }

    async def process(
        self,
        chunks: List[AudioChunk],
        job_dir: Optional[Path] = None,  # v3.1.0: 用于保存检查点
        diagnosed_indices: Optional[set] = None  # v3.1.0: 已诊断的索引（用于恢复）
    ) -> List[AudioChunk]:
        """
        批量分诊所有chunk

        V3.1.2+dev.20260109.01: 支持智能探针模式和命令行进度条

        Args:
            chunks: 待分诊的AudioChunk列表
            job_dir: 任务目录（可选，v3.1.0 用于保存检查点）
            diagnosed_indices: 已诊断的chunk索引集合（可选，v3.1.0 用于恢复）

        Returns:
            带有分诊结果标记的AudioChunk列表
        """
        if not chunks:
            self.logger.warning("收到空的chunk列表，跳过频谱分诊")
            return chunks

        self.logger.info(f"开始频谱分诊，共 {len(chunks)} 个chunk")

        token = self.cancellation_token  # v3.1.0: 简化引用
        diagnosed_indices = diagnosed_indices or set()
        sample_rate = chunks[0].sample_rate if chunks else 16000

        # V3.1.1+dev.20260108.06: 收集分诊详细信息
        triage_log = []

        # V3.1.2+dev.20260109.01: 智能探针模式
        if self.use_smart_probe:
            return await self._process_with_smart_probe(
                chunks, job_dir, diagnosed_indices, triage_log
            )

        # V3.1.2+dev.20260109.01: 标准模式 + 进度条
        return await self._process_standard(
            chunks, job_dir, diagnosed_indices, triage_log, sample_rate, token
        )

    async def _process_standard(
        self,
        chunks: List[AudioChunk],
        job_dir: Optional[Path],
        diagnosed_indices: set,
        triage_log: list,
        sample_rate: int,
        token
    ) -> List[AudioChunk]:
        """
        标准分诊模式（逐个处理）

        V3.1.2+dev.20260109.01: 新增，支持命令行进度条
        """
        # V3.1.2+dev.20260109.01: 创建进度条
        iterator = enumerate(chunks)
        if self.show_progress and TQDM_AVAILABLE:
            iterator = enumerate(tqdm(
                chunks,
                desc="频谱分诊",
                unit="chunk",
                ncols=80,
                leave=False
            ))

        # v3.1.0: 逐个处理 chunk，支持中断
        for i, chunk in iterator:
            # v3.1.0: 跳过已诊断的 chunk（用于恢复）
            if i in diagnosed_indices:
                self.logger.debug(f"跳过已诊断的 chunk {i}")
                continue

            # v3.1.0: 单个 chunk 分诊（快速，不需要原子区域）
            diagnosis = self.classifier.diagnose_chunk(
                audio=chunk.audio,
                chunk_index=i,
                sr=sample_rate
            )

            # 设置分诊结果
            chunk.needs_separation = diagnosis.need_separation
            chunk.recommended_model = diagnosis.recommended_model
            chunk.spectrum_diagnosis = diagnosis

            # V3.1.1+dev.20260108.06: 收集分诊详细信息
            chunk_log = {
                "chunk_index": i,
                "start_time": round(chunk.start, 3),
                "end_time": round(chunk.end, 3),
                "duration": round(chunk.end - chunk.start, 3),
                "need_separation": diagnosis.need_separation,
                "recommended_model": diagnosis.recommended_model,
                "reason": diagnosis.reason,
                "snr": round(diagnosis.snr, 2) if diagnosis.snr is not None else None,
                "c50": round(diagnosis.c50, 2) if diagnosis.c50 is not None else None,
                "snr_level": diagnosis.snr_level,
                "c50_level": diagnosis.c50_level,
                "triage_layer": diagnosis.triage_layer,
            }
            triage_log.append(chunk_log)

            # v3.1.0: 每个 chunk 之间检查取消/暂停并保存检查点
            if token and job_dir:
                # 每 5 个 chunk 保存一次检查点（避免频繁 I/O）
                if (i + 1) % 5 == 0 or i == len(chunks) - 1:
                    diagnosed_indices.add(i)
                    checkpoint_data = {
                        "spectral_triage": {
                            "diagnosed_indices": list(diagnosed_indices),
                            "diagnosed_count": len(diagnosed_indices),
                            "total_chunks": len(chunks)
                        }
                    }
                    token.check_and_save(checkpoint_data, job_dir)

        # 保存缓存快照
        self._last_triage_log = triage_log
        self._last_probe_state = None

        # 统计和日志保存
        return self._finalize_triage(chunks, triage_log, job_dir)

    async def _process_with_smart_probe(
        self,
        chunks: List[AudioChunk],
        job_dir: Optional[Path],
        diagnosed_indices: set,
        triage_log: list
    ) -> List[AudioChunk]:
        """
        智能探针分诊模式

        V3.1.2+dev.20260109.01: 新增
        使用中心扩散探针策略快速判断是否需要全量分离
        """
        smart_probe = self._get_smart_probe()
        if smart_probe is None:
            self.logger.warning("智能探针服务不可用，回退到标准模式")
            return await self._process_standard(
                chunks, job_dir, diagnosed_indices, triage_log,
                chunks[0].sample_rate if chunks else 16000,
                self.cancellation_token
            )

        self.logger.info(f"使用智能探针模式进行分诊，共 {len(chunks)} 个chunk")

        # V3.1.2+dev.20260109.01: 创建进度条回调
        pbar = None
        if self.show_progress and TQDM_AVAILABLE:
            pbar = tqdm(
                total=len(chunks),
                desc="智能探针分诊",
                unit="chunk",
                ncols=80,
                leave=False
            )

        def progress_callback(current, total):
            if pbar:
                pbar.n = current
                pbar.refresh()

        # 执行智能探针检测
        decision, cache, sequence = smart_probe.run_probe(chunks, progress_callback)
        self._last_probe_state = {
            "decision": decision,
            "snr_threshold": smart_probe.threshold,
            "max_step_chunks": smart_probe.max_step,
            "sequence": sequence,
            "cache": cache,
            "cursor": len(sequence),
        }

        if pbar:
            pbar.close()

        # 根据探针结果标记所有 chunks
        if decision == "SEPARATE_ALL":
            self.logger.info("智能探针判定：需要全量分离")
            for chunk in chunks:
                chunk.needs_separation = True
                chunk.recommended_model = "htdemucs"
                # 如果该 chunk 已被探针检测，使用缓存的结果
                if chunk.index in cache:
                    result = cache[chunk.index]
                    triage_log.append({
                        "chunk_index": chunk.index,
                        "start_time": round(chunk.start, 3),
                        "end_time": round(chunk.end, 3),
                        "duration": round(chunk.end - chunk.start, 3),
                        "need_separation": True,
                        "recommended_model": "htdemucs",
                        "reason": f"[智能探针] 全量分离 (SNR={result['snr']:.1f}dB)",
                        "snr": round(result['snr'], 2),
                        "c50": round(result['c50'], 2),
                        "snr_level": None,
                        "c50_level": None,
                        "triage_layer": 0,  # 探针模式
                    })
        else:
            self.logger.info("智能探针判定：纯净视频，无需分离")
            for chunk in chunks:
                chunk.needs_separation = False
                chunk.recommended_model = None
                # 如果该 chunk 已被探针检测，使用缓存的结果
                if chunk.index in cache:
                    result = cache[chunk.index]
                    triage_log.append({
                        "chunk_index": chunk.index,
                        "start_time": round(chunk.start, 3),
                        "end_time": round(chunk.end, 3),
                        "duration": round(chunk.end - chunk.start, 3),
                        "need_separation": False,
                        "recommended_model": None,
                        "reason": f"[智能探针] 纯净视频 (SNR={result['snr']:.1f}dB)",
                        "snr": round(result['snr'], 2),
                        "c50": round(result['c50'], 2),
                        "snr_level": None,
                        "c50_level": None,
                        "triage_layer": 0,  # 探针模式
                    })

        self._last_triage_log = triage_log
        return self._finalize_triage(chunks, triage_log, job_dir)

    def _finalize_triage(
        self,
        chunks: List[AudioChunk],
        triage_log: list,
        job_dir: Optional[Path]
    ) -> List[AudioChunk]:
        """
        完成分诊统计和日志保存

        V3.1.2+dev.20260109.01: 新增，统一处理统计和日志
        """
        # 统计
        need_sep_count = sum(1 for c in chunks if c.needs_separation)
        self.logger.info(
            f"频谱分诊完成: {need_sep_count}/{len(chunks)} 个chunk需要分离"
        )

        # 详细统计
        if need_sep_count > 0:
            htdemucs_count = sum(
                1 for c in chunks
                if c.needs_separation and c.recommended_model == 'htdemucs'
            )
            mdx_extra_count = sum(
                1 for c in chunks
                if c.needs_separation and c.recommended_model == 'mdx_extra'
            )
            self.logger.info(
                f"推荐模型分布: htdemucs={htdemucs_count}, mdx_extra={mdx_extra_count}"
            )

        # V3.1.1+dev.20260108.06: 保存分诊详细日志
        if job_dir and triage_log:
            self._save_triage_log(job_dir, triage_log, chunks)

        return chunks

    def _save_triage_log(self, job_dir: Path, triage_log: list, chunks: List[AudioChunk]):
        """
        保存分诊详细日志到任务目录

        V3.1.1+dev.20260108.06: 新增功能
        保存每个 Chunk 的 SNR/C50/决策层级等详细信息

        Args:
            job_dir: 任务目录
            triage_log: 分诊日志列表
            chunks: AudioChunk 列表
        """
        try:
            # 计算统计信息
            stats = self.get_statistics(chunks)

            # Layer 分布统计
            layer_stats = {
                "layer_1": sum(1 for log in triage_log if log.get("triage_layer") == 1),
                "layer_2": sum(1 for log in triage_log if log.get("triage_layer") == 2),
                "layer_3": sum(1 for log in triage_log if log.get("triage_layer") == 3),
                "layer_0": sum(1 for log in triage_log if log.get("triage_layer") == 0),
            }

            # SNR/C50 统计
            snr_values = [log["snr"] for log in triage_log if log.get("snr") is not None]
            c50_values = [log["c50"] for log in triage_log if log.get("c50") is not None]

            snr_stats = {}
            if snr_values:
                snr_stats = {
                    "min": round(min(snr_values), 2),
                    "max": round(max(snr_values), 2),
                    "avg": round(sum(snr_values) / len(snr_values), 2),
                }

            c50_stats = {}
            if c50_values:
                c50_stats = {
                    "min": round(min(c50_values), 2),
                    "max": round(max(c50_values), 2),
                    "avg": round(sum(c50_values) / len(c50_values), 2),
                }

            # 构建完整日志
            full_log = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_chunks": len(chunks),
                    "use_snr_triage": self.use_snr_triage,
                },
                "summary": {
                    **stats,
                    "layer_distribution": layer_stats,
                    "snr_statistics": snr_stats,
                    "c50_statistics": c50_stats,
                },
                "chunks": triage_log
            }

            # 保存到文件
            log_file = job_dir / "triage_log.json"
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(full_log, f, indent=2, ensure_ascii=False)

            self.logger.info(f"分诊详细日志已保存: {log_file}")

        except Exception as e:
            self.logger.warning(f"保存分诊日志失败: {e}")

    def get_statistics(self, chunks: List[AudioChunk]) -> dict:
        """
        获取分诊统计信息

        Args:
            chunks: 已分诊的AudioChunk列表

        Returns:
            统计信息字典
        """
        total = len(chunks)
        need_sep = sum(1 for c in chunks if c.needs_separation)
        htdemucs = sum(
            1 for c in chunks
            if c.needs_separation and c.recommended_model == 'htdemucs'
        )
        mdx_extra = sum(
            1 for c in chunks
            if c.needs_separation and c.recommended_model == 'mdx_extra'
        )

        return {
            "total_chunks": total,
            "need_separation": need_sep,
            "no_separation": total - need_sep,
            "recommended_htdemucs": htdemucs,
            "recommended_mdx_extra": mdx_extra,
            "separation_ratio": need_sep / total if total > 0 else 0.0
        }
