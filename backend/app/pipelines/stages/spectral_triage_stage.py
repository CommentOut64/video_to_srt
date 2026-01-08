"""
SpectralTriageStage - 频谱分诊阶段

负责为每个AudioChunk进行频谱分析，判断是否需要人声分离，并推荐合适的模型。

V3.7 更新：
- 集成 CancellationToken 支持暂停/取消
- 支持逐 Chunk 中断和检查点保存

V3.1.1+dev.20260108.02 更新：
- 支持 use_snr_triage 配置参数
- 默认启用 SNR+C50 三层决策策略
"""

import logging
from typing import List, Optional, TYPE_CHECKING
from pathlib import Path

from app.services.audio.chunk_engine import AudioChunk
from app.services.audio_spectrum_classifier import AudioSpectrumClassifier, get_spectrum_classifier

# V3.7: 导入取消令牌
if TYPE_CHECKING:
    from app.utils.cancellation_token import CancellationToken


class SpectralTriageStage:
    """
    频谱分诊阶段

    职责：
    - 为每个AudioChunk进行频谱分析
    - 判断是否需要人声分离
    - 推荐合适的分离模型（htdemucs/mdx_extra）

    V3.7: 支持 CancellationToken 实现暂停/取消/断点续传
    原子单位：单个 Chunk 分诊，可在每个 Chunk 之间中断

    V3.1.1+dev.20260108.02: 支持 SNR+C50 三层决策策略配置
    """

    def __init__(
        self,
        classifier: Optional[AudioSpectrumClassifier] = None,
        threshold: float = 0.35,
        logger: Optional[logging.Logger] = None,
        cancellation_token: Optional["CancellationToken"] = None,  # V3.7: 新增
        use_snr_triage: bool = True  # V3.1.1+dev.20260108.02: 新增，默认启用
    ):
        """
        初始化频谱分诊阶段

        Args:
            classifier: 频谱分类器实例，如果为None则使用全局单例
            threshold: 分诊阈值，默认0.35
            logger: 日志记录器，如果为None则创建新的
            cancellation_token: 取消令牌（可选，V3.7）
            use_snr_triage: 是否启用 SNR+C50 三层决策策略（默认 True，V3.1.1+dev.20260108.02）
        """
        # V3.1.1+dev.20260108.02: 根据配置决定是否启用 SNR 策略
        self.classifier = classifier or get_spectrum_classifier(use_snr_strategy=use_snr_triage)
        self.threshold = threshold
        self.logger = logger or logging.getLogger(__name__)
        self.cancellation_token = cancellation_token  # V3.7
        self.use_snr_triage = use_snr_triage  # V3.1.1+dev.20260108.02

    async def process(
        self,
        chunks: List[AudioChunk],
        job_dir: Optional[Path] = None,  # V3.7: 用于保存检查点
        diagnosed_indices: Optional[set] = None  # V3.7: 已诊断的索引（用于恢复）
    ) -> List[AudioChunk]:
        """
        批量分诊所有chunk

        Args:
            chunks: 待分诊的AudioChunk列表
            job_dir: 任务目录（可选，V3.7 用于保存检查点）
            diagnosed_indices: 已诊断的chunk索引集合（可选，V3.7 用于恢复）

        Returns:
            带有分诊结果标记的AudioChunk列表
        """
        if not chunks:
            self.logger.warning("收到空的chunk列表，跳过频谱分诊")
            return chunks

        self.logger.info(f"开始频谱分诊，共 {len(chunks)} 个chunk")

        token = self.cancellation_token  # V3.7: 简化引用
        diagnosed_indices = diagnosed_indices or set()
        sample_rate = chunks[0].sample_rate if chunks else 16000

        # V3.7: 逐个处理 chunk，支持中断
        for i, chunk in enumerate(chunks):
            # V3.7: 跳过已诊断的 chunk（用于恢复）
            if i in diagnosed_indices:
                self.logger.debug(f"跳过已诊断的 chunk {i}")
                continue

            # V3.7: 单个 chunk 分诊（快速，不需要原子区域）
            diagnosis = self.classifier.diagnose_chunk(
                audio=chunk.audio,
                chunk_index=i,
                sr=sample_rate
            )

            # 设置分诊结果
            chunk.needs_separation = diagnosis.need_separation
            chunk.recommended_model = diagnosis.recommended_model
            chunk.spectrum_diagnosis = diagnosis

            # V3.7: 每个 chunk 之间检查取消/暂停并保存检查点
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

        return chunks

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
