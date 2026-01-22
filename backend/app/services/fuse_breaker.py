"""
熔断决策器（v2.1 概念重构版）

职责：判断是否需要升级分离模型
触发条件：检测到 BGM/Noise 标签 + 低置信度
动作：回溯到原始音频，使用升级的模型重新分离

注意：Whisper复核不在此处理，那是后处理增强阶段的事
"""
import logging
from typing import Optional
from app.models.circuit_breaker_models import (
    FuseAction, FuseDecision, ChunkProcessState, SeparationLevel
)

logger = logging.getLogger(__name__)


class FuseBreaker:
    """
    熔断决策器（v2.1）

    仅负责判断是否需要升级分离模型
    """

    def __init__(
        self,
        fuse_confidence_threshold: float = 0.5,  # 低于此值且有BGM标签才考虑熔断
        bgm_tags: tuple = ('BGM', 'Noise', 'Music', 'Applause')  # 触发熔断的事件标签
    ):
        self.fuse_confidence_threshold = fuse_confidence_threshold
        self.bgm_tags = bgm_tags

    def should_fuse(
        self,
        chunk_state: ChunkProcessState,
        confidence: float,
        event_tag: Optional[str]
    ) -> FuseDecision:
        """
        判断是否需要熔断（升级分离模型）

        Args:
            chunk_state: Chunk 处理状态
            confidence: SenseVoice 转录置信度
            event_tag: SenseVoice 检测到的事件标签

        Returns:
            FuseDecision: 熔断决策
        """
        # 1. 置信度足够高，不需要熔断
        if confidence >= self.fuse_confidence_threshold:
            return FuseDecision(
                action=FuseAction.ACCEPT,
                reason=f"置信度 {confidence:.2f} >= {self.fuse_confidence_threshold}"
            )

        # 2. 没有 BGM/Noise 标签，不需要熔断（可能只是说话不清晰）
        if event_tag not in self.bgm_tags:
            return FuseDecision(
                action=FuseAction.ACCEPT,
                reason=f"无BGM/Noise标签，不触发熔断（置信度低可由后处理增强补救）"
            )

        # 3. 检查是否可以升级分离
        if not chunk_state.can_upgrade_separation():
            return FuseDecision(
                action=FuseAction.ACCEPT,
                reason=f"无法升级分离（已达止损点或最高级别），接受当前结果"
            )

        # 4. 触发熔断：升级分离模型
        next_level = chunk_state.get_next_separation_level()
        return FuseDecision(
            action=FuseAction.UPGRADE_SEPARATION,
            reason=f"检测到 {event_tag} + 低置信度 {confidence:.2f}，升级分离模型",
            next_separation_level=next_level
        )


def execute_fuse_upgrade(
    chunk_state: ChunkProcessState,
    next_level: SeparationLevel,
    demucs_service
) -> ChunkProcessState:
    """
    执行熔断升级：使用原始音频重新分离

    Args:
        chunk_state: Chunk 处理状态
        next_level: 目标分离级别
        demucs_service: Demucs 服务实例

    Returns:
        ChunkProcessState: 更新后的状态
    """
    logger.info(
        f"Chunk {chunk_state.chunk_index} 熔断升级: "
        f"{chunk_state.separation_level.value} → {next_level.value}"
    )

    # 关键：使用原始音频（分离前）进行重新分离
    original_audio = chunk_state.original_audio
    if original_audio is None:
        logger.error("熔断失败：原始音频引用丢失")
        return chunk_state

    # 选择模型
    model_name = "htdemucs" if next_level == SeparationLevel.HTDEMUCS else "mdx_extra"

    # 执行分离（使用 Chunk 的采样率）
    separated_audio = demucs_service.separate_chunk(
        audio=original_audio,
        model=model_name,
        sr=chunk_state.sample_rate
    )

    # 更新状态
    chunk_state.current_audio = separated_audio
    chunk_state.separation_level = next_level
    chunk_state.separation_model_used = model_name
    chunk_state.fuse_retry_count += 1

    logger.info(f"Chunk {chunk_state.chunk_index} 熔断升级完成，重试次数: {chunk_state.fuse_retry_count}")

    return chunk_state


# ========== 单例访问 ==========

_fuse_breaker_instance = None


def get_fuse_breaker() -> FuseBreaker:
    """获取熔断决策器单例"""
    global _fuse_breaker_instance
    if _fuse_breaker_instance is None:
        _fuse_breaker_instance = FuseBreaker()
    return _fuse_breaker_instance
