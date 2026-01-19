"""
AlignmentWorker - 对齐层 Worker（CPU）

职责：
1. 执行双流对齐（三级降级策略）
2. 分句 + 语义分组（使用慢流策略）
3. 推送定稿到 SSE（Chunk 级别批量替换）
4. 填充 ProcessingContext.final_sentences

特点：
- 质量优先（~0.5秒/Chunk）
- 三级降级：双模态对齐 → Whisper 伪对齐 → SenseVoice 草稿
- 分句策略：依赖 Whisper 的精准标点
- 语义分组：依赖语义完整性（续接词、从句判断）
"""
import copy
import logging
from typing import Optional

from app.schemas.pipeline_context import ProcessingContext
from app.services.alignment.default_aligner import DefaultAligner, AlignmentLevel
from app.services.alignment.alignment_service import AlignmentConfig
from app.services.sentence_splitter import SplitConfig
from app.services.semantic_grouper import GroupConfig
from app.services.streaming_subtitle import get_streaming_subtitle_manager


class AlignmentWorker:
    """
    AlignmentWorker - 对齐层 Worker（CPU）

    在三级流水线中负责：
    1. 双流对齐（三级降级）
    2. 慢流分句（依赖标点）
    3. 推送定稿（Chunk 级别）
    """

    def __init__(
        self,
        job_id: str,
        final_split_config: Optional[SplitConfig] = None,
        final_group_config: Optional[GroupConfig] = None,
        alignment_config: Optional[AlignmentConfig] = None,
        enable_semantic_grouping: bool = True,
        alignment_score_threshold: float = 0.3,
        enable_fallback: bool = True,
        aligner: Optional[DefaultAligner] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化 AlignmentWorker

        Args:
            job_id: 任务 ID
            final_split_config: 慢流分句配置
            final_group_config: 慢流语义分组配置
            alignment_config: 对齐服务配置
            enable_semantic_grouping: 是否启用语义分组
            alignment_score_threshold: 对齐质量阈值
            enable_fallback: 是否启用降级策略
            aligner: 对齐服务（默认实现）
            logger: 日志记录器
        """
        self.job_id = job_id
        self.enable_semantic_grouping = enable_semantic_grouping
        self.alignment_score_threshold = alignment_score_threshold
        self.enable_fallback = enable_fallback
        self.logger = logger or logging.getLogger(__name__)

        # 初始化对齐服务（封装旧逻辑）
        if alignment_config is None:
            alignment_config = AlignmentConfig()
        self.aligner = aligner or DefaultAligner(
            alignment_config=alignment_config,
            final_split_config=final_split_config,
            final_group_config=final_group_config,
            is_enable_semantic_grouping=enable_semantic_grouping,
            alignment_score_threshold=alignment_score_threshold,
            is_enable_fallback=enable_fallback,
            logger=self.logger,
        )

        # 获取流式字幕管理器
        self.subtitle_manager = get_streaming_subtitle_manager(job_id)

    async def process(self, ctx: ProcessingContext):
        """
        处理单个 Chunk（对齐层）

        流程：
        1. 【V3.10】快速路径：Whisper 跳过时直接使用 SenseVoice
        2. 双流对齐（三级降级）
        3. 分句 + 语义分组
        4. 推送定稿
        5. 填充 ctx.final_sentences

        Args:
            ctx: 处理上下文
        """
        chunk = ctx.audio_chunk

        # V3.10: 快速路径 - SlowWorker 跳过时直接使用 SenseVoice
        if ctx.whisper_skipped:
            self.logger.info(f"Chunk {ctx.chunk_index}: Whisper 跳过，直接使用 SenseVoice 定稿")
            final_sentences = self.aligner.split_sensevoice_only(ctx.sv_result, chunk)

            ctx.final_sentences = final_sentences

            # 推送定稿
            sentences_for_manager = copy.deepcopy(final_sentences)
            self.subtitle_manager.replace_chunk(ctx.chunk_index, sentences_for_manager)

            self.logger.debug(
                f"Chunk {ctx.chunk_index}: SenseVoice 定稿已推送 "
                f"({len(final_sentences)} 个句子) [智能补刀-跳过]"
            )
            return

        # 阶段 1: 双流对齐（三级降级策略）
        self.logger.debug(f"Chunk {ctx.chunk_index}: 双流对齐")

        whisper_result = ctx.whisper_result
        sv_result = ctx.sv_result

        final_sentences, alignment_level = await self.aligner.align(
            whisper_result,
            sv_result,
            chunk,
        )

        ctx.final_sentences = final_sentences

        # 阶段 2: 推送定稿（使用 Chunk 级别的批量替换）
        # V3.8 调试日志：记录对齐结果状态
        self.logger.debug(
            f"Chunk {ctx.chunk_index}: 对齐完成 - "
            f"final_sentences={len(final_sentences)}, "
            f"alignment_level={alignment_level.value}, "
            f"whisper_text_len={len(whisper_result.get('text', ''))}, "
            f"sv_text_clean_len={len(sv_result.get('text_clean', ''))}"
        )

        # V3.8 修复：如果定稿句子为空，尝试从草稿中恢复
        if not final_sentences:
            self.logger.error(
                f"Chunk {ctx.chunk_index}: 定稿句子为空！"
                f"Whisper文本长度={len(whisper_result.get('text', ''))}, "
                f"SenseVoice文本长度={len(sv_result.get('text_clean', ''))}, "
                f"对齐级别={alignment_level.value}"
            )

            # V3.8 修复：尝试从 subtitle_manager 中获取草稿句子作为兜底
            draft_indices = self.subtitle_manager.chunk_sentences.get(ctx.chunk_index, [])
            if draft_indices:
                draft_sentences = []
                for idx in draft_indices:
                    if idx in self.subtitle_manager.sentences:
                        # 深拷贝草稿句子，设置为定稿状态
                        draft_sentence = copy.deepcopy(self.subtitle_manager.sentences[idx])
                        draft_sentence.is_finalized = True
                        draft_sentence.is_draft = False
                        draft_sentences.append(draft_sentence)

                if draft_sentences:
                    self.logger.warning(
                        f"Chunk {ctx.chunk_index}: 使用草稿句子作为兜底 ({len(draft_sentences)} 个句子)"
                    )
                    final_sentences = draft_sentences
                    ctx.final_sentences = final_sentences

        # V3.8 修复竞态条件：深拷贝 final_sentences 再传给 subtitle_manager
        # 避免 subtitle_manager 修改句子对象影响 ctx.final_sentences
        sentences_for_manager = copy.deepcopy(final_sentences)
        self.subtitle_manager.replace_chunk(ctx.chunk_index, sentences_for_manager)

        self.logger.debug(
            f"Chunk {ctx.chunk_index}: 定稿已推送 "
            f"({len(final_sentences)} 个句子, 对齐级别={alignment_level.value})"
        )

