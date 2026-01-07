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
from typing import Dict, List, Optional, Any
from enum import Enum

from app.schemas.pipeline_context import ProcessingContext
from app.services.audio.chunk_engine import AudioChunk
from app.services.alignment.alignment_service import AlignmentService, AlignmentConfig
from app.services.sentence_splitter import SentenceSplitter, SplitConfig
from app.services.semantic_grouper import SemanticGrouper, GroupConfig
from app.services.streaming_subtitle import StreamingSubtitleManager, get_streaming_subtitle_manager
from app.services.pseudo_alignment import PseudoAlignment
from app.models.sensevoice_models import SentenceSegment, TextSource, WordTimestamp


class AlignmentLevel(Enum):
    """对齐级别（三级降级策略）"""
    DUAL_MODAL = "dual_modal"           # Level 1: 双模态对齐（黄金标准）
    WHISPER_PSEUDO = "whisper_pseudo"   # Level 2: Whisper 伪对齐（银标准）
    SENSEVOICE_ONLY = "sensevoice_only" # Level 3: SenseVoice 草稿（铜标准）


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
        alignment_service: Optional[AlignmentService] = None,
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
            alignment_service: 对齐服务
            logger: 日志记录器
        """
        self.job_id = job_id
        self.enable_semantic_grouping = enable_semantic_grouping
        self.alignment_score_threshold = alignment_score_threshold
        self.enable_fallback = enable_fallback
        self.logger = logger or logging.getLogger(__name__)

        # 初始化对齐服务
        if alignment_config is None:
            alignment_config = AlignmentConfig()
        self.alignment_service = alignment_service or AlignmentService(
            config=alignment_config,
            logger=self.logger
        )

        # 初始化慢流分句器（依赖 Whisper 的精准标点）
        if final_split_config is None:
            final_split_config = SplitConfig(
                prefer_punctuation_break=True,   # 依赖标点
                use_dynamic_pause=True,
                pause_threshold=0.5,
                max_duration=5.0,                # 软上限（监控点）
                enable_hard_limit=True,          # 启用硬上限（异常保护，设置很宽松）
                hard_limit_duration=20.0,        # 20秒硬上限（远大于正常句子）
                delay_split_to_punctuation=True, # 延迟切分到标点
                delay_split_max_wait=15.0,       # 允许延迟15秒（总计20秒）
                merge_short_sentences=True
            )
        self.final_splitter = SentenceSplitter(final_split_config)

        # 初始化慢流语义分组器（依赖语义完整性）
        if final_group_config is None:
            final_group_config = GroupConfig(
                max_group_gap=2.0,
                max_group_duration=10.0,
                max_group_sentences=5,
                enable_overlap_detection=True
            )
        self.final_grouper = SemanticGrouper(final_group_config)

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
            final_sentences = self._split_sentences_from_sv(ctx.sv_result, chunk)

            # 设置为定稿状态
            for sentence in final_sentences:
                sentence.is_finalized = True
                sentence.is_draft = False

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

        final_sentences, alignment_level = await self._align_and_fallback(
            whisper_result,
            sv_result,
            chunk
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

    async def _align_and_fallback(
        self,
        whisper_result: Dict[str, Any],
        sv_result: Dict[str, Any],
        chunk: AudioChunk
    ) -> tuple[List[SentenceSegment], AlignmentLevel]:
        """
        双流对齐 + 三级降级策略 + 分句

        Level 1: 双模态对齐（黄金标准）
        Level 2: Whisper 伪对齐（银标准）
        Level 3: SenseVoice 草稿（铜标准）

        Args:
            whisper_result: Whisper 推理结果
            sv_result: SenseVoice 推理结果
            chunk: AudioChunk

        Returns:
            tuple[List[SentenceSegment], AlignmentLevel]: 最终句子列表和对齐级别
        """
        # 获取 Whisper 检测到的语言，用于语义完整性检查
        detected_language = whisper_result.get('language', 'auto')
        self.final_splitter.config.language = detected_language
        self.logger.debug(f"使用 Whisper 检测到的语言: {detected_language}")

        # ========== 早期拦截：长度异常检测 ==========
        whisper_text = whisper_result.get('text', '').strip()
        sv_text_clean = sv_result.get('text_clean', '').strip()

        # 检测0: Whisper 单词数过少（幻觉检测）
        # 统计Whisper识别出的单词数，如果少于2个，可能是幻觉
        if whisper_text:
            # 简单的单词统计：按空格分割（适用于英文）或按字符统计（适用于中文）
            word_count = len(whisper_text.split())
            if word_count < 2 and sv_text_clean:
                self.logger.warning(
                    f"Whisper 单词数过少（{word_count} < 2），可能是幻觉，直接降级到 Level 3: "
                    f"whisper='{whisper_text}', sv='{sv_text_clean[:50]}...'"
                )
                # 直接降级到 Level 3（SenseVoice 草稿）
                sentences = self._split_sentences_from_sv(sv_result, chunk)
                if sentences:  # 只有当SenseVoice有结果时才返回
                    for sentence in sentences:
                        sentence.is_finalized = True
                        sentence.is_draft = False
                    return sentences, AlignmentLevel.SENSEVOICE_ONLY
                else:
                    # SenseVoice也为空，返回空列表（删除该段字幕）
                    self.logger.warning(
                        f"Whisper和SenseVoice都没有有效内容，返回空列表"
                    )
                    return [], AlignmentLevel.SENSEVOICE_ONLY

        # 检测 Whisper 输出是否异常
        if sv_text_clean:  # 仅当 SenseVoice 有输出时检查
            len_w = len(whisper_text)
            len_sv = len(sv_text_clean)

            # 检测1: Whisper 长度暴涨（幻觉）
            # 公式: len(whisper) > 3 * len(sensevoice) + 10
            if len_w > 3 * len_sv + 10:
                # 检查 Whisper 置信度
                whisper_confidence = whisper_result.get('confidence', 0.5)

                # 置信度阈值：0.5 对应 avg_logprob ≈ -0.5
                if whisper_confidence < 0.5:
                    self.logger.warning(
                        f"Whisper 长度暴涨且置信度低，直接降级到 Level 3: "
                        f"len(whisper)={len_w} > 3*{len_sv}+10, "
                        f"confidence={whisper_confidence:.2f} < 0.5, "
                        f"whisper='{whisper_text[:50]}...', sv='{sv_text_clean[:50]}...'"
                    )
                    # 直接降级到 Level 3（SenseVoice 草稿）
                    sentences = self._split_sentences_from_sv(sv_result, chunk)
                    for sentence in sentences:
                        sentence.is_finalized = True
                        sentence.is_draft = False
                    return sentences, AlignmentLevel.SENSEVOICE_ONLY
                else:
                    # 特权放行：置信度高，可能是 SenseVoice 漏识别
                    self.logger.info(
                        f"Whisper 长度暴涨但置信度高，特权放行: "
                        f"len(whisper)={len_w} > 3*{len_sv}+10, "
                        f"confidence={whisper_confidence:.2f} >= 0.5"
                    )

            # 检测2: Whisper 过短（识别失败或漏识别）
            # V3.1.0 修复：将阈值从 0.3 提升到 0.65
            # 原因：双模态对齐以 Whisper 文本为准，如果 Whisper 明显短于 SenseVoice，
            #       会导致 SenseVoice 多识别的内容被丢弃（如 Chunk 58: 146 vs 262 字符）
            # 公式: len(whisper) < len(sensevoice) * 0.65
            elif len_w < len_sv * 0.65:
                self.logger.warning(
                    f"Whisper 输出明显短于 SenseVoice，直接降级到 Level 3: "
                    f"len(whisper)={len_w} < {len_sv} * 0.65 = {len_sv * 0.65:.0f}, "
                    f"ratio={len_w/len_sv:.2f}, "
                    f"whisper='{whisper_text[:50] if whisper_text else '(空)'}', "
                    f"sv='{sv_text_clean[:50]}...'"
                )
                # 直接降级到 Level 3（SenseVoice 草稿）
                sentences = self._split_sentences_from_sv(sv_result, chunk)
                for sentence in sentences:
                    sentence.is_finalized = True
                    sentence.is_draft = False
                return sentences, AlignmentLevel.SENSEVOICE_ONLY

        try:
            # Level 1: 尝试双模态对齐
            sv_words_data = sv_result.get('words', [])

            if not whisper_text or not sv_words_data:
                raise ValueError("Whisper 或 SenseVoice 结果为空")

            # 转换 SenseVoice words 为 WordTimestamp 列表
            sv_tokens = [
                WordTimestamp(
                    word=w.get('word', ''),
                    start=w.get('start', 0.0),
                    end=w.get('end', 0.0),
                    confidence=w.get('confidence', 1.0)
                )
                for w in sv_words_data
            ]

            # 执行对齐
            aligned_subtitle = await self.alignment_service.align(
                whisper_text=whisper_text,
                sv_tokens=sv_tokens,
                vad_range=(0.0, chunk.duration),  # 使用相对时间
                chunk_offset=chunk.start,
                audio_array=chunk.audio,
                sample_rate=chunk.sample_rate
            )

            # 检查对齐质量
            if aligned_subtitle.alignment_score < self.alignment_score_threshold:
                raise ValueError(f"对齐质量过低: {aligned_subtitle.alignment_score:.2f}")

            # 转换对齐结果为 WordTimestamp 列表
            aligned_words = [
                WordTimestamp(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    confidence=w.final_confidence,
                    is_pseudo=w.is_pseudo
                )
                for w in aligned_subtitle.words
            ]

            # 重新分句（基于 Whisper 的标点，使用慢流分句器）
            sentences = self.final_splitter.split(aligned_words, whisper_text)

            # 语义分组（使用慢流语义分组器）
            if self.enable_semantic_grouping:
                sentences = self.final_grouper.group(sentences)

            # 设置状态
            for sentence in sentences:
                sentence.source = TextSource.WHISPER_PATCH
                sentence.is_finalized = True
                sentence.is_draft = False
                sentence.alignment_score = aligned_subtitle.alignment_score
                sentence.matched_ratio = aligned_subtitle.matched_ratio
                sentence.whisper_text = whisper_text

            self.logger.debug(
                f"双模态对齐成功: alignment_score={aligned_subtitle.alignment_score:.2f}, "
                f"matched_ratio={aligned_subtitle.matched_ratio:.2f}, "
                f"分句数={len(sentences)}"
            )

            return sentences, AlignmentLevel.DUAL_MODAL

        except Exception as e:
            self.logger.warning(f"双模态对齐失败: {e}, 降级到 Whisper 伪对齐")

            if not self.enable_fallback:
                raise

            try:
                # Level 2: Whisper 伪对齐
                whisper_text = whisper_result.get('text', '').strip()

                if not whisper_text:
                    raise ValueError("Whisper 结果为空")

                # 使用伪对齐生成字级时间戳
                words = PseudoAlignment.apply(
                    original_start=0.0,
                    original_end=chunk.duration,
                    new_text=whisper_text
                )

                # 调整时间偏移
                for word in words:
                    word.start += chunk.start
                    word.end += chunk.start

                # 分句（使用慢流分句器）
                sentences = self.final_splitter.split(words, whisper_text)

                # 语义分组（使用慢流语义分组器）
                if self.enable_semantic_grouping:
                    sentences = self.final_grouper.group(sentences)

                # 设置状态
                for sentence in sentences:
                    sentence.source = TextSource.WHISPER_PATCH
                    sentence.is_finalized = True
                    sentence.is_draft = False
                    sentence.alignment_score = 0.5  # 伪对齐的默认分数
                    sentence.whisper_text = whisper_text

                self.logger.debug(f"Whisper 伪对齐成功, 分句数={len(sentences)}")

                return sentences, AlignmentLevel.WHISPER_PSEUDO

            except Exception as e2:
                self.logger.error(f"Whisper 伪对齐失败: {e2}, 降级到 SenseVoice 草稿")

                # Level 3: SenseVoice 草稿
                try:
                    sentences = self._split_sentences_from_sv(sv_result, chunk)

                    # 设置为定稿状态（虽然是草稿，但作为最终结果）
                    for sentence in sentences:
                        sentence.is_finalized = True
                        sentence.is_draft = False

                    self.logger.debug(f"使用 SenseVoice 草稿作为最终结果, 分句数={len(sentences)}")

                    return sentences, AlignmentLevel.SENSEVOICE_ONLY

                except Exception as e3:
                    # Level 4: 最终兜底方案
                    self.logger.error(f"SenseVoice 草稿也失败: {e3}, 使用最终兜底方案")

                    # 尝试使用任何可用的文本
                    fallback_text = whisper_text or sv_text_clean

                    if not fallback_text:
                        # 如果两者都为空，返回空列表（删除该段字幕）
                        self.logger.warning(
                            f"Whisper 和 SenseVoice 都没有识别出内容，返回空列表"
                        )
                        return [], AlignmentLevel.SENSEVOICE_ONLY

                    # 创建覆盖整个Chunk的单句字幕
                    # 注：SentenceSegment 和 TextSource 已在文件顶部导入
                    sentence = SentenceSegment(
                        text=fallback_text,
                        start=chunk.start,
                        end=chunk.start + chunk.duration,
                        source=TextSource.SENSEVOICE_DRAFT,
                        is_finalized=True,
                        is_draft=False
                    )

                    self.logger.warning(
                        f"兜底字幕: [{chunk.start:.2f}s - {chunk.start + chunk.duration:.2f}s] "
                        f"text='{fallback_text[:30]}...'"
                    )

                    return [sentence], AlignmentLevel.SENSEVOICE_ONLY

    def _split_sentences_from_sv(
        self,
        sv_result: Dict[str, Any],
        chunk: AudioChunk
    ) -> List[SentenceSegment]:
        """
        从 SenseVoice 结果分句（降级方案）

        Args:
            sv_result: SenseVoice 推理结果
            chunk: AudioChunk

        Returns:
            List[SentenceSegment]: 分句后的句子列表
        """
        text_clean = sv_result.get('text_clean', '')
        words_data = sv_result.get('words', [])

        # 转换 words 为 WordTimestamp 列表
        words = [
            WordTimestamp(
                word=w.get('word', ''),
                start=w.get('start', 0.0),
                end=w.get('end', 0.0),
                confidence=w.get('confidence', 1.0)
            )
            for w in words_data
        ]

        if not words:
            # V3.8 修复: words 为空但 text_clean 有值时，创建覆盖整个 chunk 的单句
            # 避免"有文本但无字级时间戳"导致字幕丢失
            if text_clean and text_clean.strip():
                self.logger.warning(
                    f"SenseVoice 没有字级时间戳但有文本，创建兜底单句: "
                    f"text='{text_clean[:50]}...', chunk=[{chunk.start:.2f}s, {chunk.end:.2f}s]"
                )
                # 创建覆盖整个 chunk 的单句
                fallback_sentence = SentenceSegment(
                    text=text_clean.strip(),
                    start=chunk.start,
                    end=chunk.end,
                    words=[],  # 无字级时间戳
                    source=TextSource.SENSEVOICE,
                    confidence=sv_result.get('confidence', 0.5),
                    is_finalized=True,
                    is_draft=False
                )
                return [fallback_sentence]
            else:
                self.logger.warning("SenseVoice 结果没有字级时间戳且无文本，无法分句")
                return []

        # 使用慢流分句器（但是基于 SenseVoice 的数据）
        sentences = self.final_splitter.split(words, text_clean)

        # 语义分组
        if self.enable_semantic_grouping:
            sentences = self.final_grouper.group(sentences)

        # 调整时间偏移和设置状态
        for sentence in sentences:
            sentence.start += chunk.start
            sentence.end += chunk.start
            sentence.source = TextSource.SENSEVOICE

            # 调整 words 的时间偏移
            for word in sentence.words:
                word.start += chunk.start
                word.end += chunk.start

        return sentences
