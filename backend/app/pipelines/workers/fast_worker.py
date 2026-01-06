"""
FastWorker - 快流推理 Worker（CPU）

职责：
1. 执行 SenseVoice ONNX 推理（CPU）
2. 立即分句（Layer 1 + Layer 2）
3. 立即推送草稿到 SSE（确保用户体验）
4. 填充 ProcessingContext.sv_result
5. 熔断回溯（可选）：监控转录质量，自动升级分离模型

特点：
- 速度优先（~1秒/Chunk）
- 分句策略：主要依赖 VAD 停顿，不依赖标点
- 语义分组：依赖物理约束（时间间隔、句子长度）
- 熔断回溯：低置信度+BGM标签时自动升级分离

V3.5 更新：
- 支持 is_final_output 参数，极速模式下直接输出定稿
"""
import copy
import logging
from typing import Dict, List, Optional, Any

from app.schemas.pipeline_context import ProcessingContext
from app.services.audio.chunk_engine import AudioChunk
from app.services.inference.sensevoice_executor import SenseVoiceExecutor
from app.services.sentence_splitter import SentenceSplitter, SplitConfig
from app.services.semantic_grouper import SemanticGrouper, GroupConfig
from app.services.streaming_subtitle import StreamingSubtitleManager, get_streaming_subtitle_manager
from app.models.sensevoice_models import SentenceSegment, TextSource, WordTimestamp
from app.services.fuse_breaker_v2 import FuseBreakerV2
from app.services.demucs_service import DemucsService, get_demucs_service
from app.models.circuit_breaker_models import FuseAction


class FastWorker:
    """
    FastWorker - 快流推理 Worker（CPU）

    在三级流水线中负责：
    1. SenseVoice 推理（CPU ONNX）
    2. 快速分句（依赖 VAD 停顿）
    3. 立即推送草稿（不等待 Whisper）
    """

    def __init__(
        self,
        job_id: str,
        sensevoice_language: str = "auto",
        draft_split_config: Optional[SplitConfig] = None,
        draft_group_config: Optional[GroupConfig] = None,
        enable_semantic_grouping: bool = True,
        sensevoice_executor: Optional[SenseVoiceExecutor] = None,
        # 熔断回溯相关参数
        enable_fuse_breaker: bool = False,
        fuse_max_retry: int = 1,
        fuse_confidence_threshold: float = 0.5,
        fuse_auto_upgrade: bool = False,
        demucs_service: Optional[DemucsService] = None,
        # V3.5: 极速模式参数
        is_final_output: bool = False,
        # V3.1.0: 跨 chunk 合并参数
        enable_cross_chunk_merge: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化 FastWorker

        Args:
            job_id: 任务 ID
            sensevoice_language: SenseVoice 语言设置
            draft_split_config: 快流分句配置
            draft_group_config: 快流语义分组配置
            enable_semantic_grouping: 是否启用语义分组
            sensevoice_executor: SenseVoice 执行器
            enable_fuse_breaker: 是否启用熔断回溯（默认False，保持向后兼容）
            fuse_max_retry: 熔断最大重试次数（默认1，只升级到 HTDEMUCS）
            fuse_confidence_threshold: 熔断置信度阈值
            fuse_auto_upgrade: 是否启用第二次自动升级到 MDX_EXTRA（默认False）
            demucs_service: Demucs服务实例
            is_final_output: 是否为最终输出（极速模式下为True，输出定稿而非草稿）
            enable_cross_chunk_merge: 是否启用跨chunk合并（默认False，仅SenseVoice模式推荐启用）
            logger: 日志记录器
        """
        self.job_id = job_id
        self.sensevoice_language = sensevoice_language
        self.enable_semantic_grouping = enable_semantic_grouping
        self.is_final_output = is_final_output
        self.logger = logger or logging.getLogger(__name__)

        # V3.1.0: 跨 chunk 合并机制
        self.enable_cross_chunk_merge = enable_cross_chunk_merge
        self.pending_sentence = None  # 缓存上一个 chunk 的最后一句（如果语义不完整）

        if is_final_output:
            self.logger.info("FastWorker 运行在极速模式：输出为定稿")

        if enable_cross_chunk_merge:
            self.logger.info("跨 chunk 合并已启用：将自动合并语义不完整的句子")

        # 初始化执行器
        self.sensevoice_executor = sensevoice_executor or SenseVoiceExecutor()

        # 初始化熔断决策器（新增）
        self.enable_fuse_breaker = enable_fuse_breaker
        if enable_fuse_breaker:
            self.fuse_breaker = FuseBreakerV2(
                max_retry=fuse_max_retry,
                confidence_threshold=fuse_confidence_threshold,
                auto_upgrade=fuse_auto_upgrade,
                logger=self.logger
            )
            self.demucs_service = demucs_service or get_demucs_service()
            self.logger.info(
                f"熔断回溯已启用: max_retry={fuse_max_retry}, "
                f"threshold={fuse_confidence_threshold}, auto_upgrade={fuse_auto_upgrade}"
            )
        else:
            self.fuse_breaker = None
            self.demucs_service = None

        # 初始化快流分句器（默认配置，主要依赖 VAD 停顿）
        if draft_split_config is None:
            draft_split_config = SplitConfig(
                language="en",                   # 默认使用英文策略（非中文时使用）
                prefer_punctuation_break=False,  # 不依赖标点
                use_dynamic_pause=True,          # 使用动态停顿
                pause_threshold=0.5,
                max_duration=5.0,
                enable_hard_limit=True,          # 启用硬上限（快流需要兜底保护）
                hard_limit_duration=10.0,        # 硬上限 10 秒
                merge_short_sentences=True
            )
        self.draft_splitter = SentenceSplitter(draft_split_config)

        # V3.9: 中文专用分句器（优先标点断句）
        chinese_split_config = SplitConfig(
            language="zh",
            prefer_punctuation_break=True,       # 优先在标点处断句
            delay_split_to_punctuation=True,     # 延迟到标点切分
            delay_split_max_wait=8.0,            # 等待标点的最大时长（秒）
            use_dynamic_pause=True,
            pause_threshold=0.3,                 # 降低停顿阈值
            max_duration=12.0,                   # 中文句子可以更长
            enable_hard_limit=True,
            hard_limit_duration=20.0,            # 硬上限提高到20秒（避免VAD合并导致的强制切分）
            merge_short_sentences=True,
            min_duration_threshold=1.5           # 短句合并阈值
        )
        self.chinese_splitter = SentenceSplitter(chinese_split_config)
        self.logger.debug("中文专用分句器已初始化")

        # 初始化快流语义分组器（主要依赖物理约束）
        if draft_group_config is None:
            draft_group_config = GroupConfig(
                max_group_gap=2.0,
                max_group_duration=10.0,
                max_group_sentences=5,
                enable_overlap_detection=True
            )
        self.draft_grouper = SemanticGrouper(draft_group_config)

        # 获取流式字幕管理器
        self.subtitle_manager = get_streaming_subtitle_manager(job_id)

    async def process(self, ctx: ProcessingContext):
        """
        处理单个 Chunk（快流）

        流程：
        1. SenseVoice 推理
        2. 熔断决策（如果启用）
        3. 升级分离（如果需要）
        4. 分句（Layer 1 + Layer 2）
        5. 立即推送草稿
        6. 填充 ctx.sv_result

        Args:
            ctx: 处理上下文
        """
        chunk = ctx.audio_chunk

        # 熔断循环（如果启用熔断回溯）
        if self.enable_fuse_breaker:
            await self._process_with_fuse(ctx)
        else:
            await self._process_without_fuse(ctx)

    async def _process_without_fuse(self, ctx: ProcessingContext):
        """
        不带熔断回溯的处理流程

        Args:
            ctx: 处理上下文
        """
        chunk = ctx.audio_chunk

        # 阶段 1: SenseVoice 推理
        self.logger.debug(f"Chunk {ctx.chunk_index}: SenseVoice 快流推理")
        sv_result = await self._run_sensevoice(chunk)

        # V3.8 修复竞态条件：深拷贝 sv_result，避免下游修改影响其他协程
        ctx.sv_result = copy.deepcopy(sv_result)

        # 阶段 2: 分句（Layer 1 + Layer 2）
        # 极速模式下 is_draft=False，表示这是定稿
        is_draft = not self.is_final_output
        sentences = self._split_sentences(sv_result, chunk, is_draft=is_draft)

        # V3.1.0: 跨 chunk 合并（仅在启用时）
        if self.enable_cross_chunk_merge and sentences:
            # 如果有缓存的句子，与第一句合并
            if self.pending_sentence:
                sentences[0] = self._merge_sentences(self.pending_sentence, sentences[0])
                self.pending_sentence = None

            # 检查最后一句是否语义完整
            if self._is_sentence_incomplete(sentences[-1]):
                # 缓存最后一句，不推送
                self.pending_sentence = sentences[-1]
                sentences = sentences[:-1]
                self.logger.debug(f"[跨chunk] 缓存最后一句，等待下一个chunk")

        # 阶段 3: 推送
        if self.is_final_output:
            # 极速模式：推送定稿
            self.subtitle_manager.add_finalized_sentences(ctx.chunk_index, sentences)
            self.logger.debug(
                f"Chunk {ctx.chunk_index}: 定稿已推送 ({len(sentences)} 个句子) [极速模式]"
            )
        else:
            # 补刀模式：推送草稿
            self.subtitle_manager.add_draft_sentences(ctx.chunk_index, sentences)
            self.logger.debug(
                f"Chunk {ctx.chunk_index}: 草稿已推送 ({len(sentences)} 个句子)"
            )

    async def _process_with_fuse(self, ctx: ProcessingContext):
        """
        带熔断回溯的处理流程（新增）

        流程：
        1. 熔断循环：SenseVoice推理 → 熔断决策 → 升级分离（如果需要）
        2. 分句和推送

        Args:
            ctx: 处理上下文
        """
        chunk = ctx.audio_chunk

        # 熔断循环
        while True:
            # 阶段 1: SenseVoice 推理
            self.logger.debug(
                f"Chunk {ctx.chunk_index}: SenseVoice 快流推理 "
                f"(重试次数: {chunk.fuse_retry_count})"
            )
            sv_result = await self._run_sensevoice(chunk)
            ctx.sv_result = sv_result

            # 阶段 2: 熔断决策
            decision = self.fuse_breaker.should_fuse(chunk, sv_result)

            if decision.action == FuseAction.ACCEPT:
                # 接受结果，退出循环
                self.logger.debug(
                    f"Chunk {ctx.chunk_index}: 熔断决策 - 接受结果 ({decision.reason})"
                )
                break

            elif decision.action == FuseAction.UPGRADE_SEPARATION:
                # 升级分离，继续循环
                self.logger.info(
                    f"Chunk {ctx.chunk_index}: 熔断决策 - 升级分离 "
                    f"({decision.reason})"
                )

                try:
                    # 执行升级分离
                    chunk = await self.fuse_breaker.execute_upgrade(
                        chunk=chunk,
                        target_level=decision.target_level,
                        demucs_service=self.demucs_service
                    )
                    # 更新ctx中的chunk引用
                    ctx.audio_chunk = chunk

                    # 继续循环，重新推理
                    continue

                except Exception as e:
                    # 升级失败，接受当前结果
                    self.logger.error(
                        f"Chunk {ctx.chunk_index}: 升级分离失败 - {e}，接受当前结果"
                    )
                    break

            else:
                # 未知动作，接受结果
                self.logger.warning(
                    f"Chunk {ctx.chunk_index}: 未知熔断动作 - {decision.action}，接受当前结果"
                )
                break

        # 阶段 3: 分句（Layer 1 + Layer 2）
        draft_sentences = self._split_sentences(sv_result, chunk, is_draft=True)

        # V3.1.0: 跨 chunk 合并（仅在启用时）
        if self.enable_cross_chunk_merge and draft_sentences:
            # 如果有缓存的句子，与第一句合并
            if self.pending_sentence:
                draft_sentences[0] = self._merge_sentences(self.pending_sentence, draft_sentences[0])
                self.pending_sentence = None

            # 检查最后一句是否语义完整
            if self._is_sentence_incomplete(draft_sentences[-1]):
                # 缓存最后一句，不推送
                self.pending_sentence = draft_sentences[-1]
                draft_sentences = draft_sentences[:-1]
                self.logger.debug(f"[跨chunk] 缓存最后一句，等待下一个chunk")

        # 阶段 4: 立即推送草稿（关键：不等待 Whisper）
        self.subtitle_manager.add_draft_sentences(ctx.chunk_index, draft_sentences)

        self.logger.info(
            f"Chunk {ctx.chunk_index}: 草稿已推送 ({len(draft_sentences)} 个句子), "
            f"分离级别: {chunk.separation_level.value}, 重试次数: {chunk.fuse_retry_count}"
        )

    async def _run_sensevoice(self, chunk: AudioChunk) -> Dict[str, Any]:
        """
        运行 SenseVoice 推理

        Args:
            chunk: AudioChunk

        Returns:
            Dict: SenseVoice 推理结果
        """
        result = await self.sensevoice_executor.execute(
            audio_array=chunk.audio,
            sample_rate=chunk.sample_rate,
            language=self.sensevoice_language,
            use_itn=True
        )
        return result

    def _split_sentences(
        self,
        sv_result: Dict[str, Any],
        chunk: AudioChunk,
        is_draft: bool = True
    ) -> List[SentenceSegment]:
        """
        对 SenseVoice 结果进行分句（快流策略）

        流程：
        1. Layer 1: SentenceSplitter 分句（主要依赖 VAD 停顿）
        2. Layer 2: SemanticGrouper 语义分组（依赖物理约束）
        3. 调整时间偏移

        Args:
            sv_result: SenseVoice 推理结果
            chunk: AudioChunk
            is_draft: 是否为草稿

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
                    is_finalized=not is_draft,
                    is_draft=is_draft
                )
                return [fallback_sentence]
            else:
                self.logger.warning("SenseVoice 结果没有字级时间戳且无文本，无法分句")
                return []

        # V3.9: 根据语言选择分句器
        detected_language = sv_result.get('language', 'auto')
        is_chinese = detected_language in {'zh', 'yue'}

        # Layer 1: 分句（根据语言选择分句器）
        if is_chinese:
            sentences = self.chinese_splitter.split(words, text_clean)
        else:
            sentences = self.draft_splitter.split(words, text_clean)

        # Layer 2: 语义分组（如果启用）
        if self.enable_semantic_grouping:
            sentences = self.draft_grouper.group(sentences)
            self.logger.debug(f"快流语义分组: {len(sentences)} 个句子（物理约束）")

        # 调整时间偏移和设置状态
        for sentence in sentences:
            sentence.start += chunk.start
            sentence.end += chunk.start
            sentence.is_draft = is_draft
            sentence.is_finalized = not is_draft
            sentence.source = TextSource.SENSEVOICE

            # 调整 words 的时间偏移
            for word in sentence.words:
                word.start += chunk.start
                word.end += chunk.start

        self.logger.debug(f"快流分句完成: {len(sentences)} 个句子")

        return sentences

    def _is_sentence_incomplete(self, sentence: SentenceSegment) -> bool:
        """
        检查句子是否语义不完整（V3.1.0）

        根据文本内容自动检测语言，选择合适的语义完整性检查策略。

        Args:
            sentence: 句子对象

        Returns:
            bool: True 表示语义不完整，False 表示完整
        """
        if not sentence or not sentence.text:
            return False

        text = sentence.text.strip()
        
        # 检测文本是否包含中文字符
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
        
        # 根据语言选择分句器的策略
        if has_chinese:
            strategy = self.chinese_splitter.config.get_strategy()
        else:
            strategy = self.draft_splitter.config.get_strategy()
        
        is_incomplete = strategy.is_incomplete_ending(text)

        if is_incomplete:
            self.logger.debug(f"[跨chunk检查] 句子语义不完整: '{sentence.text[-20:]}'")

        return is_incomplete

    def _merge_sentences(self, sent1: SentenceSegment, sent2: SentenceSegment) -> SentenceSegment:
        """
        合并两个句子（V3.1.0）

        用于跨 chunk 合并：将上一个 chunk 的最后一句与当前 chunk 的第一句合并。

        Args:
            sent1: 第一个句子（上一个 chunk 的最后一句）
            sent2: 第二个句子（当前 chunk 的第一句）

        Returns:
            SentenceSegment: 合并后的句子
        """
        # 合并文本
        merged_text = sent1.text + sent2.text

        # 合并时间戳（使用 sent1 的 start，sent2 的 end）
        merged_start = sent1.start
        merged_end = sent2.end

        # 合并 words 列表
        merged_words = sent1.words + sent2.words

        # 计算平均置信度
        avg_confidence = (sent1.confidence + sent2.confidence) / 2

        # 创建合并后的句子对象（V3.1.0: 移除不存在的index参数）
        merged_sentence = SentenceSegment(
            text=merged_text,
            start=merged_start,
            end=merged_end,
            words=merged_words,
            confidence=avg_confidence,
            is_draft=sent1.is_draft,
            is_finalized=sent1.is_finalized,
            source=sent1.source,
            warning_type=sent1.warning_type or sent2.warning_type  # 保留任一警告
        )

        self.logger.debug(
            f"[跨chunk合并] 合并句子: "
            f"'{sent1.text[-15:]}' + '{sent2.text[:15]}' -> "
            f"'{merged_text[-30:]}'"
        )

        return merged_sentence
