"""
流式字幕管理系统

核心职责：
1. 管理字幕句子列表（支持原地更新）
2. 协调 SSE 事件推送（统一 Tag）
3. 支持多阶段增量更新（SV → Whisper → LLM）
"""
import copy
import threading
from typing import Dict, List, Optional
from app.models.sensevoice_models import SentenceSegment, TextSource
from app.services.sse_service import get_sse_manager
import logging

logger = logging.getLogger(__name__)


def push_subtitle_event(sse_manager, job_id: str, event_type: str, data: dict):
    """
    推送字幕事件（统一封装）

    Args:
        sse_manager: SSE 管理器
        job_id: 任务 ID
        event_type: 事件类型
        data: 事件数据
    """
    sse_manager.broadcast_sync(
        f"job:{job_id}",
        f"subtitle.{event_type}",
        data
    )


class StreamingSubtitleManager:
    """流式字幕管理器"""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.sentences: Dict[int, SentenceSegment] = {}  # key = sentence_index
        self.sentence_count = 0
        self.sse_manager = get_sse_manager()

        # Phase 4: Chunk 级别的句子索引映射
        # chunk_sentences[chunk_index] = [sentence_index_1, sentence_index_2, ...]
        self.chunk_sentences: Dict[int, List[int]] = {}

        # V3.8: 添加锁保护，防止 remove_marked_sentences 在错误时机执行
        self._lock = threading.RLock()
        # V3.8: 标记是否允许删除句子
        self._deletion_enabled = False

    def add_sentence(self, sentence: SentenceSegment) -> int:
        """
        添加新句子（SenseVoice 阶段）

        Args:
            sentence: 句子段落

        Returns:
            int: 句子索引
        """
        index = self.sentence_count
        self.sentences[index] = sentence
        self.sentence_count += 1

        # 推送 SSE 事件（使用清洗后的文本）
        sentence_dict = sentence.to_dict() if hasattr(sentence, 'to_dict') else {
            "index": index,
            "text": sentence.text_clean or sentence.text,  # 优先使用清洗后的文本
            "start": sentence.start,
            "end": sentence.end,
            "confidence": sentence.confidence,
            "source": sentence.source.value if hasattr(sentence.source, 'value') else str(sentence.source),
            "words": [w.to_dict() if hasattr(w, 'to_dict') else w for w in getattr(sentence, 'words', [])]  # 确保包含 words
        }

        push_subtitle_event(
            self.sse_manager,
            self.job_id,
            "sv_sentence",
            {
                "index": index,
                "sentence": sentence_dict,
                "source": "sensevoice"
            }
        )

        logger.debug(f"添加句子 {index}: {sentence.text[:30]}...")
        return index

    def update_sentence(
        self,
        index: int,
        new_text: str,
        source: TextSource,
        confidence: float = None,
        perplexity: float = None
    ):
        """
        更新已有句子（Whisper 补刀或 LLM 校对）

        Args:
            index: 句子索引
            new_text: 新文本
            source: 文本来源
            confidence: 新置信度（可选）
            perplexity: LLM 困惑度（可选）
        """
        if index not in self.sentences:
            logger.warning(f"句子 {index} 不存在，无法更新")
            return

        sentence = self.sentences[index]

        # 应用伪对齐
        from .pseudo_alignment import PseudoAlignment
        PseudoAlignment.apply_to_sentence(sentence, new_text, source)

        # 更新置信度和困惑度
        if confidence is not None:
            sentence.confidence = confidence
        if perplexity is not None:
            sentence.perplexity = perplexity
            sentence.warning_type = sentence.compute_warning_type()

        # 推送 SSE 事件
        event_type = {
            TextSource.WHISPER_PATCH: "whisper_patch",
            TextSource.LLM_CORRECTION: "llm_proof",
            TextSource.LLM_TRANSLATION: "llm_trans",
        }.get(source, "batch_update")

        push_subtitle_event(
            self.sse_manager,
            self.job_id,
            event_type,
            {
                "index": index,
                "sentence": sentence.to_dict(),
                "source": source.value,
                "is_update": True
            }
        )

        logger.debug(f"更新句子 {index} ({source.value}): {new_text[:30]}...")

    def set_translation(self, index: int, translation: str, confidence: float = None):
        """
        设置翻译结果

        Args:
            index: 句子索引
            translation: 翻译文本
            confidence: 翻译置信度
        """
        if index not in self.sentences:
            return

        sentence = self.sentences[index]
        sentence.translation = translation
        if confidence is not None:
            sentence.translation_confidence = confidence

        push_subtitle_event(
            self.sse_manager,
            self.job_id,
            "llm_trans",
            {
                "index": index,
                "translation": translation,
                "confidence": confidence
            }
        )

    def mark_for_deletion(self, index: int, reason: str = "garbage"):
        """
        标记句子为待删除（Whisper 仲裁后确认为垃圾）

        Args:
            index: 句子索引
            reason: 删除原因
        """
        if index not in self.sentences:
            return

        sentence = self.sentences[index]
        sentence.marked_for_deletion = True
        sentence.deletion_reason = reason

        # 推送 SSE 事件通知前端
        push_subtitle_event(
            self.sse_manager,
            self.job_id,
            "sentence_deleted",
            {
                "index": index,
                "reason": reason
            }
        )
        logger.info(f"标记删除句子 {index}: {reason}")

    def remove_marked_sentences(self) -> int:
        """
        物理删除被标记为垃圾的句子

        V3.8: 添加锁保护和删除开关，防止在流水线运行期间误删

        Returns:
            int: 删除的句子数量
        """
        # V3.8: 检查删除开关
        if not self._deletion_enabled:
            logger.warning("remove_marked_sentences: 删除功能未启用，跳过删除操作")
            return 0

        with self._lock:
            marked_indices = [
                idx for idx, s in self.sentences.items()
                if getattr(s, 'marked_for_deletion', False)
            ]

            for idx in marked_indices:
                del self.sentences[idx]

            if marked_indices:
                logger.info(f"已删除 {len(marked_indices)} 个垃圾句子: {marked_indices}")

            return len(marked_indices)

    def enable_deletion(self):
        """V3.8: 启用删除功能（在任务完成后调用）"""
        self._deletion_enabled = True
        logger.info(f"StreamingSubtitleManager: 删除功能已启用 job_id={self.job_id}")

    def disable_deletion(self):
        """V3.8: 禁用删除功能（在任务开始时调用）"""
        self._deletion_enabled = False
        logger.info(f"StreamingSubtitleManager: 删除功能已禁用 job_id={self.job_id}")

    def get_all_sentences(self) -> List[SentenceSegment]:
        """获取所有句子（按时间排序，排除已标记删除的）"""
        sentences = [
            s for s in self.sentences.values()
            if not getattr(s, 'marked_for_deletion', False)
        ]
        sentences.sort(key=lambda s: s.start)
        # V3.8 调试日志：导出时记录句子数量
        logger.debug(
            f"get_all_sentences: job_id={self.job_id}, "
            f"total_in_dict={len(self.sentences)}, "
            f"after_filter={len(sentences)}, "
            f"chunk_count={len(self.chunk_sentences)}"
        )
        return sentences

    def get_context_window(self, index: int, window_size: int = 3) -> str:
        """
        获取上下文窗口（用于 LLM 校对和 Whisper 补刀）

        Args:
            index: 当前句子索引
            window_size: 上下文窗口大小

        Returns:
            str: 上下文文本（清洗后的文本，避免传递 SenseVoice 原始 token）
        """
        context_indices = range(max(0, index - window_size), index)
        context_texts = [
            self.sentences[i].text_clean or self.sentences[i].text  # 优先使用清洗后的文本，避免传递下划线等原始 token
            for i in context_indices
            if i in self.sentences
        ]
        return " ".join(context_texts)

    # ========== Phase 4: 双流对齐专用方法 ==========

    def add_draft_sentences(
        self,
        chunk_index: int,
        sentences: List[SentenceSegment]
    ) -> List[int]:
        """
        添加草稿句子（快流推送）

        Phase 4 双流对齐专用方法。
        推送多个草稿句子，并记录 Chunk 级别的索引映射。

        V3.8: 深拷贝句子对象，避免共享引用导致的竞态条件

        Args:
            chunk_index: Chunk 索引
            sentences: 句子列表

        Returns:
            List[int]: 句子索引列表
        """
        sentence_indices = []
        sentences_to_push = []  # V3.8: 收集待推送的句子数据

        with self._lock:
            for sentence in sentences:
                # V3.8 修复：深拷贝句子对象，避免共享引用
                sentence_copy = copy.deepcopy(sentence)

                index = self.sentence_count
                self.sentences[index] = sentence_copy
                self.sentence_count += 1
                sentence_indices.append(index)

                # V3.8: 收集待推送的句子数据（在锁内准备，锁外推送）
                sentence_dict = sentence_copy.to_dict() if hasattr(sentence_copy, 'to_dict') else {
                    "index": index,
                    "text": sentence_copy.text_clean or sentence_copy.text,
                    "start": sentence_copy.start,
                    "end": sentence_copy.end,
                    "confidence": sentence_copy.confidence,
                    "source": sentence_copy.source.value if hasattr(sentence_copy.source, 'value') else str(sentence_copy.source),
                    "is_draft": True,
                    "words": [w.to_dict() if hasattr(w, 'to_dict') else w for w in getattr(sentence_copy, 'words', [])]
                }
                sentences_to_push.append((index, sentence_dict))

            # 记录 Chunk 级别的索引映射
            self.chunk_sentences[chunk_index] = sentence_indices

        # V3.8: 在锁外推送 SSE 事件，避免长时间持锁
        for index, sentence_dict in sentences_to_push:
            push_subtitle_event(
                self.sse_manager,
                self.job_id,
                "draft",
                {
                    "index": index,
                    "chunk_index": chunk_index,
                    "sentence": sentence_dict
                }
            )

        # V3.8 调试日志：确认草稿已添加到管理器
        logger.debug(
            f"add_draft_sentences: Chunk {chunk_index} 添加 {len(sentences)} 个草稿, "
            f"索引 {sentence_indices}, 当前总句子数={len(self.sentences)}"
        )

        return sentence_indices

    def replace_chunk(
        self,
        chunk_index: int,
        sentences: List[SentenceSegment]
    ) -> List[int]:
        """
        替换 Chunk 的所有句子（慢流推送）

        Phase 4 双流对齐专用方法。
        用定稿句子替换整个 Chunk 的草稿句子。

        V3.8: 添加锁保护，防止竞态条件

        流程：
        1. 删除旧的草稿句子
        2. 添加新的定稿句子
        3. 更新 Chunk 索引映射
        4. 推送 replace_chunk 事件

        Args:
            chunk_index: Chunk 索引
            sentences: 定稿句子列表

        Returns:
            List[int]: 新的句子索引列表
        """
        # 防御性检查：如果新句子列表为空，保留原有草稿，不要删除
        if not sentences:
            existing_indices = self.chunk_sentences.get(chunk_index, [])
            logger.warning(
                f"replace_chunk: Chunk {chunk_index} 的定稿句子为空，"
                f"保留原有 {len(existing_indices)} 个草稿句子以避免字幕丢失"
            )
            return existing_indices

        # V3.8: 使用锁保护整个替换过程
        with self._lock:
            # 删除旧的草稿句子
            old_indices = self.chunk_sentences.get(chunk_index, [])
            for old_index in old_indices:
                if old_index in self.sentences:
                    del self.sentences[old_index]

            # 添加新的定稿句子
            new_indices = []
            for sentence in sentences:
                index = self.sentence_count
                self.sentences[index] = sentence
                self.sentence_count += 1
                new_indices.append(index)

            # 更新 Chunk 索引映射
            self.chunk_sentences[chunk_index] = new_indices

        # 推送 SSE 事件（批量替换）- 在锁外推送，避免死锁
        sentences_data = [
            sentence.to_dict() if hasattr(sentence, 'to_dict') else {
                "index": new_indices[i],
                "text": sentence.text_clean or sentence.text,
                "start": sentence.start,
                "end": sentence.end,
                "confidence": sentence.confidence,
                "source": sentence.source.value if hasattr(sentence.source, 'value') else str(sentence.source),
                "is_draft": False,
                "is_finalized": True,
                "words": [w.to_dict() if hasattr(w, 'to_dict') else w for w in getattr(sentence, 'words', [])]
            }
            for i, sentence in enumerate(sentences)
        ]

        push_subtitle_event(
            self.sse_manager,
            self.job_id,
            "replace_chunk",
            {
                "chunk_index": chunk_index,
                "old_indices": old_indices,
                "new_indices": new_indices,
                "sentences": sentences_data
            }
        )

        # V3.8 调试日志：确认替换成功
        logger.debug(
            f"replace_chunk: Chunk {chunk_index} 替换完成 - "
            f"删除 {len(old_indices)} 个草稿 {old_indices}, "
            f"添加 {len(new_indices)} 个定稿 {new_indices}, "
            f"当前总句子数={len(self.sentences)}"
        )

        return new_indices

    def add_finalized_sentences(
        self,
        chunk_index: int,
        sentences: List[SentenceSegment]
    ) -> List[int]:
        """
        添加定稿句子（极速模式专用）

        V3.5 新增: 极速模式下 FastWorker 直接输出定稿，不经过 SlowWorker。
        与 add_draft_sentences 不同，这里直接推送定稿事件。

        V3.8: 添加锁保护和深拷贝，防止竞态条件

        Args:
            chunk_index: Chunk 索引
            sentences: 定稿句子列表

        Returns:
            List[int]: 句子索引列表
        """
        sentence_indices = []
        sentences_to_push = []  # V3.8: 收集待推送的句子数据

        with self._lock:
            for sentence in sentences:
                # V3.8 修复：深拷贝句子对象，避免共享引用
                sentence_copy = copy.deepcopy(sentence)

                # 确保句子标记为定稿
                sentence_copy.is_draft = False
                sentence_copy.is_finalized = True

                index = self.sentence_count
                self.sentences[index] = sentence_copy
                self.sentence_count += 1
                sentence_indices.append(index)

                # V3.8: 收集待推送的句子数据（在锁内准备，锁外推送）
                sentence_dict = sentence_copy.to_dict() if hasattr(sentence_copy, 'to_dict') else {
                    "index": index,
                    "text": sentence_copy.text_clean or sentence_copy.text,
                    "start": sentence_copy.start,
                    "end": sentence_copy.end,
                    "confidence": sentence_copy.confidence,
                    "source": sentence_copy.source.value if hasattr(sentence_copy.source, 'value') else str(sentence_copy.source),
                    "is_draft": False,
                    "is_finalized": True,
                    "words": [w.to_dict() if hasattr(w, 'to_dict') else w for w in getattr(sentence_copy, 'words', [])]
                }
                sentences_to_push.append((index, sentence_dict))

            # 记录 Chunk 级别的索引映射
            self.chunk_sentences[chunk_index] = sentence_indices

        # V3.8: 在锁外推送 SSE 事件，避免死锁
        for index, sentence_dict in sentences_to_push:
            push_subtitle_event(
                self.sse_manager,
                self.job_id,
                "finalized",
                {
                    "index": index,
                    "chunk_index": chunk_index,
                    "sentence": sentence_dict,
                    "mode": "sensevoice_only"
                }
            )

        logger.debug(
            f"添加定稿句子 [极速模式]: Chunk {chunk_index}, "
            f"{len(sentences)} 个句子, 索引 {sentence_indices}"
        )

        return sentence_indices

    # ========== V3.1.0: 字幕持久化方法 ==========

    def to_checkpoint_data(self) -> dict:
        """
        V3.1.0: 导出字幕快照用于 Checkpoint 保存

        返回完整的字幕状态，包括：
        - sentences_snapshot: 所有句子的序列化数据
        - sentence_count: 全局句子计数器
        - chunk_sentences_map: Chunk 到句子索引的映射

        Returns:
            dict: 可直接保存到 Checkpoint 的字幕数据
        """
        sentences_snapshot = []
        for idx, sentence in self.sentences.items():
            # 使用 SentenceSegment.to_dict() 序列化
            sentence_dict = sentence.to_dict() if hasattr(sentence, 'to_dict') else {
                "text": sentence.text_clean or sentence.text,
                "start": sentence.start,
                "end": sentence.end,
                "confidence": sentence.confidence,
                "source": sentence.source.value if hasattr(sentence.source, 'value') else str(sentence.source),
            }
            # 添加索引信息
            sentence_dict["_index"] = idx
            sentence_dict["_is_draft"] = getattr(sentence, 'is_draft', False)
            sentence_dict["_is_finalized"] = getattr(sentence, 'is_finalized', False)
            sentences_snapshot.append(sentence_dict)

        return {
            "sentences_snapshot": sentences_snapshot,
            "sentence_count": self.sentence_count,
            "chunk_sentences_map": self.chunk_sentences
        }

    def restore_from_checkpoint(self, checkpoint_data: dict) -> bool:
        """
        V3.1.0: 从 Checkpoint 恢复字幕状态

        恢复所有已保存的句子，并恢复索引计数器状态。
        恢复后，新添加的句子会从正确的索引继续编号，不会与已有句子冲突。

        Args:
            checkpoint_data: Checkpoint 中的字幕数据，包含：
                - sentences_snapshot: 句子快照列表
                - sentence_count: 句子计数器
                - chunk_sentences_map: Chunk 映射

        Returns:
            bool: 恢复是否成功
        """
        from app.models.sensevoice_models import SentenceSegment, TextSource, WarningType, WordTimestamp

        try:
            sentences_snapshot = checkpoint_data.get("sentences_snapshot", [])
            sentence_count = checkpoint_data.get("sentence_count", 0)
            chunk_sentences_map = checkpoint_data.get("chunk_sentences_map", {})

            if not sentences_snapshot:
                logger.info(f"[V3.1.0] 无字幕快照需要恢复: job_id={self.job_id}")
                return True

            # 恢复句子
            restored_count = 0
            for sentence_dict in sentences_snapshot:
                idx = sentence_dict.get("_index")
                if idx is None:
                    continue

                # 从字典重建 SentenceSegment
                sentence = SentenceSegment(
                    text=sentence_dict.get("original_text") or sentence_dict.get("text", ""),
                    text_clean=sentence_dict.get("text", ""),
                    start=sentence_dict.get("start", 0.0),
                    end=sentence_dict.get("end", 0.0),
                    confidence=sentence_dict.get("confidence", 1.0),
                )

                # 恢复来源
                source_str = sentence_dict.get("source", "sensevoice")
                try:
                    sentence.source = TextSource(source_str)
                except ValueError:
                    sentence.source = TextSource.SENSEVOICE

                # 恢复警告类型
                warning_str = sentence_dict.get("warning_type", "none")
                try:
                    sentence.warning_type = WarningType(warning_str)
                except ValueError:
                    sentence.warning_type = WarningType.NONE

                # 恢复其他字段
                sentence.is_modified = sentence_dict.get("is_modified", False)
                sentence.original_text = sentence_dict.get("original_text")
                sentence.whisper_alternative = sentence_dict.get("whisper_alternative")
                sentence.perplexity = sentence_dict.get("perplexity")
                sentence.translation = sentence_dict.get("translation")
                sentence.translation_confidence = sentence_dict.get("translation_confidence")
                sentence.is_draft = sentence_dict.get("_is_draft", False)
                sentence.is_finalized = sentence_dict.get("_is_finalized", False)

                # 恢复字级时间戳
                words_data = sentence_dict.get("words", [])
                sentence.words = []
                for word_dict in words_data:
                    word = WordTimestamp(
                        word=word_dict.get("word", ""),
                        start=word_dict.get("start", 0.0),
                        end=word_dict.get("end", 0.0),
                        confidence=word_dict.get("confidence", 1.0),
                        is_pseudo=word_dict.get("is_pseudo", False)
                    )
                    sentence.words.append(word)

                self.sentences[idx] = sentence
                restored_count += 1

            # 恢复计数器（关键：确保新句子索引不会冲突）
            self.sentence_count = max(sentence_count, restored_count)

            # 恢复 Chunk 映射
            # JSON 反序列化后键是 str，需要转换为 int
            if chunk_sentences_map:
                self.chunk_sentences = {
                    int(k): v for k, v in chunk_sentences_map.items()
                }

            logger.info(
                f"[V3.1.0] 字幕恢复成功: job_id={self.job_id}, "
                f"恢复了 {restored_count} 个句子, "
                f"sentence_count={self.sentence_count}, "
                f"chunk_count={len(self.chunk_sentences)}"
            )
            return True

        except Exception as e:
            logger.error(f"[V3.1.0] 字幕恢复失败: job_id={self.job_id}, error={e}", exc_info=True)
            return False

    def push_restored_subtitles_to_frontend(self):
        """
        V3.1.0: 恢复后推送所有字幕到前端

        在恢复字幕后调用此方法，将已恢复的字幕通过 SSE 推送到前端，
        确保前端状态与后端同步。
        """
        # 按 Chunk 分组推送
        for chunk_index, sentence_indices in self.chunk_sentences.items():
            sentences_data = []
            for idx in sentence_indices:
                if idx in self.sentences:
                    sentence = self.sentences[idx]
                    sentence_dict = sentence.to_dict() if hasattr(sentence, 'to_dict') else {
                        "text": sentence.text_clean or sentence.text,
                        "start": sentence.start,
                        "end": sentence.end,
                        "confidence": sentence.confidence,
                    }
                    sentence_dict["index"] = idx
                    sentence_dict["is_draft"] = getattr(sentence, 'is_draft', False)
                    sentence_dict["is_finalized"] = getattr(sentence, 'is_finalized', True)
                    sentences_data.append(sentence_dict)

            if sentences_data:
                # 推送恢复事件（使用新的事件类型，避免与实时推送混淆）
                push_subtitle_event(
                    self.sse_manager,
                    self.job_id,
                    "restored",  # 恢复事件类型
                    {
                        "chunk_index": chunk_index,
                        "sentences": sentences_data,
                        "is_restore": True
                    }
                )

        logger.info(
            f"[V3.1.0] 已推送恢复的字幕到前端: job_id={self.job_id}, "
            f"chunks={len(self.chunk_sentences)}, "
            f"total_sentences={len(self.sentences)}"
        )


# ========== 单例工厂 ==========

_subtitle_managers: Dict[str, StreamingSubtitleManager] = {}


def get_streaming_subtitle_manager(job_id: str) -> StreamingSubtitleManager:
    """获取或创建流式字幕管理器"""
    global _subtitle_managers
    if job_id not in _subtitle_managers:
        _subtitle_managers[job_id] = StreamingSubtitleManager(job_id)
    return _subtitle_managers[job_id]


def remove_streaming_subtitle_manager(job_id: str):
    """移除流式字幕管理器"""
    global _subtitle_managers
    if job_id in _subtitle_managers:
        del _subtitle_managers[job_id]
