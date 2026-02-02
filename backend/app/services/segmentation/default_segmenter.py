"""
默认分句服务（封装旧分句逻辑）。
V3.2.0+dev.20260119.06
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from app.models.sensevoice_models import SentenceSegment, TextSource, WordTimestamp
from app.services.sentence_splitter import SentenceSplitter, SplitConfig
from app.services.semantic_grouper import SemanticGrouper, GroupConfig

if TYPE_CHECKING:
    from app.services.audio.chunk_engine import AudioChunk


class DefaultSegmenter:
    """默认分句服务：封装旧分句/语义分组逻辑。"""

    def __init__(
        self,
        is_enable_semantic_grouping: bool = True,
        is_enable_cross_chunk_merge: bool = False,
        draft_split_config: Optional[SplitConfig] = None,
        chinese_split_config: Optional[SplitConfig] = None,
        draft_group_config: Optional[GroupConfig] = None,
        final_split_config: Optional[SplitConfig] = None,
        final_group_config: Optional[GroupConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.is_enable_semantic_grouping = is_enable_semantic_grouping
        self.is_enable_cross_chunk_merge = is_enable_cross_chunk_merge
        self.pending_sentence: Optional[SentenceSegment] = None

        if draft_split_config is None:
            draft_split_config = SplitConfig(
                language="en",
                prefer_punctuation_break=False,
                use_dynamic_pause=True,
                pause_threshold=0.5,
                max_duration=5.0,
                enable_hard_limit=True,
                hard_limit_duration=10.0,
                merge_short_sentences=True,
            )
        self.draft_splitter = SentenceSplitter(draft_split_config)

        if chinese_split_config is None:
            chinese_split_config = SplitConfig(
                language="zh",
                prefer_punctuation_break=True,
                delay_split_to_punctuation=True,
                delay_split_max_wait=8.0,
                use_dynamic_pause=True,
                pause_threshold=0.3,
                max_duration=12.0,
                enable_hard_limit=True,
                hard_limit_duration=20.0,
                merge_short_sentences=True,
                min_duration_threshold=1.5,
            )
        self.chinese_splitter = SentenceSplitter(chinese_split_config)

        if draft_group_config is None:
            draft_group_config = GroupConfig(
                max_group_gap=2.0,
                max_group_duration=10.0,
                max_group_sentences=5,
                enable_overlap_detection=True,
            )
        self.draft_grouper = SemanticGrouper(draft_group_config)

        if final_split_config is None:
            final_split_config = SplitConfig(
                prefer_punctuation_break=True,
                use_dynamic_pause=True,
                pause_threshold=0.5,
                max_duration=5.0,
                enable_hard_limit=True,
                hard_limit_duration=20.0,
                delay_split_to_punctuation=True,
                delay_split_max_wait=15.0,
                merge_short_sentences=True,
            )
        self.final_splitter = SentenceSplitter(final_split_config)

        if final_group_config is None:
            final_group_config = GroupConfig(
                max_group_gap=2.0,
                max_group_duration=10.0,
                max_group_sentences=5,
                enable_overlap_detection=True,
            )
        self.final_grouper = SemanticGrouper(final_group_config)

    def split_draft(
        self,
        sv_result: Dict[str, Any],
        chunk: AudioChunk,
        is_draft: bool = True,
    ) -> List[SentenceSegment]:
        """快流分句（草稿/定稿）。"""
        sentences = self._split_draft_core(sv_result, chunk, is_draft=is_draft)

        if not self.is_enable_cross_chunk_merge or not sentences:
            return sentences

        if self.pending_sentence:
            sentences[0] = self._merge_sentences(self.pending_sentence, sentences[0])
            self.pending_sentence = None

        if self._is_sentence_incomplete(sentences[-1]):
            self.pending_sentence = sentences[-1]
            sentences = sentences[:-1]
            self.logger.debug("[跨chunk] 缓存最后一句，等待下一个chunk")

        return sentences

    def split_final_from_sv(
        self,
        sv_result: Dict[str, Any],
        chunk: AudioChunk,
        language: Optional[str] = None,
    ) -> List[SentenceSegment]:
        """慢流降级：使用 SenseVoice 结果分句。"""
        text_clean = sv_result.get("text_clean", "")
        text_display = sv_result.get("text_itn_raw") or text_clean
        words_data = sv_result.get("words", [])
        words = self._build_words(words_data)

        if not words:
            if text_display and text_display.strip():
                self.logger.warning(
                    "SenseVoice 没有字级时间戳但有文本，创建兜底单句: "
                    "text='%s...', chunk=[%.2fs, %.2fs]",
                    text_display[:50],
                    chunk.start,
                    chunk.end,
                )
                return [
                    SentenceSegment(
                        text=text_display.strip(),
                        start=chunk.start,
                        end=chunk.end,
                        words=[],
                        source=TextSource.SENSEVOICE,
                        confidence=sv_result.get("confidence", 0.5),
                        is_finalized=True,
                        is_draft=False,
                    )
                ]
            self.logger.warning("SenseVoice 结果没有字级时间戳且无文本，无法分句")
            return []

        if language:
            self.final_splitter.config.language = language

        sentences = self.final_splitter.split(words, text_display)
        if self.is_enable_semantic_grouping:
            sentences = self.final_grouper.group(sentences)

        for sentence in sentences:
            sentence.start += chunk.start
            sentence.end += chunk.start
            sentence.source = TextSource.SENSEVOICE
            sentence.is_finalized = True
            sentence.is_draft = False
            for word in sentence.words:
                word.start += chunk.start
                word.end += chunk.start

        return sentences

    def _split_draft_core(
        self,
        sv_result: Dict[str, Any],
        chunk: AudioChunk,
        is_draft: bool,
    ) -> List[SentenceSegment]:
        text_clean = sv_result.get("text_clean", "")
        text_display = sv_result.get("text_itn_raw") or text_clean
        words_data = sv_result.get("words", [])
        words = self._build_words(words_data)

        if not words:
            if text_display and text_display.strip():
                self.logger.warning(
                    "SenseVoice 没有字级时间戳但有文本，创建兜底单句: "
                    "text='%s...', chunk=[%.2fs, %.2fs]",
                    text_display[:50],
                    chunk.start,
                    chunk.end,
                )
                return [
                    SentenceSegment(
                        text=text_display.strip(),
                        start=chunk.start,
                        end=chunk.end,
                        words=[],
                        source=TextSource.SENSEVOICE,
                        confidence=sv_result.get("confidence", 0.5),
                        is_finalized=not is_draft,
                        is_draft=is_draft,
                    )
                ]
            self.logger.warning("SenseVoice 结果没有字级时间戳且无文本，无法分句")
            return []

        detected_language = sv_result.get("language", "auto")
        is_chinese = detected_language in {"zh", "yue"}

        if is_chinese:
            sentences = self.chinese_splitter.split(words, text_display)
        else:
            sentences = self.draft_splitter.split(words, text_display)

        if self.is_enable_semantic_grouping:
            sentences = self.draft_grouper.group(sentences)
            self.logger.debug("快流语义分组: %d 个句子（物理约束）", len(sentences))

        for sentence in sentences:
            sentence.start += chunk.start
            sentence.end += chunk.start
            sentence.is_draft = is_draft
            sentence.is_finalized = not is_draft
            sentence.source = TextSource.SENSEVOICE
            for word in sentence.words:
                word.start += chunk.start
                word.end += chunk.start

        self.logger.debug("快流分句完成: %d 个句子", len(sentences))
        return sentences

    @staticmethod
    def _build_words(words_data: List[Dict[str, Any]]) -> List[WordTimestamp]:
        words: List[WordTimestamp] = []
        for word in words_data:
            words.append(
                WordTimestamp(
                    word=word.get("word", ""),
                    start=word.get("start", 0.0),
                    end=word.get("end", 0.0),
                    confidence=word.get("confidence", 1.0),
                    confidence_raw=word.get("confidence_raw"),
                    confidence_display_raw=word.get("confidence_display_raw"),
                    token_type=word.get("token_type"),
                )
            )
        return words

    def _is_sentence_incomplete(self, sentence: SentenceSegment) -> bool:
        """判断句子是否语义不完整，用于跨 chunk 合并。"""
        if not sentence or not sentence.text:
            return False

        text = sentence.text.strip()
        is_chinese = any("\u4e00" <= char <= "\u9fff" for char in text)

        if is_chinese:
            strategy = self.chinese_splitter.config.get_strategy()
        else:
            strategy = self.draft_splitter.config.get_strategy()

        is_incomplete = strategy.is_incomplete_ending(text)
        if is_incomplete:
            self.logger.debug("[跨chunk检查] 句子语义不完整: '%s'", sentence.text[-20:])
        return is_incomplete

    @staticmethod
    def _merge_sentences(sent1: SentenceSegment, sent2: SentenceSegment) -> SentenceSegment:
        """合并两个句子（跨 chunk 合并）。"""
        merged_text = sent1.text + sent2.text
        merged_start = sent1.start
        merged_end = sent2.end
        merged_words = sent1.words + sent2.words

        sent1_conf = sent1.confidence or 0.0
        sent2_conf = sent2.confidence or 0.0
        strict_confidence = min(sent1_conf, sent2_conf)

        sent1_display = sent1.confidence_display_raw
        sent2_display = sent2.confidence_display_raw
        if sent1_display is None:
            sent1_display = sent1_conf
        if sent2_display is None:
            sent2_display = sent2_conf
        weight1 = max(sent1.end - sent1.start, 0.0)
        weight2 = max(sent2.end - sent2.start, 0.0)
        total_weight = (weight1 if weight1 > 0.0 else 1.0) + (weight2 if weight2 > 0.0 else 1.0)
        display_raw = (
            ((sent1_display * (weight1 if weight1 > 0.0 else 1.0)) +
             (sent2_display * (weight2 if weight2 > 0.0 else 1.0))) / total_weight
        ) if total_weight > 0.0 else None

        merged_sentence = SentenceSegment(
            text=merged_text,
            start=merged_start,
            end=merged_end,
            words=merged_words,
            confidence=strict_confidence,
            confidence_display_raw=display_raw,
            is_draft=sent1.is_draft,
            is_finalized=sent1.is_finalized,
            source=sent1.source,
            warning_type=sent1.warning_type or sent2.warning_type,
        )

        return merged_sentence
