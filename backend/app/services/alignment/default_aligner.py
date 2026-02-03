"""
默认对齐服务（封装旧对齐逻辑）。
V3.2.0+dev.20260203.02
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from app.models.sensevoice_models import SentenceSegment, TextSource, WordTimestamp
from app.services.alignment.alignment_service import AlignmentService, AlignmentConfig
from app.services.pseudo_alignment import PseudoAlignment
from app.services.punctuation.base import PuncPosition
from app.services.model_runtime_config_service import get_model_runtime_config_service
from app.services.punctuation.semantic_injector import SemanticInjector
from app.services.punctuation.final_splitter import FinalSplitter, FinalSplitConfig
from app.services.semantic_grouper import SemanticGrouper, GroupConfig
from app.services.segmentation.unified_splitter import UnifiedSplitter

if TYPE_CHECKING:
    from app.services.audio.chunk_engine import AudioChunk


class AlignmentLevel(Enum):
    """对齐级别（三级降级策略）。"""

    DUAL_MODAL = "dual_modal"
    WHISPER_PSEUDO = "whisper_pseudo"
    SENSEVOICE_ONLY = "sensevoice_only"


class DefaultAligner:
    """默认对齐服务：双流对齐 + 伪对齐 + 草稿兜底。"""

    def __init__(
        self,
        alignment_config: Optional[AlignmentConfig] = None,
        final_split_config: Optional[FinalSplitConfig] = None,
        final_group_config: Optional[GroupConfig] = None,
        is_enable_semantic_grouping: bool = True,
        alignment_score_threshold: float = 0.3,
        is_enable_fallback: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.is_enable_semantic_grouping = is_enable_semantic_grouping
        self.alignment_score_threshold = alignment_score_threshold
        self.is_enable_fallback = is_enable_fallback

        if alignment_config is None:
            alignment_config = AlignmentConfig()
        self.alignment_service = AlignmentService(
            config=alignment_config,
            logger=self.logger,
        )

        if final_split_config is None:
            final_split_config = FinalSplitConfig(
                min_tokens=5,
                max_tokens=50,
                min_duration=0.5,
                max_duration=10.0,
                soft_pause=0.35,
                long_pause=0.8,
                min_mapping_coverage=0.6,
            )
        self.final_splitter = FinalSplitter(final_split_config, logger=self.logger)

        if final_group_config is None:
            final_group_config = GroupConfig(
                max_group_gap=2.0,
                max_group_duration=10.0,
                max_group_sentences=5,
                enable_overlap_detection=True,
            )
        self.final_grouper = SemanticGrouper(final_group_config)
        # V3.2.0+dev.20260202.08: 语义注入器（对齐后标点注入）
        self._semantic_injector = SemanticInjector(
            logger=self.logger,
            min_mapping_coverage=final_split_config.min_mapping_coverage,
        )
        self._unified_splitter = UnifiedSplitter(logger=self.logger)
        self._keep_sentence_end_punct = self._load_keep_sentence_end_punct()
        self.last_alignment_stats: Optional[Dict[str, Any]] = None

    @staticmethod
    def _load_keep_sentence_end_punct() -> bool:
        runtime = get_model_runtime_config_service().get_effective_runtime_global()
        punct = runtime.get("effective", {}).get("punctuation", {})
        return bool(punct.get("keep_sentence_end_punct", False))

    @staticmethod
    def _count_words_language_aware(text: str, language: str) -> int:
        """
        语言感知的单词数统计。
        V3.2.0+dev.20260203.03: 修复中文单词数统计缺陷

        Args:
            text: 待统计文本
            language: 语言代码（zh/ja/ko/en/auto等）

        Returns:
            单词数（CJK语言返回字符数，其他语言返回空格分词数）
        """
        if not text:
            return 0

        # CJK 语言（中日韩）使用字符数统计
        if language in ("zh", "ja", "ko", "auto"):
            import re
            # 移除标点和空格，统计实际字符数
            clean_text = re.sub(r'[^\w]', '', text)
            return len(clean_text)

        # 其他语言使用空格分词
        return len(text.split())

    async def align(
        self,
        whisper_result: Dict[str, Any],
        sv_result: Dict[str, Any],
        chunk: AudioChunk,
        vad_intervals: Optional[List[Tuple[float, float]]] = None,
        *,
        punctuation_positions: Optional[List[PuncPosition]] = None,
        punctuation_clean_text: Optional[str] = None,
    ) -> Tuple[List[SentenceSegment], AlignmentLevel]:
        """执行双流对齐并完成降级兜底。"""
        self.last_alignment_stats = None
        detected_language = whisper_result.get("language", "auto")
        self.final_splitter.set_language(detected_language)
        self.logger.debug("使用 Whisper 检测到的语言: %s", detected_language)

        whisper_text = whisper_result.get("text", "").strip()
        whisper_text_raw = whisper_result.get("text_raw", "").strip()
        sv_text_clean = sv_result.get("text_clean", "").strip()

        # V3.2.0+dev.20260203.03: 组合方案 - 使用原始文本 + 语言感知统计
        if whisper_text or whisper_text_raw:
            # 保护机制：清洗前后长度差异过大，说明可能是幻觉，用清洗后文本
            if whisper_text_raw and whisper_text:
                len_diff_ratio = abs(len(whisper_text_raw) - len(whisper_text)) / max(len(whisper_text_raw), 1)
                text_for_check = whisper_text if len_diff_ratio > 0.5 else whisper_text_raw
            else:
                text_for_check = whisper_text_raw or whisper_text

            # 语言感知的单词数/字符数统计
            word_count = self._count_words_language_aware(text_for_check, detected_language)
            min_threshold = 2  # 中英文统一阈值（字符数/单词数）

            if word_count < min_threshold and sv_text_clean:
                self.logger.warning(
                    "Whisper 文本过短（%d < %d, language=%s），直接降级到 SenseVoice",
                    word_count,
                    min_threshold,
                    detected_language,
                )
                sentences = self.split_sensevoice_only(sv_result, chunk)
                return sentences, AlignmentLevel.SENSEVOICE_ONLY

        if sv_text_clean:
            len_whisper = len(whisper_text)
            len_sv = len(sv_text_clean)

            if len_whisper > 3 * len_sv + 10:
                whisper_confidence = whisper_result.get("confidence", 0.5)
                if whisper_confidence < 0.5:
                    self.logger.warning(
                        "Whisper 长度暴涨且置信度低，直接降级到 SenseVoice",
                    )
                    sentences = self.split_sensevoice_only(sv_result, chunk)
                    return sentences, AlignmentLevel.SENSEVOICE_ONLY

            elif len_whisper < len_sv * 0.65:
                self.logger.warning(
                    "Whisper 输出明显短于 SenseVoice，直接降级到 SenseVoice",
                )
                sentences = self.split_sensevoice_only(sv_result, chunk)
                return sentences, AlignmentLevel.SENSEVOICE_ONLY

        try:
            sv_words_data = sv_result.get("words", [])
            if not whisper_text or not sv_words_data:
                raise ValueError("Whisper 或 SenseVoice 结果为空")

            sv_tokens = self._build_words(sv_words_data)

            aligned_subtitle = await self.alignment_service.align(
                whisper_text=whisper_text,
                sv_tokens=sv_tokens,
                vad_range=(0.0, chunk.duration),
                chunk_offset=chunk.start,
                audio_array=chunk.audio,
                sample_rate=chunk.sample_rate,
                vad_intervals=vad_intervals,
            )

            self.last_alignment_stats = {
                "coverage": aligned_subtitle.coverage,
                "gap_ratio": aligned_subtitle.gap_ratio,
                "alignment_score": aligned_subtitle.alignment_score,
                "gap_positions": aligned_subtitle.gap_positions,
                "gap_resolution": aligned_subtitle.gap_resolution,
            }

            if aligned_subtitle.alignment_score < self.alignment_score_threshold:
                raise ValueError(f"对齐质量过低: {aligned_subtitle.alignment_score:.2f}")

            aligned_words = aligned_subtitle.words
            words_for_split = [
                WordTimestamp(
                    word=word.word,
                    start=word.start,
                    end=word.end,
                    confidence=word.final_confidence,
                    is_pseudo=word.is_pseudo,
                )
                for word in aligned_words
            ]

            text_for_split = whisper_text
            injection_stats: Dict[str, Any] = {
                "injection_positions_total": len(punctuation_positions or []),
                "injection_unmatched_total": 0,
                "injection_miss_ratio": 0.0,
                "injection_mapping_coverage": 0.0,
                "injection_blocked": 0.0,
            }
            if punctuation_positions and punctuation_clean_text is not None:
                # V3.2.0+dev.20260203.04: 添加标点映射调试日志
                aligned_text_concat = "".join(w.word for w in aligned_words)
                self.logger.debug(
                    "标点映射基准检查: whisper_text='%s', punct_clean_text='%s', aligned_concat='%s', "
                    "whisper==punct=%s, aligned==punct=%s, punct_positions=%d",
                    whisper_text[:50],
                    punctuation_clean_text[:50],
                    aligned_text_concat[:50],
                    whisper_text == punctuation_clean_text,
                    aligned_text_concat == punctuation_clean_text,
                    len(punctuation_positions),
                )
                injection = self._semantic_injector.inject(
                    aligned_words,
                    punctuation_clean_text,
                    punctuation_positions,
                    language=detected_language,
                )
                injection_stats["injection_unmatched_total"] = len(injection.unmatched_positions)
                injection_stats["injection_miss_ratio"] = (
                    len(injection.unmatched_positions) / max(len(punctuation_positions), 1)
                )
                injection_stats["injection_mapping_coverage"] = injection.mapping_coverage
                injection_stats["injection_blocked"] = 1.0 if injection.is_mapping_blocked else 0.0
                if injection.annotated_words and not injection.is_mapping_blocked:
                    words_for_split = self._semantic_injector.build_word_timestamps(
                        aligned_words,
                        injection.annotated_words,
                    )
                    text_for_split = injection.punctuated_text or text_for_split

            sentences = self._split_with_final(
                words_for_split,
                text_for_split,
                punctuation_positions=punctuation_positions,
                punctuation_clean_text=punctuation_clean_text,
            )
            if not self._keep_sentence_end_punct:
                self._strip_sentence_end_punct(sentences)
            for sentence in sentences:
                sentence.source = TextSource.WHISPER_PATCH
                sentence.is_finalized = True
                sentence.is_draft = False
                sentence.alignment_score = aligned_subtitle.alignment_score
                sentence.matched_ratio = aligned_subtitle.matched_ratio
                sentence.whisper_text = whisper_text

            split_stats = self.final_splitter.last_split_stats or {}
            if self.last_alignment_stats is None:
                self.last_alignment_stats = {}
            self.last_alignment_stats.update(injection_stats)
            for key, value in split_stats.items():
                self.last_alignment_stats[f"split_{key}"] = value

            self.logger.debug(
                "双模态对齐成功: alignment_score=%.2f, matched_ratio=%.2f, 分句数=%d",
                aligned_subtitle.alignment_score,
                aligned_subtitle.matched_ratio,
                len(sentences),
            )

            return sentences, AlignmentLevel.DUAL_MODAL

        except Exception as exc:
            self.logger.warning("双模态对齐失败: %s，降级到 Whisper 伪对齐", exc)
            if not self.is_enable_fallback:
                raise

            try:
                if not whisper_text:
                    raise ValueError("Whisper 结果为空")

                words = PseudoAlignment.apply(
                    original_start=0.0,
                    original_end=chunk.duration,
                    new_text=whisper_text,
                )

                for word in words:
                    word.start += chunk.start
                    word.end += chunk.start

                sentences = self._split_with_final(
                    words,
                    whisper_text,
                    punctuation_positions=punctuation_positions,
                    punctuation_clean_text=punctuation_clean_text,
                )
                if not self._keep_sentence_end_punct:
                    self._strip_sentence_end_punct(sentences)
                for sentence in sentences:
                    sentence.source = TextSource.WHISPER_PATCH
                    sentence.is_finalized = True
                    sentence.is_draft = False
                    sentence.alignment_score = 0.5
                    sentence.whisper_text = whisper_text

                self.logger.debug("Whisper 伪对齐成功, 分句数=%d", len(sentences))
                split_stats = self.final_splitter.last_split_stats or {}
                if self.last_alignment_stats is None:
                    self.last_alignment_stats = {}
                for key, value in split_stats.items():
                    self.last_alignment_stats[f"split_{key}"] = value
                return sentences, AlignmentLevel.WHISPER_PSEUDO

            except Exception as exc2:
                self.logger.error("Whisper 伪对齐失败: %s，降级到 SenseVoice", exc2)
                sentences = self.split_sensevoice_only(sv_result, chunk)
                return sentences, AlignmentLevel.SENSEVOICE_ONLY

    def split_sensevoice_only(
        self,
        sv_result: Dict[str, Any],
        chunk: AudioChunk,
    ) -> List[SentenceSegment]:
        """将 SenseVoice 结果作为最终输出（统一切分出口）。"""
        return self._unified_splitter.split_draft_from_sv(
            sv_result,
            chunk_start=chunk.start,
            chunk_end=chunk.end,
            is_final_output=True,
        )

    def _split_with_final(
        self,
        words: List[WordTimestamp],
        text: str,
        *,
        punctuation_positions: Optional[List[PuncPosition]] = None,
        punctuation_clean_text: Optional[str] = None,
    ) -> List[SentenceSegment]:
        sentences = self.final_splitter.split(
            words,
            clean_text=punctuation_clean_text,
            punctuation_positions=punctuation_positions,
        )
        if self.is_enable_semantic_grouping:
            sentences = self.final_grouper.group(sentences)
        return sentences

    @staticmethod
    def _strip_sentence_end_punct(sentences: List[SentenceSegment]) -> None:
        """
        智能清理句末标点（保留情感标点）

        V3.2.0+dev.20260203.06: 增强句末标点清理
        - 总是清理：句号（.、。）、逗号（,、，）、顿号（、）、分号（;、；）
        - 总是保留：问号（?、？）、感叹号（!、！）
        """
        for sentence in sentences:
            sentence.text = _strip_trailing_punct_smart(sentence.text)
            if sentence.text_clean:
                sentence.text_clean = _strip_trailing_punct_smart(sentence.text_clean)
            if sentence.words:
                last_word = sentence.words[-1]
                last_word.word = _strip_trailing_punct_smart(last_word.word)

    def _split_from_sv(
        self,
        sv_result: Dict[str, Any],
        chunk: AudioChunk,
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
                        is_finalized=True,
                        is_draft=False,
                    )
                ]
            self.logger.warning("SenseVoice 结果没有字级时间戳且无文本，无法分句")
            return []

        sentences = self._split_with_final(
            words,
            text_display,
            punctuation_clean_text=text_display,
        )

        for sentence in sentences:
            sentence.start += chunk.start
            sentence.end += chunk.start
            sentence.source = TextSource.SENSEVOICE
            for word in sentence.words:
                word.start += chunk.start
                word.end += chunk.start

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
                )
            )
        return words


def _strip_trailing_punct_smart(text: Optional[str]) -> str:
    """
    智能清理句末标点（保留情感标点）

    V3.2.0+dev.20260203.06: 智能标点清理策略
    - 清理：句号（.、。）、逗号（,、，）、顿号（、）、分号（;、；）
    - 保留：问号（?、？）、感叹号（!、！）

    Args:
        text: 待清理文本

    Returns:
        清理后的文本
    """
    if not text:
        return ""

    # 需要清理的标点（句号、逗号、顿号、分号）
    punct_to_remove = {"。", ".", "，", ",", "、", "；", ";"}

    idx = len(text) - 1
    while idx >= 0 and text[idx] in punct_to_remove:
        idx -= 1

    return text[: idx + 1]
