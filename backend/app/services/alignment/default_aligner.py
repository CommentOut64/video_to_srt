"""
默认对齐服务（封装旧对齐逻辑）。
V3.2.0+dev.20260203.03
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from app.models.sensevoice_models import SentenceSegment, TextSource, WordTimestamp
from app.services.alignment.alignment_service import AlignmentService, AlignmentConfig
from app.services.pseudo_alignment import PseudoAlignment
from app.services.punctuation.base import PuncPosition
from app.services.punctuation.semantic_injector import SemanticInjector
from app.services.punctuation.final_splitter import FinalSplitter, FinalSplitConfig
from app.services.semantic_grouper import SemanticGrouper, GroupConfig

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
        self.last_alignment_stats: Optional[Dict[str, Any]] = None

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
        sv_text_clean = sv_result.get("text_clean", "").strip()

        if whisper_text:
            word_count = len(whisper_text.split())
            if word_count < 2 and sv_text_clean:
                self.logger.warning(
                    "Whisper 单词数过少（%d < 2），直接降级到 SenseVoice",
                    word_count,
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
                injection = self._semantic_injector.inject(
                    aligned_words,
                    punctuation_clean_text,
                    punctuation_positions,
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
        """将 SenseVoice 结果作为最终输出（极速模式，跳过二次切分）。

        V3.2.0+dev.20260201.12: 修复词中间切分问题
        - sensevoice_only 模式直接使用 SenseVoice 的原始 sentences
        - 不调用 final_splitter 进行二次切分
        - 避免 Phase F 标点后处理导致的切分点错位
        """
        # 直接使用 SenseVoice 的 sentences，不进行二次切分
        sentences = sv_result.get("sentences", [])

        if not sentences:
            # 兜底：如果 SenseVoice 没有返回 sentences，使用 words 创建单个句子
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

            # 创建单个句子包含所有 words
            sentences = [
                SentenceSegment(
                    text=text_display.strip(),
                    start=words[0].start,
                    end=words[-1].end,
                    words=words,
                    source=TextSource.SENSEVOICE,
                    confidence=sv_result.get("confidence", 0.5),
                    is_finalized=True,
                    is_draft=False,
                )
            ]

        # 调整时间戳到全局坐标
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
