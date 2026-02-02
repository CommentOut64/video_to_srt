"""
AlignmentService - 双流对齐算法

Phase 3 实现 - 2025-12-10

实现 Needleman-Wunsch 序列对齐算法，用于将 Whisper 文本对齐到 SenseVoice 时间轴。
支持静音区硬约束、能量锚点校准、VAD 边界校准和 Gap 填补。
"""

import logging
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import numpy as np

from app.models.confidence_models import (
    AlignedWord,
    AlignedSubtitle,
    AlignmentStatus,
    ConfidenceLevel
)
from app.models.sensevoice_models import WordTimestamp
from app.services.alignment.gap_resolver import GapResolver
from app.services.alignment.quality_stats import QualityStatsCalculator
from app.utils.text_utils import smart_join_words


@dataclass
class AlignmentConfig:
    """对齐算法配置"""
    # Needleman-Wunsch 评分参数
    match_score: int = 2           # 匹配得分
    mismatch_penalty: int = -1     # 不匹配惩罚
    gap_penalty: int = -2          # Gap 惩罚
    
    # 静音区约束
    enable_silence_constraint: bool = True
    silence_penalty: float = -10.0  # 静音区对齐惩罚
    
    # 能量锚点校准
    enable_energy_anchor: bool = True
    energy_threshold: float = 0.02  # 能量阈值
    
    # VAD 边界校准
    enable_vad_calibration: bool = True
    vad_tolerance: float = 0.1      # VAD 边界容差（秒）
    
    # 置信度融合
    sv_weight: float = 0.4          # SenseVoice 置信度权重
    whisper_weight: float = 0.6     # Whisper 置信度权重


class AlignmentService:
    """
    双流对齐服务
    
    使用 Needleman-Wunsch 算法将 Whisper 文本对齐到 SenseVoice 时间轴。
    """
    
    def __init__(
        self,
        config: Optional[AlignmentConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化对齐服务
        
        Args:
            config: 对齐算法配置
            logger: 日志记录器
        """
        self.config = config or AlignmentConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.gap_resolver = GapResolver(logger=self.logger)
        self.quality_stats_calculator = QualityStatsCalculator(logger=self.logger)

    async def align(
        self,
        whisper_text: str,
        sv_tokens: List[WordTimestamp],
        vad_range: Tuple[float, float],
        chunk_offset: float = 0.0,
        audio_array: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        vad_intervals: Optional[List[Tuple[float, float]]] = None,
    ) -> AlignedSubtitle:
        """
        执行双流对齐

        Args:
            whisper_text: Whisper 识别的文本（权威文本）
            sv_tokens: SenseVoice 的字级时间戳列表（权威时间）
            vad_range: VAD 检测的语音范围 (start, end)
            chunk_offset: Chunk 在完整音频中的偏移量
            audio_array: 音频数组（用于能量锚点校准）
            sample_rate: 采样率

        Returns:
            AlignedSubtitle: 对齐后的字幕段
        """
        self.logger.info(f"开始对齐: Whisper='{whisper_text}', SV tokens={len(sv_tokens)}")

        # 1. 文本预处理
        whisper_words = self._tokenize(whisper_text)
        sv_words = [token.word for token in sv_tokens]

        self.logger.debug(f"Whisper words: {whisper_words}")
        self.logger.debug(f"SV words: {sv_words}")

        # 2. Needleman-Wunsch 序列对齐
        alignment_path = self._needleman_wunsch(whisper_words, sv_words)

        # 3. 生成对齐后的字级时间戳
        aligned_words = self._generate_aligned_words(
            alignment_path,
            whisper_words,
            sv_tokens,
            vad_range,
            chunk_offset
        )

        # 4. Gap 处理（插值/合并/降级）
        gap_result = self.gap_resolver.resolve_gaps(
            aligned_words,
            vad_intervals=vad_intervals,
            vad_range=vad_range,
        )
        aligned_words = gap_result.words

        # 5. 能量锚点校准（如果提供了音频）
        if self.config.enable_energy_anchor and audio_array is not None:
            aligned_words = self._apply_energy_anchor(
                aligned_words,
                audio_array,
                sample_rate
            )

        # 6. VAD 边界校准
        if self.config.enable_vad_calibration:
            aligned_words = self._apply_vad_calibration(
                aligned_words,
                vad_range
            )

        # 7. 构建 AlignedSubtitle
        result = self._build_aligned_subtitle(
            aligned_words,
            whisper_text,
            sv_tokens,
            vad_range
        )
        quality_stats = self.quality_stats_calculator.compute(
            aligned_words,
            total_tokens=len(sv_tokens),
            gap_positions=gap_result.gap_positions,
            gap_resolution=gap_result.resolution,
        )
        result.coverage = quality_stats.coverage
        result.gap_ratio = quality_stats.gap_ratio
        result.alignment_score = quality_stats.alignment_score
        result.gap_positions = quality_stats.gap_positions
        result.gap_resolution = quality_stats.gap_resolution

        self.logger.info(
            f"对齐完成: {len(aligned_words)} 个词, "
            f"匹配率={result.matched_ratio:.2%}, "
            f"平均置信度={result.avg_confidence:.2f}"
        )

        return result

    def _tokenize(self, text: str) -> List[str]:
        """
        文本分词

        简单的空格分词，支持中英文混合。

        Args:
            text: 输入文本

        Returns:
            List[str]: 词列表
        """
        words = text.strip().split()
        words = [w for w in words if w]
        return words

    def _needleman_wunsch(
        self,
        seq1: List[str],
        seq2: List[str]
    ) -> List[Tuple[Optional[int], Optional[int]]]:
        """
        Needleman-Wunsch 全局序列对齐算法

        Args:
            seq1: 序列1（Whisper 词列表）
            seq2: 序列2（SenseVoice 词列表）

        Returns:
            List[Tuple[Optional[int], Optional[int]]]: 对齐路径
                - (i, j): seq1[i] 对齐到 seq2[j]
                - (i, None): seq1[i] 插入（SenseVoice 漏字）
                - (None, j): seq2[j] 删除（SenseVoice 幻觉）
        """
        m, n = len(seq1), len(seq2)

        # 初始化得分矩阵和回溯矩阵
        score = np.zeros((m + 1, n + 1), dtype=int)
        traceback = np.zeros((m + 1, n + 1), dtype=int)

        # 初始化第一行和第一列
        for i in range(1, m + 1):
            score[i][0] = score[i-1][0] + self.config.gap_penalty
            traceback[i][0] = 1

        for j in range(1, n + 1):
            score[0][j] = score[0][j-1] + self.config.gap_penalty
            traceback[0][j] = 2

        # 填充得分矩阵
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if self._is_match(seq1[i-1], seq2[j-1]):
                    match = score[i-1][j-1] + self.config.match_score
                else:
                    match = score[i-1][j-1] + self.config.mismatch_penalty

                delete = score[i-1][j] + self.config.gap_penalty
                insert = score[i][j-1] + self.config.gap_penalty

                max_score = max(match, delete, insert)
                score[i][j] = max_score

                if max_score == match:
                    traceback[i][j] = 0
                elif max_score == delete:
                    traceback[i][j] = 1
                else:
                    traceback[i][j] = 2

        # 回溯生成对齐路径
        alignment_path = []
        i, j = m, n

        while i > 0 or j > 0:
            if i == 0:
                alignment_path.append((None, j - 1))
                j -= 1
            elif j == 0:
                alignment_path.append((i - 1, None))
                i -= 1
            else:
                direction = traceback[i][j]
                if direction == 0:
                    alignment_path.append((i - 1, j - 1))
                    i -= 1
                    j -= 1
                elif direction == 1:
                    alignment_path.append((i - 1, None))
                    i -= 1
                else:
                    alignment_path.append((None, j - 1))
                    j -= 1

        alignment_path.reverse()
        return alignment_path

    def _is_match(self, word1: str, word2: str) -> bool:
        """
        判断两个词是否匹配

        支持大小写不敏感和简单的相似度判断。

        Args:
            word1: 词1
            word2: 词2

        Returns:
            bool: 是否匹配
        """
        w1 = word1.lower().strip()
        w2 = word2.lower().strip()

        if w1 == w2:
            return True

        if len(w1) > 3 and len(w2) > 3:
            edit_dist = self._levenshtein_distance(w1, w2)
            if edit_dist <= 1:
                return True

        return False

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        计算编辑距离

        Args:
            s1: 字符串1
            s2: 字符串2

        Returns:
            int: 编辑距离
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],
                        dp[i][j-1],
                        dp[i-1][j-1]
                    )

        return dp[m][n]

    def _generate_aligned_words(
        self,
        alignment_path: List[Tuple[Optional[int], Optional[int]]],
        whisper_words: List[str],
        sv_tokens: List[WordTimestamp],
        vad_range: Tuple[float, float],
        chunk_offset: float
    ) -> List[AlignedWord]:
        """
        根据对齐路径生成对齐后的字级时间戳

        Args:
            alignment_path: 对齐路径
            whisper_words: Whisper 词列表
            sv_tokens: SenseVoice 时间戳列表
            vad_range: VAD 范围
            chunk_offset: Chunk 偏移量

        Returns:
            List[AlignedWord]: 对齐后的词列表
        """
        aligned_words = []

        for whisper_idx, sv_idx in alignment_path:
            if whisper_idx is not None and sv_idx is not None:
                whisper_word = whisper_words[whisper_idx]
                sv_token = sv_tokens[sv_idx]

                if self._is_match(whisper_word, sv_token.word):
                    status = AlignmentStatus.MATCHED
                else:
                    status = AlignmentStatus.SUBSTITUTED

                aligned_word = AlignedWord(
                    word=whisper_word,
                    start=sv_token.start + chunk_offset,
                    end=sv_token.end + chunk_offset,
                    sv_confidence=sv_token.confidence,
                    whisper_confidence=1.0,
                    final_confidence=self._compute_final_confidence(
                        sv_token.confidence,
                        1.0
                    ),
                    alignment_status=status,
                    is_pseudo=False,
                    sv_original=sv_token.word,
                    whisper_original=whisper_word
                )

                aligned_words.append(aligned_word)

            elif whisper_idx is not None:
                whisper_word = whisper_words[whisper_idx]

                start, end = self._estimate_timestamp_for_insertion(
                    whisper_idx,
                    aligned_words,
                    sv_tokens,
                    vad_range,
                    chunk_offset
                )

                aligned_word = AlignedWord(
                    word=whisper_word,
                    start=start,
                    end=end,
                    sv_confidence=0.0,
                    whisper_confidence=1.0,
                    final_confidence=self._compute_final_confidence(0.0, 1.0),
                    alignment_status=AlignmentStatus.INSERTED,
                    is_pseudo=True,
                    sv_original=None,
                    whisper_original=whisper_word
                )

                aligned_words.append(aligned_word)

        return aligned_words

    def _compute_final_confidence(
        self,
        sv_confidence: float,
        whisper_confidence: float
    ) -> float:
        """
        计算最终置信度

        融合 SenseVoice 和 Whisper 的置信度。

        Args:
            sv_confidence: SenseVoice 置信度
            whisper_confidence: Whisper 置信度

        Returns:
            float: 最终置信度
        """
        return (
            self.config.sv_weight * sv_confidence +
            self.config.whisper_weight * whisper_confidence
        )

    def _estimate_timestamp_for_insertion(
        self,
        whisper_idx: int,
        aligned_words: List[AlignedWord],
        sv_tokens: List[WordTimestamp],
        vad_range: Tuple[float, float],
        chunk_offset: float
    ) -> Tuple[float, float]:
        """
        为插入的词估算时间戳

        在相邻词之间均匀分配时间。

        Args:
            whisper_idx: Whisper 词索引
            aligned_words: 已对齐的词列表
            sv_tokens: SenseVoice 时间戳列表
            vad_range: VAD 范围
            chunk_offset: Chunk 偏移量

        Returns:
            Tuple[float, float]: (start, end)
        """
        prev_end = vad_range[0] + chunk_offset
        next_start = vad_range[1] + chunk_offset

        if aligned_words:
            prev_end = aligned_words[-1].end

        avg_duration = 0.3
        start = prev_end
        end = min(start + avg_duration, next_start)

        return start, end

    def _apply_energy_anchor(
        self,
        aligned_words: List[AlignedWord],
        audio_array: np.ndarray,
        sample_rate: int
    ) -> List[AlignedWord]:
        """
        应用能量锚点校准

        利用音频能量峰值重新定位词边界。

        Args:
            aligned_words: 对齐后的词列表
            audio_array: 音频数组
            sample_rate: 采样率

        Returns:
            List[AlignedWord]: 校准后的词列表
        """
        energy = np.abs(audio_array)

        for word in aligned_words:
            if word.is_pseudo:
                continue

            start_sample = int(word.start * sample_rate)
            end_sample = int(word.end * sample_rate)

            if start_sample >= len(energy) or end_sample > len(energy):
                continue

            word_energy = energy[start_sample:end_sample]

            if len(word_energy) == 0:
                continue

            peak_idx = np.argmax(word_energy)
            peak_time = word.start + (peak_idx / sample_rate)

            word_center = (word.start + word.end) / 2
            if abs(peak_time - word_center) > 0.1:
                duration = word.end - word.start
                word.start = peak_time - duration / 2
                word.end = peak_time + duration / 2

        return aligned_words

    def _apply_vad_calibration(
        self,
        aligned_words: List[AlignedWord],
        vad_range: Tuple[float, float]
    ) -> List[AlignedWord]:
        """
        应用 VAD 边界校准

        确保所有词的时间戳在 VAD 检测的语音范围内。

        Args:
            aligned_words: 对齐后的词列表
            vad_range: VAD 范围 (start, end)

        Returns:
            List[AlignedWord]: 校准后的词列表
        """
        vad_start, vad_end = vad_range

        for word in aligned_words:
            if word.start < vad_start:
                word.start = vad_start

            if word.end > vad_end:
                word.end = vad_end

            if word.start >= word.end:
                word.end = word.start + 0.1

        return aligned_words

    def _build_aligned_subtitle(
        self,
        aligned_words: List[AlignedWord],
        whisper_text: str,
        sv_tokens: List[WordTimestamp],
        vad_range: Tuple[float, float]
    ) -> AlignedSubtitle:
        """
        构建 AlignedSubtitle

        Args:
            aligned_words: 对齐后的词列表
            whisper_text: Whisper 文本
            sv_tokens: SenseVoice 时间戳列表
            vad_range: VAD 范围

        Returns:
            AlignedSubtitle: 对齐后的字幕段
        """
        if not aligned_words:
            return AlignedSubtitle(
                text="",
                start=vad_range[0],
                end=vad_range[1],
                words=[],
                vad_start=vad_range[0],
                vad_end=vad_range[1]
            )

        subtitle = AlignedSubtitle(
            text=whisper_text,
            start=aligned_words[0].start,
            end=aligned_words[-1].end,
            words=aligned_words,
            sv_text=smart_join_words(sv_tokens),
            whisper_text=whisper_text,
            vad_start=vad_range[0],
            vad_end=vad_range[1],
            is_draft=False,
            is_finalized=True
        )

        subtitle.compute_statistics()
        subtitle.alignment_score = self._compute_alignment_score(aligned_words)

        return subtitle

    def _compute_alignment_score(self, aligned_words: List[AlignedWord]) -> float:
        """
        计算对齐质量分数

        基于匹配率和置信度计算。

        Args:
            aligned_words: 对齐后的词列表

        Returns:
            float: 对齐质量分数 (0-1)
        """
        if not aligned_words:
            return 0.0

        matched_count = sum(
            1 for w in aligned_words
            if w.alignment_status == AlignmentStatus.MATCHED
        )
        match_ratio = matched_count / len(aligned_words)

        avg_confidence = sum(w.final_confidence for w in aligned_words) / len(aligned_words)

        score = 0.6 * match_ratio + 0.4 * avg_confidence

        return score
