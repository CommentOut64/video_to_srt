"""
Whisper 缓冲池服务

解决 Whisper 在极短音频上产生幻觉的根本问题。
通过累积多个短 Chunk，拼接后一次性推理，利用 Whisper 的长上下文能力。

架构设计:
    SenseVoice (快流): 保持现状，来一个 chunk 跑一个，保证秒级上屏
    Whisper (慢流): 使用缓冲池，累积足够音频后批量推理

触发条件 (满足任一即触发):
    1. Buffer 音频时长 > 5秒
    2. 累计 VAD Chunk 数量 > 3个
    3. 遇到长静音 (>1s) 或文件结束

工作流程:
    1. Chunk 入池 -> 累积音频和元数据
    2. 触发条件满足 -> 拼接音频 -> Whisper 推理 -> 获取长文本
    3. 对齐回填 -> 将长文本映射回原始 Chunk 时间戳
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from app.services.alignment.nw_v2_core import (
    NeedlemanWunschScoreConfig,
    NeedlemanWunschV2Core,
    ZeroPriorProvider,
)
from app.services.model_runtime_config_service import get_model_runtime_config_service

logger = logging.getLogger(__name__)


@dataclass
class BufferedChunk:
    """缓冲池中的 Chunk 单元"""
    index: int                      # 原始 Chunk 索引
    start: float                    # 起始时间（秒）
    end: float                      # 结束时间（秒）
    audio: np.ndarray              # 音频数据 (16kHz)
    sensevoice_text: str = ""      # SenseVoice 识别的文本
    sensevoice_confidence: float = 0.0  # SenseVoice 置信度

    @property
    def duration(self) -> float:
        """Chunk 时长"""
        return self.end - self.start


@dataclass
class WhisperBufferConfig:
    """缓冲池配置"""
    # 触发条件
    min_duration_sec: float = 5.0       # 最小累积时长（秒）
    max_chunk_count: int = 3            # 最大累积 Chunk 数量
    silence_trigger_sec: float = 1.0    # 长静音触发阈值（秒）

    # 安全限制
    max_duration_sec: float = 30.0      # 最大累积时长（防止内存溢出）
    max_buffer_size: int = 10           # 最大缓冲 Chunk 数量

    # 对齐参数
    min_word_duration: float = 0.05     # 最小词时长（秒）
    alignment_tolerance: float = 0.3    # 对齐容差（秒）
    # V3.2.0+dev.20260215.11: NW 统一内核参数
    nw_v2_enable: Optional[bool] = None
    nw_match_score: float = 2.0
    nw_mismatch_penalty: float = -1.0
    nw_gap_penalty: float = -2.0
    nw_v2_confidence_alpha: float = 0.6
    nw_v2_prior_beta: float = 0.4


class WhisperBufferPool:
    """
    Whisper 缓冲池

    累积多个短 Chunk，拼接后一次性推理，利用 Whisper 长上下文能力。
    """

    def __init__(self, config: WhisperBufferConfig = None):
        """
        初始化缓冲池

        Args:
            config: 缓冲池配置
        """
        self.config = config or WhisperBufferConfig()
        self._buffer: List[BufferedChunk] = []
        self._sample_rate = 16000

        logger.info(
            f"WhisperBufferPool 初始化: "
            f"min_duration={self.config.min_duration_sec}s, "
            f"max_chunks={self.config.max_chunk_count}"
        )

    def add_chunk(
        self,
        index: int,
        start: float,
        end: float,
        audio: np.ndarray,
        sensevoice_text: str = "",
        sensevoice_confidence: float = 0.0
    ) -> bool:
        """
        添加 Chunk 到缓冲池

        Args:
            index: Chunk 索引
            start: 起始时间（秒）
            end: 结束时间（秒）
            audio: 音频数据
            sensevoice_text: SenseVoice 识别文本
            sensevoice_confidence: SenseVoice 置信度

        Returns:
            bool: 是否应该触发 Whisper 推理
        """
        chunk = BufferedChunk(
            index=index,
            start=start,
            end=end,
            audio=audio,
            sensevoice_text=sensevoice_text,
            sensevoice_confidence=sensevoice_confidence
        )
        self._buffer.append(chunk)

        logger.debug(
            f"Chunk {index} 入池: [{start:.2f}s, {end:.2f}s], "
            f"累积时长={self.total_duration:.2f}s, 数量={len(self._buffer)}"
        )

        return self.should_trigger()

    def should_trigger(self, has_long_silence: bool = False, is_eof: bool = False) -> bool:
        """
        判断是否应该触发 Whisper 推理

        触发条件 (满足任一):
        1. 累积时长 >= min_duration_sec
        2. 累积 Chunk 数量 >= max_chunk_count
        3. 遇到长静音或文件结束
        4. 达到安全上限

        Args:
            has_long_silence: 是否遇到长静音
            is_eof: 是否文件结束

        Returns:
            bool: 是否应该触发
        """
        if not self._buffer:
            return False

        # 条件1: 累积时长达标
        if self.total_duration >= self.config.min_duration_sec:
            logger.debug(f"触发条件: 累积时长 {self.total_duration:.2f}s >= {self.config.min_duration_sec}s")
            return True

        # 条件2: Chunk 数量达标
        if len(self._buffer) >= self.config.max_chunk_count:
            logger.debug(f"触发条件: Chunk 数量 {len(self._buffer)} >= {self.config.max_chunk_count}")
            return True

        # 条件3: 长静音或文件结束
        if has_long_silence or is_eof:
            logger.debug(f"触发条件: 长静音={has_long_silence}, EOF={is_eof}")
            return True

        # 条件4: 安全上限
        if self.total_duration >= self.config.max_duration_sec:
            logger.warning(f"触发条件: 达到安全上限 {self.config.max_duration_sec}s")
            return True

        if len(self._buffer) >= self.config.max_buffer_size:
            logger.warning(f"触发条件: 达到缓冲上限 {self.config.max_buffer_size} 个 Chunk")
            return True

        return False

    @property
    def total_duration(self) -> float:
        """缓冲池中音频的总时长"""
        if not self._buffer:
            return 0.0
        return self._buffer[-1].end - self._buffer[0].start

    @property
    def chunk_count(self) -> int:
        """缓冲池中的 Chunk 数量"""
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        """缓冲池是否为空"""
        return len(self._buffer) == 0

    def get_concatenated_audio(self, preserve_gaps: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        获取拼接后的完整音频

        Args:
            preserve_gaps: 是否保留 Chunk 之间的真实时间间隔（插入静音）

        Returns:
            Tuple[np.ndarray, float, float]: (拼接音频, 起始时间, 结束时间)
        """
        if not self._buffer:
            return np.array([], dtype=np.float32), 0.0, 0.0

        start_time = self._buffer[0].start
        end_time = self._buffer[-1].end

        # V3.2.0+dev.20260201.07: 可选保留间隔拼接
        if not preserve_gaps:
            # 按时间顺序拼接音频（不保留间隔）
            audio_segments = [chunk.audio for chunk in self._buffer]
            concatenated = np.concatenate(audio_segments)
            return concatenated, start_time, end_time

        # 保留间隔：在相邻 Chunk 之间插入静音
        audio_parts: List[np.ndarray] = []
        current_time = start_time
        for chunk in self._buffer:
            if chunk.start > current_time:
                gap = chunk.start - current_time
                if gap > 0:
                    gap_samples = int(gap * self._sample_rate)
                    if gap_samples > 0:
                        audio_parts.append(np.zeros(gap_samples, dtype=chunk.audio.dtype))
            audio_parts.append(chunk.audio)
            current_time = chunk.end

        if audio_parts:
            concatenated = np.concatenate(audio_parts)
        else:
            concatenated = np.array([], dtype=np.float32)
        return concatenated, start_time, end_time

    def get_chunk_boundaries(self) -> List[Dict[str, Any]]:
        """
        获取所有 Chunk 的边界信息（用于对齐）

        Returns:
            List[Dict]: Chunk 边界列表
        """
        return [
            {
                "index": chunk.index,
                "start": chunk.start,
                "end": chunk.end,
                "duration": chunk.duration,
                "sensevoice_text": chunk.sensevoice_text,
                "sensevoice_confidence": chunk.sensevoice_confidence,
            }
            for chunk in self._buffer
        ]

    def flush(self) -> List[BufferedChunk]:
        """
        清空缓冲池并返回所有 Chunk

        Returns:
            List[BufferedChunk]: 缓冲池中的所有 Chunk
        """
        chunks = self._buffer.copy()
        self._buffer.clear()
        logger.debug(f"缓冲池已清空，释放 {len(chunks)} 个 Chunk")
        return chunks

    def clear(self):
        """清空缓冲池（不返回数据）"""
        self._buffer.clear()


class WhisperBufferAligner:
    """
    Whisper 长文本对齐器 (V2: 使用 Needleman-Wunsch 算法)

    将 Whisper 输出的长文本对齐回填到原始 Chunk 时间戳。
    核心策略: 使用 SenseVoice 文本作为锚点，通过序列对齐算法匹配 Whisper 词。
    """

    def __init__(self, config: WhisperBufferConfig = None):
        """
        初始化对齐器

        Args:
            config: 配置
        """
        self.config = config or WhisperBufferConfig()
        self._is_nw_v2_enabled = self._resolve_nw_v2_enabled(self.config)
        self._nw_core = NeedlemanWunschV2Core(
            NeedlemanWunschScoreConfig(
                base_match_score=float(self.config.nw_match_score),
                base_mismatch_penalty=float(self.config.nw_mismatch_penalty),
                base_gap_penalty=float(self.config.nw_gap_penalty),
                confidence_alpha=float(self.config.nw_v2_confidence_alpha),
                prior_beta=float(self.config.nw_v2_prior_beta),
                is_enable_weighted_scoring=self._is_nw_v2_enabled,
            )
        )
        self._prior_provider = ZeroPriorProvider()
        logger.info("WhisperBufferAligner NW 内核配置: nw_v2_enable=%s", self._is_nw_v2_enabled)

    @staticmethod
    def _resolve_nw_v2_enabled(config: WhisperBufferConfig) -> bool:
        if config.nw_v2_enable is not None:
            return bool(config.nw_v2_enable)
        try:
            runtime = get_model_runtime_config_service().get_effective_runtime_global()
            effective = runtime.get("effective", {}) if isinstance(runtime, dict) else {}
            m2 = effective.get("m2", {}) if isinstance(effective, dict) else {}
            return bool(m2.get("nw_v2.enable", False))
        except Exception:
            return False

    def align_text_to_chunks(
        self,
        whisper_result: Dict[str, Any],
        chunk_boundaries: List[Dict[str, Any]],
        pool_start_time: float
    ) -> List[Dict[str, Any]]:
        """
        将 Whisper 长文本对齐到原始 Chunk (V2: Needleman-Wunsch)

        新策略:
        1. 收集所有 Whisper 词（带时间戳）
        2. 收集所有 SenseVoice 词（作为锚点）
        3. 使用 Needleman-Wunsch 进行全局序列对齐
        4. 根据对齐结果，将 Whisper 词分配到对应的 Chunk

        Args:
            whisper_result: Whisper 转录结果
            chunk_boundaries: Chunk 边界列表
            pool_start_time: 缓冲池起始时间

        Returns:
            List[Dict]: 每个 Chunk 对应的 Whisper 文本
        """
        segments = whisper_result.get("segments", [])
        full_text = whisper_result.get("text", "").strip()

        if not segments or not chunk_boundaries:
            logger.warning("对齐失败: segments 或 chunk_boundaries 为空")
            return self._fallback_alignment(full_text, chunk_boundaries)

        # 收集 Whisper 词
        whisper_words = self._collect_whisper_words(segments, pool_start_time)

        if not whisper_words:
            logger.warning("Whisper 没有输出词，使用兜底对齐")
            return self._fallback_alignment(full_text, chunk_boundaries)

        # 收集 SenseVoice 词（按 Chunk 组织）
        sv_words_by_chunk = self._collect_sensevoice_words(chunk_boundaries)
        all_sv_words = [w for words in sv_words_by_chunk for w in words]

        if not all_sv_words:
            logger.warning("SenseVoice 没有词，使用时间戳对齐")
            return self._timestamp_based_alignment(whisper_words, chunk_boundaries)

        # Needleman-Wunsch 序列对齐
        alignment_path = self._needleman_wunsch(
            [w["word"] for w in whisper_words],
            [w["word"] for w in all_sv_words],
            seq1_confidences=[w.get("probability") for w in whisper_words],
            seq2_confidences=[w.get("confidence") for w in all_sv_words],
        )

        # 根据对齐路径分配 Whisper 词到 Chunk
        return self._distribute_by_alignment(
            alignment_path,
            whisper_words,
            all_sv_words,
            sv_words_by_chunk,
            chunk_boundaries
        )

    def _collect_whisper_words(
        self,
        segments: List[Dict],
        pool_start_time: float
    ) -> List[Dict[str, Any]]:
        """收集 Whisper 所有词"""
        all_words = []
        for seg in segments:
            words = seg.get("words", [])
            for w in words:
                word_text = w.get("word", "").strip()
                if word_text:
                    all_words.append({
                        "word": word_text,
                        "start": w.get("start", 0) + pool_start_time,
                        "end": w.get("end", 0) + pool_start_time,
                        "probability": w.get("probability", 0.5)
                    })
        return all_words

    def _collect_sensevoice_words(
        self,
        chunk_boundaries: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        收集 SenseVoice 词，按 Chunk 组织

        Returns:
            List[List[Dict]]: 每个 Chunk 的词列表
        """
        result = []
        for chunk in chunk_boundaries:
            sv_text = chunk.get("sensevoice_text", "")
            chunk_start = chunk["start"]
            chunk_end = chunk["end"]
            chunk_duration = chunk_end - chunk_start
            chunk_confidence = float(chunk.get("sensevoice_confidence", 0.0) or 0.0)

            # 简单分词
            words = sv_text.strip().split()
            chunk_words = []

            if words and chunk_duration > 0:
                # 均匀分配时间戳
                word_duration = chunk_duration / len(words)
                for i, word in enumerate(words):
                    chunk_words.append({
                        "word": word,
                        "start": chunk_start + i * word_duration,
                        "end": chunk_start + (i + 1) * word_duration,
                        "chunk_index": chunk["index"],
                        "confidence": chunk_confidence,
                    })

            result.append(chunk_words)
        return result

    def _needleman_wunsch(
        self,
        seq1: List[str],
        seq2: List[str],
        *,
        seq1_confidences: Optional[List[Optional[float]]] = None,
        seq2_confidences: Optional[List[Optional[float]]] = None,
    ) -> List[Tuple[Optional[int], Optional[int]]]:
        """
        Needleman-Wunsch 全局序列对齐算法（统一 V1/V2 内核）。

        Args:
            seq1: Whisper 词列表
            seq2: SenseVoice 词列表

        Returns:
            对齐路径: [(whisper_idx, sv_idx), ...]
            - (i, j): seq1[i] 对齐到 seq2[j]
            - (i, None): Whisper 词在 SenseVoice 中没有对应（插入）
            - (None, j): SenseVoice 词在 Whisper 中没有对应（删除）
        """
        return self._nw_core.align(
            seq1,
            seq2,
            seq1_confidences=seq1_confidences,
            seq2_confidences=seq2_confidences,
            prior_provider=self._prior_provider,
            match_fn=self._is_word_match,
        )

    def _is_word_match(self, word1: str, word2: str) -> bool:
        """判断两个词是否匹配（大小写不敏感 + 编辑距离容错）"""
        w1 = word1.lower().strip().rstrip('.,;:!?')
        w2 = word2.lower().strip().rstrip('.,;:!?')

        if w1 == w2:
            return True

        # 编辑距离容错（长词允许1个字符差异）
        if len(w1) > 3 and len(w2) > 3:
            if abs(len(w1) - len(w2)) <= 1:
                diff_count = sum(1 for a, b in zip(w1, w2) if a != b)
                if diff_count <= 1:
                    return True

        return False

    def _distribute_by_alignment(
        self,
        alignment_path: List[Tuple[Optional[int], Optional[int]]],
        whisper_words: List[Dict],
        all_sv_words: List[Dict],
        sv_words_by_chunk: List[List[Dict]],
        chunk_boundaries: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        根据对齐路径将 Whisper 词分配到各个 Chunk

        核心逻辑:
        - 如果 Whisper 词对齐到了某个 SenseVoice 词，就分配到该 SV 词所在的 Chunk
        - 如果 Whisper 词没有对齐（插入），就分配到最近的已对齐词所在的 Chunk
        """
        # 构建 SV 词索引到 Chunk 索引的映射
        sv_idx_to_chunk = {}
        global_sv_idx = 0
        for chunk_idx, chunk_words in enumerate(sv_words_by_chunk):
            for _ in chunk_words:
                sv_idx_to_chunk[global_sv_idx] = chunk_idx
                global_sv_idx += 1

        # 为每个 Whisper 词分配 Chunk
        whisper_to_chunk = {}
        last_known_chunk = 0

        for whisper_idx, sv_idx in alignment_path:
            if whisper_idx is not None:
                if sv_idx is not None and sv_idx in sv_idx_to_chunk:
                    # 对齐到了 SV 词，使用其 Chunk
                    chunk_idx = sv_idx_to_chunk[sv_idx]
                    whisper_to_chunk[whisper_idx] = chunk_idx
                    last_known_chunk = chunk_idx
                else:
                    # 没有对齐，使用最近的已知 Chunk
                    whisper_to_chunk[whisper_idx] = last_known_chunk

        # 按 Chunk 收集 Whisper 词（保持原始顺序）
        chunk_whisper_words = {i: [] for i in range(len(chunk_boundaries))}
        # 按 whisper_idx 顺序遍历，确保词序正确
        for whisper_idx in sorted(whisper_to_chunk.keys()):
            chunk_idx = whisper_to_chunk[whisper_idx]
            if 0 <= chunk_idx < len(chunk_boundaries):
                chunk_whisper_words[chunk_idx].append(whisper_words[whisper_idx])

        # 生成结果
        result = []
        for chunk_idx, chunk in enumerate(chunk_boundaries):
            words_in_chunk = chunk_whisper_words.get(chunk_idx, [])

            # 拼接文本
            chunk_text = " ".join(w["word"] for w in words_in_chunk).strip()

            # 计算平均置信度
            avg_prob = 0.5
            if words_in_chunk:
                avg_prob = sum(w["probability"] for w in words_in_chunk) / len(words_in_chunk)

            result.append({
                "chunk_index": chunk["index"],
                "start": chunk["start"],
                "end": chunk["end"],
                "whisper_text": chunk_text,
                "whisper_confidence": avg_prob,
                "sensevoice_text": chunk.get("sensevoice_text", ""),
                "sensevoice_confidence": chunk.get("sensevoice_confidence", 0.0),
                "word_count": len(words_in_chunk)
            })

        logger.debug(f"Needleman-Wunsch 对齐完成: {len(whisper_words)} Whisper词 -> {len(chunk_boundaries)} Chunks")
        return result

    def _timestamp_based_alignment(
        self,
        whisper_words: List[Dict],
        chunk_boundaries: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        基于时间戳的对齐（兜底方案，当 SenseVoice 没有词时使用）

        改进: 使用重叠比例而非中心点
        """
        result = []
        for chunk in chunk_boundaries:
            chunk_start = chunk["start"]
            chunk_end = chunk["end"]
            chunk_duration = chunk_end - chunk_start

            chunk_words = []
            for w in whisper_words:
                # 计算词与 Chunk 的重叠
                overlap_start = max(w["start"], chunk_start)
                overlap_end = min(w["end"], chunk_end)
                overlap = max(0, overlap_end - overlap_start)

                word_duration = w["end"] - w["start"]
                if word_duration > 0:
                    # 至少 30% 重叠才算属于这个 Chunk
                    overlap_ratio = overlap / word_duration
                    if overlap_ratio >= 0.3:
                        chunk_words.append(w)

            chunk_text = " ".join(w["word"] for w in chunk_words).strip()
            avg_prob = 0.5
            if chunk_words:
                avg_prob = sum(w["probability"] for w in chunk_words) / len(chunk_words)

            result.append({
                "chunk_index": chunk["index"],
                "start": chunk_start,
                "end": chunk_end,
                "whisper_text": chunk_text,
                "whisper_confidence": avg_prob,
                "sensevoice_text": chunk.get("sensevoice_text", ""),
                "sensevoice_confidence": chunk.get("sensevoice_confidence", 0.0),
                "word_count": len(chunk_words)
            })

        return result

    # 保留原有的 _word_level_alignment 作为备用（已弃用）
    def _word_level_alignment_deprecated(
        self,
        segments: List[Dict],
        chunk_boundaries: List[Dict],
        pool_start_time: float
    ) -> List[Dict[str, Any]]:
        """
        [已弃用] 词级时间戳对齐 - 简单的中心点匹配，易导致错位
        """
        all_words = []
        for seg in segments:
            words = seg.get("words", [])
            for w in words:
                all_words.append({
                    "word": w.get("word", ""),
                    "start": w.get("start", 0) + pool_start_time,
                    "end": w.get("end", 0) + pool_start_time,
                    "probability": w.get("probability", 0.5)
                })

        result = []
        for chunk in chunk_boundaries:
            chunk_start = chunk["start"]
            chunk_end = chunk["end"]

            chunk_words = []
            for w in all_words:
                word_center = (w["start"] + w["end"]) / 2
                if chunk_start <= word_center < chunk_end:
                    chunk_words.append(w)

            chunk_text = "".join(w["word"] for w in chunk_words).strip()
            avg_prob = 0.5
            if chunk_words:
                avg_prob = sum(w["probability"] for w in chunk_words) / len(chunk_words)

            result.append({
                "chunk_index": chunk["index"],
                "start": chunk_start,
                "end": chunk_end,
                "whisper_text": chunk_text,
                "whisper_confidence": avg_prob,
                "sensevoice_text": chunk.get("sensevoice_text", ""),
                "sensevoice_confidence": chunk.get("sensevoice_confidence", 0.0),
                "word_count": len(chunk_words)
            })

        return result

    def _segment_level_alignment(
        self,
        segments: List[Dict],
        chunk_boundaries: List[Dict],
        pool_start_time: float
    ) -> List[Dict[str, Any]]:
        """
        段级对齐

        将 Whisper segments 根据时间分配到对应的 Chunk。
        """
        result = []

        for chunk in chunk_boundaries:
            chunk_start = chunk["start"]
            chunk_end = chunk["end"]
            chunk_texts = []

            for seg in segments:
                seg_start = seg.get("start", 0) + pool_start_time
                seg_end = seg.get("end", 0) + pool_start_time
                seg_text = seg.get("text", "").strip()

                # 计算重叠
                overlap_start = max(chunk_start, seg_start)
                overlap_end = min(chunk_end, seg_end)
                overlap_duration = max(0, overlap_end - overlap_start)

                seg_duration = seg_end - seg_start
                if seg_duration > 0 and overlap_duration > 0:
                    # 重叠比例
                    overlap_ratio = overlap_duration / seg_duration
                    if overlap_ratio > 0.3:  # 至少 30% 重叠
                        chunk_texts.append(seg_text)

            chunk_text = " ".join(chunk_texts).strip()

            # 估算置信度
            avg_logprob = -0.5
            matching_segs = [
                seg for seg in segments
                if seg.get("start", 0) + pool_start_time < chunk_end
                and seg.get("end", 0) + pool_start_time > chunk_start
            ]
            if matching_segs:
                avg_logprob = sum(seg.get("avg_logprob", -0.5) for seg in matching_segs) / len(matching_segs)

            confidence = min(1.0, max(0.0, 1.0 + avg_logprob))

            result.append({
                "chunk_index": chunk["index"],
                "start": chunk_start,
                "end": chunk_end,
                "whisper_text": chunk_text,
                "whisper_confidence": confidence,
                "sensevoice_text": chunk.get("sensevoice_text", ""),
                "sensevoice_confidence": chunk.get("sensevoice_confidence", 0.0),
            })

        return result

    def _fallback_alignment(
        self,
        full_text: str,
        chunk_boundaries: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        兜底对齐: 均匀分配文本

        当无法进行精确对齐时，按时间比例分配文本。
        """
        if not chunk_boundaries:
            return []

        total_duration = sum(c["end"] - c["start"] for c in chunk_boundaries)

        # 简单按字符数比例分配
        words = full_text.split()
        total_words = len(words)

        result = []
        word_index = 0

        for chunk in chunk_boundaries:
            chunk_duration = chunk["end"] - chunk["start"]
            # 按时间比例分配词数
            chunk_word_count = int(total_words * (chunk_duration / total_duration)) if total_duration > 0 else 0
            chunk_word_count = max(1, chunk_word_count) if words else 0

            chunk_words = words[word_index:word_index + chunk_word_count]
            chunk_text = " ".join(chunk_words)
            word_index += chunk_word_count

            result.append({
                "chunk_index": chunk["index"],
                "start": chunk["start"],
                "end": chunk["end"],
                "whisper_text": chunk_text,
                "whisper_confidence": 0.5,  # 兜底对齐置信度较低
                "sensevoice_text": chunk.get("sensevoice_text", ""),
                "sensevoice_confidence": chunk.get("sensevoice_confidence", 0.0),
            })

        # 处理剩余词
        if word_index < len(words) and result:
            result[-1]["whisper_text"] += " " + " ".join(words[word_index:])

        return result


class WhisperBufferService:
    """
    Whisper 缓冲池服务

    整合缓冲池和对齐器，提供完整的缓冲推理流程。
    """

    def __init__(self, config: WhisperBufferConfig = None):
        """
        初始化服务

        Args:
            config: 配置
        """
        self.config = config or WhisperBufferConfig()
        self.pool = WhisperBufferPool(self.config)
        self.aligner = WhisperBufferAligner(self.config)

    def add_chunk(
        self,
        index: int,
        start: float,
        end: float,
        audio: np.ndarray,
        sensevoice_text: str = "",
        sensevoice_confidence: float = 0.0
    ) -> bool:
        """
        添加 Chunk 到缓冲池

        Returns:
            bool: 是否应该触发 Whisper 推理
        """
        return self.pool.add_chunk(
            index=index,
            start=start,
            end=end,
            audio=audio,
            sensevoice_text=sensevoice_text,
            sensevoice_confidence=sensevoice_confidence
        )

    def should_trigger(self, has_long_silence: bool = False, is_eof: bool = False) -> bool:
        """判断是否应该触发 Whisper 推理"""
        return self.pool.should_trigger(has_long_silence, is_eof)

    def process_buffer(
        self,
        whisper_service,
        language: str = None,
        initial_prompt: str = None
    ) -> List[Dict[str, Any]]:
        """
        处理缓冲池: 拼接音频 -> Whisper 推理 -> 对齐回填

        Args:
            whisper_service: Whisper 服务实例
            language: 语言代码
            initial_prompt: 上下文提示

        Returns:
            List[Dict]: 每个 Chunk 的对齐结果
        """
        if self.pool.is_empty:
            return []

        # 1. 获取拼接音频
        audio, pool_start, pool_end = self.pool.get_concatenated_audio()
        chunk_boundaries = self.pool.get_chunk_boundaries()

        logger.debug(
            f"Whisper 缓冲池推理: {len(chunk_boundaries)} 个 Chunk, "
            f"时长 {pool_end - pool_start:.2f}s [{pool_start:.2f}s - {pool_end:.2f}s]"
        )

        # 2. Whisper 推理
        result = whisper_service.transcribe(
            audio=audio,
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=True,  # 启用词级时间戳以支持精确对齐
            vad_filter=False  # 已经是 VAD 切片，不需要再过滤
        )

        whisper_text = result.get("text", "").strip()
        logger.debug(f"Whisper 缓冲池输出: '{whisper_text[:100]}...' (共 {len(whisper_text)} 字符)")

        # 3. 对齐回填
        aligned_results = self.aligner.align_text_to_chunks(
            whisper_result=result,
            chunk_boundaries=chunk_boundaries,
            pool_start_time=pool_start
        )

        # 4. 清空缓冲池
        self.pool.flush()

        return aligned_results

    def flush_remaining(
        self,
        whisper_service,
        language: str = None,
        initial_prompt: str = None
    ) -> List[Dict[str, Any]]:
        """
        处理缓冲池中的剩余内容（文件结束时调用）

        即使未达到触发条件，也强制处理剩余的 Chunk。
        """
        if self.pool.is_empty:
            return []

        logger.info(f"处理缓冲池剩余内容: {self.pool.chunk_count} 个 Chunk")
        return self.process_buffer(whisper_service, language, initial_prompt)

    def clear(self):
        """清空缓冲池"""
        self.pool.clear()

    @property
    def is_empty(self) -> bool:
        """缓冲池是否为空"""
        return self.pool.is_empty

    @property
    def chunk_count(self) -> int:
        """缓冲池中的 Chunk 数量"""
        return self.pool.chunk_count

    @property
    def total_duration(self) -> float:
        """缓冲池中音频的总时长"""
        return self.pool.total_duration
