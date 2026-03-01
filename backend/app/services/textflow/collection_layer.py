"""
集合层统一入口（真实实现）。
V3.2.0+dev.20260215.24
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.logging import resolve_loguru_logger
from app.models.confidence_models import AlignedWord, AlignmentStatus
from app.models.sensevoice_models import WordTimestamp
from app.services.alignment.alignment_service import AlignmentService, AlignmentConfig
from app.services.alignment.gap_resolver import GapResolver, GapResolution
from app.services.alignment.quality_stats import QualityStatsCalculator
from app.services.alignment.types import (
    AlignedFacts,
    AlignmentResult,
    AnnotatedWord,
    CollectionLayerInput,
    CollectionLayerOutput,
    TextTrack,
)
from app.services.pseudo_alignment import PseudoAlignment
from app.services.text_pipeline_config import TextPipelineConfig, AlignmentLayerConfig
from app.services.timeline.segmentation_service import map_frame_times_to_word_boundaries


class CollectionAlignmentProcessor:
    """集合层处理器：对齐 + Gap 修复 + 质量统计。"""

    def __init__(
        self,
        *,
        logger: Optional[Any] = None,
        config_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._logger = resolve_loguru_logger(
            logger,
            __name__,
            layer="集合层",
            processor_name="alignment_processor",
        )
        self._config = self._load_alignment_config(config_override)
        self._is_m2_nw_v2_enabled = bool(TextPipelineConfig.from_runtime().m2.is_nw_v2_enabled)
        self._gap_resolver = GapResolver(
            gap_ratio_low=self._config.gap_ratio_low,
            gap_ratio_mid=self._config.gap_ratio_mid,
            min_valid_neighbors=self._config.min_valid_neighbors,
            min_word_duration=self._config.min_word_duration_ms / 1000.0,
            logger=self._logger,
        )
        self._quality_stats = QualityStatsCalculator(logger=self._logger)
        self._alignment_service = AlignmentService(
            config=AlignmentConfig(
                is_enable_nw_v2=self._is_m2_nw_v2_enabled,
            ),
            logger=self._logger,
            gap_resolver=self._gap_resolver,
            quality_stats_calculator=self._quality_stats,
        )
        self._logger.info(
            "集合层 NW 内核配置: nw_v2_enable={}",
            self._is_m2_nw_v2_enabled,
        )

    def process(self, data: CollectionLayerInput) -> CollectionLayerOutput:
        """执行集合层对齐并返回 AlignmentResult。"""
        track = data.chosen_text_track
        if not track or not track.text_clean:
            empty = AlignmentResult(
                aligned_words=[],
                alignment_score=0.0,
                gap_ratio=0.0,
                gap_positions=[],
                resolution=None,
                coverage=0.0,
            )
            return CollectionLayerOutput(alignment_result=empty)

        clean_text = track.text_clean
        sv_words = data.sv_words or []
        vad_range = self._resolve_vad_range(sv_words, data.vad_intervals)

        if data.is_fast_only_mode:
            fast_only_result = self._build_fast_only_identity_result(sv_words=sv_words)
            if fast_only_result is not None:
                self._logger.info(
                    "集合层轻量模式命中: words={} score={:.2f} coverage={:.2f}",
                    len(fast_only_result.aligned_words),
                    fast_only_result.alignment_score,
                    fast_only_result.coverage,
                )
                return CollectionLayerOutput(alignment_result=fast_only_result)

        if not self._config.is_enabled:
            fallback = self._build_pseudo_result(clean_text, vad_range)
            return CollectionLayerOutput(alignment_result=fallback)

        if not self._config.use_sv_timebase:
            self._logger.warning("集合层对齐关闭 SV 时间基准，改用伪对齐")
            fallback = self._build_pseudo_result(clean_text, vad_range)
            return CollectionLayerOutput(alignment_result=fallback)

        if not sv_words:
            fallback = self._build_pseudo_result(clean_text, vad_range)
            return CollectionLayerOutput(alignment_result=fallback)

        tokens = self._alignment_service._tokenize(clean_text)
        token_confidences = self._build_token_confidences(track, clean_text, tokens)

        result = self._alignment_service.align_clean_text(
            clean_text,
            sv_words,
            vad_range=vad_range,
            vad_intervals=data.vad_intervals,
            token_confidences=token_confidences,
        )

        if not result.aligned_words:
            result = self._build_pseudo_result(clean_text, vad_range)

        self._logger.info(
            "集合层对齐完成: words={} score={:.2f} gap_ratio={:.2f} coverage={:.2f}",
            len(result.aligned_words),
            result.alignment_score,
            result.gap_ratio,
            result.coverage,
        )
        return CollectionLayerOutput(alignment_result=result)

    def _build_fast_only_identity_result(
        self,
        *,
        sv_words: Sequence[WordTimestamp],
    ) -> Optional[AlignmentResult]:
        """Fast-Only 轻量对齐：直接复用快流词时间，避免重复 NW 对齐开销。"""
        aligned_words: List[AlignedWord] = []
        for word in sv_words or []:
            text = str(getattr(word, "word", "") or "").strip()
            if not text:
                continue
            start = float(getattr(word, "start", 0.0) or 0.0)
            end = float(getattr(word, "end", start) or start)
            if end <= start:
                end = start + 1e-3

            confidence_raw = getattr(word, "confidence", None)
            confidence = (
                float(confidence_raw)
                if isinstance(confidence_raw, (int, float))
                else None
            )
            aligned_words.append(
                AlignedWord(
                    word=text,
                    start=start,
                    end=end,
                    sv_confidence=confidence,
                    whisper_confidence=None,
                    final_confidence=confidence if confidence is not None else 1.0,
                    confidence_source=str(getattr(word, "confidence_source", "") or "fast"),
                    alignment_status=AlignmentStatus.MATCHED,
                    is_pseudo=False,
                    sv_original=text,
                    whisper_original=text,
                )
            )

        if not aligned_words:
            return None

        quality = self._quality_stats.compute(
            aligned_words,
            total_tokens=len(aligned_words),
            gap_positions=[],
            gap_resolution=None,
        )
        return AlignmentResult(
            aligned_words=aligned_words,
            alignment_score=quality.alignment_score,
            gap_ratio=quality.gap_ratio,
            gap_positions=quality.gap_positions,
            resolution=quality.gap_resolution,
            coverage=quality.coverage,
        )

    @staticmethod
    def _resolve_vad_range(
        sv_words: Sequence[WordTimestamp],
        vad_intervals: Optional[Sequence[Tuple[float, float]]],
    ) -> Optional[Tuple[float, float]]:
        if sv_words:
            return sv_words[0].start, sv_words[-1].end
        if vad_intervals:
            start = min(interval[0] for interval in vad_intervals)
            end = max(interval[1] for interval in vad_intervals)
            return start, end
        return None

    def _build_pseudo_result(
        self,
        clean_text: str,
        vad_range: Optional[Tuple[float, float]],
    ) -> AlignmentResult:
        if not clean_text or vad_range is None:
            return AlignmentResult(
                aligned_words=[],
                alignment_score=0.0,
                gap_ratio=0.0,
                gap_positions=[],
                resolution=None,
                coverage=0.0,
            )

        pseudo_words = PseudoAlignment.apply(
            original_start=vad_range[0],
            original_end=vad_range[1],
            new_text=clean_text,
            default_confidence=None,
        )
        aligned_words = [
            AlignedWord(
                word=word.word,
                start=word.start,
                end=word.end,
                sv_confidence=None,
                whisper_confidence=None,
                final_confidence=None,
                confidence_source="unknown",
                alignment_status=AlignmentStatus.PSEUDO,
                is_pseudo=True,
                sv_original=None,
                whisper_original=word.word,
            )
            for word in pseudo_words
        ]
        quality = self._quality_stats.compute(
            aligned_words,
            total_tokens=len(aligned_words),
            gap_positions=[0] if aligned_words else [],
            gap_resolution=GapResolution.DEGRADED if aligned_words else None,
        )
        return AlignmentResult(
            aligned_words=aligned_words,
            alignment_score=quality.alignment_score,
            gap_ratio=quality.gap_ratio,
            gap_positions=quality.gap_positions,
            resolution=quality.gap_resolution,
            coverage=quality.coverage,
        )

    @staticmethod
    def _build_token_confidences(
        track: TextTrack,
        clean_text: str,
        tokens: Sequence[str],
    ) -> Optional[List[Optional[float]]]:
        if not track.clean_to_word or not track.word_confidences:
            return None

        confidences: List[Optional[float]] = []
        cursor = 0
        for token in tokens:
            if not token:
                confidences.append(None)
                continue
            while cursor < len(clean_text) and clean_text[cursor].isspace():
                cursor += 1
            if cursor >= len(clean_text):
                confidences.append(None)
                continue

            idx = clean_text.find(token, cursor)
            if idx == -1 and clean_text[cursor: cursor + len(token)] == token:
                idx = cursor
            if idx == -1:
                confidences.append(None)
                continue

            word_idx = None
            end_idx = min(idx + len(token), len(track.clean_to_word))
            for pos in range(idx, end_idx):
                mapped = track.clean_to_word[pos]
                if mapped is not None:
                    word_idx = mapped
                    break
            if word_idx is None or word_idx >= len(track.word_confidences):
                confidences.append(None)
            else:
                confidences.append(track.word_confidences[word_idx])
            cursor = idx + len(token)
        return confidences

    @staticmethod
    def _load_alignment_config(config_override: Optional[Dict[str, Any]]) -> AlignmentLayerConfig:
        if config_override is None:
            return TextPipelineConfig.from_runtime().alignment
        return AlignmentLayerConfig.from_runtime(config_override)


@dataclass
class CollectionFactBuilderConfig:
    """集合层配置。"""

    anchor_snap_tolerance_sec: float = 0.22
    is_enable_time_mapping: bool = False
    time_axis_version: str = "m2_nw_v2"


class CollectionFactBuilder:
    """
    集合层事实构建器（Builder Pattern）。

    Why:
    - 将快慢词、turn、时间映射拼装收敛到单入口，避免编排层散落组装。
    - 只生产事实对象，不参与切分与裁决，便于后续阶段复用。
    """

    def __init__(self, config: Optional[CollectionFactBuilderConfig] = None) -> None:
        self.config = config or CollectionFactBuilderConfig()

    def build(
        self,
        *,
        annotated_words: Sequence[AnnotatedWord],
        alignment_result: Optional[AlignmentResult],
        speaker_turns: Sequence[Dict[str, Any]],
        fast_draft_cuts: Sequence[float],
        pyannote_frame_times: Sequence[float],
    ) -> AlignedFacts:
        normalized_words = list(annotated_words or [])
        normalized_turns = self._normalize_speaker_turns(speaker_turns)
        normalized_cuts = self._normalize_cut_times(fast_draft_cuts)
        normalized_frames = self._normalize_cut_times(pyannote_frame_times)
        if not normalized_frames:
            # Why: 阶段3仅要求契约可观测；无 pyannote 帧时使用快流边界作为最小候选。
            normalized_frames = list(normalized_cuts)
        boundaries = self._collect_word_boundaries(normalized_words)
        time_mappings = map_frame_times_to_word_boundaries(
            frame_times=normalized_frames,
            word_boundaries=boundaries,
            tolerance_sec=float(self.config.anchor_snap_tolerance_sec),
            is_enable_mapping=bool(self.config.is_enable_time_mapping),
        )

        alignment_score = (
            float(alignment_result.alignment_score)
            if alignment_result is not None
            else 0.0
        )
        gap_ratio = (
            float(alignment_result.gap_ratio)
            if alignment_result is not None
            else 0.0
        )
        gap_positions = (
            list(alignment_result.gap_positions)
            if alignment_result is not None
            else []
        )

        return AlignedFacts(
            annotated_words=normalized_words,
            alignment_score=alignment_score,
            gap_ratio=gap_ratio,
            gap_positions=gap_positions,
            speaker_turns=normalized_turns,
            fast_draft_cuts=normalized_cuts,
            time_axis_version=str(self.config.time_axis_version),
            time_mappings=time_mappings,
        )

    @staticmethod
    def _normalize_speaker_turns(
        turns: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in turns or []:
            if not isinstance(item, dict):
                continue
            turn_id = str(item.get("turn_id", "") or "").strip()
            speaker_id = str(item.get("speaker_id", "") or "unknown").strip() or "unknown"
            start_raw = item.get("start")
            end_raw = item.get("end")
            if start_raw is None or end_raw is None:
                continue
            start = float(start_raw)
            end = float(end_raw)
            if end <= start:
                continue
            normalized.append(
                {
                    "turn_id": turn_id,
                    "speaker_id": speaker_id,
                    "start": start,
                    "end": end,
                    "source": str(item.get("source", "") or ""),
                    "boundary_confidence": (
                        float(item.get("boundary_confidence"))
                        if item.get("boundary_confidence") is not None
                        else 0.0
                    ),
                }
            )
        return normalized

    @staticmethod
    def _collect_word_boundaries(
        words: Sequence[AnnotatedWord],
    ) -> List[float]:
        boundaries: List[float] = []
        for word in words:
            if word.start is not None:
                boundaries.append(float(word.start))
            if word.end is not None:
                boundaries.append(float(word.end))
        return sorted(set(boundaries))

    @staticmethod
    def _normalize_cut_times(values: Sequence[float]) -> List[float]:
        normalized: List[float] = []
        for value in values or []:
            try:
                normalized.append(float(value))
            except (TypeError, ValueError):
                continue
        return sorted(set(normalized))


# 兼容旧命名（用于过渡期）。
AlignmentProcessor = CollectionAlignmentProcessor
FactBuilder = CollectionFactBuilder
FactBuilderConfig = CollectionFactBuilderConfig

__all__ = [
    "CollectionAlignmentProcessor",
    "CollectionFactBuilder",
    "CollectionFactBuilderConfig",
    "AlignmentProcessor",
    "FactBuilder",
    "FactBuilderConfig",
]
