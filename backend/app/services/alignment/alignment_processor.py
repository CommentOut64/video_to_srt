"""
L4 对齐层处理器（AlignmentProcessor）。
V3.2.0+dev.20260205.09
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.logging import resolve_loguru_logger
from app.models.confidence_models import AlignedWord, AlignmentStatus
from app.models.sensevoice_models import WordTimestamp
from app.services.alignment.alignment_service import AlignmentService, AlignmentConfig
from app.services.alignment.gap_resolver import GapResolver, GapResolution
from app.services.alignment.quality_stats import QualityStatsCalculator
from app.services.alignment.types import AlignmentResult, L4Input, L4Output, TextTrack
from app.services.pseudo_alignment import PseudoAlignment
from app.services.text_pipeline_config import TextPipelineConfig, AlignmentLayerConfig


class AlignmentProcessor:
    """L4 对齐层处理器：对齐 + Gap 修复 + 质量统计。"""

    def __init__(
        self,
        *,
        logger: Optional[Any] = None,
        config_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._logger = resolve_loguru_logger(
            logger,
            __name__,
            layer="L4",
            processor_name="alignment_processor",
        )
        self._config = self._load_alignment_config(config_override)
        self._gap_resolver = GapResolver(
            gap_ratio_low=self._config.gap_ratio_low,
            gap_ratio_mid=self._config.gap_ratio_mid,
            min_valid_neighbors=self._config.min_valid_neighbors,
            min_word_duration=self._config.min_word_duration_ms / 1000.0,
            logger=self._logger,
        )
        self._quality_stats = QualityStatsCalculator(logger=self._logger)
        self._alignment_service = AlignmentService(
            config=AlignmentConfig(),
            logger=self._logger,
            gap_resolver=self._gap_resolver,
            quality_stats_calculator=self._quality_stats,
        )

    def process(self, data: L4Input) -> L4Output:
        """执行 L4 对齐并返回 AlignmentResult。"""
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
            return L4Output(alignment_result=empty)

        clean_text = track.text_clean
        sv_words = data.sv_words or []
        vad_range = self._resolve_vad_range(sv_words, data.vad_intervals)

        if not self._config.is_enabled:
            fallback = self._build_pseudo_result(clean_text, vad_range)
            return L4Output(alignment_result=fallback)

        if not self._config.use_sv_timebase:
            self._logger.warning("L4 对齐关闭 SV 时间基准，改用伪对齐")
            fallback = self._build_pseudo_result(clean_text, vad_range)
            return L4Output(alignment_result=fallback)

        if not sv_words:
            fallback = self._build_pseudo_result(clean_text, vad_range)
            return L4Output(alignment_result=fallback)

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
            "L4 对齐完成: words={} score={:.2f} gap_ratio={:.2f} coverage={:.2f}",
            len(result.aligned_words),
            result.alignment_score,
            result.gap_ratio,
            result.coverage,
        )
        return L4Output(alignment_result=result)

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
