"""
L2 文本选文处理器（单路径）。
V3.2.0+dev.20260216.01
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from app.core.logging import resolve_loguru_logger
from app.services.alignment.types import L2Input, L2Output, QualitySignals, TextTrack
from app.services.text_pipeline_config import TextPipelineConfig


@dataclass
class ArbitrationResult:
    """仲裁结果（包含决策上下文）。"""

    chosen_source: str
    reason: str
    sv_score: float
    wh_score: float
    coverage: float
    error_code: Optional[str] = None
    gap_positions: List[int] = field(default_factory=list)


class TextArbiterProcessor:
    """处理器模式：单路径选文，输出明确来源与原因。"""

    _DEFAULT_CONFIG: Dict[str, Any] = {
        "enable": True,
        "text_source_preference": "auto",
        "min_length_ratio": 0.65,
        "max_length_ratio": 3.0,
        "low_confidence_threshold": 0.5,
        "hallucination_block": True,
    }

    def __init__(
        self,
        *,
        logger: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._logger = resolve_loguru_logger(
            logger,
            __name__,
            layer="选文层",
            processor_name="text_arbiter_processor",
        )
        self._config_override = dict(config) if config else None

    def process(self, data: L2Input) -> L2Output:
        """执行 L2 仲裁，输出 chosen_text_track 与 ArbitrationResult。"""
        sv_track = data.sv_track
        whisper_track = data.whisper_track
        quality = data.quality_signals
        config = self._resolve_config()
        preference = self._normalize_preference(config.get("text_source_preference"))
        is_enabled = bool(config.get("enable", True))

        sv_text = self._get_text(sv_track)
        wh_text = self._get_text(whisper_track)
        coverage = self._compute_coverage(sv_text, wh_text)

        if not sv_track and not whisper_track:
            error_code = "E_L2_ARBITRATION_NO_TEXT"
            result = self._build_result(
                chosen_source="fast",
                reason="missing_tracks",
                quality=quality,
                coverage=coverage,
                error_code=error_code,
            )
            self._logger.warning("选文层缺失输入，回退 fast")
            return L2Output(chosen_text_track=None, arbitration_result=result)

        chosen_source, reason = self._decide_source(
            is_enabled=is_enabled,
            preference=preference,
            quality=quality,
            config=config,
            sv_text=sv_text,
            wh_text=wh_text,
        )

        chosen_track = self._select_track(chosen_source, sv_track, whisper_track)
        chosen_text = self._get_text(chosen_track)
        error_code = None
        if not chosen_text:
            error_code = "E_L2_ARBITRATION_NO_TEXT"
            fallback_track = sv_track if self._get_text(sv_track) else None
            if fallback_track:
                chosen_track = fallback_track
                chosen_source = "fast"
                reason = f"{reason}|fallback_fast"
            else:
                chosen_track = None
                chosen_source = "fast"
                reason = f"{reason}|fallback_fast_missing"

        chosen_track = self._clone_track(chosen_track) if chosen_track else None

        if chosen_track:
            chosen_track.source = "chosen"

        result = self._build_result(
            chosen_source=chosen_source,
            reason=reason,
            quality=quality,
            coverage=coverage,
            error_code=error_code,
        )
        self._log_result(
            chosen_source=chosen_source,
            reason=reason,
            sv_text=sv_text,
            wh_text=wh_text,
            chosen_text=self._get_text(chosen_track),
            result=result,
        )
        return L2Output(chosen_text_track=chosen_track, arbitration_result=result)

    def _decide_source(
        self,
        *,
        is_enabled: bool,
        preference: str,
        quality: QualitySignals,
        config: Dict[str, Any],
        sv_text: str,
        wh_text: str,
    ) -> tuple[str, str]:
        if not is_enabled:
            chosen_source = "slow" if preference == "slow" else "fast"
            return chosen_source, "disabled_preference"

        if quality.is_hallucination and bool(config.get("hallucination_block", True)):
            self._logger.debug("选文层门控: hallucination_flag=true")
            return "fast", "hallucination"

        if quality.is_repetition:
            self._logger.debug("选文层门控: repetition_flag=true")
            return "fast", "repetition"

        length_ratio = self._resolve_length_ratio(quality.length_ratio, sv_text, wh_text)
        min_ratio = float(config.get("min_length_ratio", 0.65))
        max_ratio = float(config.get("max_length_ratio", 3.0))
        if length_ratio < min_ratio:
            self._logger.debug("选文层门控: length_ratio={:.2f} < {:.2f}", length_ratio, min_ratio)
            return "slow", "length_ratio_low"
        if length_ratio > max_ratio:
            self._logger.debug("选文层门控: length_ratio={:.2f} > {:.2f}", length_ratio, max_ratio)
            return "fast", "length_ratio_high"

        slow_conf = float(quality.confidence_slow or 0.0)
        low_conf = float(config.get("low_confidence_threshold", 0.5))
        if slow_conf < low_conf:
            self._logger.debug("选文层门控: slow_conf={:.2f} < {:.2f}", slow_conf, low_conf)
            return "fast", "low_confidence_slow"

        if preference == "fast":
            return "fast", "preference_fast"
        if preference == "slow":
            return "slow", "preference_slow"
        return "slow", "auto_slow"

    def _resolve_config(self) -> Dict[str, Any]:
        if self._config_override is not None:
            return self._apply_defaults(self._config_override)
        pipeline_config = TextPipelineConfig.from_runtime()
        return pipeline_config.arbitration.to_dict()

    def _apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(self._DEFAULT_CONFIG)
        merged.update({k: v for k, v in config.items() if v is not None})
        return merged

    @staticmethod
    def _normalize_preference(value: Optional[str]) -> str:
        if not value:
            return "auto"
        normalized = str(value).lower()
        if normalized in {"fast", "slow", "auto"}:
            return normalized
        return "auto"

    @staticmethod
    def _select_track(
        source: str,
        sv_track: Optional[TextTrack],
        whisper_track: Optional[TextTrack],
    ) -> Optional[TextTrack]:
        if source == "fast":
            return sv_track
        if source == "slow":
            return whisper_track
        return whisper_track or sv_track

    @staticmethod
    def _clone_track(track: TextTrack) -> TextTrack:
        return replace(
            track,
            clean_to_word=list(track.clean_to_word),
            punct_positions=list(track.punct_positions),
        )

    @staticmethod
    def _get_text(track: Optional[TextTrack]) -> str:
        if not track:
            return ""
        return str(track.text_clean or track.text_itn_raw or track.raw_text or "")

    @staticmethod
    def _resolve_length_ratio(length_ratio: float, sv_text: str, wh_text: str) -> float:
        if length_ratio > 0:
            return length_ratio
        if not sv_text or not wh_text:
            return 0.0
        return len(sv_text) / max(len(wh_text), 1)

    @staticmethod
    def _compute_coverage(sv_text: str, wh_text: str) -> float:
        """使用序列相似度估算覆盖率。"""
        if not sv_text or not wh_text:
            return 0.0
        normalized_sv = " ".join(sv_text.split())
        normalized_wh = " ".join(wh_text.split())
        return SequenceMatcher(None, normalized_sv, normalized_wh).ratio()

    @staticmethod
    def _build_result(
        *,
        chosen_source: str,
        reason: str,
        quality: QualitySignals,
        coverage: float,
        error_code: Optional[str] = None,
    ) -> ArbitrationResult:
        return ArbitrationResult(
            chosen_source=chosen_source,
            reason=reason,
            sv_score=float(quality.confidence_fast or 0.0),
            wh_score=float(quality.confidence_slow or 0.0),
            coverage=coverage,
            error_code=error_code,
            gap_positions=[],
        )

    def _log_result(
        self,
        *,
        chosen_source: str,
        reason: str,
        sv_text: str,
        wh_text: str,
        chosen_text: str,
        result: ArbitrationResult,
    ) -> None:
        message = (
            "选文层完成 chosen_source={} reason={} coverage={:.2f} sv_score={:.2f} "
            "wh_score={:.2f} input_len=({}, {}) output_len={}"
        )
        if result.error_code:
            message += " error_code={}"
            self._logger.info(
                message,
                chosen_source,
                reason,
                result.coverage,
                result.sv_score,
                result.wh_score,
                len(sv_text),
                len(wh_text),
                len(chosen_text),
                result.error_code,
            )
        else:
            self._logger.info(
                message,
                chosen_source,
                reason,
                result.coverage,
                result.sv_score,
                result.wh_score,
                len(sv_text),
                len(wh_text),
                len(chosen_text),
            )


class Arbiter(TextArbiterProcessor):
    """兼容旧命名（已统一为 TextArbiterProcessor）。"""
