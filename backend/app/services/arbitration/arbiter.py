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
    is_edge_selection_bypassed: bool = False
    forced_source: Optional[str] = None
    edge_selection_mode: str = "auto"


class TextArbiterProcessor:
    """处理器模式：单路径选文，输出明确来源与原因。"""

    _DEFAULT_CONFIG: Dict[str, Any] = {
        "enable": True,
        "text_source_preference": "auto",
        "edge_selection_mode": "auto",
        "min_length_ratio": 0.65,
        "max_length_ratio": 3.0,
        "low_confidence_threshold": 0.5,
        "hallucination_block": True,
        # V3.2.4+dev.20260301.05:
        # 慢流优先策略在“快流包含慢流完整内容 + 额外有效尾句”时会漏句，
        # 因此引入 fast_tail_guard，保护跨说话人切换附近的快流补尾信息。
        "enable_fast_tail_guard": True,
        "fast_tail_min_extra_chars": 10,
        "fast_tail_min_wh_coverage": 0.92,
        "fast_tail_min_fast_confidence": 0.55,
        "fast_tail_max_slow_advantage": 0.2,
        "fast_tail_require_end_match_ratio": 0.9,
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
        edge_selection_mode = self._normalize_edge_selection_mode(
            data.edge_selection_mode or config.get("edge_selection_mode")
        )
        is_enabled = bool(config.get("enable", True))

        sv_text = self._get_text(sv_track)
        wh_text = self._get_text(whisper_track)
        coverage = self._compute_coverage(sv_text, wh_text)

        forced_source = self._resolve_forced_source(edge_selection_mode)
        if forced_source is not None:
            chosen_track = self._select_track(forced_source, sv_track, whisper_track)
            chosen_text = self._get_text(chosen_track)
            error_code = None
            reason = f"forced_{forced_source}"
            if not chosen_text:
                chosen_track = None
                reason = "forced_source_missing"
                error_code = "E_L2_ARBITRATION_FORCED_SOURCE_MISSING"
            else:
                chosen_track = self._clone_track(chosen_track)
                chosen_track.source = "chosen"
            result = self._build_result(
                chosen_source=forced_source,
                reason=reason,
                quality=quality,
                coverage=coverage,
                error_code=error_code,
                is_edge_selection_bypassed=True,
                forced_source=forced_source,
                edge_selection_mode=edge_selection_mode,
            )
            self._log_result(
                chosen_source=forced_source,
                reason=reason,
                sv_text=sv_text,
                wh_text=wh_text,
                chosen_text=self._get_text(chosen_track),
                result=result,
            )
            return L2Output(chosen_text_track=chosen_track, arbitration_result=result)

        if not sv_track and not whisper_track:
            error_code = "E_L2_ARBITRATION_NO_TEXT"
            result = self._build_result(
                chosen_source="fast",
                reason="missing_tracks",
                quality=quality,
                coverage=coverage,
                error_code=error_code,
                edge_selection_mode=edge_selection_mode,
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
            edge_selection_mode=edge_selection_mode,
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

        if self._is_fast_tail_guard_triggered(
            quality=quality,
            config=config,
            sv_text=sv_text,
            wh_text=wh_text,
        ):
            return "fast", "fast_tail_guard"

        if preference == "fast":
            return "fast", "preference_fast"
        if preference == "slow":
            return "slow", "preference_slow"
        return "slow", "auto_slow"

    def _is_fast_tail_guard_triggered(
        self,
        *,
        quality: QualitySignals,
        config: Dict[str, Any],
        sv_text: str,
        wh_text: str,
    ) -> bool:
        """检测“快流包含慢流且额外尾句”场景，避免 auto_slow 造成漏句。"""
        if not bool(config.get("enable_fast_tail_guard", True)):
            return False
        normalized_sv = self._normalize_text(sv_text)
        normalized_wh = self._normalize_text(wh_text)
        if not normalized_sv or not normalized_wh:
            return False
        if len(normalized_sv) <= len(normalized_wh):
            return False

        fast_conf = float(quality.confidence_fast or 0.0)
        slow_conf = float(quality.confidence_slow or 0.0)
        min_fast_conf = float(config.get("fast_tail_min_fast_confidence", 0.55))
        max_slow_advantage = float(config.get("fast_tail_max_slow_advantage", 0.2))
        if fast_conf < min_fast_conf:
            return False
        if (slow_conf - fast_conf) > max_slow_advantage:
            return False

        matcher = SequenceMatcher(None, normalized_wh, normalized_sv)
        blocks = matcher.get_matching_blocks()
        content_blocks = [block for block in blocks if block.size > 0]
        if not content_blocks:
            return False

        covered_wh_chars = sum(block.size for block in content_blocks)
        covered_wh_ratio = covered_wh_chars / max(len(normalized_wh), 1)
        min_wh_coverage = float(config.get("fast_tail_min_wh_coverage", 0.92))
        if covered_wh_ratio < min_wh_coverage:
            return False

        last_block = max(content_blocks, key=lambda item: item.a + item.size)
        wh_end_ratio = (last_block.a + last_block.size) / max(len(normalized_wh), 1)
        required_end_match_ratio = float(config.get("fast_tail_require_end_match_ratio", 0.9))
        if wh_end_ratio < required_end_match_ratio:
            return False

        tail_start = last_block.b + last_block.size
        fast_tail = normalized_sv[tail_start:].strip(" ,.;:!?")
        min_tail_chars = int(config.get("fast_tail_min_extra_chars", 10))
        if len(fast_tail) < min_tail_chars:
            return False
        if not any(ch.isalnum() for ch in fast_tail):
            return False

        self._logger.info(
            "选文层 fast_tail_guard 命中: wh_coverage={:.2f} wh_end_ratio={:.2f} "
            "tail_chars={} slow_conf={:.2f} fast_conf={:.2f}",
            covered_wh_ratio,
            wh_end_ratio,
            len(fast_tail),
            slow_conf,
            fast_conf,
        )
        return True

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
    def _normalize_edge_selection_mode(value: Optional[str]) -> str:
        if not value:
            return "auto"
        normalized = str(value).strip().lower()
        if normalized in {"auto", "force_fast", "force_slow"}:
            return normalized
        return "auto"

    @staticmethod
    def _resolve_forced_source(edge_selection_mode: str) -> Optional[str]:
        if edge_selection_mode == "force_fast":
            return "fast"
        if edge_selection_mode == "force_slow":
            return "slow"
        return None

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
    def _normalize_text(text: str) -> str:
        return " ".join(str(text or "").strip().lower().split())

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
        is_edge_selection_bypassed: bool = False,
        forced_source: Optional[str] = None,
        edge_selection_mode: str = "auto",
    ) -> ArbitrationResult:
        return ArbitrationResult(
            chosen_source=chosen_source,
            reason=reason,
            sv_score=float(quality.confidence_fast or 0.0),
            wh_score=float(quality.confidence_slow or 0.0),
            coverage=coverage,
            error_code=error_code,
            gap_positions=[],
            is_edge_selection_bypassed=bool(is_edge_selection_bypassed),
            forced_source=forced_source,
            edge_selection_mode=edge_selection_mode,
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
