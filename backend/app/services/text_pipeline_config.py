"""
文本处理统一参数入口（TextPipelineConfig）。
V3.2.0+dev.20260214.10
"""
# V3.2.0+dev.20260205.09: 接入 L4 对齐层参数。
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from app.services.model_runtime_config_service import get_model_runtime_config_service


_PUNCT_WIDTH_OPTIONS = {"auto", "full", "half"}


def _read_runtime_value(raw: Dict[str, Any], key: str, default: Any) -> Any:
    """读取运行参数，兼容 dotted / nested / underscore 三种口径。"""
    if key in raw:
        return raw[key]
    if "." in key:
        current: Any = raw
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if current is not None:
            return current
    alt_key = key.replace(".", "_")
    if alt_key in raw:
        return raw[alt_key]
    return default


def _read_runtime_optional(raw: Dict[str, Any], key: str) -> Tuple[Any, bool]:
    """
    读取运行参数的“显式覆盖值”。

    说明：
    - 用于“yaml 默认值 + 运行参数覆盖”的场景；
    - 仅当 key 在 override 中显式出现（dotted/nested/underscore 任一口径）才返回 found=True；
    - 若显式出现但 value 为 None，视为“清空覆盖”，返回 found=False（让上游回退到默认）。
    """
    if key in raw:
        value = raw[key]
        return (value, value is not None)
    if "." in key:
        current: Any = raw
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if current is not None:
            return (current, True)
    alt_key = key.replace(".", "_")
    if alt_key in raw:
        value = raw[alt_key]
        return (value, value is not None)
    return (None, False)


@dataclass
class PunctuationSchedulerOverride:
    """L3 标点调度覆盖项（仅当运行参数显式设置时生效）。"""

    mode: Optional[str] = None
    fast_confidence_threshold: Optional[float] = None
    alignment_coverage_threshold: Optional[float] = None
    arbiter_conflict_threshold: Optional[float] = None
    max_slow_retries: Optional[int] = None


@dataclass
class PunctuationPostprocessModeOverride:
    """后处理模式覆盖项（fast/dual）。"""

    candidate_min_conf: Optional[float] = None
    add_mid_conf: Optional[float] = None
    add_end_conf: Optional[float] = None
    keep_raw_mid_conf: Optional[float] = None
    keep_raw_end_conf: Optional[float] = None
    drop_raw_mid_conf: Optional[float] = None
    drop_raw_end_conf: Optional[float] = None
    question_gate_min_conf: Optional[float] = None
    pause_end_min_sec: Optional[float] = None


@dataclass
class PunctuationPostprocessSharedOverride:
    """后处理 shared 覆盖项。"""

    candidate_window_words: Optional[int] = None
    conflict_window_chars: Optional[int] = None
    max_repeat_punct: Optional[int] = None
    allowed_punct_zh: Optional[str] = None
    allowed_punct_en: Optional[str] = None
    comma_guard_enabled: Optional[bool] = None
    comma_guard_min_conf: Optional[float] = None
    comma_guard_min_chars: Optional[int] = None
    comma_guard_min_words: Optional[int] = None
    comma_guard_pause_min_sec: Optional[float] = None


@dataclass
class PunctuationRuntimeOverrides:
    """标点运行参数覆盖集合（用于覆盖 punctuation.yaml 的默认值）。"""

    scheduler: PunctuationSchedulerOverride
    postprocess_fast: PunctuationPostprocessModeOverride
    postprocess_dual: PunctuationPostprocessModeOverride
    postprocess_shared: PunctuationPostprocessSharedOverride

    @classmethod
    def from_runtime(cls, runtime: Optional[Dict[str, Any]] = None) -> "PunctuationRuntimeOverrides":
        if runtime is None:
            runtime = get_model_runtime_config_service().get_effective_runtime_global()
        override = runtime.get("override", {}) if isinstance(runtime, dict) else {}
        punct = override.get("punctuation", {}) if isinstance(override, dict) else {}
        punct = punct if isinstance(punct, dict) else {}

        mode_raw, has_mode = _read_runtime_optional(punct, "scheduler.mode")
        mode = str(mode_raw).lower() if has_mode and mode_raw is not None else None
        if mode not in {"fast_only", "dual", "smart_review"}:
            mode = None

        fast_thr, has_fast_thr = _read_runtime_optional(punct, "scheduler.fast_confidence_threshold")
        cov_thr, has_cov_thr = _read_runtime_optional(punct, "scheduler.alignment_coverage_threshold")
        conflict_thr, has_conflict_thr = _read_runtime_optional(punct, "scheduler.arbiter_conflict_threshold")
        max_retry, has_max_retry = _read_runtime_optional(punct, "scheduler.max_slow_retries")

        scheduler = PunctuationSchedulerOverride(
            mode=mode,
            fast_confidence_threshold=float(fast_thr) if has_fast_thr else None,
            alignment_coverage_threshold=float(cov_thr) if has_cov_thr else None,
            arbiter_conflict_threshold=float(conflict_thr) if has_conflict_thr else None,
            max_slow_retries=int(max_retry) if has_max_retry else None,
        )

        def mode_override(prefix: str) -> PunctuationPostprocessModeOverride:
            def read_float(key: str) -> Optional[float]:
                value, found = _read_runtime_optional(punct, f"postprocess.{prefix}.{key}")
                return float(value) if found else None

            return PunctuationPostprocessModeOverride(
                candidate_min_conf=read_float("candidate_min_conf"),
                add_mid_conf=read_float("add_mid_conf"),
                add_end_conf=read_float("add_end_conf"),
                keep_raw_mid_conf=read_float("keep_raw_mid_conf"),
                keep_raw_end_conf=read_float("keep_raw_end_conf"),
                drop_raw_mid_conf=read_float("drop_raw_mid_conf"),
                drop_raw_end_conf=read_float("drop_raw_end_conf"),
                question_gate_min_conf=read_float("question_gate_min_conf"),
                pause_end_min_sec=read_float("pause_end_min_sec"),
            )

        def shared_override() -> PunctuationPostprocessSharedOverride:
            def read_int(key: str) -> Optional[int]:
                value, found = _read_runtime_optional(punct, f"postprocess.shared.{key}")
                return int(value) if found else None

            def read_float(key: str) -> Optional[float]:
                value, found = _read_runtime_optional(punct, f"postprocess.shared.{key}")
                return float(value) if found else None

            def read_str(key: str) -> Optional[str]:
                value, found = _read_runtime_optional(punct, f"postprocess.shared.{key}")
                return str(value) if found else None

            def read_bool(key: str) -> Optional[bool]:
                value, found = _read_runtime_optional(punct, f"postprocess.shared.{key}")
                return bool(value) if found else None

            return PunctuationPostprocessSharedOverride(
                candidate_window_words=read_int("candidate_window_words"),
                conflict_window_chars=read_int("conflict_window_chars"),
                max_repeat_punct=read_int("max_repeat_punct"),
                allowed_punct_zh=read_str("allowed_punct_zh"),
                allowed_punct_en=read_str("allowed_punct_en"),
                comma_guard_enabled=read_bool("comma_guard_enabled"),
                comma_guard_min_conf=read_float("comma_guard_min_conf"),
                comma_guard_min_chars=read_int("comma_guard_min_chars"),
                comma_guard_min_words=read_int("comma_guard_min_words"),
                comma_guard_pause_min_sec=read_float("comma_guard_pause_min_sec"),
            )

        return cls(
            scheduler=scheduler,
            postprocess_fast=mode_override("fast"),
            postprocess_dual=mode_override("dual"),
            postprocess_shared=shared_override(),
        )


@dataclass
class NormalizationConfig:
    """L1 规范化层参数。"""

    is_itn_enabled: bool = True
    itn_quality_min_ratio: float = 0.30
    itn_quality_max_ratio: float = 3.00
    is_decimal_protection_enabled: bool = True
    is_safe_punct_enabled: bool = True
    is_abbrev_dot_protection_enabled: bool = True
    is_hyphen_protection_enabled: bool = True
    is_cjk_single_digit_to_zh: bool = True
    is_cjk_digit_merge: bool = True
    is_collapse_spaces: bool = True
    punct_width: str = "auto"

    @classmethod
    def from_runtime(cls, raw: Optional[Dict[str, Any]]) -> "NormalizationConfig":
        raw = raw or {}
        width_raw = str(_read_runtime_value(raw, "punct_width", cls.punct_width) or cls.punct_width)
        width = width_raw.lower()
        if width not in _PUNCT_WIDTH_OPTIONS:
            width = cls.punct_width
        return cls(
            is_itn_enabled=bool(_read_runtime_value(raw, "itn.enable", cls.is_itn_enabled)),
            itn_quality_min_ratio=float(
                _read_runtime_value(raw, "itn.quality_min_ratio", cls.itn_quality_min_ratio)
            ),
            itn_quality_max_ratio=float(
                _read_runtime_value(raw, "itn.quality_max_ratio", cls.itn_quality_max_ratio)
            ),
            is_decimal_protection_enabled=bool(
                _read_runtime_value(raw, "decimal_protection.enable", cls.is_decimal_protection_enabled)
            ),
            is_safe_punct_enabled=bool(
                _read_runtime_value(raw, "safe_punct.enable", cls.is_safe_punct_enabled)
            ),
            is_abbrev_dot_protection_enabled=bool(
                _read_runtime_value(
                    raw,
                    "abbrev_dot_protection.enable",
                    cls.is_abbrev_dot_protection_enabled,
                )
            ),
            is_hyphen_protection_enabled=bool(
                _read_runtime_value(
                    raw,
                    "hyphen_protection.enable",
                    cls.is_hyphen_protection_enabled,
                )
            ),
            is_cjk_single_digit_to_zh=bool(
                _read_runtime_value(raw, "cjk_single_digit_to_zh", cls.is_cjk_single_digit_to_zh)
            ),
            is_cjk_digit_merge=bool(
                _read_runtime_value(raw, "cjk_digit_merge", cls.is_cjk_digit_merge)
            ),
            is_collapse_spaces=bool(
                _read_runtime_value(raw, "collapse_spaces", cls.is_collapse_spaces)
            ),
            punct_width=width,
        )


@dataclass
class AlignmentLayerConfig:
    """L4 对齐层参数。"""

    is_enabled: bool = True
    score_threshold: float = 0.3
    gap_ratio_low: float = 0.1
    gap_ratio_mid: float = 0.3
    gap_ratio_max: float = 0.4
    min_valid_neighbors: int = 1
    min_word_duration_ms: int = 100
    use_sv_timebase: bool = True
    # V3.2.0+dev.20260206.02: 双轨实验配置（仅影响 L4-L6 CPU 后处理，不并行占用 GPU）。
    is_enable_dual_time_experiment: bool = False
    dual_time_mode: str = "off"
    is_dual_time_write_debug_srt: bool = False
    dual_time_boundary_tolerance_ms: int = 250
    dual_time_active_min_boundary_f1: float = 0.85

    @classmethod
    def from_runtime(cls, raw: Optional[Dict[str, Any]]) -> "AlignmentLayerConfig":
        raw = raw or {}
        dual_time_mode = str(
            _read_runtime_value(raw, "dual_time_mode", cls.dual_time_mode)
            or cls.dual_time_mode
        ).lower()
        if dual_time_mode not in {"off", "shadow", "active"}:
            dual_time_mode = cls.dual_time_mode
        tolerance_ms = int(
            _read_runtime_value(
                raw,
                "dual_time_boundary_tolerance_ms",
                cls.dual_time_boundary_tolerance_ms,
            )
        )
        tolerance_ms = max(50, tolerance_ms)
        active_min_f1 = float(
            _read_runtime_value(
                raw,
                "dual_time_active_min_boundary_f1",
                cls.dual_time_active_min_boundary_f1,
            )
        )
        active_min_f1 = min(1.0, max(0.0, active_min_f1))
        return cls(
            is_enabled=bool(_read_runtime_value(raw, "enable", cls.is_enabled)),
            score_threshold=float(_read_runtime_value(raw, "score_threshold", cls.score_threshold)),
            gap_ratio_low=float(_read_runtime_value(raw, "gap_ratio_low", cls.gap_ratio_low)),
            gap_ratio_mid=float(_read_runtime_value(raw, "gap_ratio_mid", cls.gap_ratio_mid)),
            gap_ratio_max=float(_read_runtime_value(raw, "gap_ratio_max", cls.gap_ratio_max)),
            min_valid_neighbors=int(_read_runtime_value(raw, "min_valid_neighbors", cls.min_valid_neighbors)),
            min_word_duration_ms=int(_read_runtime_value(raw, "min_word_duration_ms", cls.min_word_duration_ms)),
            use_sv_timebase=bool(_read_runtime_value(raw, "use_sv_timebase", cls.use_sv_timebase)),
            is_enable_dual_time_experiment=bool(
                _read_runtime_value(
                    raw,
                    "enable_dual_time_experiment",
                    cls.is_enable_dual_time_experiment,
                )
            ),
            dual_time_mode=dual_time_mode,
            is_dual_time_write_debug_srt=bool(
                _read_runtime_value(
                    raw,
                    "dual_time_write_debug_srt",
                    cls.is_dual_time_write_debug_srt,
                )
            ),
            dual_time_boundary_tolerance_ms=tolerance_ms,
            dual_time_active_min_boundary_f1=active_min_f1,
        )


@dataclass
class SegmentationLayerConfig:
    """L6 切分层参数。"""

    is_enabled: bool = True
    min_chars: int = 6
    min_duration_sec: float = 0.8
    max_duration_sec: float = 12.0
    max_tokens: int = 40
    long_pause_sec: float = 0.8
    soft_pause_sec: float = 0.4
    short_merge_max_chars: int = 22
    is_keep_sentence_end_punct: bool = False

    final_min_tokens: int = 5
    final_max_tokens: int = 50
    final_min_duration: float = 0.5
    final_max_duration: float = 10.0
    final_soft_pause: float = 0.35
    final_long_pause: float = 0.8
    final_min_mapping_coverage: float = 0.6
    is_enable_soft_cut: bool = True
    is_enable_soft_cut_overlap_degrade: bool = False
    soft_cut_plan_provider: str = "m1_internal"
    soft_cut_plan_provider_class: str = ""

    @classmethod
    def from_runtime(
        cls,
        raw: Optional[Dict[str, Any]],
        punctuation_raw: Optional[Dict[str, Any]] = None,
    ) -> "SegmentationLayerConfig":
        raw = raw or {}
        punctuation_raw = punctuation_raw or {}

        keep_sentence_end_punct = _read_runtime_value(
            raw,
            "keep_sentence_end_punct",
            _read_runtime_value(
                punctuation_raw,
                "keep_sentence_end_punct",
                cls.is_keep_sentence_end_punct,
            ),
        )

        return cls(
            is_enabled=bool(_read_runtime_value(raw, "enable", cls.is_enabled)),
            min_chars=int(_read_runtime_value(raw, "min_chars", cls.min_chars)),
            min_duration_sec=float(
                _read_runtime_value(raw, "min_duration_sec", cls.min_duration_sec)
            ),
            max_duration_sec=float(
                _read_runtime_value(raw, "max_duration_sec", cls.max_duration_sec)
            ),
            max_tokens=int(_read_runtime_value(raw, "max_tokens", cls.max_tokens)),
            long_pause_sec=float(
                _read_runtime_value(raw, "long_pause_sec", cls.long_pause_sec)
            ),
            soft_pause_sec=float(
                _read_runtime_value(raw, "soft_pause_sec", cls.soft_pause_sec)
            ),
            short_merge_max_chars=int(
                _read_runtime_value(raw, "short_merge_max_chars", cls.short_merge_max_chars)
            ),
            is_keep_sentence_end_punct=bool(keep_sentence_end_punct),
            final_min_tokens=int(
                _read_runtime_value(raw, "final.min_tokens", cls.final_min_tokens)
            ),
            final_max_tokens=int(
                _read_runtime_value(raw, "final.max_tokens", cls.final_max_tokens)
            ),
            final_min_duration=float(
                _read_runtime_value(raw, "final.min_duration", cls.final_min_duration)
            ),
            final_max_duration=float(
                _read_runtime_value(raw, "final.max_duration", cls.final_max_duration)
            ),
            final_soft_pause=float(
                _read_runtime_value(raw, "final.soft_pause", cls.final_soft_pause)
            ),
            final_long_pause=float(
                _read_runtime_value(raw, "final.long_pause", cls.final_long_pause)
            ),
            final_min_mapping_coverage=float(
                _read_runtime_value(
                    raw,
                    "final.min_mapping_coverage",
                    cls.final_min_mapping_coverage,
                )
            ),
            is_enable_soft_cut=bool(
                _read_runtime_value(
                    raw,
                    "soft_cut.enable",
                    cls.is_enable_soft_cut,
                )
            ),
            is_enable_soft_cut_overlap_degrade=bool(
                _read_runtime_value(
                    raw,
                    "soft_cut.overlap_degrade_enable",
                    cls.is_enable_soft_cut_overlap_degrade,
                )
            ),
            soft_cut_plan_provider=str(
                _read_runtime_value(
                    raw,
                    "soft_cut.plan_provider",
                    cls.soft_cut_plan_provider,
                )
                or cls.soft_cut_plan_provider
            ).strip(),
            soft_cut_plan_provider_class=str(
                _read_runtime_value(
                    raw,
                    "soft_cut.plan_provider_class",
                    cls.soft_cut_plan_provider_class,
                )
                or cls.soft_cut_plan_provider_class
            ).strip(),
        )


@dataclass
class TextPipelineConfig:
    """文本处理流水线参数入口（L1/L2/L3/L4/L6）。"""

    normalization: NormalizationConfig
    arbitration: "ArbitrationConfig"
    punctuation: "PunctuationConfig"
    alignment: AlignmentLayerConfig
    segmentation: SegmentationLayerConfig

    @classmethod
    def from_runtime(cls, runtime: Optional[Dict[str, Any]] = None) -> "TextPipelineConfig":
        if runtime is None:
            runtime = get_model_runtime_config_service().get_effective_runtime_global()
        effective = runtime.get("effective", {}) if isinstance(runtime, dict) else {}
        normalization_raw = effective.get("normalization", {}) if isinstance(effective, dict) else {}
        arbitration_raw = effective.get("arbitration", {}) if isinstance(effective, dict) else {}
        punctuation_raw = effective.get("punctuation", {}) if isinstance(effective, dict) else {}
        alignment_raw = effective.get("alignment", {}) if isinstance(effective, dict) else {}
        segmentation_raw = effective.get("segmentation", {}) if isinstance(effective, dict) else {}
        return cls(
            normalization=NormalizationConfig.from_runtime(normalization_raw),
            arbitration=ArbitrationConfig.from_runtime(arbitration_raw),
            punctuation=PunctuationConfig.from_runtime(punctuation_raw),
            alignment=AlignmentLayerConfig.from_runtime(alignment_raw),
            segmentation=SegmentationLayerConfig.from_runtime(
                segmentation_raw,
                punctuation_raw=punctuation_raw,
            ),
        )


@dataclass
class ArbitrationConfig:
    """L2 文本仲裁层参数。"""

    is_enabled: bool = True
    text_source_preference: str = "auto"
    min_length_ratio: float = 0.65
    max_length_ratio: float = 3.0
    low_confidence_threshold: float = 0.5
    is_hallucination_block: bool = True

    @classmethod
    def from_runtime(cls, raw: Optional[Dict[str, Any]]) -> "ArbitrationConfig":
        raw = raw or {}
        preference = str(
            _read_runtime_value(raw, "text_source_preference", cls.text_source_preference)
            or cls.text_source_preference
        ).lower()
        if preference not in {"fast", "slow", "auto"}:
            preference = cls.text_source_preference
        return cls(
            is_enabled=bool(_read_runtime_value(raw, "enable", cls.is_enabled)),
            text_source_preference=preference,
            min_length_ratio=float(
                _read_runtime_value(raw, "min_length_ratio", cls.min_length_ratio)
            ),
            max_length_ratio=float(
                _read_runtime_value(raw, "max_length_ratio", cls.max_length_ratio)
            ),
            low_confidence_threshold=float(
                _read_runtime_value(raw, "low_confidence_threshold", cls.low_confidence_threshold)
            ),
            is_hallucination_block=bool(
                _read_runtime_value(raw, "hallucination_block", cls.is_hallucination_block)
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable": self.is_enabled,
            "text_source_preference": self.text_source_preference,
            "min_length_ratio": self.min_length_ratio,
            "max_length_ratio": self.max_length_ratio,
            "low_confidence_threshold": self.low_confidence_threshold,
            "hallucination_block": self.is_hallucination_block,
        }


@dataclass
class PunctuationConfig:
    """L3 标点层参数。"""

    is_enabled: bool = True
    source_preference: str = "merged"
    min_confidence: float = 0.35
    sentence_end_min_confidence: float = 0.55
    raw_source_min_mapping_coverage: float = 0.6
    raw_source_max_weak_ratio: float = 0.8  # 已废弃 (V3.2.0+dev.20260205.06): 保留用于向后兼容，但不再使用
    sv_fallback_mode: str = "strict"

    @classmethod
    def from_runtime(cls, raw: Optional[Dict[str, Any]]) -> "PunctuationConfig":
        raw = raw or {}
        enable = _read_runtime_value(raw, "enable", None)
        if enable is None:
            enable = _read_runtime_value(raw, "enable_punctuation", cls.is_enabled)
        preference = (
            _read_runtime_value(raw, "source_preference", None)
            or _read_runtime_value(raw, "fallback_priority", None)
            or cls.source_preference
        )
        normalized = str(preference).lower()
        if normalized not in {"fast", "slow", "merged"}:
            normalized = cls.source_preference
        min_conf = _read_runtime_value(raw, "min_confidence", None)
        if min_conf is None:
            min_conf = _read_runtime_value(raw, "confidence_threshold", cls.min_confidence)
        end_min_conf = _read_runtime_value(raw, "sentence_end_min_confidence", cls.sentence_end_min_confidence)
        raw_min_cov = _read_runtime_value(
            raw,
            "raw_source.min_mapping_coverage",
            cls.raw_source_min_mapping_coverage,
        )
        raw_max_weak = _read_runtime_value(
            raw,
            "raw_source.max_weak_ratio",
            cls.raw_source_max_weak_ratio,
        )
        sv_fallback_mode = str(
            _read_runtime_value(raw, "sv_fallback_mode", cls.sv_fallback_mode)
        ).lower()
        if sv_fallback_mode not in {"strict", "tolerant"}:
            sv_fallback_mode = cls.sv_fallback_mode
        return cls(
            is_enabled=bool(enable),
            source_preference=normalized,
            min_confidence=float(min_conf),
            sentence_end_min_confidence=float(end_min_conf),
            raw_source_min_mapping_coverage=float(raw_min_cov),
            raw_source_max_weak_ratio=float(raw_max_weak),
            sv_fallback_mode=sv_fallback_mode,
        )

