"""
文本处理统一参数入口（TextPipelineConfig）。
V3.2.0+dev.20260204.04
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

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
class TextPipelineConfig:
    """文本处理流水线参数入口（L1/L2）。"""

    normalization: NormalizationConfig
    arbitration: "ArbitrationConfig"

    @classmethod
    def from_runtime(cls, runtime: Optional[Dict[str, Any]] = None) -> "TextPipelineConfig":
        if runtime is None:
            runtime = get_model_runtime_config_service().get_effective_runtime_global()
        effective = runtime.get("effective", {}) if isinstance(runtime, dict) else {}
        normalization_raw = effective.get("normalization", {}) if isinstance(effective, dict) else {}
        arbitration_raw = effective.get("arbitration", {}) if isinstance(effective, dict) else {}
        return cls(
            normalization=NormalizationConfig.from_runtime(normalization_raw),
            arbitration=ArbitrationConfig.from_runtime(arbitration_raw),
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
