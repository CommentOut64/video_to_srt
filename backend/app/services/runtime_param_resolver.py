"""
运行参数解析器

用于将统一运行参数落地到具体模型/组件的配置对象。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.core.spectrum_thresholds import SpectrumThresholds
from app.services.audio.vad_service import VADConfig, VADMethod
from app.services.model_manager_v2 import get_model_manager_v2
from app.services.model_runtime_config_service import get_model_runtime_config_service

logger = logging.getLogger(__name__)


def _merge_params(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """合并运行参数（overrides 优先，忽略 None）。"""
    merged = dict(base)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                merged[key] = value
    return merged


def _guess_runtime_group(model_id: str) -> Optional[str]:
    """根据模型ID推断运行参数分组。"""
    if model_id.startswith("whisper-"):
        return "whisper"
    if model_id.startswith("sensevoice"):
        return "sensevoice"
    if model_id.startswith("demucs-"):
        return "demucs"
    if "vad" in model_id:
        return "vad"
    if model_id.startswith("punct-"):
        return "punctuation"
    return None


def get_runtime_group(group: str) -> Dict[str, Any]:
    """获取指定运行参数分组（全局生效）。"""
    service = get_model_runtime_config_service()
    runtime = service.get_effective_runtime_global()
    effective = runtime.get("effective", {})
    return dict(effective.get(group, {}))


def get_runtime_group_for_model(model_id: str) -> Dict[str, Any]:
    """获取指定模型的运行参数（含 per_model 覆盖）。"""
    service = get_model_runtime_config_service()
    try:
        spec = get_model_manager_v2().registry.get(model_id)
        runtime = service.get_effective_runtime_for_model(spec)
        return dict(runtime.get("effective", {}))
    except Exception as exc:
        group = _guess_runtime_group(model_id)
        if group:
            logger.debug("运行参数回退到分组: model_id=%s group=%s reason=%s", model_id, group, exc)
            return get_runtime_group(group)
        logger.debug("运行参数分组无法推断: model_id=%s reason=%s", model_id, exc)
        return {}


def get_effective_model_config(model_id: str) -> Dict[str, Any]:
    """获取模型管理参数（device/compute_type等）。"""
    service = get_model_runtime_config_service()
    spec = get_model_manager_v2().registry.get(model_id)
    effective = service.get_effective_model(spec).get("effective", {})
    return dict(effective)


def build_vad_config(overrides: Optional[Dict[str, Any]] = None) -> VADConfig:
    """根据运行参数构建 VADConfig。"""
    runtime = get_runtime_group("vad")
    merged = _merge_params(runtime, overrides)
    base = VADConfig()

    method_raw = merged.get("method", base.method.value if isinstance(base.method, VADMethod) else "silero")
    try:
        method = VADMethod(method_raw)
    except ValueError:
        logger.warning("无效VAD方法，回退默认值: %s", method_raw)
        method = VADMethod.SILERO

    return VADConfig(
        method=method,
        hf_token=merged.get("hf_token", base.hf_token),
        onset=merged.get("onset", base.onset),
        offset=merged.get("offset", base.offset),
        chunk_size=merged.get("chunk_size", base.chunk_size),
        min_speech_duration_ms=merged.get("min_speech_duration_ms", base.min_speech_duration_ms),
        min_silence_duration_ms=merged.get("min_silence_duration_ms", base.min_silence_duration_ms),
        speech_pad_ms=merged.get("speech_pad_ms", base.speech_pad_ms),
        merge_max_gap=merged.get("merge_max_gap", base.merge_max_gap),
        merge_max_duration=merged.get("merge_max_duration", base.merge_max_duration),
        merge_min_fragment=merged.get("merge_min_fragment", base.merge_min_fragment),
        smart_target_duration=merged.get("smart_target_duration", base.smart_target_duration),
        smart_max_duration=merged.get("smart_max_duration", base.smart_max_duration),
        smart_min_gap_to_split=merged.get("smart_min_gap_to_split", base.smart_min_gap_to_split),
    )


def build_vad_config_for_profile(
    profile: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> VADConfig:
    """根据运行参数构建指定预设的 VADConfig。"""
    config = build_vad_config()
    if profile == "sensevoice":
        runtime = get_runtime_group("vad")
        config.merge_max_gap = runtime.get("sensevoice_merge_max_gap", config.merge_max_gap)
        config.merge_max_duration = runtime.get("sensevoice_merge_max_duration", config.merge_max_duration)
        config.smart_target_duration = runtime.get(
            "sensevoice_smart_target_duration",
            config.smart_target_duration,
        )
    elif profile != "whisper":
        logger.debug("未知 VAD profile，回退默认配置: %s", profile)

    if overrides:
        for key, value in overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)

    return config


def build_spectrum_thresholds(overrides: Optional[Dict[str, Any]] = None) -> SpectrumThresholds:
    """根据配置构建频谱阈值。"""
    base = SpectrumThresholds()
    merged = _merge_params(dict(base.__dict__), overrides)
    return SpectrumThresholds(**merged)


def get_demucs_runtime_params() -> Dict[str, Any]:
    """获取 Demucs 运行参数。"""
    return get_runtime_group("demucs")


def get_smart_probe_runtime_params() -> Dict[str, Any]:
    """获取 SmartProbe 运行参数。"""
    return get_runtime_group("smart_probe")


def get_yamnet_runtime_params() -> Dict[str, Any]:
    """获取 YAMNet 运行参数。"""
    return get_runtime_group("yamnet")
