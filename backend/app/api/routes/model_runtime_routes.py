"""
模型运行参数 API 路由
为前端提供读写运行参数能力。
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from app.services.model_runtime_config_service import get_model_runtime_config_service
from app.services.model_manager_v2 import get_model_manager_v2

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])

_DEVICE_OPTIONS = {"auto", "cuda", "cpu"}
_COMPUTE_TYPES = {"auto", "int8", "int8_float16", "float16", "float32", "fp32"}
_SENSEVOICE_LANGUAGES = {"auto", "zh", "en", "yue", "ja", "ko", "nospeech"}
_PUNCT_ALIGNMENT_METHODS = {"anchor", "dtw", "levenshtein"}
_DEMUCS_MODELS = {"htdemucs", "htdemucs_ft", "mdx_extra", "mdx_extra_q"}


class _RuntimeBase(BaseModel):
    """运行参数基础模型（启用别名与字段校验）。"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True, extra="forbid")


class WhisperRuntimeParams(_RuntimeBase):
    language: Optional[str] = Field(default=None)
    initial_prompt: Optional[str] = Field(default=None)
    is_word_timestamps: Optional[bool] = Field(default=None, alias="word_timestamps")
    beam_size: Optional[int] = Field(default=None, ge=1)
    is_vad_filter: Optional[bool] = Field(default=None, alias="vad_filter")
    vad_parameters: Optional[Dict[str, float]] = Field(default=None)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    is_condition_on_previous_text: Optional[bool] = Field(default=None, alias="condition_on_previous_text")
    suppress_tokens: Optional[List[int]] = Field(default=None)
    repetition_penalty: Optional[float] = Field(default=None, ge=0.0)
    no_repeat_ngram_size: Optional[int] = Field(default=None, ge=0)

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if value == "auto":
            return value
        if 2 <= len(value) <= 10 and value.replace("-", "").isalnum():
            return value.lower()
        raise ValueError("language 必须是 auto 或 2-10 位语言码")

    @field_validator("vad_parameters")
    @classmethod
    def _validate_vad_parameters(cls, value: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if value is None:
            return value
        for key, item in value.items():
            if not isinstance(item, (int, float)):
                raise ValueError(f"vad_parameters.{key} 必须为数值")
            if item < 0:
                raise ValueError(f"vad_parameters.{key} 必须为非负数")
        return value

    @field_validator("suppress_tokens")
    @classmethod
    def _validate_suppress_tokens(cls, value: Optional[List[int]]) -> Optional[List[int]]:
        if value is None:
            return value
        for item in value:
            if not isinstance(item, int) or item < 0:
                raise ValueError("suppress_tokens 仅允许非负整数")
        return value


class SenseVoiceRuntimeParams(_RuntimeBase):
    language: Optional[str] = Field(default=None)
    is_use_itn: Optional[bool] = Field(default=None, alias="use_itn")
    is_ban_emo_unk: Optional[bool] = Field(default=None, alias="ban_emo_unk")
    model_type: Optional[str] = Field(default=None)
    device: Optional[str] = Field(default=None)
    batch_size: Optional[int] = Field(default=None, ge=1)
    is_quantize: Optional[bool] = Field(default=None, alias="quantize")

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value_lower = value.lower()
        if value_lower not in _SENSEVOICE_LANGUAGES:
            raise ValueError("SenseVoice language 仅支持 auto/zh/en/yue/ja/ko/nospeech")
        return value_lower

    @field_validator("model_type")
    @classmethod
    def _validate_model_type(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value_lower = value.lower()
        if value_lower not in {"quantized", "fp32"}:
            raise ValueError("model_type 仅支持 quantized 或 fp32")
        return value_lower

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value_lower = value.lower()
        if value_lower not in _DEVICE_OPTIONS:
            raise ValueError("device 仅支持 auto/cuda/cpu")
        return value_lower


class DemucsRuntimeParams(_RuntimeBase):
    model_name: Optional[str] = Field(default=None)
    device: Optional[str] = Field(default=None)
    shifts: Optional[int] = Field(default=None, ge=1, le=5)
    overlap: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    segment_length: Optional[int] = Field(default=None, ge=1)
    segment_buffer_sec: Optional[float] = Field(default=None, ge=0.0)
    bgm_sample_duration: Optional[float] = Field(default=None, ge=1.0)
    bgm_light_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    bgm_heavy_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("model_name")
    @classmethod
    def _validate_model_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value_lower = value.lower()
        if value_lower not in _DEMUCS_MODELS:
            raise ValueError("model_name 仅支持 htdemucs/htdemucs_ft/mdx_extra/mdx_extra_q")
        return value_lower

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value_lower = value.lower()
        if value_lower not in _DEVICE_OPTIONS:
            raise ValueError("device 仅支持 auto/cuda/cpu")
        return value_lower


class VADRuntimeParams(_RuntimeBase):
    method: Optional[str] = Field(default=None)
    hf_token: Optional[str] = Field(default=None)
    onset: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    offset: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    chunk_size: Optional[int] = Field(default=None, ge=1)
    min_speech_duration_ms: Optional[int] = Field(default=None, ge=0)
    min_silence_duration_ms: Optional[int] = Field(default=None, ge=0)
    speech_pad_ms: Optional[int] = Field(default=None, ge=0)
    merge_max_gap: Optional[float] = Field(default=None, ge=0.0)
    merge_max_duration: Optional[float] = Field(default=None, ge=0.0)
    merge_min_fragment: Optional[float] = Field(default=None, ge=0.0)
    smart_target_duration: Optional[float] = Field(default=None, ge=0.0)
    smart_max_duration: Optional[float] = Field(default=None, ge=0.0)
    smart_min_gap_to_split: Optional[float] = Field(default=None, ge=0.0)
    sensevoice_merge_max_gap: Optional[float] = Field(default=None, ge=0.0)
    sensevoice_merge_max_duration: Optional[float] = Field(default=None, ge=0.0)
    sensevoice_smart_target_duration: Optional[float] = Field(default=None, ge=0.0)

    @field_validator("method")
    @classmethod
    def _validate_method(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value_lower = value.lower()
        if value_lower not in {"silero", "pyannote"}:
            raise ValueError("method 仅支持 silero 或 pyannote")
        return value_lower

    @model_validator(mode="after")
    def _validate_smart_limits(self):
        if (
            self.smart_target_duration is not None
            and self.smart_max_duration is not None
            and self.smart_max_duration < self.smart_target_duration
        ):
            raise ValueError("smart_max_duration 必须 >= smart_target_duration")
        if (
            self.sensevoice_smart_target_duration is not None
            and self.smart_max_duration is not None
            and self.smart_max_duration < self.sensevoice_smart_target_duration
        ):
            raise ValueError("smart_max_duration 必须 >= sensevoice_smart_target_duration")
        return self


class SpectrumRuntimeParams(_RuntimeBase):
    is_use_yamnet: Optional[bool] = Field(default=None, alias="use_yamnet")
    is_use_snr_strategy: Optional[bool] = Field(default=None, alias="use_snr_strategy")
    harmonic_ratio_music: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    spectral_centroid_music_low: Optional[float] = Field(default=None, ge=0.0)
    spectral_centroid_music_high: Optional[float] = Field(default=None, ge=0.0)
    energy_variance_music: Optional[float] = Field(default=None, ge=0.0)
    onset_strength_music: Optional[float] = Field(default=None, ge=0.0)
    zcr_noise_high: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    zcr_variance_noise: Optional[float] = Field(default=None, ge=0.0)
    high_freq_ratio_noise: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    spectral_flatness_noise: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    music_score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    noise_score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    clean_score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    heavy_bgm_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    light_bgm_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    snr_high_threshold: Optional[float] = Field(default=None, ge=0.0)
    snr_low_threshold: Optional[float] = Field(default=None, ge=0.0)
    c50_good_threshold: Optional[float] = Field(default=None)
    c50_bad_threshold: Optional[float] = Field(default=None)
    spectral_contrast_low: Optional[float] = Field(default=None, ge=0.0)
    spectral_contrast_critical: Optional[float] = Field(default=None, ge=0.0)
    spectral_flatness_high: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_threshold_pairs(self):
        if (
            self.spectral_centroid_music_low is not None
            and self.spectral_centroid_music_high is not None
            and self.spectral_centroid_music_high < self.spectral_centroid_music_low
        ):
            raise ValueError("spectral_centroid_music_high 必须 >= spectral_centroid_music_low")
        if (
            self.heavy_bgm_threshold is not None
            and self.light_bgm_threshold is not None
            and self.heavy_bgm_threshold < self.light_bgm_threshold
        ):
            raise ValueError("heavy_bgm_threshold 必须 >= light_bgm_threshold")
        if (
            self.snr_high_threshold is not None
            and self.snr_low_threshold is not None
            and self.snr_high_threshold < self.snr_low_threshold
        ):
            raise ValueError("snr_high_threshold 必须 >= snr_low_threshold")
        if (
            self.c50_good_threshold is not None
            and self.c50_bad_threshold is not None
            and self.c50_good_threshold < self.c50_bad_threshold
        ):
            raise ValueError("c50_good_threshold 必须 >= c50_bad_threshold")
        if (
            self.spectral_contrast_low is not None
            and self.spectral_contrast_critical is not None
            and self.spectral_contrast_low < self.spectral_contrast_critical
        ):
            raise ValueError("spectral_contrast_low 必须 >= spectral_contrast_critical")
        return self


class SmartProbeRuntimeParams(_RuntimeBase):
    snr_threshold: Optional[float] = Field(default=None, ge=0.0)


class YAMNetRuntimeParams(_RuntimeBase):
    acappella_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    music_max_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    music_avg_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    speech_max_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    speech_max_music_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    speech_dominant_delta: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    speech_dominant_music_max: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    probe_window_count: Optional[int] = Field(default=None, ge=1, le=5)
    probe_window_duration_sec: Optional[float] = Field(default=None, ge=0.5, le=2.0)


class PunctuationRuntimeParams(_RuntimeBase):
    is_enable_punctuation: Optional[bool] = Field(default=None, alias="enable_punctuation")
    default_language: Optional[str] = Field(default=None)
    is_fallback_to_multilingual: Optional[bool] = Field(default=None, alias="fallback_to_multilingual")
    is_cache_models: Optional[bool] = Field(default=None, alias="cache_models")
    max_cached_models: Optional[int] = Field(default=None, ge=1)
    device: Optional[str] = Field(default=None)
    num_threads: Optional[int] = Field(default=None, ge=1)
    is_use_int8: Optional[bool] = Field(default=None, alias="use_int8")
    batch_size: Optional[int] = Field(default=None, ge=1)
    max_sequence_length: Optional[int] = Field(default=None, ge=1)
    is_enable_arbitration: Optional[bool] = Field(default=None, alias="enable_arbitration")
    confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    vad_tolerance: Optional[float] = Field(default=None, ge=0.0)
    min_sentence_gap: Optional[float] = Field(default=None, ge=0.0)
    is_hallucination_check: Optional[bool] = Field(default=None, alias="hallucination_check")
    is_require_vad_pause: Optional[bool] = Field(default=None, alias="require_vad_pause")
    is_cross_validation: Optional[bool] = Field(default=None, alias="cross_validation")
    whisper_confidence_min: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    alignment_method: Optional[str] = Field(default=None)
    anchor_confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_local_alignment_length: Optional[int] = Field(default=None, ge=1)
    is_enable_semantic_buffer: Optional[bool] = Field(default=None, alias="enable_semantic_buffer")
    max_buffer_duration: Optional[float] = Field(default=None, ge=0.0)
    hard_limit_duration: Optional[float] = Field(default=None, ge=0.0)
    min_chunk_duration: Optional[float] = Field(default=None, ge=0.0)
    is_prefer_punctuation_split: Optional[bool] = Field(default=None, alias="prefer_punctuation_split")

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value_lower = value.lower()
        if value_lower not in {"cpu", "cuda"}:
            raise ValueError("device 仅支持 cpu 或 cuda")
        return value_lower

    @field_validator("alignment_method")
    @classmethod
    def _validate_alignment_method(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value_lower = value.lower()
        if value_lower not in _PUNCT_ALIGNMENT_METHODS:
            raise ValueError("alignment_method 仅支持 anchor/dtw/levenshtein")
        return value_lower


class PipelineRuntimeParams(_RuntimeBase):
    batch_size: Optional[int] = Field(default=None, ge=1)
    is_word_timestamps: Optional[bool] = Field(default=None, alias="word_timestamps")


class RuntimeGroupUpdateRequest(_RuntimeBase):
    whisper: Optional[WhisperRuntimeParams] = None
    sensevoice: Optional[SenseVoiceRuntimeParams] = None
    demucs: Optional[DemucsRuntimeParams] = None
    vad: Optional[VADRuntimeParams] = None
    spectrum: Optional[SpectrumRuntimeParams] = None
    smart_probe: Optional[SmartProbeRuntimeParams] = None
    yamnet: Optional[YAMNetRuntimeParams] = None
    punctuation: Optional[PunctuationRuntimeParams] = None
    pipeline: Optional[PipelineRuntimeParams] = None


_RUNTIME_GROUP_MODELS = {
    "whisper": WhisperRuntimeParams,
    "sensevoice": SenseVoiceRuntimeParams,
    "demucs": DemucsRuntimeParams,
    "vad": VADRuntimeParams,
    "spectrum": SpectrumRuntimeParams,
    "smart_probe": SmartProbeRuntimeParams,
    "yamnet": YAMNetRuntimeParams,
    "punctuation": PunctuationRuntimeParams,
    "pipeline": PipelineRuntimeParams,
}


def _validate_runtime_group_payload(group: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    model_cls = _RUNTIME_GROUP_MODELS.get(group)
    if not model_cls:
        raise ValueError(f"不支持的运行参数分组: {group}")
    model = model_cls(**payload)
    return model.model_dump(by_alias=True, exclude_unset=True)


_PARAM_SCHEMA: Dict[str, Any] = {
    "global": {
        "device_preference": {"type": "enum", "enum": ["auto", "cuda", "cpu"], "default": "auto"},
        "allow_download": {"type": "bool", "default": False},
        "max_vram_mb": {"type": "int", "min": 0, "default": 8000},
        "reserved_vram_mb": {"type": "int", "min": 0, "default": 500},
        "max_models": {"type": "int", "min": 1, "default": 3},
        "cpu_threads": {"type": "int", "min": 1, "default": 4},
        "cpu_affinity_strategy": {"type": "enum", "enum": ["auto", "half", "custom"], "default": None},
        "onnx_intra_threads": {"type": "int", "min": 1, "default": 1},
        "onnx_inter_threads": {"type": "int", "min": 1, "default": 1},
    },
    "per_model": {
        "device": {"type": "enum", "enum": ["auto", "cuda", "cpu"], "default": "auto"},
        "compute_type": {
            "type": "enum",
            "enum": ["auto", "int8", "int8_float16", "float16", "float32", "fp32"],
            "default": "auto",
        },
        "cpu_threads": {"type": "int", "min": 1, "default": None},
        "keep_resident": {"type": "bool", "default": False},
        "evict_priority": {"type": "int", "default": 0},
        "max_concurrency": {"type": "int", "min": 1, "default": 1},
    },
    "runtime": {
        "whisper": {
            "language": {"type": "string", "default": "auto", "pattern": "[a-z0-9-]{2,10}|auto"},
            "initial_prompt": {"type": "string", "default": None},
            "word_timestamps": {"type": "bool", "default": False},
            "beam_size": {"type": "int", "min": 1, "default": 5},
            "vad_filter": {"type": "bool", "default": True},
            "vad_parameters": {"type": "object", "default": None},
            "temperature": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
            "condition_on_previous_text": {"type": "bool", "default": True},
            "suppress_tokens": {"type": "int_list", "min": 0, "default": None},
            "repetition_penalty": {"type": "float", "min": 0.0, "default": 1.0},
            "no_repeat_ngram_size": {"type": "int", "min": 0, "default": 0},
        },
        "sensevoice": {
            "language": {
                "type": "enum",
                "enum": ["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],
                "default": "auto",
            },
            "use_itn": {"type": "bool", "default": True},
            "ban_emo_unk": {"type": "bool", "default": False},
            "model_type": {"type": "enum", "enum": ["quantized", "fp32"], "default": "quantized"},
            "device": {"type": "enum", "enum": ["auto", "cuda", "cpu"], "default": "cpu"},
            "batch_size": {"type": "int", "min": 1, "default": 1},
            "quantize": {"type": "bool", "default": True},
        },
        "demucs": {
            "model_name": {
                "type": "enum",
                "enum": ["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_extra_q"],
                "default": "htdemucs",
            },
            "device": {"type": "enum", "enum": ["auto", "cuda", "cpu"], "default": "cuda"},
            "shifts": {"type": "int", "min": 1, "max": 5, "default": 1},
            "overlap": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "segment_length": {"type": "int", "min": 1, "default": 10},
            "segment_buffer_sec": {"type": "float", "min": 0.0, "default": 2.0},
            "bgm_sample_duration": {"type": "float", "min": 1.0, "default": 10.0},
            "bgm_light_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.02},
            "bgm_heavy_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.15},
        },
        "vad": {
            "method": {"type": "enum", "enum": ["silero", "pyannote"], "default": "silero"},
            "hf_token": {"type": "string", "default": None},
            "onset": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.4},
            "offset": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.4},
            "chunk_size": {"type": "int", "min": 1, "default": 30},
            "min_speech_duration_ms": {"type": "int", "min": 0, "default": 250},
            "min_silence_duration_ms": {"type": "int", "min": 0, "default": 400},
            "speech_pad_ms": {"type": "int", "min": 0, "default": 300},
            "merge_max_gap": {"type": "float", "min": 0.0, "default": 1.0},
            "merge_max_duration": {"type": "float", "min": 0.0, "default": 12.0},
            "merge_min_fragment": {"type": "float", "min": 0.0, "default": 1.0},
            "smart_target_duration": {"type": "float", "min": 0.0, "default": 12.0},
            "smart_max_duration": {"type": "float", "min": 0.0, "default": 30.0},
            "smart_min_gap_to_split": {"type": "float", "min": 0.0, "default": 0.3},
            "sensevoice_merge_max_gap": {"type": "float", "min": 0.0, "default": 0.3},
            "sensevoice_merge_max_duration": {"type": "float", "min": 0.0, "default": 8.0},
            "sensevoice_smart_target_duration": {"type": "float", "min": 0.0, "default": 8.0},
        },
        "spectrum": {
            "use_yamnet": {"type": "bool", "default": True},
            "use_snr_strategy": {"type": "bool", "default": True},
            "harmonic_ratio_music": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.6},
            "spectral_centroid_music_low": {"type": "float", "min": 0.0, "default": 1500},
            "spectral_centroid_music_high": {"type": "float", "min": 0.0, "default": 4000},
            "energy_variance_music": {"type": "float", "min": 0.0, "default": 0.25},
            "onset_strength_music": {"type": "float", "min": 0.0, "default": 0.3},
            "zcr_noise_high": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.15},
            "zcr_variance_noise": {"type": "float", "min": 0.0, "default": 0.02},
            "high_freq_ratio_noise": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.4},
            "spectral_flatness_noise": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "music_score_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.35},
            "noise_score_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.45},
            "clean_score_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.7},
            "heavy_bgm_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.6},
            "light_bgm_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.35},
            "snr_high_threshold": {"type": "float", "min": 0.0, "default": 40.0},
            "snr_low_threshold": {"type": "float", "min": 0.0, "default": 25.0},
            "c50_good_threshold": {"type": "float", "default": 13.70},
            "c50_bad_threshold": {"type": "float", "default": -12.33},
            "spectral_contrast_low": {"type": "float", "min": 0.0, "default": 19.84},
            "spectral_contrast_critical": {"type": "float", "min": 0.0, "default": 13.59},
            "spectral_flatness_high": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.29},
        },
        "smart_probe": {
            "snr_threshold": {"type": "float", "min": 0.0, "default": 15.0},
        },
        "yamnet": {
            "acappella_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.3},
            "music_max_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.15},
            "music_avg_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.10},
            "speech_max_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.8},
            "speech_max_music_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.1},
            "speech_dominant_delta": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.3},
            "speech_dominant_music_max": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.15},
            "probe_window_count": {"type": "int", "min": 1, "max": 5, "default": 3},
            "probe_window_duration_sec": {"type": "float", "min": 0.5, "max": 2.0, "default": 0.975},
        },
        "punctuation": {
            "enable_punctuation": {"type": "bool", "default": True},
            "default_language": {"type": "string", "default": "zh"},
            "fallback_to_multilingual": {"type": "bool", "default": True},
            "cache_models": {"type": "bool", "default": True},
            "max_cached_models": {"type": "int", "min": 1, "default": 3},
            "device": {"type": "enum", "enum": ["cpu", "cuda"], "default": "cpu"},
            "num_threads": {"type": "int", "min": 1, "default": 4},
            "use_int8": {"type": "bool", "default": True},
            "batch_size": {"type": "int", "min": 1, "default": 1},
            "max_sequence_length": {"type": "int", "min": 1, "default": 512},
            "enable_arbitration": {"type": "bool", "default": True},
            "confidence_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.6},
            "vad_tolerance": {"type": "float", "min": 0.0, "default": 0.3},
            "min_sentence_gap": {"type": "float", "min": 0.0, "default": 1.5},
            "hallucination_check": {"type": "bool", "default": True},
            "require_vad_pause": {"type": "bool", "default": True},
            "cross_validation": {"type": "bool", "default": True},
            "whisper_confidence_min": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.7},
            "alignment_method": {"type": "enum", "enum": ["anchor", "dtw", "levenshtein"], "default": "anchor"},
            "anchor_confidence_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.9},
            "max_local_alignment_length": {"type": "int", "min": 1, "default": 50},
            "enable_semantic_buffer": {"type": "bool", "default": True},
            "max_buffer_duration": {"type": "float", "min": 0.0, "default": 15.0},
            "hard_limit_duration": {"type": "float", "min": 0.0, "default": 20.0},
            "min_chunk_duration": {"type": "float", "min": 0.0, "default": 1.0},
            "prefer_punctuation_split": {"type": "bool", "default": True},
        },
        "pipeline": {
            "batch_size": {"type": "int", "min": 1, "default": 16},
            "word_timestamps": {"type": "bool", "default": False},
        },
    },
}


class GlobalRuntimeUpdateRequest(BaseModel):
    """全局运行参数更新请求。"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    device_preference: Optional[str] = Field(default=None, description="auto/cuda/cpu")
    is_allow_download: Optional[bool] = Field(default=None, alias="allow_download")
    max_vram_mb: Optional[int] = Field(default=None, ge=0)
    reserved_vram_mb: Optional[int] = Field(default=None, ge=0)
    max_models: Optional[int] = Field(default=None, ge=1)
    cpu_threads: Optional[int] = Field(default=None, ge=1)
    cpu_affinity_strategy: Optional[str] = Field(default=None)
    onnx_intra_threads: Optional[int] = Field(default=None, ge=1)
    onnx_inter_threads: Optional[int] = Field(default=None, ge=1)
    runtime: Optional[RuntimeGroupUpdateRequest] = None

    @field_validator("device_preference")
    @classmethod
    def _validate_device_preference(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value_lower = value.lower()
        if value_lower not in _DEVICE_OPTIONS:
            raise ValueError("device_preference 仅支持 auto/cuda/cpu")
        return value_lower

    @field_validator("cpu_affinity_strategy")
    @classmethod
    def _validate_cpu_affinity_strategy(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value_lower = value.lower()
        if value_lower not in {"auto", "half", "custom"}:
            raise ValueError("cpu_affinity_strategy 仅支持 auto/half/custom")
        return value_lower


class ModelRuntimeUpdateRequest(BaseModel):
    """单模型运行参数更新请求。"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    device: Optional[str] = Field(default=None, description="auto/cuda/cpu")
    compute_type: Optional[str] = Field(default=None)
    cpu_threads: Optional[int] = Field(default=None, ge=1)
    is_keep_resident: Optional[bool] = Field(default=None, alias="keep_resident")
    evict_priority: Optional[int] = Field(default=None)
    max_concurrency: Optional[int] = Field(default=None, ge=1)
    runtime: Optional[Dict[str, Any]] = None

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value_lower = value.lower()
        if value_lower not in _DEVICE_OPTIONS:
            raise ValueError("device 仅支持 auto/cuda/cpu")
        return value_lower

    @field_validator("compute_type")
    @classmethod
    def _validate_compute_type(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value_lower = value.lower()
        if value_lower not in _COMPUTE_TYPES:
            raise ValueError("compute_type 仅支持 auto/int8/int8_float16/float16/float32/fp32")
        return value_lower


@router.get("/runtime")
async def get_runtime_config() -> Dict[str, Any]:
    """获取所有模型的运行参数（含全局配置与单模型覆盖）。"""
    service = get_model_runtime_config_service()
    manager = get_model_manager_v2()
    models = {}
    for spec in manager.registry.list():
        models[spec.id] = service.get_effective_model(spec)

    return {
        "success": True,
        "global": service.get_effective_global(),
        "runtime": service.get_effective_runtime_global(),
        "models": models,
    }


@router.get("/runtime/{model_id}")
async def get_runtime_config_for_model(model_id: str) -> Dict[str, Any]:
    """获取单模型运行参数。"""
    service = get_model_runtime_config_service()
    manager = get_model_manager_v2()
    try:
        spec = manager.registry.get(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "success": True,
        "model_id": model_id,
        "config": service.get_effective_model(spec),
    }


@router.put("/runtime")
async def update_runtime_global(req: GlobalRuntimeUpdateRequest) -> Dict[str, Any]:
    """更新全局运行参数（部分更新）。"""
    service = get_model_runtime_config_service()
    updates = req.model_dump(by_alias=True, exclude_unset=True)
    runtime_updates = updates.pop("runtime", None)
    updated = service.update_global(updates)
    if runtime_updates:
        service.update_runtime_global(runtime_updates)
    return {
        "success": True,
        "global": updated.to_dict(),
        "runtime": service.get_effective_runtime_global(),
    }


@router.put("/runtime/{model_id}")
async def update_runtime_model(model_id: str, req: ModelRuntimeUpdateRequest) -> Dict[str, Any]:
    """更新单模型运行参数（部分更新）。"""
    service = get_model_runtime_config_service()
    manager = get_model_manager_v2()
    try:
        manager.registry.get(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    updates = req.model_dump(by_alias=True, exclude_unset=True)
    runtime_payload = updates.pop("runtime", None)
    updated = service.update_model(model_id, updates)

    runtime_override: Dict[str, Any] = {}
    if runtime_payload is not None:
        spec = manager.registry.get(model_id)
        runtime_group = service.get_effective_runtime_for_model(spec).get("group")
        if not runtime_group:
            raise HTTPException(status_code=400, detail="该模型不支持独立运行参数覆盖")
        try:
            validated = _validate_runtime_group_payload(runtime_group, runtime_payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        runtime_override = service.update_model_runtime(model_id, validated)
    return {
        "success": True,
        "model_id": model_id,
        "override": updated.to_dict(),
        "runtime_override": runtime_override,
        "runtime": service.get_effective_runtime_for_model(manager.registry.get(model_id)),
    }


@router.post("/runtime/apply")
async def apply_runtime_config() -> Dict[str, Any]:
    """
    应用运行参数配置（更新模型管理器预算与下载策略）。

    说明：
    - 当前只更新 ModelManagerV2 的预算与下载策略；
    - 已加载模型是否重载由前端决定（可在此接口后显式触发卸载/重载）。
    """
    service = get_model_runtime_config_service()
    manager = get_model_manager_v2()
    global_effective = service.get_effective_global()
    effective = global_effective.get("effective", {})

    if "max_models" in effective and effective["max_models"]:
        manager.budget.max_models = int(effective["max_models"])
        manager.cache.capacity = manager.budget.max_models
    if "max_vram_mb" in effective and effective["max_vram_mb"]:
        manager.budget.max_vram_mb = int(effective["max_vram_mb"])
    if "reserved_vram_mb" in effective and effective["reserved_vram_mb"] is not None:
        manager.budget.reserved_vram_mb = int(effective["reserved_vram_mb"])
    if "allow_download" in effective and effective["allow_download"] is not None:
        manager.downloader.allow_download = bool(effective["allow_download"])

    return {
        "success": True,
        "applied": True,
        "global": global_effective,
    }


@router.get("/params")
async def get_runtime_params_schema() -> Dict[str, Any]:
    """返回运行参数字段与范围定义，供前端生成表单。"""
    return {
        "success": True,
        "global": _PARAM_SCHEMA["global"],
        "per_model": _PARAM_SCHEMA["per_model"],
        "runtime": _PARAM_SCHEMA["runtime"],
    }
