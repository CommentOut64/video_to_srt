"""
模型运行参数 API 路由
为前端提供读写运行参数能力。
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List, Literal

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


class SanitizePattern(BaseModel):
    """Whisper 清洗规则模式。"""

    type: Literal["phrase", "regex"] = Field(
        default="phrase",
        description="匹配类型：phrase=短语匹配，regex=正则表达式",
    )
    value: str = Field(..., min_length=1, description="匹配内容")
    position: Literal["prefix", "suffix", "any"] = Field(
        default="any",
        description="匹配位置：prefix=前缀，suffix=后缀，any=任意位置",
    )
    flags: Optional[str] = Field(
        default="",
        description="正则标志（如 i 表示忽略大小写）",
    )


class WhisperSanitizeRuntimeParams(_RuntimeBase):
    is_enabled: Optional[bool] = Field(
        default=None,
        alias="enabled",
        description="是否启用 Whisper 清洗",
    )
    min_text_length: Optional[int] = Field(
        default=None,
        ge=0,
        description="最小有效文本长度",
    )
    patterns: Optional[List[SanitizePattern]] = Field(
        default=None,
        description="清洗规则列表",
    )


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
    global_group_duration_sec: Optional[float] = Field(default=None, ge=60.0)
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


class TimelineRuntimeParams(_RuntimeBase):
    is_enabled: Optional[bool] = Field(default=None, alias="enabled")
    device: Optional[str] = Field(default=None)
    is_diarization_enabled: Optional[bool] = Field(default=None, alias="diarization_enabled")
    diarization_model_id: Optional[str] = Field(default=None)
    diarization_local_path: Optional[str] = Field(default=None)
    diarization_hf_token: Optional[str] = Field(default=None)
    diarization_max_speakers: Optional[int] = Field(default=None, ge=1)
    diarization_min_speakers: Optional[int] = Field(default=None, ge=1)
    diarization_num_speakers: Optional[int] = Field(default=None, ge=1)
    segmentation_boundary_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    segmentation_min_boundary_interval_sec: Optional[float] = Field(default=None, ge=0.0)
    min_support_turns: Optional[int] = Field(default=None, ge=1)
    min_total_duration: Optional[float] = Field(default=None, ge=0.0)
    merge_similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    long_pause_cut_sec: Optional[float] = Field(default=None, ge=0.0)

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value_lower = value.lower()
        if value_lower not in _DEVICE_OPTIONS:
            raise ValueError("device 仅支持 auto/cuda/cpu")
        return value_lower

    @model_validator(mode="after")
    def _validate_speaker_constraints(self):
        if self.diarization_num_speakers is not None:
            if (
                self.diarization_min_speakers is not None
                and self.diarization_num_speakers < self.diarization_min_speakers
            ):
                raise ValueError("diarization_num_speakers 必须 >= diarization_min_speakers")
            if (
                self.diarization_max_speakers is not None
                and self.diarization_num_speakers > self.diarization_max_speakers
            ):
                raise ValueError("diarization_num_speakers 必须 <= diarization_max_speakers")
        if (
            self.diarization_min_speakers is not None
            and self.diarization_max_speakers is not None
            and self.diarization_min_speakers > self.diarization_max_speakers
        ):
            raise ValueError("diarization_min_speakers 必须 <= diarization_max_speakers")
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
    fallback_priority: Optional[str] = Field(default=None)
    source_preference: Optional[str] = Field(default=None)
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
    min_subtitle_chars_zh_ja: Optional[int] = Field(default=None, ge=1)
    min_subtitle_words_en: Optional[int] = Field(default=None, ge=1)
    min_subtitle_duration_sec: Optional[float] = Field(default=None, ge=0.0)
    fast_delay_budget_sec: Optional[float] = Field(default=None, ge=0.0)
    is_force_split_on_sentence_end_punct: Optional[bool] = Field(
        default=None,
        alias="force_split_on_sentence_end_punct",
    )

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value_lower = value.lower()
        if value_lower not in {"cpu", "cuda"}:
            raise ValueError("device 仅支持 cpu 或 cuda")
        return value_lower

    @field_validator("fallback_priority")
    @classmethod
    def _validate_fallback_priority(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value_lower = value.lower()
        if value_lower not in {"fast", "slow"}:
            raise ValueError("fallback_priority 仅支持 fast 或 slow")
        return value_lower

    @field_validator("source_preference")
    @classmethod
    def _validate_source_preference(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value_lower = value.lower()
        if value_lower not in {"fast", "slow", "merged"}:
            raise ValueError("source_preference 仅支持 fast/slow/merged")
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


class NormalizationRuntimeParams(_RuntimeBase):
    is_itn_enable: Optional[bool] = Field(default=None, alias="itn.enable")
    itn_quality_min_ratio: Optional[float] = Field(default=None, ge=0.0, alias="itn.quality_min_ratio")
    itn_quality_max_ratio: Optional[float] = Field(default=None, ge=0.0, alias="itn.quality_max_ratio")
    is_decimal_protection_enable: Optional[bool] = Field(default=None, alias="decimal_protection.enable")
    is_safe_punct_enable: Optional[bool] = Field(default=None, alias="safe_punct.enable")
    is_abbrev_dot_protection_enable: Optional[bool] = Field(default=None, alias="abbrev_dot_protection.enable")
    is_hyphen_protection_enable: Optional[bool] = Field(default=None, alias="hyphen_protection.enable")
    is_cjk_single_digit_to_zh: Optional[bool] = Field(default=None, alias="cjk_single_digit_to_zh")
    is_cjk_digit_merge: Optional[bool] = Field(default=None, alias="cjk_digit_merge")
    is_collapse_spaces: Optional[bool] = Field(default=None, alias="collapse_spaces")
    punct_width: Optional[str] = Field(default=None, alias="punct_width")

    @field_validator("punct_width")
    @classmethod
    def _validate_punct_width(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value_lower = value.lower()
        if value_lower not in {"auto", "full", "half"}:
            raise ValueError("punct_width 仅支持 auto/full/half")
        return value_lower


class ArbitrationRuntimeParams(_RuntimeBase):
    is_enable: Optional[bool] = Field(default=None, alias="enable")
    text_source_preference: Optional[str] = Field(default=None)
    min_length_ratio: Optional[float] = Field(default=None, ge=0.0)
    max_length_ratio: Optional[float] = Field(default=None, ge=0.0)
    low_confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    is_hallucination_block: Optional[bool] = Field(default=None, alias="hallucination_block")

    @field_validator("text_source_preference")
    @classmethod
    def _validate_text_source_preference(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value_lower = value.lower()
        if value_lower not in {"fast", "slow", "auto"}:
            raise ValueError("text_source_preference 仅支持 fast/slow/auto")
        return value_lower


class SegmentationRuntimeParams(_RuntimeBase):
    is_enable: Optional[bool] = Field(default=None, alias="enable")
    min_chars: Optional[int] = Field(default=None, ge=1)
    min_duration_sec: Optional[float] = Field(default=None, ge=0.0)
    max_duration_sec: Optional[float] = Field(default=None, ge=0.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    long_pause_sec: Optional[float] = Field(default=None, ge=0.0)
    soft_pause_sec: Optional[float] = Field(default=None, ge=0.0)
    short_merge_max_chars: Optional[int] = Field(default=None, ge=1)
    is_keep_sentence_end_punct: Optional[bool] = Field(default=None, alias="keep_sentence_end_punct")
    is_enable_soft_cut: Optional[bool] = Field(default=None, alias="soft_cut.enable")
    is_enable_soft_cut_overlap_degrade: Optional[bool] = Field(
        default=None,
        alias="soft_cut.overlap_degrade_enable",
    )
    soft_cut_plan_provider: Optional[str] = Field(default=None, alias="soft_cut.plan_provider")
    soft_cut_plan_provider_class: Optional[str] = Field(default=None, alias="soft_cut.plan_provider_class")
    soft_cut_priority_active_profile: Optional[str] = Field(
        default=None,
        alias="soft_cut.priority.active_profile",
    )
    soft_cut_priority_profiles: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="soft_cut.priority.profiles",
    )

    @model_validator(mode="after")
    def _validate_duration_limits(self):
        if (
            self.max_duration_sec is not None
            and self.min_duration_sec is not None
            and self.max_duration_sec < self.min_duration_sec
        ):
            raise ValueError("max_duration_sec 必须 >= min_duration_sec")
        return self

    @field_validator("soft_cut_priority_active_profile")
    @classmethod
    def _validate_soft_cut_priority_active_profile(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        normalized = value.strip()
        if not normalized:
            raise ValueError("soft_cut.priority.active_profile 不能为空字符串")
        return normalized

    @field_validator("soft_cut_priority_profiles")
    @classmethod
    def _validate_soft_cut_priority_profiles(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if value is None:
            return value
        if not isinstance(value, dict):
            raise ValueError("soft_cut.priority.profiles 必须为对象")
        return value


class RuntimeGroupUpdateRequest(_RuntimeBase):
    whisper: Optional[WhisperRuntimeParams] = None
    whisper_sanitize: Optional[WhisperSanitizeRuntimeParams] = None
    sensevoice: Optional[SenseVoiceRuntimeParams] = None
    demucs: Optional[DemucsRuntimeParams] = None
    vad: Optional[VADRuntimeParams] = None
    timeline: Optional[TimelineRuntimeParams] = None
    smart_probe: Optional[SmartProbeRuntimeParams] = None
    yamnet: Optional[YAMNetRuntimeParams] = None
    punctuation: Optional[PunctuationRuntimeParams] = None
    pipeline: Optional[PipelineRuntimeParams] = None
    normalization: Optional[NormalizationRuntimeParams] = None
    arbitration: Optional[ArbitrationRuntimeParams] = None
    segmentation: Optional[SegmentationRuntimeParams] = None


_RUNTIME_GROUP_MODELS = {
    "whisper": WhisperRuntimeParams,
    "whisper_sanitize": WhisperSanitizeRuntimeParams,
    "sensevoice": SenseVoiceRuntimeParams,
    "demucs": DemucsRuntimeParams,
    "vad": VADRuntimeParams,
    "timeline": TimelineRuntimeParams,
    "smart_probe": SmartProbeRuntimeParams,
    "yamnet": YAMNetRuntimeParams,
    "punctuation": PunctuationRuntimeParams,
    "pipeline": PipelineRuntimeParams,
    "normalization": NormalizationRuntimeParams,
    "arbitration": ArbitrationRuntimeParams,
    "segmentation": SegmentationRuntimeParams,
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
    "resident": {
        "models": {"type": "string_list", "default": []},
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
        "whisper_sanitize": {
            "enabled": {"type": "bool", "default": True},
            "min_text_length": {"type": "int", "min": 0, "default": 2},
            "patterns": {"type": "object", "default": []},
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
            "global_group_duration_sec": {"type": "float", "min": 60.0, "default": 1800.0},
            "segment_buffer_sec": {"type": "float", "min": 0.0, "default": 2.0},
            "bgm_sample_duration": {"type": "float", "min": 1.0, "default": 10.0},
            "bgm_light_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.02},
            "bgm_heavy_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.15},
        },
        "timeline": {
            "enabled": {"type": "bool", "default": True},
            "device": {"type": "enum", "enum": ["auto", "cuda", "cpu"], "default": "cuda"},
            "diarization_enabled": {"type": "bool", "default": False},
            "diarization_model_id": {"type": "string", "default": "pyannote-speaker-diarization-community-1"},
            "diarization_local_path": {"type": "string", "default": ""},
            "diarization_hf_token": {"type": "string", "default": None},
            "diarization_max_speakers": {"type": "int", "min": 1, "default": None},
            "diarization_min_speakers": {"type": "int", "min": 1, "default": None},
            "diarization_num_speakers": {"type": "int", "min": 1, "default": 3},
            "segmentation_boundary_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.55},
            "segmentation_min_boundary_interval_sec": {"type": "float", "min": 0.0, "default": 0.2},
            "min_support_turns": {"type": "int", "min": 1, "default": 2},
            "min_total_duration": {"type": "float", "min": 0.0, "default": 2.0},
            "merge_similarity_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.88},
            "long_pause_cut_sec": {"type": "float", "min": 0.0, "default": 1.8},
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
            "fallback_priority": {"type": "enum", "enum": ["fast", "slow"], "default": "fast"},
            "source_preference": {
                "type": "enum",
                "enum": ["fast", "slow", "merged"],
                "default": "merged",
            },
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
            "min_subtitle_chars_zh_ja": {"type": "int", "min": 1, "default": 4},
            "min_subtitle_words_en": {"type": "int", "min": 1, "default": 4},
            "min_subtitle_duration_sec": {"type": "float", "min": 0.0, "default": 0.8},
            "fast_delay_budget_sec": {"type": "float", "min": 0.0, "default": 2.0},
            "force_split_on_sentence_end_punct": {"type": "bool", "default": True},
        },
        "arbitration": {
            "enable": {"type": "bool", "default": True},
            "text_source_preference": {
                "type": "enum",
                "enum": ["fast", "slow", "auto"],
                "default": "auto",
            },
            "min_length_ratio": {"type": "float", "min": 0.0, "default": 0.65},
            "max_length_ratio": {"type": "float", "min": 0.0, "default": 3.0},
            "low_confidence_threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "hallucination_block": {"type": "bool", "default": True},
        },
        "segmentation": {
            "enable": {"type": "bool", "default": True},
            "min_chars": {"type": "int", "min": 1, "default": 6},
            "min_duration_sec": {"type": "float", "min": 0.0, "default": 0.8},
            "max_duration_sec": {"type": "float", "min": 0.0, "default": 12.0},
            "max_tokens": {"type": "int", "min": 1, "default": 40},
            "long_pause_sec": {"type": "float", "min": 0.0, "default": 0.8},
            "soft_pause_sec": {"type": "float", "min": 0.0, "default": 0.4},
            "short_merge_max_chars": {"type": "int", "min": 1, "default": 22},
            "keep_sentence_end_punct": {"type": "bool", "default": False},
            "soft_cut.enable": {"type": "bool", "default": True},
            "soft_cut.overlap_degrade_enable": {"type": "bool", "default": False},
            "soft_cut.plan_provider": {"type": "string", "default": "m1_internal"},
            "soft_cut.plan_provider_class": {"type": "string", "default": ""},
            "soft_cut.priority.active_profile": {
                "type": "enum",
                "enum": ["punct_boost_transition", "llm_ramp_up", "llm_primary_no_punct"],
                "default": "punct_boost_transition",
            },
            "soft_cut.priority.profiles": {"type": "object", "default": {}},
        },
        "normalization": {
            "itn.enable": {"type": "bool", "default": True},
            "itn.quality_min_ratio": {"type": "float", "min": 0.0, "default": 0.30},
            "itn.quality_max_ratio": {"type": "float", "min": 0.0, "default": 3.00},
            "decimal_protection.enable": {"type": "bool", "default": True},
            "safe_punct.enable": {"type": "bool", "default": True},
            "abbrev_dot_protection.enable": {"type": "bool", "default": True},
            "hyphen_protection.enable": {"type": "bool", "default": True},
            "cjk_single_digit_to_zh": {"type": "bool", "default": True},
            "cjk_digit_merge": {"type": "bool", "default": True},
            "collapse_spaces": {"type": "bool", "default": True},
            "punct_width": {"type": "enum", "enum": ["auto", "full", "half"], "default": "auto"},
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


class ResidentModelsUpdateRequest(BaseModel):
    """强制常驻模型列表更新请求。"""

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    models: List[str] = Field(default_factory=list)
    mode: Optional[str] = Field(default="replace")

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value_lower = value.lower()
        if value_lower not in {"replace", "add", "remove"}:
            raise ValueError("mode 仅支持 replace/add/remove")
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
        "resident": service.get_resident_models(),
    }


@router.get("/resident")
async def get_resident_models() -> Dict[str, Any]:
    """获取强制常驻模型列表。"""
    service = get_model_runtime_config_service()
    manager = get_model_manager_v2()
    resident = service.get_resident_models()
    return {
        "success": True,
        "resident": resident,
        "effective": manager.get_force_resident_models(),
    }


@router.put("/resident")
async def update_resident_models(req: ResidentModelsUpdateRequest) -> Dict[str, Any]:
    """更新强制常驻模型列表。"""
    service = get_model_runtime_config_service()
    manager = get_model_manager_v2()

    requested = [model_id.strip() for model_id in req.models if model_id and model_id.strip()]
    for model_id in requested:
        try:
            manager.registry.get(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    current = service.get_resident_models().get("models", [])
    mode = (req.mode or "replace").lower()
    if mode == "replace":
        next_models = requested
    elif mode == "add":
        next_models = list(dict.fromkeys(current + requested))
    else:
        remove_set = set(requested)
        next_models = [model_id for model_id in current if model_id not in remove_set]

    updated = service.update_resident_models(next_models)
    manager.set_force_resident_models(updated)

    return {
        "success": True,
        "resident": {"models": updated},
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
    global_effective = manager.refresh_runtime_config()

    return {
        "success": True,
        "applied": True,
        "global": global_effective,
        "resident": service.get_resident_models(),
    }


@router.get("/params")
async def get_runtime_params_schema() -> Dict[str, Any]:
    """返回运行参数字段与范围定义，供前端生成表单。"""
    return {
        "success": True,
        "global": _PARAM_SCHEMA["global"],
        "resident": _PARAM_SCHEMA["resident"],
        "per_model": _PARAM_SCHEMA["per_model"],
        "runtime": _PARAM_SCHEMA["runtime"],
    }
