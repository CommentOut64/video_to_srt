"""
模型运行参数配置服务
提供运行参数读写、合并优先级与有效配置计算。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from app.core.config import config
from app.core.asr.model_spec import ModelSpec
from app.models.model_runtime_models import (
    GlobalRuntimeConfig,
    ModelRuntimeConfig,
    ModelRuntimeOverride,
)

logger = logging.getLogger(__name__)

_CONFIG_VERSION = "1.1"
_ENV_PREFIX = "MODEL_RUNTIME_"


class ModelRuntimeConfigService:
    """
    模型运行参数配置服务。

    采用单例模式：避免多实例并发写入配置文件。
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._lock = threading.RLock()
        self._config_file = config_path or self._resolve_config_path()
        self._ensure_config_file()

    @staticmethod
    def _resolve_config_path() -> Path:
        env_path = os.getenv("MODEL_RUNTIME_CONFIG_PATH")
        if env_path:
            return Path(env_path)
        return config.BASE_DIR / "model_runtime_config.json"

    def _ensure_config_file(self) -> None:
        if self._config_file.exists():
            return
        default_config = {
            "version": _CONFIG_VERSION,
            "global": {},
            "runtime": {},
            "per_model": {},
            "resident_models": [],
        }
        self._save_raw_config(default_config)

    def _load_raw_config(self) -> Dict[str, Any]:
        with self._lock:
            try:
                content = self._config_file.read_text(encoding="utf-8")
                data = json.loads(content) if content else {}
                if "global" not in data:
                    data["global"] = {}
                if "runtime" not in data:
                    data["runtime"] = {}
                if "per_model" not in data:
                    data["per_model"] = {}
                if "resident_models" not in data:
                    data["resident_models"] = []
                if "version" not in data:
                    data["version"] = _CONFIG_VERSION
                return data
            except Exception as exc:
                logger.error("加载运行参数配置失败: %s", exc)
                return {"version": _CONFIG_VERSION, "global": {}, "runtime": {}, "per_model": {}}

    def _save_raw_config(self, data: Dict[str, Any]) -> None:
        with self._lock:
            self._config_file.parent.mkdir(parents=True, exist_ok=True)
            self._config_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    @staticmethod
    def _parse_global_config(raw: Dict[str, Any]) -> GlobalRuntimeConfig:
        return GlobalRuntimeConfig(
            device_preference=raw.get("device_preference"),
            is_allow_download=raw.get("allow_download"),
            max_vram_mb=raw.get("max_vram_mb"),
            reserved_vram_mb=raw.get("reserved_vram_mb"),
            max_models=raw.get("max_models"),
            cpu_threads=raw.get("cpu_threads"),
            cpu_affinity_strategy=raw.get("cpu_affinity_strategy"),
            onnx_intra_threads=raw.get("onnx_intra_threads"),
            onnx_inter_threads=raw.get("onnx_inter_threads"),
        )

    @staticmethod
    def _parse_model_override(raw: Dict[str, Any]) -> ModelRuntimeOverride:
        return ModelRuntimeOverride(
            device=raw.get("device"),
            compute_type=raw.get("compute_type"),
            cpu_threads=raw.get("cpu_threads"),
            is_keep_resident=raw.get("keep_resident"),
            evict_priority=raw.get("evict_priority"),
            max_concurrency=raw.get("max_concurrency"),
        )

    def get_config(self) -> ModelRuntimeConfig:
        raw = self._load_raw_config()
        global_config = self._parse_global_config(raw.get("global", {}))
        per_model = {
            model_id: self._parse_model_override(data)
            for model_id, data in raw.get("per_model", {}).items()
        }
        return ModelRuntimeConfig(
            global_config=global_config,
            per_model=per_model,
            runtime=raw.get("runtime", {}),
        )

    def get_resident_models(self) -> Dict[str, Any]:
        """获取强制常驻模型列表。"""
        raw = self._load_raw_config()
        models = raw.get("resident_models", [])
        return {
            "models": list(models) if isinstance(models, list) else [],
        }

    def update_resident_models(self, model_ids: List[str]) -> List[str]:
        """更新强制常驻模型列表。"""
        raw = self._load_raw_config()
        unique_ids = list(dict.fromkeys(model_ids))
        raw["resident_models"] = unique_ids
        self._save_raw_config(raw)
        return unique_ids

    def update_global(self, updates: Dict[str, Any]) -> GlobalRuntimeConfig:
        raw = self._load_raw_config()
        global_raw = raw.get("global", {})
        for key, value in updates.items():
            if value is None:
                global_raw.pop(key, None)
            else:
                global_raw[key] = value
        raw["global"] = global_raw
        self._save_raw_config(raw)
        return self._parse_global_config(global_raw)

    def update_model(self, model_id: str, updates: Dict[str, Any]) -> ModelRuntimeOverride:
        raw = self._load_raw_config()
        per_model = raw.get("per_model", {})
        model_raw = per_model.get(model_id, {})
        for key, value in updates.items():
            if value is None:
                model_raw.pop(key, None)
            else:
                model_raw[key] = value
        if model_raw:
            per_model[model_id] = model_raw
        else:
            per_model.pop(model_id, None)
        raw["per_model"] = per_model
        self._save_raw_config(raw)
        return self._parse_model_override(model_raw)

    def update_runtime_global(self, updates: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        raw = self._load_raw_config()
        runtime_raw = raw.get("runtime", {})
        for group, payload in updates.items():
            group_raw = runtime_raw.get(group, {})
            for key, value in payload.items():
                if value is None:
                    group_raw.pop(key, None)
                else:
                    group_raw[key] = value
            if group_raw:
                runtime_raw[group] = group_raw
            else:
                runtime_raw.pop(group, None)
        raw["runtime"] = runtime_raw
        self._save_raw_config(raw)
        return runtime_raw

    def update_model_runtime(self, model_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        raw = self._load_raw_config()
        per_model = raw.get("per_model", {})
        model_raw = per_model.get(model_id, {})
        runtime_raw = model_raw.get("runtime", {})
        for key, value in updates.items():
            if value is None:
                runtime_raw.pop(key, None)
            else:
                runtime_raw[key] = value
        if runtime_raw:
            model_raw["runtime"] = runtime_raw
        else:
            model_raw.pop("runtime", None)
        if model_raw:
            per_model[model_id] = model_raw
        else:
            per_model.pop(model_id, None)
        raw["per_model"] = per_model
        self._save_raw_config(raw)
        return runtime_raw

    def get_model_override(self, model_id: str) -> ModelRuntimeOverride:
        raw = self._load_raw_config()
        per_model = raw.get("per_model", {})
        return self._parse_model_override(per_model.get(model_id, {}))

    @staticmethod
    def _read_env_global() -> Dict[str, Any]:
        def get_env(key: str) -> Optional[str]:
            return os.getenv(f"{_ENV_PREFIX}{key}")

        def to_bool(value: Optional[str]) -> Optional[bool]:
            if value is None:
                return None
            value_lower = value.strip().lower()
            if value_lower in {"1", "true", "yes", "on"}:
                return True
            if value_lower in {"0", "false", "no", "off"}:
                return False
            return None

        def to_int(value: Optional[str]) -> Optional[int]:
            if value is None:
                return None
            try:
                return int(value)
            except ValueError:
                return None

        return {
            "device_preference": get_env("DEVICE_PREFERENCE"),
            "allow_download": to_bool(get_env("ALLOW_DOWNLOAD")),
            "max_vram_mb": to_int(get_env("MAX_VRAM_MB")),
            "reserved_vram_mb": to_int(get_env("RESERVED_VRAM_MB")),
            "max_models": to_int(get_env("MAX_MODELS")),
            "cpu_threads": to_int(get_env("CPU_THREADS")),
            "cpu_affinity_strategy": get_env("CPU_AFFINITY_STRATEGY"),
            "onnx_intra_threads": to_int(get_env("ONNX_INTRA_THREADS")),
            "onnx_inter_threads": to_int(get_env("ONNX_INTER_THREADS")),
        }

    @staticmethod
    def _read_env_runtime() -> Dict[str, Dict[str, Any]]:
        def norm_str(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            value = value.strip()
            return value or None

        return {
            "sensevoice": {
                "device": norm_str(os.getenv("SENSEVOICE_DEVICE")),
                "model_type": norm_str(os.getenv("SENSEVOICE_MODEL_TYPE")),
            },
        }

    @staticmethod
    def _runtime_defaults() -> Dict[str, Dict[str, Any]]:
        return {
            "whisper": {
                "language": "auto",
                "initial_prompt": None,
                "word_timestamps": False,
                "beam_size": 5,
                "vad_filter": True,
                "vad_parameters": None,
                "temperature": 0.0,
                "condition_on_previous_text": True,
                "suppress_tokens": None,
                "repetition_penalty": 1.0,
                "no_repeat_ngram_size": 0,
            },
            "sensevoice": {
                "language": "auto",
                "use_itn": True,
                "ban_emo_unk": False,
                "model_type": "quantized",
                "device": "cpu",
                "batch_size": 1,
                "quantize": True,
            },
            "demucs": {
                "model_name": "htdemucs",
                "device": "cuda",
                "shifts": 1,
                "overlap": 0.5,
                "segment_length": 10,
                "segment_buffer_sec": 2.0,
                "bgm_sample_duration": 10.0,
                "bgm_light_threshold": 0.02,
                "bgm_heavy_threshold": 0.15,
            },
            "vad": {
                "method": "silero",
                "hf_token": None,
                "onset": 0.4,
                "offset": 0.4,
                "chunk_size": 30,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 400,
                "speech_pad_ms": 300,
                "merge_max_gap": 1.0,
                "merge_max_duration": 12.0,
                "merge_min_fragment": 1.0,
                "smart_target_duration": 12.0,
                "smart_max_duration": 30.0,
                "smart_min_gap_to_split": 0.3,
                "sensevoice_merge_max_gap": 0.3,
                "sensevoice_merge_max_duration": 8.0,
                "sensevoice_smart_target_duration": 8.0,
            },
            "smart_probe": {
                "snr_threshold": 15.0,
            },
            "yamnet": {
                "acappella_threshold": 0.3,
                "music_max_threshold": 0.15,
                "music_avg_threshold": 0.10,
                "speech_max_threshold": 0.8,
                "speech_max_music_threshold": 0.1,
                "speech_dominant_delta": 0.3,
                "speech_dominant_music_max": 0.15,
                "probe_window_count": 3,
                "probe_window_duration_sec": 0.975,
            },
            "punctuation": {
                "enable_punctuation": True,
                "default_language": "zh",
                "fallback_priority": "fast",
                "cache_models": True,
                "max_cached_models": 3,
                "device": "cpu",
                "num_threads": 4,
                "use_int8": True,
                "batch_size": 1,
                "max_sequence_length": 512,
                "enable_arbitration": True,
                "confidence_threshold": 0.6,
                "vad_tolerance": 0.3,
                "min_sentence_gap": 1.5,
                "hallucination_check": True,
                "require_vad_pause": True,
                "cross_validation": True,
                "whisper_confidence_min": 0.7,
                "alignment_method": "anchor",
                "anchor_confidence_threshold": 0.9,
                "max_local_alignment_length": 50,
                "enable_semantic_buffer": True,
                "max_buffer_duration": 15.0,
                "hard_limit_duration": 20.0,
                "min_chunk_duration": 1.0,
                "prefer_punctuation_split": True,
            },
            "pipeline": {
                "batch_size": 16,
                "word_timestamps": False,
            },
        }

    @staticmethod
    def _get_hardware_recommendation() -> Dict[str, Any]:
        try:
            from app.services.hardware_profile_service import get_hardware_profile_provider

            provider = get_hardware_profile_provider()
            return provider.get_runtime_recommendation()
        except Exception as exc:
            logger.debug("硬件推荐参数获取失败（忽略）: %s", exc)
            return {}

    @staticmethod
    def _resolve_value(
        field: str,
        model_value: Optional[Any],
        global_value: Optional[Any],
        env_value: Optional[Any],
        hardware_value: Optional[Any],
        default_value: Any,
    ) -> Tuple[Any, str]:
        if model_value is not None:
            return model_value, "model_override"
        if global_value is not None:
            return global_value, "global_override"
        if env_value is not None:
            return env_value, "env"
        if hardware_value is not None:
            return hardware_value, "hardware"
        return default_value, "default"

    @staticmethod
    def _resolve_runtime_value(
        field: str,
        model_value: Optional[Any],
        runtime_value: Optional[Any],
        env_value: Optional[Any],
        default_value: Any,
    ) -> Tuple[Any, str]:
        if model_value is not None:
            return model_value, "model_override"
        if runtime_value is not None:
            return runtime_value, "runtime_override"
        if env_value is not None:
            return env_value, "env"
        return default_value, "default"

    @staticmethod
    def _resolve_runtime_group(spec: ModelSpec) -> Optional[str]:
        if spec.kind == "asr":
            if spec.id.startswith("whisper-"):
                return "whisper"
            if spec.id.startswith("sensevoice"):
                return "sensevoice"
        if spec.kind == "separation":
            return "demucs"
        if spec.kind == "vad":
            return "vad"
        if spec.kind == "punct":
            return "punctuation"
        return None

    def get_effective_global(self) -> Dict[str, Any]:
        raw = self._load_raw_config()
        global_raw = raw.get("global", {})
        env_raw = self._read_env_global()
        hardware_raw = self._get_hardware_recommendation()

        defaults = {
            "device_preference": "auto",
            "allow_download": False,
            "max_vram_mb": 8000,
            "reserved_vram_mb": 500,
            "max_models": 3,
            "cpu_threads": 4,
            "cpu_affinity_strategy": None,
            "onnx_intra_threads": 1,
            "onnx_inter_threads": 1,
        }

        effective = {}
        sources = {}
        for key, default_value in defaults.items():
            value, source = self._resolve_value(
                key,
                None,
                global_raw.get(key),
                env_raw.get(key),
                hardware_raw.get(key),
                default_value,
            )
            effective[key] = value
            sources[key] = source

        return {
            "effective": effective,
            "sources": sources,
            "override": global_raw,
        }

    def get_effective_runtime_global(self) -> Dict[str, Any]:
        raw = self._load_raw_config()
        runtime_raw = raw.get("runtime", {})
        env_raw = self._read_env_runtime()
        defaults = self._runtime_defaults()

        effective: Dict[str, Dict[str, Any]] = {}
        sources: Dict[str, Dict[str, str]] = {}

        for group, default_values in defaults.items():
            group_override = runtime_raw.get(group, {})
            group_env = env_raw.get(group, {})
            effective_group: Dict[str, Any] = {}
            sources_group: Dict[str, str] = {}
            for key, default_value in default_values.items():
                value, source = self._resolve_runtime_value(
                    key,
                    None,
                    group_override.get(key),
                    group_env.get(key),
                    default_value,
                )
                effective_group[key] = value
                sources_group[key] = source
            effective[group] = effective_group
            sources[group] = sources_group

        return {
            "effective": effective,
            "sources": sources,
            "override": runtime_raw,
            "defaults": defaults,
        }

    def get_effective_runtime_for_model(self, spec: ModelSpec) -> Dict[str, Any]:
        group = self._resolve_runtime_group(spec)
        if not group:
            return {
                "group": None,
                "effective": {},
                "sources": {},
                "override": {},
                "defaults": {},
            }

        raw = self._load_raw_config()
        runtime_raw = raw.get("runtime", {})
        env_raw = self._read_env_runtime()
        defaults = self._runtime_defaults().get(group, {})

        per_model_raw = raw.get("per_model", {}).get(spec.id, {})
        per_model_runtime = per_model_raw.get("runtime", {})
        group_override = runtime_raw.get(group, {})
        group_env = env_raw.get(group, {})

        effective: Dict[str, Any] = {}
        sources: Dict[str, str] = {}
        for key, default_value in defaults.items():
            value, source = self._resolve_runtime_value(
                key,
                per_model_runtime.get(key),
                group_override.get(key),
                group_env.get(key),
                default_value,
            )
            effective[key] = value
            sources[key] = source

        return {
            "group": group,
            "effective": effective,
            "sources": sources,
            "override": per_model_runtime,
            "defaults": defaults,
        }

    def get_effective_model(self, spec: ModelSpec) -> Dict[str, Any]:
        raw = self._load_raw_config()
        global_raw = raw.get("global", {})
        per_model_raw = raw.get("per_model", {}).get(spec.id, {})
        env_raw = self._read_env_global()
        hardware_raw = self._get_hardware_recommendation()

        default_device = spec.default_device or "auto"
        default_compute = spec.compute_type or "auto"
        default_cpu_threads = spec.resources.get("cpu_threads") if spec.resources else None
        if default_cpu_threads is None:
            default_cpu_threads = hardware_raw.get("cpu_threads", 4)

        effective = {}
        sources = {}

        # device
        device_value, device_source = self._resolve_value(
            "device",
            per_model_raw.get("device"),
            global_raw.get("device_preference"),
            env_raw.get("device_preference"),
            hardware_raw.get("device_preference"),
            default_device,
        )
        effective["device"] = device_value
        sources["device"] = device_source

        # compute_type
        compute_default = default_compute
        try:
            if spec.kind == "asr" and "whisper" in spec.id:
                from app.services.whisper_service import get_auto_compute_type

                compute_default = get_auto_compute_type(device_value)
        except Exception:
            compute_default = default_compute

        compute_value, compute_source = self._resolve_value(
            "compute_type",
            per_model_raw.get("compute_type"),
            None,
            None,
            None,
            compute_default,
        )
        effective["compute_type"] = compute_value
        sources["compute_type"] = compute_source

        # cpu_threads
        cpu_value, cpu_source = self._resolve_value(
            "cpu_threads",
            per_model_raw.get("cpu_threads"),
            global_raw.get("cpu_threads"),
            env_raw.get("cpu_threads"),
            hardware_raw.get("cpu_threads"),
            default_cpu_threads,
        )
        effective["cpu_threads"] = cpu_value
        sources["cpu_threads"] = cpu_source

        # keep_resident
        is_keep_default = False
        is_keep_value, keep_source = self._resolve_value(
            "keep_resident",
            per_model_raw.get("keep_resident"),
            None,
            None,
            None,
            is_keep_default,
        )
        effective["keep_resident"] = is_keep_value
        sources["keep_resident"] = keep_source

        # evict_priority
        evict_default = 0
        evict_value, evict_source = self._resolve_value(
            "evict_priority",
            per_model_raw.get("evict_priority"),
            None,
            None,
            None,
            evict_default,
        )
        effective["evict_priority"] = evict_value
        sources["evict_priority"] = evict_source

        # max_concurrency
        concurrency_default = 1
        concurrency_value, concurrency_source = self._resolve_value(
            "max_concurrency",
            per_model_raw.get("max_concurrency"),
            None,
            None,
            None,
            concurrency_default,
        )
        effective["max_concurrency"] = concurrency_value
        sources["max_concurrency"] = concurrency_source

        per_model_override = {
            key: value for key, value in per_model_raw.items() if key != "runtime"
        }
        return {
            "model_id": spec.id,
            "kind": spec.kind,
            "framework": spec.framework,
            "effective": effective,
            "sources": sources,
            "override": per_model_override,
            "defaults": {
                "device": default_device,
                "compute_type": default_compute,
                "cpu_threads": default_cpu_threads,
                "keep_resident": is_keep_default,
                "evict_priority": evict_default,
                "max_concurrency": concurrency_default,
            },
            "resources": spec.resources,
            "features": spec.features,
            "runtime": self.get_effective_runtime_for_model(spec),
        }


_runtime_config_service: Optional[ModelRuntimeConfigService] = None


def get_model_runtime_config_service() -> ModelRuntimeConfigService:
    """获取运行参数配置服务（单例）。"""
    global _runtime_config_service
    if _runtime_config_service is None:
        _runtime_config_service = ModelRuntimeConfigService()
    return _runtime_config_service


def reset_model_runtime_config_service() -> None:
    """重置运行参数配置服务（用于测试）。"""
    global _runtime_config_service
    _runtime_config_service = None
