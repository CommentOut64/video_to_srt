"""
标点服务配置加载器。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from app.core.config import config

logger = logging.getLogger(__name__)

_PUNCT_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def _default_config() -> Dict[str, Any]:
    return {
        "default_language": "zh",
        "fallback_priority": "fast",
        "chinese": {
            "model_id": "punct-ct-transformer-zh",
            "itn_first": True,
            "fallback_on_oov": True,
        },
        "english": {
            "model_id": "punct-distilbert-en",
            "overlap_words": 10,
        },
        "japanese": {
            "model_id": "punct-pcs-47lang",
        },
        "scheduler": {
            "mode": "fast_only",
            "fast_confidence_threshold": 0.55,
            "alignment_coverage_threshold": 0.7,
            "arbiter_conflict_threshold": 0.35,
            "max_slow_retries": 1,
        },
        "postprocess": {
            "fast": {
                "candidate_min_conf": 0.30,
                "add_mid_conf": 0.70,
                "add_end_conf": 0.75,
                "keep_raw_mid_conf": 0.60,
                "keep_raw_end_conf": 0.70,
                "drop_raw_mid_conf": 0.35,
                "drop_raw_end_conf": 0.50,
                "question_gate_min_conf": 0.85,
                "pause_end_min_sec": 0.35,
            },
            "dual": {
                "candidate_min_conf": 0.25,
                "add_mid_conf": 0.60,
                "add_end_conf": 0.65,
                "keep_raw_mid_conf": 0.55,
                "keep_raw_end_conf": 0.60,
                "drop_raw_mid_conf": 0.30,
                "drop_raw_end_conf": 0.40,
                "question_gate_min_conf": 0.75,
                "pause_end_min_sec": 0.25,
            },
            "shared": {
                "candidate_window_words": 1,
                "conflict_window_chars": 1,
                "max_repeat_punct": 1,
                "allowed_punct_zh": "。？！；：，、",
                "allowed_punct_en": ".!?,;:",
                "comma_guard_enabled": False,
                "comma_guard_min_conf": 0.85,
                "comma_guard_min_chars": 6,
                "comma_guard_min_words": 4,
                "comma_guard_pause_min_sec": 0.25,
            },
        },
    }


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_punctuation_config() -> Dict[str, Any]:
    base = Path(config.BASE_DIR)
    path = base / "backend" / "app" / "config" / "punctuation.yaml"
    defaults = _default_config()
    if not path.exists():
        logger.warning("未找到标点配置文件: %s", path)
        return defaults
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            logger.warning("标点配置格式异常，使用默认值: %s", path)
            return defaults
        return _deep_merge(defaults, data)
    except Exception as exc:
        logger.warning("加载标点配置失败，使用默认值: %s", exc)
        return defaults


def get_punctuation_config() -> Dict[str, Any]:
    global _PUNCT_CONFIG_CACHE
    if _PUNCT_CONFIG_CACHE is None:
        _PUNCT_CONFIG_CACHE = load_punctuation_config()
    return _PUNCT_CONFIG_CACHE


def reset_punctuation_config_cache() -> None:
    global _PUNCT_CONFIG_CACHE
    _PUNCT_CONFIG_CACHE = None
