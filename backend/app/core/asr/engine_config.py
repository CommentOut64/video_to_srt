"""
ASR engine configuration loader.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

_ENGINE_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def load_engine_config(
    config_path: Path,
    extra_dir: Optional[Path] = None,
    *,
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    base_dir = base_dir or config_path.parent.parent.parent
    data: Dict[str, Any] = {
        "engines": {},
        "profiles": {},
        "plugins": {"module_dirs": [], "modules": []},
    }

    def merge_config(config_data: Any) -> None:
        if not isinstance(config_data, dict):
            return
        engines = config_data.get("engines") or {}
        if isinstance(engines, dict):
            data["engines"].update(engines)
        profiles = config_data.get("profiles") or {}
        if isinstance(profiles, dict):
            data["profiles"].update(profiles)
        plugins = config_data.get("plugins") or {}
        if isinstance(plugins, dict):
            module_dirs = plugins.get("module_dirs") or []
            if isinstance(module_dirs, list):
                data["plugins"]["module_dirs"].extend(module_dirs)
            modules = plugins.get("modules") or []
            if isinstance(modules, list):
                data["plugins"]["modules"].extend(modules)

    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                merge_config(yaml.safe_load(handle) or {})
        except Exception as exc:
            logger.warning("Failed to load ASR engine config: %s", exc)
    else:
        logger.warning("ASR engine config not found: %s", config_path)

    if extra_dir and extra_dir.exists():
        patterns = [str(extra_dir / "*.yaml"), str(extra_dir / "*.yml")]
        for pattern in patterns:
            for path_str in sorted(glob.glob(pattern)):
                path = Path(path_str)
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        merge_config(yaml.safe_load(handle) or {})
                except Exception as exc:
                    logger.warning("Failed to load ASR engine config: %s", exc)

    normalized_dirs: List[Path] = []
    for raw in data["plugins"].get("module_dirs", []):
        if not isinstance(raw, str):
            continue
        raw_path = Path(raw)
        normalized_dirs.append(raw_path if raw_path.is_absolute() else (base_dir / raw_path))
    data["plugins"]["module_dirs"] = normalized_dirs

    modules = [m for m in data["plugins"].get("modules", []) if isinstance(m, str)]
    data["plugins"]["modules"] = modules

    return data


def get_engine_config() -> Dict[str, Any]:
    global _ENGINE_CONFIG_CACHE
    if _ENGINE_CONFIG_CACHE is None:
        from app.core.config import config

        base_dir = Path(config.BASE_DIR)
        _ENGINE_CONFIG_CACHE = load_engine_config(
            base_dir / "backend" / "app" / "config" / "asr_engines.yaml",
            extra_dir=base_dir / "backend" / "app" / "config" / "asr_engines.d",
            base_dir=base_dir,
        )
    return _ENGINE_CONFIG_CACHE


def reset_engine_config_cache() -> None:
    global _ENGINE_CONFIG_CACHE
    _ENGINE_CONFIG_CACHE = None


def normalize_engine_ref(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        return {"id": raw, "params": {}}
    if isinstance(raw, dict):
        engine_id = raw.get("id") or raw.get("engine")
        if not engine_id:
            return None
        params = raw.get("params") or {}
        return {"id": engine_id, "params": params if isinstance(params, dict) else {}}
    return None
