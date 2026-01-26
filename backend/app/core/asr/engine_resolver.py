"""
ASR engine resolver based on config profiles.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Tuple

from app.core.asr.engine import ASREngine
from app.core.asr.engine_config import get_engine_config, normalize_engine_ref
from app.engines.factory import ASREngineFactory


_PLACEHOLDER_RE = re.compile(r"^\$\{(.+)\}$")


class EngineResolver:
    """Resolve draft/patch engines from config profiles."""

    def __init__(
        self,
        *,
        config_data: Optional[Dict[str, Any]] = None,
        hardware_profile_provider: Optional[object] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config_data = config_data or get_engine_config()
        self._hardware_profile_provider = hardware_profile_provider
        self._logger = logger or logging.getLogger(__name__)

    def resolve_profile_engines(
        self,
        profile: str,
        *,
        job: object,
        optimization_config: Optional[object] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ) -> Tuple[Optional[ASREngine], Optional[ASREngine]]:
        profiles = self._config_data.get("profiles") or {}
        profile_spec = profiles.get(profile)
        if not isinstance(profile_spec, dict):
            raise ValueError(f"Unknown ASR profile: {profile}")

        runtime_device = self._resolve_device(
            device=device,
            optimization_config=optimization_config,
        )
        context = {
            "job": job,
            "transcription": getattr(getattr(job, "settings", None), "transcription", None),
            "runtime": {
                "device": runtime_device,
                "compute_type": compute_type,
            },
        }

        draft_engine = self._build_engine(profile_spec.get("draft_engine"), context)
        patch_engine = self._build_engine(profile_spec.get("patch_engine"), context)
        return draft_engine, patch_engine

    def _resolve_device(
        self,
        *,
        device: Optional[str],
        optimization_config: Optional[object],
    ) -> str:
        if device:
            return device
        if optimization_config and hasattr(optimization_config, "recommended_device"):
            resolved = getattr(optimization_config, "recommended_device")
            if resolved:
                return resolved
        if self._hardware_profile_provider:
            try:
                hardware_info = self._hardware_profile_provider.get_hardware_info(
                    is_force_refresh=False
                )
                optimization = self._hardware_profile_provider.get_optimization_config(
                    hardware_info
                )
                if getattr(optimization, "recommended_device", None):
                    return optimization.recommended_device
            except Exception as exc:
                self._logger.warning("Failed to resolve device from hardware profile: %s", exc)
        return "cuda"

    def _build_engine(
        self,
        raw_ref: Any,
        context: Dict[str, Any],
    ) -> Optional[ASREngine]:
        engine_ref = normalize_engine_ref(raw_ref)
        if not engine_ref:
            return None
        params = self._resolve_params(engine_ref.get("params") or {}, context)
        return ASREngineFactory.create(engine_ref["id"], **params)

    def _resolve_params(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        resolved: Dict[str, Any] = {}
        for key, value in params.items():
            resolved[key] = self._resolve_value(value, context)
        return resolved

    def _resolve_value(self, value: Any, context: Dict[str, Any]) -> Any:
        if isinstance(value, dict):
            return {k: self._resolve_value(v, context) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_value(item, context) for item in value]
        if isinstance(value, str):
            match = _PLACEHOLDER_RE.match(value.strip())
            if not match:
                return value
            expr = match.group(1)
            path, _, default = expr.partition("|")
            resolved = self._resolve_path(context, path.strip())
            if resolved is None:
                if default:
                    return self._parse_default(default.strip())
                return None
            return resolved
        return value

    @staticmethod
    def _parse_default(raw: str) -> Any:
        lowered = raw.lower()
        if lowered in {"none", "null"}:
            return None
        return raw

    @staticmethod
    def _resolve_path(source: Any, path: str) -> Any:
        current = source
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = getattr(current, part, None)
            if current is None:
                return None
        return current
