"""
ASR engine factory.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from app.core.asr.engine import ASREngine
from app.core.asr.engine_config import get_engine_config, normalize_engine_ref

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineRegistration:
    engine_class: Type[ASREngine]
    defaults: Dict[str, Any]
    components: Dict[str, Any]


class ASREngineFactory:
    """ASR engine factory with config-driven registration."""

    _engines: Dict[str, EngineRegistration] = {}
    _initialized: bool = False
    _init_lock = threading.Lock()

    @classmethod
    def register(
        cls,
        name: str,
        engine_class: Type[ASREngine],
        *,
        defaults: Optional[Dict[str, Any]] = None,
        components: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register engine implementation."""
        existing = cls._engines.get(name)
        if existing and existing.engine_class is not engine_class:
            raise ValueError(f"引擎已注册，禁止覆盖: {name}")
        cls._engines[name] = EngineRegistration(
            engine_class=engine_class,
            defaults=defaults or {},
            components=components or {},
        )

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> ASREngine:
        """Create engine instance."""
        cls._ensure_initialized()
        if name not in cls._engines:
            available = ", ".join(sorted(cls._engines)) or "无"
            raise ValueError(f"未知引擎: {name}，已注册: {available}")
        registration = cls._engines[name]
        params: Dict[str, Any] = dict(registration.defaults)
        params.update(kwargs)
        cls._inject_components(registration, params)
        return registration.engine_class(**params)

    @classmethod
    def list_engines(cls) -> List[str]:
        """List registered engines."""
        cls._ensure_initialized()
        return sorted(cls._engines.keys())

    @classmethod
    def _ensure_initialized(cls) -> None:
        if cls._initialized:
            return
        with cls._init_lock:
            if cls._initialized:
                return
            config_data = get_engine_config()
            cls._register_from_config(config_data)
            cls._load_plugins(config_data)
            cls._initialized = True

    @classmethod
    def _register_from_config(cls, config_data: Dict[str, Any]) -> None:
        engines = config_data.get("engines") or {}
        if not isinstance(engines, dict):
            return
        for name, spec in engines.items():
            if not isinstance(spec, dict):
                continue
            class_path = spec.get("class_path")
            if not class_path:
                logger.warning("ASR engine missing class_path: %s", name)
                continue
            defaults = spec.get("defaults") if isinstance(spec.get("defaults"), dict) else {}
            components = spec.get("components") if isinstance(spec.get("components"), dict) else {}
            try:
                engine_class = cls._import_class(class_path)
            except Exception as exc:
                logger.warning("Failed to import engine %s: %s", name, exc)
                continue
            cls.register(name, engine_class, defaults=defaults, components=components)

    @classmethod
    def _inject_components(
        cls,
        registration: EngineRegistration,
        params: Dict[str, Any],
    ) -> None:
        for param_name, raw_ref in (registration.components or {}).items():
            if param_name in params and params[param_name] is not None:
                continue
            engine_ref = normalize_engine_ref(raw_ref)
            if not engine_ref:
                continue
            component_params = engine_ref.get("params") or {}
            params[param_name] = cls.create(engine_ref["id"], **component_params)

    @classmethod
    def _import_class(cls, class_path: str) -> Type[ASREngine]:
        module_path, _, class_name = class_path.rpartition(".")
        if not module_path:
            raise ValueError(f"Invalid class path: {class_path}")
        module = importlib.import_module(module_path)
        engine_class = getattr(module, class_name, None)
        if engine_class is None:
            raise ImportError(f"Class not found: {class_path}")
        return engine_class

    @classmethod
    def _load_plugins(cls, config_data: Dict[str, Any]) -> None:
        plugins = config_data.get("plugins") or {}
        module_names = plugins.get("modules") or []
        module_dirs = plugins.get("module_dirs") or []

        for module_name in module_names:
            if not isinstance(module_name, str):
                continue
            try:
                importlib.import_module(module_name)
            except Exception as exc:
                logger.warning("Failed to import ASR plugin module %s: %s", module_name, exc)

        for module_dir in module_dirs:
            path = Path(module_dir)
            if not path.exists():
                continue
            for file_path in sorted(path.glob("*.py")):
                if file_path.name == "__init__.py":
                    continue
                cls._import_module_from_path(file_path)

    @classmethod
    def _import_module_from_path(cls, file_path: Path) -> None:
        from app.core.config import config

        base_dir = Path(config.BASE_DIR)
        app_root = base_dir / "backend" / "app"
        module_name = None
        try:
            relative = file_path.resolve().relative_to(app_root)
            module_name = "app." + ".".join(relative.with_suffix("").parts)
        except Exception:
            module_name = f"asr_plugin_{file_path.stem}_{abs(hash(str(file_path)))}"

        if module_name in sys.modules:
            return
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
        except Exception as exc:
            logger.warning("Failed to import ASR plugin file %s: %s", file_path, exc)
