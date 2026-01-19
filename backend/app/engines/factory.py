"""
ASR 引擎工厂。
V3.2.0+dev.20260119.02
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from app.core.asr.engine import ASREngine


class ASREngineFactory:
    """ASR 引擎工厂（工厂模式：集中管理创建，避免业务层耦合具体实现）。"""

    _engines: Dict[str, Type[ASREngine]] = {}

    @classmethod
    def register(cls, name: str, engine_class: Type[ASREngine]) -> None:
        """注册引擎实现。"""
        if name in cls._engines and cls._engines[name] is not engine_class:
            raise ValueError(f"引擎已注册，禁止覆盖: {name}")
        cls._engines[name] = engine_class

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> ASREngine:
        """创建引擎实例。"""
        if name not in cls._engines:
            available = ", ".join(sorted(cls._engines)) or "无"
            raise ValueError(f"未知引擎: {name}，已注册: {available}")
        return cls._engines[name](**kwargs)

    @classmethod
    def list_engines(cls) -> List[str]:
        """列出已注册引擎。"""
        return sorted(cls._engines.keys())


def register_default_engines() -> None:
    """注册内置引擎实现。"""
    from app.engines.sensevoice_engine import SenseVoiceEngine
    from app.engines.whisper_engine import WhisperEngine
    from app.engines.dual_stream_engine import DualStreamEngine
    from app.engines.dummy_engine import DummyEngine

    ASREngineFactory.register("sensevoice", SenseVoiceEngine)
    ASREngineFactory.register("whisper", WhisperEngine)
    ASREngineFactory.register("dual_stream", DualStreamEngine)
    ASREngineFactory.register("dummy", DummyEngine)


register_default_engines()
