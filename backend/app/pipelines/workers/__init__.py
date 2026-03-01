"""
Workers package for async dual pipeline

包含两个推理 Worker：
- FastWorker: SenseVoice 快流推理
- SlowWorker: Whisper 慢流推理
"""
__all__ = ["FastWorker", "SlowWorker"]


def __getattr__(name: str):
    """延迟导入，避免无关依赖在导入时被强制加载。"""
    if name == "FastWorker":
        from .fast_worker import FastWorker

        return FastWorker
    if name == "SlowWorker":
        from .slow_worker import SlowWorker

        return SlowWorker
    raise AttributeError(f"module 'app.pipelines.workers' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
