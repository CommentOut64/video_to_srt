"""
Workers package for async dual pipeline

包含两个推理 Worker：
- FastWorker: SenseVoice 快流推理
- SlowWorker: Whisper 慢流推理
"""
from .fast_worker import FastWorker
from .slow_worker import SlowWorker
__all__ = ['FastWorker', 'SlowWorker']
