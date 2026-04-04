"""快流适配器导出。"""

from app.services.timeanchored_alignment.adapters.fast.contracts import FastTimeAdapter
from app.services.timeanchored_alignment.adapters.fast.sensevoice_time_adapter import SenseVoiceTimeAdapter

__all__ = ["FastTimeAdapter", "SenseVoiceTimeAdapter"]
