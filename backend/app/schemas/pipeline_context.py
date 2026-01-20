"""
流水线处理上下文

用于三级流水线架构中的数据传递
"""
from dataclasses import dataclass, field
from typing import Optional, List, Any
import numpy as np


@dataclass
class ProcessingContext:
    """
    流水线处理上下文

    这是唯一在流水线中流动的对象，替代散乱的参数传递。
    在三级流水线（FastWorker → SlowWorker → 对齐阶段）中传递数据。

    Attributes:
        job_id: 任务 ID
        chunk_index: Chunk 索引
        audio_chunk: AudioChunk 对象
        full_audio_array: 完整音频数组（用于 Audio Overlap）

        sv_result: FastWorker 产出（SenseVoice 推理结果）
        whisper_result: SlowWorker 产出（Whisper 推理结果）
        final_sentences: 对齐阶段产出（最终句子列表）

        is_end: 结束流标记（用于通知下游阶段停止）
        error: 异常携带（用于异常传播）
    """
    # 基础信息
    job_id: str
    chunk_index: int
    audio_chunk: Any  # AudioChunk 对象

    # 音频上下文（用于 Audio Overlap）
    full_audio_array: Optional[np.ndarray] = None  # 完整音频数组（16kHz，单声道）
    full_audio_sr: int = 16000                     # 完整音频采样率

    # 阶段产物
    sv_result: Optional[dict] = None      # FastWorker 产出
    whisper_result: Optional[dict] = None # SlowWorker 产出
    final_sentences: List[Any] = field(default_factory=list)  # 对齐阶段产出

    # 控制信号
    is_end: bool = False                  # 结束流标记
    error: Optional[Exception] = None     # 异常携带

    # 智能补刀标记 (V3.10)
    whisper_skipped: bool = False         # SlowWorker 是否跳过（智能补刀模式下 SenseVoice 质量足够高时跳过 Whisper）
