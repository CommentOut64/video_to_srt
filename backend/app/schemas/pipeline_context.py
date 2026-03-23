"""
流水线处理上下文

用于三级流水线架构中的数据传递
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Any, Dict, TYPE_CHECKING
import numpy as np

from app.services.alignment.types import PunctTrack, TextTrackBundle

if TYPE_CHECKING:
    from app.services.timeanchored_alignment.contracts import (
        LanguageRun,
        PipelineReport,
        PronunciationPackage,
        ProtectedSpan,
        TextTruthPackage,
        TimeBasePackage,
    )

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
    job_dir: Optional[Path] = None
    debug_punctuation: bool = False
    edge_selection_mode: str = "auto"

    # 音频上下文（用于 Audio Overlap）
    full_audio_array: Optional[np.ndarray] = None  # 完整音频数组（16kHz，单声道）
    full_audio_sr: int = 16000                     # 完整音频采样率

    # 阶段产物
    sv_result: Optional[dict] = None      # FastWorker 产出
    whisper_result: Optional[dict] = None # SlowWorker 产出
    final_sentences: List[Any] = field(default_factory=list)  # 对齐阶段产出
    arbitration_result: Optional[Any] = None  # V3.2.0+dev.20260202.08: 仲裁结果（Phase G-3）
    text_tracks: Optional[TextTrackBundle] = None  # V3.2.0+dev.20260203.03: 三轨文本
    punct_track: Optional[PunctTrack] = None  # V3.2.0+dev.20260204.05: 标点前置域轨道
    finalization_metrics: Dict[str, Any] = field(default_factory=dict)  # V3.2.0+dev.20260203.03
    time_base_chunk: Optional["TimeBasePackage"] = None  # V3.3.0: 时间锚定主链时间基底
    time_base_report: Optional[Dict[str, Any]] = None  # V3.3.0: 时间基底构建报告（轻量）
    text_truth: Optional["TextTruthPackage"] = None  # V3.3.0 Phase3: 文本真相包
    protected_spans: List["ProtectedSpan"] = field(default_factory=list)  # V3.3.0 Phase3: 保护结构 span
    language_runs: List["LanguageRun"] = field(default_factory=list)  # V3.3.0 Phase3: run级语言切分结果
    pronunciation_package: Optional["PronunciationPackage"] = None  # V3.3.0 Phase3: 轻量发音前端产物
    slow_window_meta: Dict[str, Any] = field(default_factory=dict)  # V3.3.0 Phase3: 组窗元数据
    pronunciation_report: Dict[str, Any] = field(default_factory=dict)  # V3.3.0 Phase3: 发音前端报告
    hetero_alignment_result: Optional[Dict[str, Any]] = None  # V3.3.0 Phase7: 新链运行摘要
    hetero_alignment_report: Optional[Dict[str, Any]] = None  # V3.3.0 Phase7: 新链结构化报告
    hetero_route: Optional[str] = None  # V3.3.0 Phase7: 新链路由结果
    slow_window_id: Optional[str] = None  # V3.3.0 Phase2: 组窗ID透传
    slow_window_flush_reason: Optional[str] = None  # V3.3.0 Phase2: 组窗flush原因透传
    slow_window_is_mixed: bool = False  # V3.3.0 Phase2: mixed窗口回退标记

    # 控制信号
    is_end: bool = False                  # 结束流标记
    error: Optional[Exception] = None     # 异常携带

    # 智能复核标记 (V3.10)
    whisper_skipped: bool = False         # SlowWorker 是否跳过（智能复核模式下 SenseVoice 质量足够高时跳过 Whisper）

    def release_preparation_artifacts(self) -> None:
        """
        释放准备层中间对象，避免大型结构进入长期上下文。

        约束：
        - ctc_logits / top_candidates 不应在终稿提交后继续驻留。
        - 文本真相、语言 run、发音前端中间对象可在收口后立即释放。
        """
        self.text_truth = None
        self.protected_spans = []
        self.language_runs = []
        self.pronunciation_package = None
        self.slow_window_meta = {}
        self.pronunciation_report = {}

        if isinstance(self.sv_result, dict):
            for key in (
                "ctc_logits",
                "top_candidates",
                "ctc_top_candidates",
                "compact_trace",
                "acoustic_trace",
            ):
                self.sv_result.pop(key, None)
