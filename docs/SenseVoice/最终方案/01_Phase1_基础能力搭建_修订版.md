# Phase 1: 基础能力搭建（修订版 v2.1）

> 目标：搭建 SenseVoice 服务基础框架，建立时空解耦架构基础
>
> 工期：2-3天
>
> 版本更新：整合 [06_转录层深度优化_时空解耦架构](./06_转录层深度优化_时空解耦架构.md) 设计

---

## ⚠️ 重要修订

### v2.1 新增（VAD优先 + 频谱分诊）

- ✅ **说明**：阈值配置体系分为两部分
  - 本阶段 `thresholds.py`：置信度、困惑度、警告高亮阈值
  - Phase 2 `spectrum_thresholds.py`：频谱分诊阈值（ZCR、谱质心、谐波比等）

### v2.0 新增（时空解耦架构）

- ✅ **新增**：扩展 `SentenceSegment` 数据模型（source, warning_type, perplexity 等）
- ✅ **新增**：伪对齐算法 `pseudo_alignment.py`
- ✅ **新增**：智能进度系统 `progress_tracker.py`
- ✅ **新增**：SSE 事件 Tag 统一设计
- ✅ **新增**：阈值配置体系 `thresholds.py`
- ✅ **新增**：JobSettings 预设配置扩展

### v1.0 基础修订

- ❌ **删除**：不再创建 `hardware_detector.py`（已存在）
- ✅ **复用**：使用现有的 `hardware_service.py`
- ✅ **扩展**：扩展现有的 `hardware_models.py`

---

## 一、任务清单

| 任务 | 文件 | 优先级 | 状态 |
|------|------|--------|------|
| SenseVoice ONNX 服务 | `sensevoice_onnx_service.py` | P0 | 新建 |
| **扩展数据模型（时空解耦）** | `sensevoice_models.py` | **P0** | **修改** |
| 分句算法 | `sentence_splitter.py` | P0 | 新建 |
| 文本清洗与标点统一 | `text_normalizer.py` | P0 | 新建 |
| **伪对齐算法** | `pseudo_alignment.py` | **P0** | **新建** |
| **智能进度系统** | `progress_tracker.py` | **P0** | **新建** |
| **阈值配置体系** | `thresholds.py` | **P0** | **新建** |
| **SSE 事件 Tag 扩展** | `sse_service.py` | **P0** | **修改** |
| 扩展硬件模型 | `hardware_models.py` | P1 | 修改 |
| 显存释放机制 | `model_preload_manager.py` | P0 | 修改 |

---

## 二、数据模型定义（时空解耦版）

**路径**: `backend/app/models/sensevoice_models.py`

```python
"""
SenseVoice 数据模型定义（时空解耦版 v2.0）

核心设计：
- SenseVoice 为时间领主，提供绝对时间轴基准
- Whisper 为听觉补丁，仅提供文本
- LLM 为逻辑胶水，校对/翻译
"""
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class TextSource(Enum):
    """文本来源"""
    SENSEVOICE = "sensevoice"            # SenseVoice 原始输出
    WHISPER_PATCH = "whisper_patch"      # Whisper 复核替换
    LLM_CORRECTION = "llm_correction"    # LLM 校对修正
    LLM_TRANSLATION = "llm_translation"  # LLM 翻译


class WarningType(Enum):
    """警告类型（用于高亮系统）"""
    NONE = "none"                                   # 无警告
    LOW_TRANSCRIPTION_CONFIDENCE = "low_transcription"  # 转录置信度低
    HIGH_PROOFREAD_PERPLEXITY = "high_perplexity"      # 校对困惑度高
    BOTH = "both"                                   # 两者都有问题


@dataclass
class SenseVoiceONNXConfig:
    """SenseVoice ONNX 配置"""
    model_path: str = "models/sensevoice_small_int8.onnx"
    use_gpu: bool = True
    fallback_to_cpu: bool = True
    num_threads: int = 4
    batch_size: int = 1
    quantization: str = "int8"
    enable_graph_optimization: bool = True
    optimization_level: int = 99


@dataclass
class WordTimestamp:
    """字级时间戳（扩展版）"""
    word: str
    start: float
    end: float
    confidence: float = 1.0
    is_pseudo: bool = False              # 是否为伪对齐生成
    # ========== 新增：警告字段 ==========
    warning_type: WarningType = WarningType.NONE
    perplexity: Optional[float] = None   # LLM 校对时的困惑度


@dataclass
class SenseVoiceResult:
    """SenseVoice 转录结果"""
    text: str
    text_clean: str
    confidence: float
    words: List[WordTimestamp]
    start: float
    end: float
    language: str = "auto"
    emotion: Optional[str] = None
    event: Optional[str] = None
    raw_result: Optional[dict] = None


@dataclass
class SentenceSegment:
    """句级字幕段（时空解耦版）"""
    text: str
    start: float
    end: float
    words: List[WordTimestamp]
    confidence: float = 1.0

    # ========== 时空解耦字段 ==========
    source: TextSource = TextSource.SENSEVOICE   # 文本来源
    is_modified: bool = False                     # 是否被修改过
    original_text: Optional[str] = None           # 修改前的原始文本
    whisper_alternative: Optional[str] = None     # Whisper 备选文本

    # ========== 警告与校对字段 ==========
    warning_type: WarningType = WarningType.NONE  # 句级警告类型
    perplexity: Optional[float] = None            # LLM 校对困惑度
    translation: Optional[str] = None             # 翻译结果
    translation_confidence: Optional[float] = None  # 翻译置信度

    def mark_as_modified(self, new_text: str, source: TextSource):
        """标记为已修改"""
        if not self.is_modified:
            self.original_text = self.text
        self.text = new_text
        self.source = source
        self.is_modified = True

    def compute_warning_type(self) -> WarningType:
        """根据置信度和困惑度计算警告类型"""
        has_low_confidence = self.confidence < 0.6
        has_high_perplexity = self.perplexity is not None and self.perplexity > 50.0

        if has_low_confidence and has_high_perplexity:
            return WarningType.BOTH
        elif has_low_confidence:
            return WarningType.LOW_TRANSCRIPTION_CONFIDENCE
        elif has_high_perplexity:
            return WarningType.HIGH_PROOFREAD_PERPLEXITY
        return WarningType.NONE

    def to_dict(self) -> dict:
        """转换为字典格式（用于 SSE 推送）"""
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "source": self.source.value,
            "is_modified": self.is_modified,
            "original_text": self.original_text,
            "warning_type": self.warning_type.value,
            "perplexity": self.perplexity,
            "translation": self.translation,
            "words": [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "confidence": w.confidence,
                    "is_pseudo": w.is_pseudo
                }
                for w in self.words
            ]
        }
```

---

## 三、伪对齐算法（新增）

**路径**: `backend/app/services/pseudo_alignment.py`

```python
"""
伪对齐算法（Pseudo-Alignment）

核心原理：
- SenseVoice 确定的时间窗口（start/end）不可变
- 当 Whisper/LLM 替换文本后，新字符均匀分布在原时间窗口内
- 生成的字级时间戳标记为 is_pseudo=True
"""
from typing import List
from ..models.sensevoice_models import WordTimestamp, TextSource
import logging

logger = logging.getLogger(__name__)


class PseudoAlignment:
    """伪对齐器"""

    @staticmethod
    def apply(
        original_start: float,
        original_end: float,
        new_text: str,
        default_confidence: float = 1.0
    ) -> List[WordTimestamp]:
        """
        将新文本均匀映射到原时间段内

        Args:
            original_start: 原始起始时间（由 SenseVoice 确定，不可变）
            original_end: 原始结束时间（由 SenseVoice 确定，不可变）
            new_text: 替换后的新文本
            default_confidence: 默认置信度（修正后通常为 1.0）

        Returns:
            List[WordTimestamp]: 伪对齐的字级时间戳列表
        """
        duration = original_end - original_start

        # 过滤空白字符，保留实际字符
        chars = [c for c in new_text if not c.isspace()]
        char_count = len(chars)

        if char_count == 0:
            logger.warning("伪对齐：文本为空，跳过")
            return []

        # 计算每个字符的时长
        step = duration / char_count
        result = []

        for i, char in enumerate(chars):
            w_start = original_start + (i * step)
            w_end = w_start + step

            result.append(WordTimestamp(
                word=char,
                start=round(w_start, 3),
                end=round(w_end, 3),
                confidence=default_confidence,
                is_pseudo=True  # 标记为伪对齐生成
            ))

        logger.debug(
            f"伪对齐完成: {char_count} 字符, "
            f"时间窗口 {original_start:.2f}-{original_end:.2f}s"
        )

        return result

    @staticmethod
    def apply_to_sentence(
        sentence,  # SentenceSegment
        new_text: str,
        source: TextSource
    ):
        """
        对句子应用伪对齐

        Args:
            sentence: 原始句子段落
            new_text: 替换后的文本
            source: 文本来源

        Returns:
            修改后的 SentenceSegment（原对象被修改）
        """
        # 生成新的字级时间戳
        new_words = PseudoAlignment.apply(
            original_start=sentence.start,
            original_end=sentence.end,
            new_text=new_text
        )

        # 更新句子
        sentence.mark_as_modified(new_text, source)
        sentence.words = new_words

        return sentence


# ========== 单例访问 ==========

_pseudo_alignment_instance = None


def get_pseudo_alignment() -> PseudoAlignment:
    """获取伪对齐器单例"""
    global _pseudo_alignment_instance
    if _pseudo_alignment_instance is None:
        _pseudo_alignment_instance = PseudoAlignment()
    return _pseudo_alignment_instance
```

---

## 四、智能进度系统（新增）

**路径**: `backend/app/services/progress_tracker.py`

```python
"""
智能进度追踪系统

核心特性：
1. 根据预设配置动态调整各阶段权重
2. 支持流式更新（SV、Whisper、LLM 各阶段独立更新）
3. 统一 SSE 事件 tag 设计
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProcessPhase(Enum):
    """处理阶段枚举"""
    PENDING = "pending"
    EXTRACT = "extract"
    BGM_DETECT = "bgm_detect"
    DEMUCS = "demucs"
    VAD = "vad"
    SENSEVOICE = "sensevoice"
    WHISPER_PATCH = "whisper"
    LLM_PROOF = "llm_proof"
    LLM_TRANS = "llm_trans"
    SRT = "srt"
    COMPLETE = "complete"


@dataclass
class PhaseProgress:
    """阶段进度"""
    phase: ProcessPhase
    weight: float
    total_items: int = 0
    completed_items: int = 0
    is_active: bool = False
    message: str = ""


@dataclass
class PresetWeights:
    """预设权重配置"""
    extract: float = 5
    bgm_detect: float = 2
    demucs: float = 8
    vad: float = 5
    sensevoice: float = 40
    whisper: float = 0
    llm_proof: float = 0
    llm_trans: float = 0
    srt: float = 10

    @classmethod
    def from_preset(cls, preset_id: str) -> 'PresetWeights':
        """根据预设ID生成权重配置"""
        presets = {
            'default': cls(sensevoice=50, srt=10),
            'preset1': cls(sensevoice=35, whisper=20, srt=10),
            'preset2': cls(sensevoice=30, whisper=15, llm_proof=15, srt=10),
            'preset3': cls(sensevoice=25, whisper=15, llm_proof=25, srt=10),
            'preset4': cls(sensevoice=20, whisper=10, llm_proof=20, llm_trans=15, srt=10),
            'preset5': cls(sensevoice=22, whisper=12, llm_proof=20, llm_trans=8, srt=10),
        }
        return presets.get(preset_id, cls())

    def total_weight(self) -> float:
        """计算总权重"""
        return (
            self.extract + self.bgm_detect + self.demucs +
            self.vad + self.sensevoice + self.whisper +
            self.llm_proof + self.llm_trans + self.srt
        )


class ProgressTracker:
    """智能进度追踪器"""

    def __init__(self, job_id: str, preset_id: str = 'default'):
        self.job_id = job_id
        self.weights = PresetWeights.from_preset(preset_id)
        self.total_weight = self.weights.total_weight()
        self.phases: Dict[ProcessPhase, PhaseProgress] = {}
        self._init_phases()
        self.completed_weight = 0.0
        self.current_phase = ProcessPhase.PENDING

    def _init_phases(self):
        """初始化各阶段"""
        weight_map = {
            ProcessPhase.EXTRACT: self.weights.extract,
            ProcessPhase.BGM_DETECT: self.weights.bgm_detect,
            ProcessPhase.DEMUCS: self.weights.demucs,
            ProcessPhase.VAD: self.weights.vad,
            ProcessPhase.SENSEVOICE: self.weights.sensevoice,
            ProcessPhase.WHISPER_PATCH: self.weights.whisper,
            ProcessPhase.LLM_PROOF: self.weights.llm_proof,
            ProcessPhase.LLM_TRANS: self.weights.llm_trans,
            ProcessPhase.SRT: self.weights.srt,
        }
        for phase, weight in weight_map.items():
            self.phases[phase] = PhaseProgress(phase=phase, weight=weight)

    def start_phase(self, phase: ProcessPhase, total_items: int = 1, message: str = ""):
        """开始某个阶段"""
        if phase in self.phases:
            self.phases[phase].is_active = True
            self.phases[phase].total_items = total_items
            self.phases[phase].completed_items = 0
            self.phases[phase].message = message
            self.current_phase = phase

    def update_phase(self, phase: ProcessPhase, completed: int = None, increment: int = None, message: str = None):
        """更新阶段进度"""
        if phase not in self.phases:
            return
        pp = self.phases[phase]
        if completed is not None:
            pp.completed_items = completed
        elif increment is not None:
            pp.completed_items += increment
        if message is not None:
            pp.message = message

    def complete_phase(self, phase: ProcessPhase):
        """完成某个阶段"""
        if phase in self.phases:
            pp = self.phases[phase]
            pp.is_active = False
            pp.completed_items = pp.total_items
            self.completed_weight += pp.weight

    def get_overall_progress(self) -> float:
        """获取总体进度百分比"""
        progress = self.completed_weight
        if self.current_phase in self.phases:
            pp = self.phases[self.current_phase]
            if pp.is_active and pp.total_items > 0:
                phase_progress = pp.completed_items / pp.total_items
                progress += pp.weight * phase_progress
        return round((progress / self.total_weight) * 100, 1)

    def to_sse_data(self) -> dict:
        """生成 SSE 推送数据"""
        return {
            "job_id": self.job_id,
            "phase": self.current_phase.value,
            "percent": self.get_overall_progress(),
            "message": self.phases.get(self.current_phase, PhaseProgress(
                phase=self.current_phase, weight=0
            )).message,
        }


# ========== 单例工厂 ==========

_tracker_instances: Dict[str, ProgressTracker] = {}


def get_progress_tracker(job_id: str, preset_id: str = None) -> ProgressTracker:
    """获取或创建进度追踪器"""
    global _tracker_instances
    if job_id not in _tracker_instances:
        if preset_id is None:
            preset_id = 'default'
        _tracker_instances[job_id] = ProgressTracker(job_id, preset_id)
    return _tracker_instances[job_id]


def remove_progress_tracker(job_id: str):
    """移除进度追踪器"""
    global _tracker_instances
    if job_id in _tracker_instances:
        del _tracker_instances[job_id]
```

---

## 五、阈值配置体系（新增）

本阶段定义**置信度与困惑度阈值**，用于熔断决策和警告高亮。

> **注意**：频谱分诊相关阈值（音乐检测、噪音检测）定义在 [Phase 2 的 spectrum_thresholds.py](./02_Phase2_智能熔断集成_修订版.md#32-分诊阈值配置) 中。

**路径**: `backend/app/core/thresholds.py`

```python
"""
置信度与困惑度阈值配置

基于实际测试和业界经验设定阈值：
- 转录置信度：基于 Whisper avg_logprob 和 no_speech_prob
- 校对困惑度：基于 LLM 输出的 perplexity
"""
from dataclasses import dataclass
from enum import Enum


class ConfidenceLevel(Enum):
    """置信度等级"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CRITICAL = "critical"


@dataclass
class ThresholdConfig:
    """阈值配置"""

    # ========== 转录置信度阈值 ==========
    sv_confidence_high: float = 0.85
    sv_confidence_medium: float = 0.6
    sv_confidence_low: float = 0.4

    # Whisper 阈值
    whisper_logprob_good: float = -0.5
    whisper_logprob_ok: float = -0.8
    whisper_no_speech_medium: float = 0.6

    # ========== 校对困惑度阈值 ==========
    llm_perplexity_good: float = 35.0
    llm_perplexity_acceptable: float = 50.0
    llm_perplexity_poor: float = 80.0

    # ========== 触发阈值 ==========
    whisper_patch_trigger_confidence: float = 0.6
    llm_proof_trigger_confidence: float = 0.7
    llm_proof_trigger_modified: bool = True

    # ========== 警告高亮阈值 ==========
    word_warning_confidence: float = 0.5
    word_critical_confidence: float = 0.3
    sentence_warning_confidence: float = 0.6
    sentence_warning_perplexity: float = 50.0


DEFAULT_THRESHOLDS = ThresholdConfig()


def get_confidence_level(confidence: float, config: ThresholdConfig = None) -> ConfidenceLevel:
    """根据置信度获取等级"""
    if config is None:
        config = DEFAULT_THRESHOLDS
    if confidence >= config.sv_confidence_high:
        return ConfidenceLevel.HIGH
    elif confidence >= config.sv_confidence_medium:
        return ConfidenceLevel.MEDIUM
    elif confidence >= config.sv_confidence_low:
        return ConfidenceLevel.LOW
    else:
        return ConfidenceLevel.CRITICAL


def needs_whisper_patch(confidence: float, config: ThresholdConfig = None) -> bool:
    """判断是否需要 Whisper 复核"""
    if config is None:
        config = DEFAULT_THRESHOLDS
    return confidence < config.whisper_patch_trigger_confidence


def needs_llm_proof(confidence: float, is_modified: bool = False, mode: str = 'sparse', config: ThresholdConfig = None) -> bool:
    """判断是否需要 LLM 校对"""
    if config is None:
        config = DEFAULT_THRESHOLDS
    if mode == 'full':
        return True
    if mode == 'sparse':
        if confidence < config.llm_proof_trigger_confidence:
            return True
        if is_modified and config.llm_proof_trigger_modified:
            return True
    return False
```

---

## 六、SSE 事件 Tag 扩展（修改）

**路径**: `backend/app/services/sse_service.py` (扩展)

在现有 SSE 服务中添加以下事件定义：

```python
"""
SSE 事件 Tag 统一定义

事件命名规范：
- progress.{phase}: 进度更新事件
- subtitle.{action}: 字幕流式事件
- signal.{type}: 信号事件
"""

# ========== 进度事件 Tag ==========
SSE_PROGRESS_TAGS = {
    "progress.overall": "总体进度更新",
    "progress.extract": "音频提取进度",
    "progress.bgm_detect": "BGM 检测进度",
    "progress.demucs": "人声分离进度",
    "progress.vad": "VAD 分段进度",
    "progress.sensevoice": "SenseVoice 转录进度（流式）",
    "progress.whisper": "Whisper 复核进度",
    "progress.llm_proof": "LLM 校对进度",
    "progress.llm_trans": "LLM 翻译进度",
    "progress.srt": "SRT 生成进度",
}

# ========== 字幕流式事件 Tag ==========
SSE_SUBTITLE_TAGS = {
    "subtitle.sv_segment": "SenseVoice 完成一个 VAD 段",
    "subtitle.sv_sentence": "SenseVoice 完成一个句子",
    "subtitle.whisper_patch": "Whisper 复核覆盖一个句子",
    "subtitle.llm_proof": "LLM 校对覆盖一个句子",
    "subtitle.llm_trans": "LLM 翻译完成一个句子",
    "subtitle.batch_update": "批量更新多个句子",
}

# ========== 信号事件 Tag ==========
SSE_SIGNAL_TAGS = {
    "signal.job_start": "任务开始",
    "signal.job_complete": "任务完成",
    "signal.job_failed": "任务失败",
    "signal.phase_start": "阶段开始",
    "signal.phase_complete": "阶段完成",
    "signal.circuit_breaker": "熔断触发",
}


# ========== SSE 推送辅助方法 ==========

def push_progress_event(sse_manager, job_id: str, phase: str, data: dict):
    """推送进度事件"""
    tag = f"progress.{phase}"
    channel_id = f"job:{job_id}"
    sse_manager.broadcast_sync(channel_id, tag, data)


def push_subtitle_event(sse_manager, job_id: str, event_type: str, sentence_data: dict):
    """推送字幕流式事件"""
    tag = f"subtitle.{event_type}"
    channel_id = f"job:{job_id}"
    sse_manager.broadcast_sync(channel_id, tag, sentence_data)


def push_signal_event(sse_manager, job_id: str, signal_type: str, message: str = ""):
    """推送信号事件"""
    tag = f"signal.{signal_type}"
    channel_id = f"job:{job_id}"
    sse_manager.broadcast_sync(channel_id, tag, {"job_id": job_id, "signal": signal_type, "message": message})
```

---

## 七、扩展硬件模型

**修改文件**: `backend/app/models/hardware_models.py`

在 `OptimizationConfig` 类中添加 SenseVoice 相关配置：

```python
@dataclass
class OptimizationConfig:
    """基于硬件的优化配置"""
    batch_size: int = 16
    concurrency: int = 1
    use_memory_mapping: bool = False
    cpu_affinity_cores: List[int] = field(default_factory=list)
    recommended_device: str = "cpu"
    recommended_model: str = "medium"

    # ========== 新增：SenseVoice 相关配置 ==========
    enable_sensevoice: bool = True
    enable_demucs: bool = True
    demucs_model: str = "htdemucs"
    sensevoice_device: str = "cuda"
    note: str = ""
```

---

## 八、文本清洗与标点统一

**路径**: `backend/app/services/text_normalizer.py`

```python
"""
SenseVoice 文本后处理器
清洗特殊标签、重复字符，统一标点符号
"""
import re
import logging

logger = logging.getLogger(__name__)


class TextNormalizer:
    """SenseVoice 文本后处理器"""

    SPECIAL_TAGS = re.compile(r'<\|.*?\|>')
    REPEATED_CHARS = re.compile(r'(.)\1{2,}')

    PUNCTUATION_MAP = {
        '，': ',', '。': '.', '！': '!', '？': '?',
        '；': ';', '：': ':', '"': '"', '"': '"',
    }

    @staticmethod
    def clean(text: str) -> str:
        """清洗文本（移除特殊标签和异常重复）"""
        text = TextNormalizer.SPECIAL_TAGS.sub('', text)
        text = TextNormalizer.REPEATED_CHARS.sub(r'\1\1', text)
        return text.strip()

    @staticmethod
    def normalize_punctuation(text: str, to_fullwidth: bool = True) -> str:
        """统一标点符号（全角/半角）"""
        if to_fullwidth:
            mapping = {v: k for k, v in TextNormalizer.PUNCTUATION_MAP.items()}
        else:
            mapping = TextNormalizer.PUNCTUATION_MAP
        for old, new in mapping.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def process(text: str, to_fullwidth: bool = True) -> str:
        """完整处理流程"""
        text = TextNormalizer.clean(text)
        text = TextNormalizer.normalize_punctuation(text, to_fullwidth)
        return text
```

---

## 九、显存释放机制

**修改文件**: `backend/app/services/model_preload_manager.py`

```python
def unload_demucs(self):
    """卸载 Demucs 模型（显式释放显存）"""
    with self._global_lock:
        if self._demucs_model is not None:
            del self._demucs_model
            self._demucs_model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.info("PyTorch 显存已释放")
            except Exception as e:
                self.logger.warning(f"释放显存失败: {e}")
```

---

## 十、验收标准（扩展）

### 基础能力

- [ ] 硬件检测使用现有 `hardware_service.py`
- [ ] 分句算法可正确切分测试用例
- [ ] SenseVoice ONNX 服务可正常加载/卸载
- [ ] 文本清洗器可移除特殊标签
- [ ] Demucs 卸载时显存被释放

### 时空解耦架构（新增）

- [ ] `SentenceSegment` 包含 `source`、`warning_type` 等字段
- [ ] 伪对齐算法正确生成字级时间戳
- [ ] 智能进度系统根据预设动态调整权重
- [ ] SSE 事件 Tag 统一规范
- [ ] 阈值配置体系完整

---

## 十一、下一步

完成 Phase 1（修订版 v2.0）后，进入 [Phase 2: 智能熔断集成](./02_Phase2_智能熔断集成_修订版.md)
