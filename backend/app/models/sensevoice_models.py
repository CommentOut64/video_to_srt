"""
SenseVoice 数据模型定义（时空解耦版 v2.0）

核心设计：
- SenseVoice 为时间领主，提供绝对时间轴基准
- Whisper 为听觉补丁，仅提供文本
- LLM 为逻辑胶水，校对/翻译
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


@dataclass
class SenseVoiceConfig:
    """SenseVoice 配置"""
    model_dir: str = "iic/SenseVoiceSmall"  # ModelScope 模型ID
    batch_size: int = 1
    quantize: bool = True  # 使用量化模型（INT8）
    device: str = "auto"  # V3.5: auto/cuda/cpu，auto时GPU优先，无GPU降级CPU
    use_itn: bool = True  # 是否使用逆文本正则化
    language: str = "auto"  # auto, zh, en, yue, ja, ko, nospeech
    ban_emo_unk: bool = False  # 是否禁用未知情感标签


class TextSource(Enum):
    """文本来源"""
    SENSEVOICE = "sensevoice"            # SenseVoice 原始输出
    WHISPER_PATCH = "whisper_patch"      # Whisper 补刀替换
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
    # 警告字段
    warning_type: WarningType = field(default=WarningType.NONE)
    perplexity: Optional[float] = None   # LLM 校对时的困惑度

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "is_pseudo": self.is_pseudo,
            "warning_type": self.warning_type.value,
            "perplexity": self.perplexity
        }


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

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "text": self.text,
            "text_clean": self.text_clean,
            "confidence": self.confidence,
            "words": [w.to_dict() for w in self.words],
            "start": self.start,
            "end": self.end,
            "language": self.language,
            "emotion": self.emotion,
            "event": self.event
        }


@dataclass
class SentenceSegment:
    """句级字幕段（时空解耦版）"""
    text: str  # 原始文本（包含标签和连字符）
    text_clean: str = ""  # 清洗后的文本（用于展示）
    start: float = 0.0
    end: float = 0.0
    words: List[WordTimestamp] = field(default_factory=list)
    confidence: float = 1.0

    # 时空解耦字段
    source: TextSource = field(default=TextSource.SENSEVOICE)   # 文本来源
    is_modified: bool = False                     # 是否被修改过
    original_text: Optional[str] = None           # 修改前的原始文本
    whisper_alternative: Optional[str] = None     # Whisper 备选文本

    # 警告与校对字段
    warning_type: WarningType = field(default=WarningType.NONE)  # 句级警告类型
    perplexity: Optional[float] = None            # LLM 校对困惑度
    translation: Optional[str] = None             # 翻译结果
    translation_confidence: Optional[float] = None  # 翻译置信度

    # Layer 2: 语义分组字段 (Layer 1 预留)
    group_id: Optional[str] = None                # 语义组ID，同组的句子属于同一个完整语句
    is_soft_break: bool = False                   # 是否为软断点（物理断但语义连续）
    group_position: Optional[str] = None          # 在组内的位置: 'start', 'middle', 'end', 'single'

    # Phase 4: 双流对齐字段
    alignment_score: Optional[float] = None       # 对齐质量分数 (0-1)
    matched_ratio: Optional[float] = None         # 匹配比例 (0-1)
    is_draft: bool = False                        # 是否为草稿（快流）
    is_finalized: bool = False                    # 是否已定稿（慢流完成）
    sv_original_text: Optional[str] = None        # SenseVoice 原始文本（用于对比）
    whisper_text: Optional[str] = None            # Whisper 识别文本（用于对比）

    def mark_as_modified(self, new_text: str, source: TextSource):
        """标记为已修改"""
        if not self.is_modified:
            self.original_text = self.text
        self.text = new_text
        self.text_clean = new_text  # 同时更新清洗后的文本
        self.source = source
        self.is_modified = True

    def compute_warning_type(self, confidence_threshold: float = 0.6, perplexity_threshold: float = 50.0) -> WarningType:
        """根据置信度和困惑度计算警告类型"""
        has_low_confidence = self.confidence < confidence_threshold
        has_high_perplexity = self.perplexity is not None and self.perplexity > perplexity_threshold

        if has_low_confidence and has_high_perplexity:
            return WarningType.BOTH
        elif has_low_confidence:
            return WarningType.LOW_TRANSCRIPTION_CONFIDENCE
        elif has_high_perplexity:
            return WarningType.HIGH_PROOFREAD_PERPLEXITY
        return WarningType.NONE

    def update_warning_type(self, confidence_threshold: float = 0.6, perplexity_threshold: float = 50.0):
        """更新警告类型"""
        self.warning_type = self.compute_warning_type(confidence_threshold, perplexity_threshold)

    def to_dict(self) -> Dict:
        """转换为字典格式（用于 SSE 推送）"""
        return {
            "text": self.text_clean or self.text,  # 优先使用清洗后的文本
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "source": self.source.value,
            "is_modified": self.is_modified,
            "original_text": self.original_text,
            "whisper_alternative": self.whisper_alternative,
            "warning_type": self.warning_type.value,
            "perplexity": self.perplexity,
            "translation": self.translation,
            "translation_confidence": self.translation_confidence,
            "words": [w.to_dict() for w in self.words],
            # Layer 2: 语义分组相关字段
            "group_id": self.group_id,
            "is_soft_break": self.is_soft_break,
            "group_position": self.group_position
        }


@dataclass
class TranscriptionOutput:
    """转录输出结果"""
    sentences: List[SentenceSegment] = field(default_factory=list)
    language: str = "auto"
    duration: float = 0.0
    engine: str = "sensevoice"

    # 统计信息
    total_sentences: int = 0
    modified_sentences: int = 0
    low_confidence_sentences: int = 0

    def add_sentence(self, sentence: SentenceSegment):
        """添加句子"""
        self.sentences.append(sentence)
        self.total_sentences += 1
        if sentence.is_modified:
            self.modified_sentences += 1
        if sentence.confidence < 0.6:
            self.low_confidence_sentences += 1

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "sentences": [s.to_dict() for s in self.sentences],
            "language": self.language,
            "duration": self.duration,
            "engine": self.engine,
            "stats": {
                "total_sentences": self.total_sentences,
                "modified_sentences": self.modified_sentences,
                "low_confidence_sentences": self.low_confidence_sentences
            }
        }
