"""
对齐/规范化共享类型定义。
V3.2.0+dev.20260214.07
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING, Tuple

from app.models.confidence_models import AlignedWord as AlignedWord
from app.models.sensevoice_models import SentenceSegment, WordTimestamp
from app.services.alignment.gap_resolver import GapResolution
from app.services.punctuation.base import PuncPosition, WordTimestampLike

if TYPE_CHECKING:
    from app.services.arbitration.arbiter import ArbitrationResult
    from app.services.language_policy.types import LanguagePolicySnapshot
    from app.services.segmentation.soft_cut.types import CutPlan


@dataclass
class CharMapping:
    """字符映射信息（raw -> clean）。"""

    raw_idx: int
    clean_idx: Optional[int]
    punct: Optional[str] = None


@dataclass
class NormalizationResult:
    """统一规范化输出（分轨文本）。"""

    text_itn_raw: str
    text_clean: str
    char_mapping: List[CharMapping]
    raw_to_clean: List[Optional[int]]
    clean_to_raw: List[int]
    itn_fallback: bool = False
    itn_fallback_reason: Optional[str] = None
    mapping_coverage: float = 0.0


@dataclass
class TextTrack:
    """统一文本轨道（单轨）。"""

    raw_text: str
    text_itn_raw: str
    text_clean: str
    char_mapping: List[CharMapping]
    raw_to_clean: List[Optional[int]]
    clean_to_raw: List[int]
    language: str = "auto"
    source: str = ""
    itn_fallback: bool = False
    itn_fallback_reason: Optional[str] = None
    clean_to_word: List[Optional[int]] = field(default_factory=list)
    # V3.2.0+dev.20260205.09: 集合层对齐支持词级置信度来源透传。
    word_confidences: List[Optional[float]] = field(default_factory=list)
    punct_positions: List[PuncPosition] = field(default_factory=list)
    mapping_coverage: float = 0.0


@dataclass
class PunctSource:
    """标点候选来源（标点前置域内部使用）。"""

    clean_text_ref: str
    positions: List[PuncPosition]
    source: str
    confidence: float = 0.0
    model_id: str = ""


@dataclass
class PunctTrack:
    """标点轨道（标点前置域输出）。"""

    clean_text_ref: str
    positions: List[PuncPosition]
    source: str = ""
    confidence_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextTrackBundle:
    """三轨文本结构（sv/whisper/chosen）。"""

    sv_track: Optional[TextTrack] = None
    whisper_track: Optional[TextTrack] = None
    chosen_track: Optional[TextTrack] = None


@dataclass
class L1Input:
    """L1 规范化输入。"""

    sv_raw_result: Optional[Dict[str, Any]]
    wh_raw_result: Optional[Dict[str, Any]]
    language_hint: Optional[str] = None


@dataclass
class L1Output:
    """L1 规范化输出。"""

    sv_track: Optional[TextTrack] = None
    whisper_track: Optional[TextTrack] = None


@dataclass
class QualitySignals:
    """L2 仲裁质量信号（统一口径）。"""

    length_ratio: float = 1.0
    confidence_fast: float = 0.0
    confidence_slow: float = 0.0
    is_repetition: bool = False
    is_hallucination: bool = False
    is_itn_fallback: bool = False
    mapping_coverage: float = 0.0
    alignment_score: float = 0.0
    gap_ratio: float = 0.0

    @property
    def repetition_flag(self) -> bool:
        """兼容文档口径：repetition_flag。"""
        return self.is_repetition

    @property
    def hallucination_flag(self) -> bool:
        """兼容文档口径：hallucination_flag。"""
        return self.is_hallucination

    @property
    def itn_fallback_flag(self) -> bool:
        """兼容文档口径：itn_fallback_flag。"""
        return self.is_itn_fallback


@dataclass
class L2Input:
    """L2 文本仲裁输入。"""

    sv_track: Optional[TextTrack]
    whisper_track: Optional[TextTrack]
    quality_signals: QualitySignals


@dataclass
class L2Output:
    """L2 文本仲裁输出。"""

    chosen_text_track: Optional[TextTrack]
    arbitration_result: "ArbitrationResult"


@dataclass
class PunctuationPreInput:
    """标点前置域输入。"""

    chosen_text_track: Optional[TextTrack]
    sv_punct_source: Optional[PunctSource] = None
    wh_punct_source: Optional[PunctSource] = None
    word_timestamps: Optional[Sequence[WordTimestampLike]] = None


@dataclass
class PunctuationPreOutput:
    """标点前置域输出。"""

    punct_track: PunctTrack


@dataclass
class AlignmentResult:
    """集合层对齐输出结果。"""

    aligned_words: List[AlignedWord]
    alignment_score: float
    gap_ratio: float
    gap_positions: List[int]
    resolution: Optional[GapResolution] = None
    coverage: float = 0.0


@dataclass
class CollectionLayerInput:
    """集合层输入。"""

    chosen_text_track: Optional[TextTrack]
    sv_words: List[WordTimestamp]
    vad_intervals: Optional[List[Tuple[float, float]]] = None
    policy_snapshot: Optional["LanguagePolicySnapshot"] = None
    is_fast_only_mode: bool = False


@dataclass
class CollectionLayerOutput:
    """集合层输出。"""

    alignment_result: AlignmentResult


@dataclass
class AnnotatedWord:
    """语义注入后的词信息（预留）。"""

    word: str
    start: Optional[float]
    end: Optional[float]
    trailing_punct: str = ""
    confidence: Optional[float] = None
    confidence_source: Optional[str] = None
    is_pseudo: bool = False
    speaker_id: Optional[str] = None
    turn_id: Optional[str] = None
    track_id: Optional[str] = None


@dataclass
class ScoringLayerInput:
    """评分层输入。"""

    alignment_result: AlignmentResult
    punct_track: PunctTrack
    language: str = "auto"
    speaker_id: Optional[str] = None  # V3.2.0+dev.20260207.03: P0 speaker 信号链
    turn_id: Optional[str] = None  # V3.2.0+dev.20260210.09: Phase 2 turn 信号链
    policy_snapshot: Optional["LanguagePolicySnapshot"] = None


@dataclass
class ScoringLayerOutput:
    """评分层输出。"""

    annotated_words: List[AnnotatedWord]
    injection_report: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlignedFacts:
    """集合层事实契约（M2 预埋）。"""

    annotated_words: List[AnnotatedWord] = field(default_factory=list)
    alignment_score: float = 0.0
    gap_ratio: float = 0.0
    gap_positions: List[int] = field(default_factory=list)
    speaker_turns: List[Dict[str, Any]] = field(default_factory=list)
    fast_draft_cuts: List[float] = field(default_factory=list)
    time_axis_version: str = "m1_legacy"
    time_mappings: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FusedEvidence:
    """评分层统一证据契约（M2 预埋）。"""

    speaker_changes: List[Dict[str, Any]] = field(default_factory=list)
    pause_anchors: List[Dict[str, Any]] = field(default_factory=list)
    semantic_anchors: List[Dict[str, Any]] = field(default_factory=list)
    punctuation_anchors: List[Dict[str, Any]] = field(default_factory=list)
    evidence_report: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputTrace:
    """输出层追踪契约（句级切分原因 + 时间映射）。"""

    sentence_index: int
    split_reason: str = ""
    split_risk: str = ""
    window_id: str = ""
    pyannote_frame_time: Optional[float] = None
    mapped_cut_time: Optional[float] = None
    mapping_quality: str = ""
    mapping_reason: str = ""
    sentence_start: Optional[float] = None
    sentence_end: Optional[float] = None


@dataclass
class DecisionLayerInput:
    """裁决层输入。"""

    annotated_words: List[AnnotatedWord]
    vad_intervals: Optional[List[Tuple[float, float]]] = None
    cut_plan: Optional["CutPlan"] = None
    aligned_facts: Optional[AlignedFacts] = None
    fused_evidence: Optional[FusedEvidence] = None
    fallback_clean_text_ref: str = ""
    fallback_punctuation_positions: List[PuncPosition] = field(default_factory=list)
    policy_snapshot: Optional["LanguagePolicySnapshot"] = None


@dataclass
class DecisionLayerOutput:
    """裁决层输出。"""

    sentence_segments: List[SentenceSegment]
    words_for_split: List[WordTimestamp]
    segmentation_report: Dict[str, Any] = field(default_factory=dict)
    applied_cut_plan: Optional["CutPlan"] = None
    output_traces: List[OutputTrace] = field(default_factory=list)


@dataclass
class OutputLayerInput:
    """输出层输入。"""

    chunk_index: int | str
    sentence_segments: List[SentenceSegment]
    language: str = "auto"
    injection_report: Optional[Dict[str, Any]] = None
    segmentation_report: Optional[Dict[str, Any]] = None
    output_traces: Optional[List[OutputTrace]] = None


@dataclass
class OutputLayerOutput:
    """输出层输出。"""

    output_payload: Dict[str, Any] = field(default_factory=dict)
    output_traces: List[OutputTrace] = field(default_factory=list)


__all__ = [
    "AlignedWord",
    "CharMapping",
    "L1Input",
    "L1Output",
    "L2Input",
    "L2Output",
    "PunctuationPreInput",
    "PunctuationPreOutput",
    "CollectionLayerInput",
    "CollectionLayerOutput",
    "ScoringLayerInput",
    "ScoringLayerOutput",
    "DecisionLayerInput",
    "DecisionLayerOutput",
    "OutputLayerInput",
    "OutputLayerOutput",
    "NormalizationResult",
    "PunctSource",
    "PunctTrack",
    "QualitySignals",
    "AlignmentResult",
    "TextTrack",
    "TextTrackBundle",
    "AnnotatedWord",
    "AlignedFacts",
    "FusedEvidence",
    "OutputTrace",
]

