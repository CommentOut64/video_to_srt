"""
软切决策契约定义（Phase A）。
V3.2.0+dev.20260220.01
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional


class SpeakerChangeTag(str, Enum):
    """SpeakerChangeEvidence 标签枚举，禁止自由文本。"""

    REBOUND_MERGED = "rebound_merged"
    SUSPECTED_MISSED = "suspected_missed"
    CROSS_CHUNK_DEFERRED = "cross_chunk_deferred"


class EvidenceLevel(str, Enum):
    """说话人变化证据等级。"""

    HIGH = "high"
    MID = "mid"
    LOW = "low"


class AnchorType(str, Enum):
    """锚点类型。"""

    PAUSE_ANCHOR = "pause_anchor"
    WORD_BOUNDARY = "word_boundary"
    SEMANTIC_ANCHOR = "semantic_anchor"
    PUNCTUATION_ANCHOR = "punctuation_anchor"


class SplitEvidenceSource(str, Enum):
    """切分证据来源。"""

    SPEAKER = "speaker"
    PAUSE = "pause"
    PUNCTUATION = "punctuation"
    SEMANTIC = "semantic"
    LLM = "llm"
    FAST_DRAFT = "fast_draft"
    FORCE = "force"


class CutWindowState(str, Enum):
    """待切窗口状态。"""

    OPEN = "open"
    RESOLVED = "resolved"
    FORCED = "forced"
    EXPIRED = "expired"


class DeferredCutState(str, Enum):
    """延迟切分状态。"""

    PENDING = "pending"
    RESOLVED = "resolved"
    FORCED = "forced"
    EXPIRED = "expired"


@dataclass(frozen=True)
class SourcePriorityRule:
    """来源级优先级规则。"""

    enabled: bool = True
    weight: float = 0.5
    min_confidence: float = 0.0
    trigger_threshold: float = 0.0

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "SourcePriorityRule":
        payload = dict(raw or {})
        return cls(
            enabled=bool(payload.get("enabled", True)),
            weight=max(0.0, min(1.0, float(payload.get("weight", 0.5)))),
            min_confidence=max(0.0, min(1.0, float(payload.get("min_confidence", 0.0)))),
            trigger_threshold=max(0.0, min(1.0, float(payload.get("trigger_threshold", 0.0)))),
        )


@dataclass(frozen=True)
class SoftCutPriorityProfile:
    """soft-cut 优先级 profile。"""

    profile_name: str
    tiebreak_order: list[SplitEvidenceSource] = field(default_factory=list)
    merge_window_ms: int = 120
    source_rules: dict[SplitEvidenceSource, SourcePriorityRule] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        *,
        profile_name: str,
        raw: Optional[Mapping[str, Any]],
    ) -> "SoftCutPriorityProfile":
        payload = dict(raw or {})
        raw_order = list(payload.get("tiebreak_order") or [])
        order: list[SplitEvidenceSource] = []
        for item in raw_order:
            try:
                order.append(SplitEvidenceSource(str(item).strip().lower()))
            except ValueError:
                continue
        if not order:
            order = [
                SplitEvidenceSource.SPEAKER,
                SplitEvidenceSource.FAST_DRAFT,
                SplitEvidenceSource.PUNCTUATION,
                SplitEvidenceSource.PAUSE,
                SplitEvidenceSource.SEMANTIC,
                SplitEvidenceSource.LLM,
            ]

        raw_sources = payload.get("source_rules")
        if raw_sources is None:
            raw_sources = payload.get("source")
        source_rules: dict[SplitEvidenceSource, SourcePriorityRule] = {}
        if isinstance(raw_sources, Mapping):
            for key, value in raw_sources.items():
                try:
                    source = SplitEvidenceSource(str(key).strip().lower())
                except ValueError:
                    continue
                source_rules[source] = SourcePriorityRule.from_dict(
                    value if isinstance(value, Mapping) else None
                )

        default_rules = _default_source_rules()
        for source, rule in default_rules.items():
            if source not in source_rules:
                source_rules[source] = rule

        merge_window_ms = int(payload.get("merge_window_ms", 120) or 120)
        merge_window_ms = max(20, min(2000, merge_window_ms))
        return cls(
            profile_name=profile_name,
            tiebreak_order=order,
            merge_window_ms=merge_window_ms,
            source_rules=source_rules,
        )

    def source_rule(self, source: SplitEvidenceSource) -> SourcePriorityRule:
        return self.source_rules.get(source, SourcePriorityRule(enabled=False))

    def source_rank(self, source: SplitEvidenceSource) -> int:
        try:
            return self.tiebreak_order.index(source)
        except ValueError:
            return len(self.tiebreak_order)


@dataclass(frozen=True)
class SplitEvidence:
    """统一切分证据结构。"""

    source: SplitEvidenceSource
    anchor_time: float
    confidence: float
    base_score: float
    final_score: float
    reason: str
    risk: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _default_source_rules() -> dict[SplitEvidenceSource, SourcePriorityRule]:
    return {
        SplitEvidenceSource.SPEAKER: SourcePriorityRule(
            enabled=True,
            weight=0.85,
            min_confidence=0.45,
            trigger_threshold=0.42,
        ),
        SplitEvidenceSource.PAUSE: SourcePriorityRule(
            enabled=True,
            weight=0.55,
            min_confidence=0.35,
            trigger_threshold=0.30,
        ),
        SplitEvidenceSource.PUNCTUATION: SourcePriorityRule(
            enabled=True,
            weight=0.75,
            min_confidence=0.35,
            trigger_threshold=0.28,
        ),
        SplitEvidenceSource.SEMANTIC: SourcePriorityRule(
            enabled=True,
            weight=0.45,
            min_confidence=0.30,
            trigger_threshold=0.24,
        ),
        SplitEvidenceSource.LLM: SourcePriorityRule(
            enabled=False,
            weight=0.0,
            min_confidence=0.0,
            trigger_threshold=1.0,
        ),
        SplitEvidenceSource.FAST_DRAFT: SourcePriorityRule(
            enabled=True,
            weight=0.78,
            min_confidence=0.40,
            trigger_threshold=0.30,
        ),
        SplitEvidenceSource.FORCE: SourcePriorityRule(
            enabled=True,
            weight=1.0,
            min_confidence=0.0,
            trigger_threshold=0.0,
        ),
    }


def default_priority_profiles() -> dict[str, SoftCutPriorityProfile]:
    """默认优先级 profile 集合。"""
    base = _default_source_rules()
    punct_boost = {
        "tiebreak_order": ["speaker", "fast_draft", "punctuation", "pause", "semantic", "llm"],
        "merge_window_ms": 120,
        "source_rules": {
            source.value: {
                "enabled": rule.enabled,
                "weight": rule.weight,
                "min_confidence": rule.min_confidence,
                "trigger_threshold": rule.trigger_threshold,
            }
            for source, rule in base.items()
        },
    }

    llm_ramp = {
        "tiebreak_order": ["speaker", "fast_draft", "llm", "punctuation", "pause", "semantic"],
        "merge_window_ms": 120,
        "source_rules": {
            source.value: {
                "enabled": rule.enabled,
                "weight": rule.weight,
                "min_confidence": rule.min_confidence,
                "trigger_threshold": rule.trigger_threshold,
            }
            for source, rule in base.items()
        },
    }
    llm_ramp["source_rules"]["llm"] = {
        "enabled": True,
        "weight": 0.78,
        "min_confidence": 0.50,
        "trigger_threshold": 0.36,
    }
    llm_ramp["source_rules"]["punctuation"] = {
        "enabled": True,
        "weight": 0.35,
        "min_confidence": 0.35,
        "trigger_threshold": 0.28,
    }

    llm_primary = {
        "tiebreak_order": ["speaker", "fast_draft", "llm", "pause", "semantic", "punctuation"],
        "merge_window_ms": 120,
        "source_rules": {
            source.value: {
                "enabled": rule.enabled,
                "weight": rule.weight,
                "min_confidence": rule.min_confidence,
                "trigger_threshold": rule.trigger_threshold,
            }
            for source, rule in base.items()
        },
    }
    llm_primary["source_rules"]["llm"] = {
        "enabled": True,
        "weight": 0.92,
        "min_confidence": 0.55,
        "trigger_threshold": 0.40,
    }
    llm_primary["source_rules"]["punctuation"] = {
        "enabled": False,
        "weight": 0.0,
        "min_confidence": 1.0,
        "trigger_threshold": 1.0,
    }

    return {
        "punct_boost_transition": SoftCutPriorityProfile.from_dict(
            profile_name="punct_boost_transition",
            raw=punct_boost,
        ),
        "llm_ramp_up": SoftCutPriorityProfile.from_dict(
            profile_name="llm_ramp_up",
            raw=llm_ramp,
        ),
        "llm_primary_no_punct": SoftCutPriorityProfile.from_dict(
            profile_name="llm_primary_no_punct",
            raw=llm_primary,
        ),
    }


def resolve_priority_profile(
    *,
    active_profile: str,
    profile_overrides: Optional[Mapping[str, Any]] = None,
) -> SoftCutPriorityProfile:
    """解析当前激活 profile，并合并覆盖配置。"""
    defaults = default_priority_profiles()
    resolved: dict[str, SoftCutPriorityProfile] = dict(defaults)
    if isinstance(profile_overrides, Mapping):
        for profile_name, raw in profile_overrides.items():
            normalized_name = str(profile_name or "").strip()
            if not normalized_name:
                continue
            mapping = raw if isinstance(raw, Mapping) else None
            base_raw: dict[str, Any] = {}
            if normalized_name in defaults:
                base_raw = {
                    "tiebreak_order": [item.value for item in defaults[normalized_name].tiebreak_order],
                    "merge_window_ms": defaults[normalized_name].merge_window_ms,
                    "source_rules": {
                        source.value: {
                            "enabled": rule.enabled,
                            "weight": rule.weight,
                            "min_confidence": rule.min_confidence,
                            "trigger_threshold": rule.trigger_threshold,
                        }
                        for source, rule in defaults[normalized_name].source_rules.items()
                    },
                }
            if mapping:
                if "tiebreak_order" in mapping:
                    base_raw["tiebreak_order"] = mapping.get("tiebreak_order")
                if "merge_window_ms" in mapping:
                    base_raw["merge_window_ms"] = mapping.get("merge_window_ms")
                if "source_rules" in mapping and isinstance(mapping["source_rules"], Mapping):
                    merged_sources = dict(base_raw.get("source_rules") or {})
                    merged_sources.update(dict(mapping["source_rules"]))
                    base_raw["source_rules"] = merged_sources
                if "source" in mapping and isinstance(mapping["source"], Mapping):
                    merged_sources = dict(base_raw.get("source_rules") or {})
                    merged_sources.update(dict(mapping["source"]))
                    base_raw["source_rules"] = merged_sources
            resolved[normalized_name] = SoftCutPriorityProfile.from_dict(
                profile_name=normalized_name,
                raw=base_raw,
            )

    normalized_active = str(active_profile or "").strip()
    if normalized_active in resolved:
        return resolved[normalized_active]
    return resolved["punct_boost_transition"]


@dataclass
class SpeakerChangeEvidence:
    """说话人变化证据。"""

    time: float
    from_speaker: str
    to_speaker: str
    pyannote_confidence: float
    pause_duration: float
    embedding_distance: float
    embedding_threshold: float
    is_abrupt_energy_shift: bool
    level: EvidenceLevel
    tags: set[SpeakerChangeTag] = field(default_factory=set)


@dataclass
class AnchorScore:
    """锚点评分。"""

    anchor_type: AnchorType
    anchor_time: float
    base_score: float
    distance_penalty: float
    final_score: float
    source: str
    evidence_source: SplitEvidenceSource = SplitEvidenceSource.SPEAKER


@dataclass
class CutWindow:
    """待切窗口。"""

    window_id: str
    trigger_time: float
    trigger_level: EvidenceLevel
    start_time: float
    end_time: float
    chunk_id: str
    trigger_source: SplitEvidenceSource = SplitEvidenceSource.SPEAKER
    trigger_score: float = 0.0
    candidate_anchors: list[AnchorScore] = field(default_factory=list)
    state: CutWindowState = CutWindowState.OPEN


@dataclass
class CutDecision:
    """切分决策。"""

    time: float
    window_id: str
    reason: str
    risk: Optional[str]
    anchor_type: AnchorType
    anchor_score: float
    depends_on_fast_draft: bool
    time_range: tuple[float, float]
    source: str = ""
    # V3.2.0+dev.20260215.10: 时间映射契约（Phase 1 预埋）。
    pyannote_frame_time: Optional[float] = None
    mapped_cut_time: Optional[float] = None
    mapping_quality: str = ""
    mapping_reason: str = ""

    def __post_init__(self) -> None:
        if not self.window_id:
            raise ValueError("CutDecision.window_id 不能为空")

    @property
    def split_reason(self) -> str:
        """兼容口径：split_reason。"""
        return self.reason

    @property
    def split_risk(self) -> str:
        """兼容口径：split_risk。"""
        return str(self.risk or "")

    @property
    def split_source(self) -> str:
        """兼容口径：split_source。"""
        return str(self.source or "")


@dataclass
class DeferredCut:
    """延迟切分。"""

    deferred_id: str
    window_id: str
    created_at: float
    expected_resolve_by: float
    state: DeferredCutState = DeferredCutState.PENDING
    resolution_decision: Optional[CutDecision] = None
    window_start: Optional[float] = None
    window_end: Optional[float] = None
    trigger_level: str = ""
    depends_on_fast_draft: bool = False


@dataclass
class CutPlan:
    """切分计划。"""

    plan_id: str
    block_id: str
    decisions: list[CutDecision] = field(default_factory=list)
    deferred_cuts: list[DeferredCut] = field(default_factory=list)
    generation_report: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "AnchorScore",
    "AnchorType",
    "SourcePriorityRule",
    "SoftCutPriorityProfile",
    "CutDecision",
    "CutPlan",
    "CutWindow",
    "CutWindowState",
    "DeferredCut",
    "DeferredCutState",
    "EvidenceLevel",
    "SplitEvidence",
    "SplitEvidenceSource",
    "SpeakerChangeEvidence",
    "SpeakerChangeTag",
    "default_priority_profiles",
    "resolve_priority_profile",
]
