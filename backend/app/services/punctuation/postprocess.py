"""
统一标点后处理模块（快流/双流共用）。
V3.2.0+dev.20260130.01
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.services.punctuation.base import PuncPosition, WordTimestampLike, apply_punctuation
from app.services.punctuation.config import get_punctuation_config


Mode = str


@dataclass
class PunctuationPostprocessConfig:
    """后处理配置（按模式加载）。"""

    candidate_min_conf: float
    add_mid_conf: float
    add_end_conf: float
    keep_raw_mid_conf: float
    keep_raw_end_conf: float
    drop_raw_mid_conf: float
    drop_raw_end_conf: float
    question_gate_min_conf: float
    pause_end_min_sec: float
    candidate_window_words: int
    conflict_window_chars: int
    max_repeat_punct: int
    allowed_punct_zh: str
    allowed_punct_en: str


@dataclass
class Decision:
    """后处理决策日志。"""

    char_index: int
    punctuation: str
    action: str
    reason: str
    raw_conf: Optional[float]
    cand_conf: Optional[float]


@dataclass
class PunctuationPostprocessResult:
    """后处理输出。"""

    final_text: str
    final_positions: List[PuncPosition]
    decision_log: List[Decision]
    metrics: Dict[str, int]


@dataclass
class _WordSpan:
    index: int
    start: int
    end: int


@dataclass
class _RawMark:
    word_index: int
    char_index: int
    punctuation: str
    raw_conf: float


@dataclass
class _MappedCandidate:
    word_index: int
    char_index: int
    punctuation: str
    confidence: float


_PUNCTUATION_SET = set(",.!?;:'\"()[]{}，。！？；：、（）【】《》“”‘’「」『』")
_LEFT_PUNCT = set("([{“‘（【《「『")
_SENTENCE_END = set("。！？.!?")
_QUESTION_PUNCT = {"?", "？"}

_DEFAULT_POSTPROCESS = {
    "fast": {
        "candidate_min_conf": 0.30,
        "add_mid_conf": 0.70,
        "add_end_conf": 0.75,
        "keep_raw_mid_conf": 0.60,
        "keep_raw_end_conf": 0.70,
        "drop_raw_mid_conf": 0.35,
        "drop_raw_end_conf": 0.50,
        "question_gate_min_conf": 0.85,
        "pause_end_min_sec": 0.35,
    },
    "dual": {
        "candidate_min_conf": 0.25,
        "add_mid_conf": 0.60,
        "add_end_conf": 0.65,
        "keep_raw_mid_conf": 0.55,
        "keep_raw_end_conf": 0.60,
        "drop_raw_mid_conf": 0.30,
        "drop_raw_end_conf": 0.40,
        "question_gate_min_conf": 0.75,
        "pause_end_min_sec": 0.25,
    },
    "shared": {
        "candidate_window_words": 1,
        "conflict_window_chars": 1,
        "max_repeat_punct": 1,
        "allowed_punct_zh": "。？！；：，、",
        "allowed_punct_en": ".!?,;:",
    },
}


def get_postprocess_config(mode: Mode, config: Optional[Dict[str, Any]] = None) -> PunctuationPostprocessConfig:
    """按模式获取后处理配置。"""
    mode_key = "dual" if str(mode).lower() == "dual" else "fast"
    data = config or get_punctuation_config()
    post_cfg = data.get("postprocess", {}) if isinstance(data, dict) else {}
    shared = dict(_DEFAULT_POSTPROCESS["shared"])
    shared.update(post_cfg.get("shared", {}) if isinstance(post_cfg, dict) else {})
    mode_cfg = dict(_DEFAULT_POSTPROCESS[mode_key])
    mode_cfg.update(post_cfg.get(mode_key, {}) if isinstance(post_cfg, dict) else {})
    return PunctuationPostprocessConfig(
        candidate_min_conf=float(mode_cfg["candidate_min_conf"]),
        add_mid_conf=float(mode_cfg["add_mid_conf"]),
        add_end_conf=float(mode_cfg["add_end_conf"]),
        keep_raw_mid_conf=float(mode_cfg["keep_raw_mid_conf"]),
        keep_raw_end_conf=float(mode_cfg["keep_raw_end_conf"]),
        drop_raw_mid_conf=float(mode_cfg["drop_raw_mid_conf"]),
        drop_raw_end_conf=float(mode_cfg["drop_raw_end_conf"]),
        question_gate_min_conf=float(mode_cfg["question_gate_min_conf"]),
        pause_end_min_sec=float(mode_cfg["pause_end_min_sec"]),
        candidate_window_words=int(shared["candidate_window_words"]),
        conflict_window_chars=int(shared["conflict_window_chars"]),
        max_repeat_punct=int(shared["max_repeat_punct"]),
        allowed_punct_zh=str(shared["allowed_punct_zh"]),
        allowed_punct_en=str(shared["allowed_punct_en"]),
    )


def build_clean_text(raw_text: str) -> Tuple[str, List[int], List[Optional[int]]]:
    """去除标点，返回 clean_text 与索引映射。"""
    clean_chars: List[str] = []
    clean_to_raw: List[int] = []
    raw_to_clean: List[Optional[int]] = []
    clean_idx = 0
    for idx, char in enumerate(raw_text):
        if char in _PUNCTUATION_SET:
            raw_to_clean.append(None)
            continue
        clean_chars.append(char)
        clean_to_raw.append(idx)
        raw_to_clean.append(clean_idx)
        clean_idx += 1
    return "".join(clean_chars), clean_to_raw, raw_to_clean


def postprocess_punctuation(
    raw_text: str,
    words: Sequence[WordTimestampLike],
    language: str,
    mode: Mode,
    candidates: Optional[Sequence[PuncPosition]],
    config: PunctuationPostprocessConfig,
) -> PunctuationPostprocessResult:
    """统一后处理入口。"""
    if not raw_text:
        return PunctuationPostprocessResult(
            final_text="",
            final_positions=[],
            decision_log=[],
            metrics=_build_metrics([]),
        )

    clean_text, clean_to_raw, raw_to_clean = build_clean_text(raw_text)
    if not clean_text:
        return PunctuationPostprocessResult(
            final_text=raw_text,
            final_positions=[],
            decision_log=[],
            metrics=_build_metrics([]),
        )

    filtered_candidates = _filter_candidates(candidates or [], language, config)
    spans = _build_word_spans(clean_text, words)
    clean_to_word = _build_clean_to_word_index(len(clean_text), spans)

    candidate_map, mapped_candidates = _map_candidates(
        filtered_candidates,
        clean_to_word,
        len(clean_text),
    )
    raw_marks = _extract_raw_marks(
        raw_text,
        raw_to_clean,
        clean_to_word,
        words,
    )

    decisions = _rule_engine(
        raw_marks=raw_marks,
        candidate_map=candidate_map,
        all_candidates=mapped_candidates,
        words=words,
        language=language,
        mode=mode,
        config=config,
        clean_text_length=len(clean_text),
    )

    final_positions = _merge_positions(decisions)
    final_positions = _normalize_positions(final_positions, config, language)
    final_text = apply_punctuation(clean_text, final_positions)
    metrics = _build_metrics(decisions)
    return PunctuationPostprocessResult(
        final_text=final_text,
        final_positions=final_positions,
        decision_log=decisions,
        metrics=metrics,
    )


def _filter_candidates(
    candidates: Sequence[PuncPosition],
    language: str,
    config: PunctuationPostprocessConfig,
) -> List[PuncPosition]:
    allowed = config.allowed_punct_zh if language in {"zh", "yue"} else config.allowed_punct_en
    allowed_set = set(allowed)
    result: List[PuncPosition] = []
    for cand in candidates:
        if cand.confidence < config.candidate_min_conf:
            continue
        if cand.punctuation not in allowed_set:
            continue
        result.append(cand)
    return result


def _build_word_spans(
    clean_text: str,
    words: Sequence[WordTimestampLike],
) -> List[_WordSpan]:
    spans: List[_WordSpan] = []
    cursor = 0
    text_len = len(clean_text)
    for idx, word in enumerate(words):
        token = _get_word_text(word)
        if not token:
            continue
        match_idx = clean_text.find(token, cursor)
        if match_idx == -1:
            match_idx = _fallback_match(clean_text, token, cursor)
        if match_idx == -1:
            match_idx = min(cursor, max(text_len - 1, 0))
        start = max(match_idx, 0)
        end = min(match_idx + len(token) - 1, max(text_len - 1, 0))
        spans.append(_WordSpan(index=idx, start=start, end=end))
        cursor = min(end + 1, text_len)
    return spans


def _fallback_match(text: str, token: str, cursor: int) -> int:
    if not token:
        return -1
    if token.isspace():
        for idx in range(cursor, len(text)):
            if text[idx].isspace():
                return idx
        return -1
    if cursor < len(text) and text[cursor: cursor + len(token)] == token:
        return cursor
    for idx in range(cursor, len(text)):
        if text[idx: idx + len(token)] == token:
            return idx
    return -1


def _build_clean_to_word_index(length: int, spans: List[_WordSpan]) -> List[Optional[int]]:
    mapping: List[Optional[int]] = [None] * length
    for span in spans:
        for idx in range(span.start, span.end + 1):
            if 0 <= idx < length:
                mapping[idx] = span.index
    return mapping


def _map_candidates(
    candidates: Sequence[PuncPosition],
    clean_to_word: Sequence[Optional[int]],
    clean_length: int,
) -> Tuple[Dict[int, _MappedCandidate], List[_MappedCandidate]]:
    candidate_map: Dict[int, _MappedCandidate] = {}
    mapped: List[_MappedCandidate] = []
    for cand in candidates:
        char_index = _clamp_char_index(cand.char_index, clean_length)
        word_index = _find_nearest_word_index(clean_to_word, char_index)
        if word_index is None:
            continue
        mapped_candidate = _MappedCandidate(
            word_index=word_index,
            char_index=char_index,
            punctuation=cand.punctuation,
            confidence=cand.confidence,
        )
        mapped.append(mapped_candidate)
        existing = candidate_map.get(word_index)
        if existing is None or existing.confidence < mapped_candidate.confidence:
            candidate_map[word_index] = mapped_candidate
    return candidate_map, mapped


def _extract_raw_marks(
    raw_text: str,
    raw_to_clean: Sequence[Optional[int]],
    clean_to_word: Sequence[Optional[int]],
    words: Sequence[WordTimestampLike],
) -> List[_RawMark]:
    marks: List[_RawMark] = []
    for idx, char in enumerate(raw_text):
        if char not in _PUNCTUATION_SET:
            continue
        clean_idx = _map_raw_punct_to_clean_index(idx, raw_text, raw_to_clean, char)
        if clean_idx is None:
            continue
        word_index = _find_nearest_word_index(clean_to_word, clean_idx)
        if word_index is None:
            continue
        raw_conf = _get_word_confidence(words[word_index])
        marks.append(
            _RawMark(
                word_index=word_index,
                char_index=clean_idx,
                punctuation=char,
                raw_conf=raw_conf,
            )
        )
    return marks


def _map_raw_punct_to_clean_index(
    raw_index: int,
    raw_text: str,
    raw_to_clean: Sequence[Optional[int]],
    punct: str,
) -> Optional[int]:
    if punct in _LEFT_PUNCT:
        for idx in range(raw_index + 1, len(raw_text)):
            mapped = raw_to_clean[idx]
            if mapped is not None:
                return mapped
        return None
    for idx in range(raw_index - 1, -1, -1):
        mapped = raw_to_clean[idx]
        if mapped is not None:
            return mapped
    return None


def _rule_engine(
    *,
    raw_marks: Sequence[_RawMark],
    candidate_map: Dict[int, _MappedCandidate],
    all_candidates: Sequence[_MappedCandidate],
    words: Sequence[WordTimestampLike],
    language: str,
    mode: Mode,
    config: PunctuationPostprocessConfig,
    clean_text_length: int,
) -> List[Decision]:
    decisions: List[Decision] = []
    used_candidates: Dict[Tuple[int, str], float] = {}

    for mark in raw_marks:
        cand = candidate_map.get(mark.word_index)
        cand_near = _find_candidate_in_window(candidate_map, mark.word_index, config.candidate_window_words)
        candidate_missing = cand_near is None
        question_ok = _pass_question_gate(
            words=words,
            word_index=mark.word_index,
            language=language,
            punctuation=mark.punctuation,
            raw_conf=mark.raw_conf,
            cand_conf=cand.confidence if cand else None,
            config=config,
        )

        if (
            mark.raw_conf < _drop_threshold(mark.punctuation, config)
            and candidate_missing
            and not question_ok
        ):
            decisions.append(
                Decision(
                    char_index=mark.char_index,
                    punctuation=mark.punctuation,
                    action="drop_raw",
                    reason="drop_low_conf_no_candidate",
                    raw_conf=mark.raw_conf,
                    cand_conf=None,
                )
            )
            continue

        if cand and _is_conflict(mark, cand, config) and _candidate_is_valid(
            cand=cand,
            words=words,
            language=language,
            config=config,
        ):
            if mark.raw_conf < _keep_threshold(mark.punctuation, config):
                decisions.append(
                    Decision(
                        char_index=cand.char_index,
                        punctuation=cand.punctuation,
                        action="add_candidate",
                        reason="replace_by_candidate",
                        raw_conf=mark.raw_conf,
                        cand_conf=cand.confidence,
                    )
                )
                used_candidates[(cand.char_index, cand.punctuation)] = cand.confidence
                continue

        decisions.append(
            Decision(
                char_index=mark.char_index,
                punctuation=mark.punctuation,
                action="keep_raw",
                reason="keep_raw",
                raw_conf=mark.raw_conf,
                cand_conf=cand.confidence if cand else None,
            )
        )

    for cand in all_candidates:
        key = (cand.char_index, cand.punctuation)
        if key in used_candidates:
            continue
        if not _candidate_is_valid(cand, words, language, config):
            continue
        if cand.char_index < 0 or cand.char_index >= clean_text_length:
            continue
        decisions.append(
            Decision(
                char_index=cand.char_index,
                punctuation=cand.punctuation,
                action="add_candidate",
                reason="add_candidate",
                raw_conf=None,
                cand_conf=cand.confidence,
            )
        )
        used_candidates[key] = cand.confidence

    return decisions


def _merge_positions(decisions: Sequence[Decision]) -> List[PuncPosition]:
    merged: Dict[Tuple[int, str], float] = {}
    for decision in decisions:
        if decision.action == "drop_raw":
            continue
        key = (decision.char_index, decision.punctuation)
        confidence = decision.cand_conf if decision.action == "add_candidate" else decision.raw_conf
        merged[key] = max(merged.get(key, 0.0), float(confidence or 1.0))
    positions = [
        PuncPosition(char_index=idx, punctuation=punct, confidence=conf)
        for (idx, punct), conf in merged.items()
    ]
    positions.sort(key=lambda item: item.char_index)
    return positions


def _normalize_positions(
    positions: Sequence[PuncPosition],
    config: PunctuationPostprocessConfig,
    language: str,
) -> List[PuncPosition]:
    if not positions:
        return []
    allowed = config.allowed_punct_zh if language in {"zh", "yue"} else config.allowed_punct_en
    allowed_set = set(allowed)
    normalized: List[PuncPosition] = []
    for pos in positions:
        if pos.punctuation not in allowed_set:
            continue
        if not normalized:
            normalized.append(pos)
            continue
        prev = normalized[-1]
        if abs(pos.char_index - prev.char_index) <= config.conflict_window_chars:
            if pos.punctuation == prev.punctuation:
                if pos.confidence > prev.confidence:
                    normalized[-1] = pos
            else:
                if pos.confidence > prev.confidence:
                    normalized[-1] = pos
            continue
        normalized.append(pos)

    return _limit_repeat_punctuation(normalized, config.max_repeat_punct)


def _limit_repeat_punctuation(
    positions: Sequence[PuncPosition],
    max_repeat: int,
) -> List[PuncPosition]:
    if max_repeat <= 0:
        return []
    output: List[PuncPosition] = []
    last_key: Optional[Tuple[int, str]] = None
    repeat_count = 0
    for pos in positions:
        key = (pos.char_index, pos.punctuation)
        if key == last_key:
            repeat_count += 1
        else:
            repeat_count = 1
        last_key = key
        if repeat_count <= max_repeat:
            output.append(pos)
    return output


def _build_metrics(decisions: Sequence[Decision]) -> Dict[str, int]:
    metrics = {
        "keep_raw": 0,
        "add_candidate": 0,
        "drop_raw": 0,
    }
    for decision in decisions:
        if decision.action in metrics:
            metrics[decision.action] += 1
    return metrics


def _candidate_is_valid(
    cand: _MappedCandidate,
    words: Sequence[WordTimestampLike],
    language: str,
    config: PunctuationPostprocessConfig,
) -> bool:
    if cand.confidence < _add_threshold(cand.punctuation, config):
        return False
    if cand.punctuation in _QUESTION_PUNCT:
        if not _pass_question_gate(
            words=words,
            word_index=cand.word_index,
            language=language,
            punctuation=cand.punctuation,
            raw_conf=None,
            cand_conf=cand.confidence,
            config=config,
        ):
            return False
    if cand.punctuation in _SENTENCE_END:
        if not _pause_ok(words, cand.word_index, config.pause_end_min_sec):
            return False
    return True


def _pass_question_gate(
    *,
    words: Sequence[WordTimestampLike],
    word_index: int,
    language: str,
    punctuation: str,
    raw_conf: Optional[float],
    cand_conf: Optional[float],
    config: PunctuationPostprocessConfig,
) -> bool:
    if punctuation not in _QUESTION_PUNCT:
        return True
    if raw_conf is not None and raw_conf >= config.question_gate_min_conf:
        return True
    if cand_conf is not None and cand_conf >= config.question_gate_min_conf:
        return True
    if _has_question_word(words, word_index, language):
        return True
    if _pause_ok(words, word_index, config.pause_end_min_sec):
        return True
    return False


def _has_question_word(
    words: Sequence[WordTimestampLike],
    word_index: int,
    language: str,
) -> bool:
    if word_index < 0 or word_index >= len(words):
        return False
    token = _get_word_text(words[word_index]).lower()
    prev_token = _get_word_text(words[word_index - 1]).lower() if word_index > 0 else ""
    if language in {"zh", "yue"}:
        markers = {"吗", "么", "呢", "嘛", "吧", "是否", "为什么", "怎么", "如何", "哪", "谁", "什么", "多少", "几"}
        return any(mark in token for mark in markers) or any(mark in prev_token for mark in markers)
    markers = {
        "what", "why", "how", "where", "who", "when", "which",
        "do", "does", "did", "is", "are", "am", "can", "could", "would", "should", "will",
    }
    return token in markers or prev_token in markers


def _pause_ok(words: Sequence[WordTimestampLike], word_index: int, threshold: float) -> bool:
    if word_index < 0 or word_index >= len(words):
        return True
    if word_index >= len(words) - 1:
        return True
    end_time = _get_word_end(words[word_index])
    next_start = _get_word_start(words[word_index + 1])
    if end_time is None or next_start is None:
        return True
    return (next_start - end_time) >= threshold


def _find_candidate_in_window(
    candidate_map: Dict[int, _MappedCandidate],
    word_index: int,
    window: int,
) -> Optional[_MappedCandidate]:
    for idx in range(word_index - window, word_index + window + 1):
        cand = candidate_map.get(idx)
        if cand:
            return cand
    return None


def _is_conflict(mark: _RawMark, cand: _MappedCandidate, config: PunctuationPostprocessConfig) -> bool:
    if cand.punctuation == mark.punctuation:
        return False
    return abs(cand.char_index - mark.char_index) <= config.conflict_window_chars


def _add_threshold(punct: str, config: PunctuationPostprocessConfig) -> float:
    return config.add_end_conf if punct in _SENTENCE_END else config.add_mid_conf


def _keep_threshold(punct: str, config: PunctuationPostprocessConfig) -> float:
    return config.keep_raw_end_conf if punct in _SENTENCE_END else config.keep_raw_mid_conf


def _drop_threshold(punct: str, config: PunctuationPostprocessConfig) -> float:
    return config.drop_raw_end_conf if punct in _SENTENCE_END else config.drop_raw_mid_conf


def _find_nearest_word_index(
    clean_to_word: Sequence[Optional[int]],
    char_index: int,
) -> Optional[int]:
    if not clean_to_word:
        return None
    if 0 <= char_index < len(clean_to_word):
        current = clean_to_word[char_index]
        if current is not None:
            return current
    left = char_index - 1
    right = char_index + 1
    while left >= 0 or right < len(clean_to_word):
        if left >= 0:
            current = clean_to_word[left]
            if current is not None:
                return current
            left -= 1
        if right < len(clean_to_word):
            current = clean_to_word[right]
            if current is not None:
                return current
            right += 1
    return None


def _clamp_char_index(char_index: int, length: int) -> int:
    if length <= 0:
        return 0
    return max(0, min(char_index, length - 1))


def _get_word_text(word: WordTimestampLike) -> str:
    if isinstance(word, dict):
        return str(word.get("word", "") or "")
    return str(getattr(word, "word", "") or "")


def _get_word_confidence(word: WordTimestampLike) -> float:
    if isinstance(word, dict):
        return float(word.get("confidence", 1.0) or 1.0)
    confidence = getattr(word, "confidence", None)
    return float(confidence) if confidence is not None else 1.0


def _get_word_start(word: WordTimestampLike) -> Optional[float]:
    if isinstance(word, dict):
        return word.get("start")
    return getattr(word, "start", None)


def _get_word_end(word: WordTimestampLike) -> Optional[float]:
    if isinstance(word, dict):
        return word.get("end")
    return getattr(word, "end", None)
