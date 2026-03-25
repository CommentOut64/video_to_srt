"""
统一标点后处理模块（快流/双流共用）。
V3.2.0+dev.20260205.02
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.services.punctuation.base import PuncPosition, WordTimestampLike, apply_punctuation
from app.services.punctuation.config import get_punctuation_config
from app.services.text_protection import should_skip_raw_punctuation
from app.services.text_pipeline_config import PunctuationRuntimeOverrides


Mode = str


_WEAK_PUNCTUATION_SET = set(",，、;；:：")
_RAW_WEAK_RATIO_THRESHOLD = 0.8


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
    is_comma_guard_enabled: bool
    comma_guard_min_conf: float
    comma_guard_min_chars: int
    comma_guard_min_words: int
    comma_guard_pause_min_sec: float


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


_PUNCTUATION_SET = set(",.!?;:\"()[]{}，。！？；：、（）【】《》“”「」『』")
_LEFT_PUNCT = set("([{“‘（【《「『")
_SENTENCE_END = set("。！？.!?")
_QUESTION_PUNCT = {"?", "？"}
_COMMA_PUNCT = {"，", "、"}
# V3.2.0+dev.20260131.05: 小数点保护（避免 1.4 被当作句号）
_DECIMAL_DOT_CHARS = {".", "。", "．"}


def _is_decimal_dot(text: str, index: int) -> bool:
    if index <= 0 or index >= len(text) - 1:
        return False
    if text[index] not in _DECIMAL_DOT_CHARS:
        return False
    return text[index - 1].isdigit() and text[index + 1].isdigit()

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
        "comma_guard_enabled": False,
        "comma_guard_min_conf": 0.85,
        "comma_guard_min_chars": 6,
        "comma_guard_min_words": 4,
        "comma_guard_pause_min_sec": 0.25,
    },
}


def get_postprocess_config(
    mode: Mode,
    config: Optional[Dict[str, Any]] = None,
    *,
    runtime: Optional[Dict[str, Any]] = None,
) -> PunctuationPostprocessConfig:
    """按模式获取后处理配置。"""
    mode_key = "dual" if str(mode).lower() == "dual" else "fast"
    data = config or get_punctuation_config()
    post_cfg = data.get("postprocess", {}) if isinstance(data, dict) else {}
    shared = dict(_DEFAULT_POSTPROCESS["shared"])
    shared.update(post_cfg.get("shared", {}) if isinstance(post_cfg, dict) else {})
    mode_cfg = dict(_DEFAULT_POSTPROCESS[mode_key])
    mode_cfg.update(post_cfg.get(mode_key, {}) if isinstance(post_cfg, dict) else {})

    # V3.2.0+dev.20260204.09: 运行参数显式覆盖（仅覆盖 override 中出现的键）
    overrides = PunctuationRuntimeOverrides.from_runtime(runtime)
    mode_override = overrides.postprocess_dual if mode_key == "dual" else overrides.postprocess_fast
    for key in (
        "candidate_min_conf",
        "add_mid_conf",
        "add_end_conf",
        "keep_raw_mid_conf",
        "keep_raw_end_conf",
        "drop_raw_mid_conf",
        "drop_raw_end_conf",
        "question_gate_min_conf",
        "pause_end_min_sec",
    ):
        value = getattr(mode_override, key)
        if value is not None:
            mode_cfg[key] = value

    shared_override = overrides.postprocess_shared
    for key in (
        "candidate_window_words",
        "conflict_window_chars",
        "max_repeat_punct",
        "allowed_punct_zh",
        "allowed_punct_en",
        "comma_guard_enabled",
        "comma_guard_min_conf",
        "comma_guard_min_chars",
        "comma_guard_min_words",
        "comma_guard_pause_min_sec",
    ):
        value = getattr(shared_override, key)
        if value is not None:
            shared[key] = value
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
        is_comma_guard_enabled=bool(shared.get("comma_guard_enabled", False)),
        comma_guard_min_conf=float(shared.get("comma_guard_min_conf", 0.85)),
        comma_guard_min_chars=int(shared.get("comma_guard_min_chars", 6)),
        comma_guard_min_words=int(shared.get("comma_guard_min_words", 4)),
        comma_guard_pause_min_sec=float(shared.get("comma_guard_pause_min_sec", 0.25)),
    )


def build_clean_text(raw_text: str) -> Tuple[str, List[int], List[Optional[int]]]:
    """去除标点，返回 clean_text 与索引映射。"""
    clean_chars: List[str] = []
    clean_to_raw: List[int] = []
    raw_to_clean: List[Optional[int]] = []
    clean_idx = 0
    for idx, char in enumerate(raw_text):
        if char in _PUNCTUATION_SET and not should_skip_raw_punctuation(raw_text, idx, char):
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
    *,
    clean_text: Optional[str] = None,
    raw_to_clean: Optional[Sequence[Optional[int]]] = None,
    clean_to_raw: Optional[Sequence[int]] = None,
) -> PunctuationPostprocessResult:
    """统一后处理入口（支持分轨文本输入）。"""
    if not raw_text:
        return PunctuationPostprocessResult(
            final_text="",
            final_positions=[],
            decision_log=[],
            metrics=_build_metrics([]),
        )

    if clean_text is None or raw_to_clean is None or clean_to_raw is None:
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
    # V3.2.0+dev.20260204.14: 保护性门控 - 弱标点过密时视为噪声，短句也防止“每词一逗号”。
    if raw_marks and words:
        word_count = len(words)
        weak_count = sum(1 for mark in raw_marks if mark.punctuation in _WEAK_PUNCTUATION_SET)
        weak_ratio = weak_count / max(word_count, 1)
        if (
            (word_count >= 6 and weak_ratio >= _RAW_WEAK_RATIO_THRESHOLD)
            or (word_count >= 4 and weak_count >= max(2, word_count - 1))
        ):
            raw_marks = [mark for mark in raw_marks if mark.punctuation not in _WEAK_PUNCTUATION_SET]

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
        if should_skip_raw_punctuation(raw_text, idx, char):
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

        # V3.2.0+dev.20260204.14 -> V3.2.0+dev.20260205.02:
        # 英文句末标点过滤仅在 fast 模式生效（SenseVoice 英文标点噪声）。
        # dual 模式下 raw_marks 来自 Whisper，英文标点质量高，无需过滤。
        if (
            mode == "fast"
            and not _is_cjk_language(language)
            and mark.punctuation in _SENTENCE_END
            and mark.punctuation not in _QUESTION_PUNCT
            and candidate_missing
            and not _pause_ok(words, mark.word_index, config.pause_end_min_sec)
        ):
            decisions.append(
                Decision(
                    char_index=mark.char_index,
                    punctuation=mark.punctuation,
                    action="drop_raw",
                    reason="drop_raw_no_candidate_no_pause",
                    raw_conf=mark.raw_conf,
                    cand_conf=None,
                )
            )
            continue

        # V3.2.0+dev.20260205.01 -> V3.2.0+dev.20260205.02:
        # 英文弱标点过滤仅在 fast 模式生效（SenseVoice "It's, still, only, ..." 噪声）。
        # dual 模式下 raw_marks 来自 Whisper，英文逗号等弱标点质量高，无需过滤。
        if mode == "fast" and not _is_cjk_language(language) and mark.punctuation in _WEAK_PUNCTUATION_SET:
            supported_by_candidate = False
            for maybe in (cand, cand_near):
                if not maybe:
                    continue
                if maybe.punctuation != mark.punctuation:
                    continue
                if abs(int(maybe.char_index) - int(mark.char_index)) <= config.conflict_window_chars:
                    supported_by_candidate = True
                    break
            if not supported_by_candidate:
                decisions.append(
                    Decision(
                        char_index=mark.char_index,
                        punctuation=mark.punctuation,
                        action="drop_raw",
                        reason="drop_raw_weak_untrusted",
                        raw_conf=mark.raw_conf,
                        cand_conf=cand.confidence if cand else None,
                    )
                )
                continue

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
    if not _comma_guard_ok(cand, words, language, config):
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
    # V3.2.0+dev.20260205.02: 句末停顿门控仅对 CJK 语言生效
    # 英文 Whisper/模型标点质量高，停顿与句边界弱相关，禁用此门控
    if cand.punctuation in _SENTENCE_END:
        if _is_cjk_language(language):
            if not _pause_ok(words, cand.word_index, config.pause_end_min_sec):
                return False
    return True


def _comma_guard_ok(
    cand: _MappedCandidate,
    words: Sequence[WordTimestampLike],
    language: str,
    config: PunctuationPostprocessConfig,
) -> bool:
    if not config.is_comma_guard_enabled:
        return True
    if language not in {"zh", "yue"}:
        return True
    if cand.punctuation not in _COMMA_PUNCT:
        return True
    if cand.confidence < config.comma_guard_min_conf:
        return False
    word_count, char_count = _count_segment_since_pause(
        words,
        cand.word_index,
        config.comma_guard_pause_min_sec,
    )
    words_ok = config.comma_guard_min_words <= 0 or word_count >= config.comma_guard_min_words
    chars_ok = config.comma_guard_min_chars <= 0 or char_count >= config.comma_guard_min_chars
    pause_after = _get_pause_after(words, cand.word_index)
    pause_ok = False
    if config.comma_guard_pause_min_sec > 0:
        pause_ok = pause_after is not None and pause_after >= config.comma_guard_pause_min_sec
    return words_ok or chars_ok or pause_ok


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


def _get_pause_after(words: Sequence[WordTimestampLike], word_index: int) -> Optional[float]:
    if word_index < 0 or word_index >= len(words) - 1:
        return None
    end_time = _get_word_end(words[word_index])
    next_start = _get_word_start(words[word_index + 1])
    if end_time is None or next_start is None:
        return None
    return max(0.0, next_start - end_time)


def _count_segment_since_pause(
    words: Sequence[WordTimestampLike],
    word_index: int,
    pause_threshold: float,
) -> Tuple[int, int]:
    if word_index < 0 or word_index >= len(words):
        return 0, 0
    if pause_threshold <= 0:
        start_index = 0
    else:
        start_index = word_index
        while start_index > 0:
            prev_end = _get_word_end(words[start_index - 1])
            curr_start = _get_word_start(words[start_index])
            if prev_end is None or curr_start is None:
                break
            if (curr_start - prev_end) >= pause_threshold:
                break
            start_index -= 1
    word_count = 0
    char_count = 0
    for idx in range(start_index, word_index + 1):
        token = _get_word_text(words[idx])
        if not token:
            continue
        word_count += 1
        char_count += _count_effective_chars(token)
    return word_count, char_count


def _count_effective_chars(token: str) -> int:
    return sum(1 for ch in token if not ch.isspace())


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


def _is_cjk_language(language: str) -> bool:
    lang = (language or "auto").lower()
    return lang.startswith(("zh", "yue", "ja", "jp", "ko"))
