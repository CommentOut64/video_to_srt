"""
L1 规范化处理器（Processor）。
V3.2.0+dev.20260204.12
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import re

from app.core.logging import resolve_loguru_logger
from app.services.alignment.text_normalizer import (
    TextNormalizer,
    get_alignment_text_normalizer,
)
from app.services.alignment.types import L1Input, L1Output, NormalizationResult, TextTrack
from app.services.punctuation.semantic_injector import SemanticInjector
from app.services.text_pipeline_config import TextPipelineConfig


_EDGE_PUNCT = set(",.!?;:\"()[]{}，。！？；：、（）【】《》“”‘’「」『』")
_WEAK_PUNCT = set(",，、;；:：")
_SPECIAL_TAG_PATTERN = re.compile(r"<\|.*?\|>")


class TextNormalizerProcessor:
    """处理器模式：统一输入输出，便于流水线接入。"""

    def __init__(
        self,
        *,
        normalizer: Optional[TextNormalizer] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self._logger = resolve_loguru_logger(
            logger,
            __name__,
            layer="L1",
            processor_name="text_normalizer_processor",
        )
        self._normalizer = normalizer or get_alignment_text_normalizer(logger=logger)

    def process(self, data: L1Input) -> L1Output:
        """执行 L1 规范化，输出 TextTrack。"""
        return L1Output(
            sv_track=self._normalize_single(data.sv_raw_result, source="sv", language_hint=data.language_hint),
            whisper_track=self._normalize_single(
                data.wh_raw_result,
                source="whisper",
                language_hint=data.language_hint,
            ),
        )

    def _normalize_single(
        self,
        raw_result: Optional[Dict[str, Any]],
        *,
        source: str,
        language_hint: Optional[str],
    ) -> Optional[TextTrack]:
        if not raw_result:
            return None
        raw_text = str(raw_result.get("raw_text") or "")
        input_text = raw_text
        if not input_text and source == "whisper":
            input_text = str(raw_result.get("min_clean_text") or "")
        if not input_text:
            self._logger.warning("L1 规范化缺失 raw_text/min_clean_text source={}", source)
            return None
        if not raw_text:
            raw_text = input_text
        language = str(raw_result.get("language") or language_hint or "auto")
        normalized = self._normalizer.normalize(input_text, language)
        track = self._build_text_track(raw_text, normalized, source=source, language=language)
        # L1 负责把“原始词级标点”提取为 clean_text 坐标（避免 L3 再反推）。
        self._attach_word_level_punctuation(track, raw_result, source=source)
        # L1 可选生成 clean_to_word 映射（供后续模块埋点/对齐参考）。
        self._attach_clean_to_word(track, raw_result, source=source)
        return track

    @staticmethod
    def _attach_word_level_punctuation(track: TextTrack, raw_result: Dict[str, Any], *, source: str) -> None:
        if not track.text_clean:
            return
        words = TextNormalizerProcessor._extract_words_for_punct(raw_result, source=source)
        try:
            punct_config = TextPipelineConfig.from_runtime().punctuation
            min_cov = float(punct_config.raw_source_min_mapping_coverage)
            max_weak = float(punct_config.raw_source_max_weak_ratio)
        except Exception:
            min_cov = 0.6
            max_weak = 0.8
        if words:
            _, positions, _ = SemanticInjector.extract_word_punctuation_positions(
                words,
                clean_text=track.text_clean,
                min_mapping_coverage=min_cov,
                max_weak_ratio=max_weak,
            )
            if positions:
                track.punct_positions = list(positions or [])
                return

        # 词级时间戳缺失时，允许使用字符映射兜底（仅在 L1 内部做，L3 禁止回看）。
        raw_text = str(track.text_itn_raw or track.raw_text or "")
        if not raw_text or not track.raw_to_clean:
            return
        clean_text, positions = SemanticInjector.extract_raw_punctuation_positions(
            raw_text,
            clean_text=track.text_clean,
            raw_to_clean=track.raw_to_clean,
        )
        if not positions or clean_text != track.text_clean:
            return
        token_count = TextNormalizerProcessor._estimate_token_count(track.text_clean)
        weak_count = sum(1 for pos in positions if pos.punctuation in _WEAK_PUNCT)
        if token_count >= 6 and max_weak > 0 and weak_count / max(token_count, 1) >= max_weak:
            return
        track.punct_positions = list(positions)

    @staticmethod
    def _attach_clean_to_word(track: TextTrack, raw_result: Dict[str, Any], *, source: str) -> None:
        clean_text = str(track.text_clean or "")
        if not clean_text:
            return
        words = TextNormalizerProcessor._extract_words_for_clean_map(raw_result, source=source)
        if not words:
            track.clean_to_word = []
            return
        track.clean_to_word = TextNormalizerProcessor._build_clean_to_word_map(clean_text, words)

    @staticmethod
    def _extract_words_for_punct(raw_result: Dict[str, Any], *, source: str) -> List[Dict[str, Any]]:
        if not isinstance(raw_result, dict):
            return []
        if source == "sv":
            words = raw_result.get("words", []) or []
            if not isinstance(words, list):
                return []
            cleaned: List[Dict[str, Any]] = []
            for item in words:
                if not isinstance(item, dict):
                    continue
                token = TextNormalizerProcessor._strip_special_tokens(str(item.get("word", "") or ""))
                if not token:
                    continue
                cleaned.append({**item, "word": token})
            return cleaned
        raw = raw_result.get("raw_result") if isinstance(raw_result, dict) else None
        segments = raw.get("segments", []) if isinstance(raw, dict) else []
        collected: List[Dict[str, Any]] = []
        for seg in segments:
            for word in seg.get("words", []) or []:
                token = TextNormalizerProcessor._strip_special_tokens(str(word.get("word", "") or ""))
                if not token:
                    continue
                collected.append(
                    {
                        "word": token,
                        "start": float(word.get("start", 0.0) or 0.0),
                        "end": float(word.get("end", 0.0) or 0.0),
                        "confidence": float(word.get("probability", 0.0) or 0.0),
                    }
                )
        return collected

    @staticmethod
    def _extract_words_for_clean_map(raw_result: Dict[str, Any], *, source: str) -> List[Dict[str, Any]]:
        words = TextNormalizerProcessor._extract_words_for_punct(raw_result, source=source)
        if not words:
            return []
        normalized: List[Dict[str, Any]] = []
        for item in words:
            token = TextNormalizerProcessor._strip_special_tokens(str(item.get("word", "") or ""))
            token = TextNormalizerProcessor._strip_edge_punct(token)
            if not token:
                continue
            normalized.append({**item, "word": token})
        return normalized

    @staticmethod
    def _strip_edge_punct(token: str) -> str:
        if not token:
            return ""
        start = 0
        end = len(token)
        while start < end and token[start] in _EDGE_PUNCT:
            start += 1
        while end > start and token[end - 1] in _EDGE_PUNCT:
            end -= 1
        return token[start:end]

    @staticmethod
    def _strip_special_tokens(token: str) -> str:
        """去除 SenseVoice 标签与 SentencePiece 符号。"""
        if not token:
            return ""
        token = _SPECIAL_TAG_PATTERN.sub("", token)
        token = token.replace("▁", " ")
        return token.strip()

    @staticmethod
    def _build_clean_to_word_map(clean_text: str, words: List[Dict[str, Any]]) -> List[Optional[int]]:
        mapping: List[Optional[int]] = [None] * len(clean_text)
        cursor = 0
        for idx, word in enumerate(words):
            token = str(word.get("word", "") or "")
            token = token.strip()
            if not token:
                continue
            while cursor < len(clean_text) and clean_text[cursor].isspace():
                cursor += 1
            if cursor >= len(clean_text):
                break
            match_idx = clean_text.find(token, cursor)
            if match_idx == -1:
                match_idx = TextNormalizerProcessor._fallback_match(clean_text, token, cursor)
            if match_idx == -1:
                continue
            start = max(match_idx, 0)
            end = min(match_idx + len(token) - 1, len(clean_text) - 1)
            for pos in range(start, end + 1):
                mapping[pos] = idx
            cursor = min(end + 1, len(clean_text))
        return mapping

    @staticmethod
    def _fallback_match(text: str, token: str, cursor: int) -> int:
        if not token:
            return -1
        if cursor < len(text) and text[cursor: cursor + len(token)] == token:
            return cursor
        for idx in range(cursor, len(text)):
            if text[idx: idx + len(token)] == token:
                return idx
        return -1

    @staticmethod
    def _estimate_token_count(clean_text: str) -> int:
        if not clean_text:
            return 0
        if " " in clean_text:
            return len([t for t in clean_text.split() if t])
        return len([ch for ch in clean_text if not ch.isspace()])

    @staticmethod
    def _build_text_track(
        raw_text: str,
        normalized: NormalizationResult,
        *,
        source: str,
        language: str,
    ) -> TextTrack:
        return TextTrack(
            raw_text=raw_text,
            text_itn_raw=normalized.text_itn_raw,
            text_clean=normalized.text_clean or normalized.text_itn_raw,
            char_mapping=normalized.char_mapping,
            raw_to_clean=normalized.raw_to_clean,
            clean_to_raw=normalized.clean_to_raw,
            language=language or "auto",
            source=source,
            itn_fallback=normalized.itn_fallback,
            itn_fallback_reason=normalized.itn_fallback_reason,
            mapping_coverage=normalized.mapping_coverage,
        )
