"""
标点前置域处理器（PunctuationProcessor）。
V3.2.0+dev.20260205.02
"""
from __future__ import annotations

from bisect import bisect_left
import difflib
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from app.core.logging import resolve_loguru_logger
from app.services.alignment.types import (
    PunctuationPreInput,
    PunctuationPreOutput,
    PunctSource,
    PunctTrack,
    TextTrack,
)
from app.services.punctuation.base import PuncPosition, PunctuationResult, WordTimestampLike, apply_punctuation
from app.services.punctuation.postprocess import (
    PunctuationPostprocessResult,
    get_postprocess_config,
    postprocess_punctuation,
)
if TYPE_CHECKING:
    from app.services.punctuation.service import PunctuationService
from app.services.text_pipeline_config import PunctuationConfig, TextPipelineConfig


_WEAK_PUNCTUATION_SET = set(",，、;；:：")


class PunctuationProcessor:
    """处理器模式：标点前置域恢复与候选合并。"""

    def __init__(
        self,
        punctuation_service: Optional["PunctuationService"] = None,
        *,
        logger: Optional[Any] = None,
        config_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._logger = resolve_loguru_logger(
            logger,
            __name__,
            layer="评分层",
            processor_name="punctuation_processor",
        )
        self._punctuation_service = punctuation_service
        self._config_override = dict(config_override) if config_override else None
        if self._punctuation_service is None:
            from app.services.punctuation.service import get_punctuation_service

            self._punctuation_service = get_punctuation_service()

    async def process(self, data: PunctuationPreInput) -> PunctuationPreOutput:
        """执行标点前置域恢复并输出 PunctTrack。"""
        track = data.chosen_text_track
        clean_text = self._get_clean_text(track)
        if not clean_text:
            return PunctuationPreOutput(
                punct_track=PunctTrack(clean_text_ref=clean_text or "", positions=[], source="empty")
            )

        config = self._resolve_config()
        if not config.is_enabled:
            return PunctuationPreOutput(
                punct_track=PunctTrack(clean_text_ref=clean_text, positions=[], source="disabled")
            )

        language = track.language if track else "auto"
        word_timestamps = list(data.word_timestamps or [])
        raw_text = self._get_raw_text(track, clean_text)

        candidates: List[PuncPosition] = []
        model_candidates: List[PuncPosition] = []
        model_result: Optional[PunctuationResult] = None
        # V3.2.0+dev.20260205.02: 标点模型仅在 fast 模式下调用
        # dual 模式下 Whisper 原始标点质量高（尤其英文），无需模型补充
        if config.source_preference == "fast":
            model_candidates, model_result = await self._restore_with_model(
                clean_text,
                language,
                word_timestamps,
            )
            candidates.extend(model_candidates)

        sv_positions = self._extract_source_positions(
            data.sv_punct_source,
            clean_text,
            allowed={"fast", "merged"},
            fallback_mode=config.sv_fallback_mode,
        )
        candidates.extend(sv_positions)
        slow_positions = self._extract_source_positions(
            data.wh_punct_source,
            clean_text,
            allowed={"slow", "merged", "slow_raw"},
            fallback_mode="strict",
        )
        if slow_positions:
            filtered = self._filter_dense_weak_positions(
                slow_positions,
                word_timestamps,
                language,
                clean_text=clean_text,
                max_weak_ratio=config.raw_source_max_weak_ratio,
            )
            slow_positions = filtered
        candidates.extend(slow_positions)

        if config.source_preference == "fast":
            candidates = self._filter_candidates_by_source(
                data.sv_punct_source,
                clean_text,
                candidates,
                fallback_keep=True,
                fallback_mode=config.sv_fallback_mode,
            )
        elif config.source_preference == "slow":
            # 注意：不能直接返回 data.wh_punct_source.positions（可能包含弱标点噪声）。
            candidates = list(slow_positions or [])

        positions, post_result = self._postprocess_candidates(
            raw_text=raw_text,
            clean_text=clean_text,
            raw_to_clean=track.raw_to_clean if track else None,
            clean_to_raw=track.clean_to_raw if track else None,
            words=word_timestamps,
            language=language,
            mode="fast" if config.source_preference == "fast" else "dual",
            candidates=candidates,
        )

        punct_track = PunctTrack(
            clean_text_ref=clean_text,
            positions=positions,
            source=self._resolve_source_label(config, candidates, positions),
            confidence_stats=self._build_confidence_stats(positions, model_result),
        )
        return PunctuationPreOutput(punct_track=punct_track)

    def _resolve_config(self) -> PunctuationConfig:
        if self._config_override is not None:
            return PunctuationConfig.from_runtime(self._config_override)
        pipeline_config = TextPipelineConfig.from_runtime()
        return pipeline_config.punctuation

    @staticmethod
    def _get_clean_text(track: Optional[TextTrack]) -> str:
        if not track:
            return ""
        return str(track.text_clean or track.text_itn_raw or track.raw_text or "")

    @staticmethod
    def _get_raw_text(track: Optional[TextTrack], fallback: str) -> str:
        if not track:
            return fallback
        return str(track.text_itn_raw or track.raw_text or fallback)

    async def _restore_with_model(
        self,
        clean_text: str,
        language: str,
        word_timestamps: Sequence[WordTimestampLike],
    ) -> tuple[List[PuncPosition], Optional[PunctuationResult]]:
        if not self._punctuation_service or not clean_text:
            return [], None
        try:
            result = await self._punctuation_service.restore(
                text=clean_text,
                language=language,
                word_timestamps=word_timestamps,
            )
            return list(result.punctuation_positions or []), result
        except Exception as exc:
            self._logger.warning("评分层标点模型恢复失败: {}", exc)
            return [], None

    @staticmethod
    def _extract_source_positions(
        source: Optional[PunctSource],
        clean_text: str,
        *,
        allowed: set[str],
        fallback_mode: str = "strict",
    ) -> List[PuncPosition]:
        if not source or not clean_text:
            return []
        if source.source not in allowed:
            return []
        return PunctuationProcessor._resolve_positions_by_mode(
            source=source,
            clean_text=clean_text,
            fallback_mode=fallback_mode,
        )

    @staticmethod
    def _filter_candidates_by_source(
        source: Optional[PunctSource],
        clean_text: str,
        candidates: List[PuncPosition],
        *,
        fallback_keep: bool,
        fallback_mode: str = "strict",
    ) -> List[PuncPosition]:
        if source and source.positions:
            mapped = PunctuationProcessor._resolve_positions_by_mode(
                source=source,
                clean_text=clean_text,
                fallback_mode=fallback_mode,
            )
            if mapped:
                return mapped
        return list(candidates) if fallback_keep else []

    # V3.2.0+dev.20260206.01: 允许 clean_text_ref 在大小写/引号形态上的等价匹配，
    # 但仍保持字符索引一一对应（长度不变），避免误放宽导致越界注入。
    @staticmethod
    def _is_compatible_clean_text_ref(source_ref: str, clean_text: str) -> bool:
        if source_ref == clean_text:
            return True
        if not source_ref or not clean_text:
            return False
        if len(source_ref) != len(clean_text):
            return False
        for source_char, target_char in zip(source_ref, clean_text):
            if source_char == target_char:
                continue
            if source_char.isspace() and target_char.isspace():
                continue
            if PunctuationProcessor._normalize_ref_char(source_char) == PunctuationProcessor._normalize_ref_char(target_char):
                continue
            return False
        return True

    
    @staticmethod
    def _resolve_positions_by_mode(
        source: PunctSource,
        clean_text: str,
        fallback_mode: str,
    ) -> List[PuncPosition]:
        if not source.positions:
            return []
        mode = str(fallback_mode or "strict").lower()
        if mode not in {"strict", "tolerant"}:
            mode = "strict"
        if mode == "strict":
            if not PunctuationProcessor._is_compatible_clean_text_ref(source.clean_text_ref, clean_text):
                return []
            return list(source.positions)
        if PunctuationProcessor._is_compatible_clean_text_ref(source.clean_text_ref, clean_text):
            return list(source.positions)
        return PunctuationProcessor._remap_positions_tolerant(
            source_ref=str(source.clean_text_ref or ""),
            target_ref=clean_text,
            positions=source.positions,
        )

    @staticmethod
    def _remap_positions_tolerant(
        source_ref: str,
        target_ref: str,
        positions: Sequence[PuncPosition],
    ) -> List[PuncPosition]:
        if not source_ref or not target_ref or not positions:
            return []
        source_chars = [PunctuationProcessor._normalize_ref_char(ch) for ch in source_ref]
        target_chars = [PunctuationProcessor._normalize_ref_char(ch) for ch in target_ref]
        matcher = difflib.SequenceMatcher(a=source_chars, b=target_chars, autojunk=False)
        if matcher.ratio() < 0.55:
            return []
        direct_map: Dict[int, int] = {}
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                continue
            span = min(i2 - i1, j2 - j1)
            for offset in range(span):
                direct_map[i1 + offset] = j1 + offset
        if not direct_map:
            return []
        mapped_indexes = sorted(direct_map.keys())
        remapped: List[PuncPosition] = []
        for position in positions:
            target_index = PunctuationProcessor._project_char_index(
                source_index=int(position.char_index),
                direct_map=direct_map,
                mapped_indexes=mapped_indexes,
                target_len=len(target_ref),
            )
            if target_index is None:
                continue
            remapped.append(
                PuncPosition(
                    char_index=target_index,
                    punctuation=position.punctuation,
                    confidence=position.confidence,
                )
            )
        if not remapped:
            return []
        dedup: Dict[tuple[int, str], PuncPosition] = {}
        for position in remapped:
            key = (position.char_index, position.punctuation)
            exists = dedup.get(key)
            if exists is None or position.confidence > exists.confidence:
                dedup[key] = position
        return sorted(dedup.values(), key=lambda item: item.char_index)

    @staticmethod
    def _project_char_index(
        source_index: int,
        direct_map: Dict[int, int],
        mapped_indexes: Sequence[int],
        target_len: int,
    ) -> Optional[int]:
        if target_len <= 0:
            return None
        if source_index in direct_map:
            return max(0, min(target_len - 1, direct_map[source_index]))
        if not mapped_indexes:
            return None
        pos = bisect_left(mapped_indexes, source_index)
        left = mapped_indexes[pos - 1] if pos > 0 else None
        right = mapped_indexes[pos] if pos < len(mapped_indexes) else None
        candidate: Optional[int]
        if left is not None and right is not None and right != left:
            left_target = direct_map[left]
            right_target = direct_map[right]
            ratio = float(source_index - left) / float(right - left)
            candidate = int(round(left_target + ratio * (right_target - left_target)))
        elif left is not None:
            candidate = direct_map[left] + (source_index - left)
        elif right is not None:
            candidate = direct_map[right] - (right - source_index)
        else:
            candidate = None
        if candidate is None:
            return None
        return max(0, min(target_len - 1, int(candidate)))
    @staticmethod
    def _normalize_ref_char(value: str) -> str:
        if not value:
            return ""
        quote_alias = {
            '’': "'",
            '‘': "'",
            '`': "'",
            '“': '"',
            '”': '"',
        }
        return quote_alias.get(value, value).lower()

    def _filter_dense_weak_positions(
        self,
        positions: Sequence[PuncPosition],
        words: Sequence[WordTimestampLike],
        language: str,
        *,
        clean_text: str,
        max_weak_ratio: float,
    ) -> List[PuncPosition]:
        if not positions:
            return list(positions)
        if PunctuationProcessor._is_cjk_language(language):
            return list(positions)
        # 英文等非CJK场景，SenseVoice/Whisper 的 words 粒度可能为子词/字符，
        # 直接用 len(words) 会导致弱标点密度被稀释，过滤失效。
        word_count = PunctuationProcessor._estimate_token_count(clean_text)
        if word_count < 3:
            return list(positions)
        weak_count = sum(1 for pos in positions if pos.punctuation in _WEAK_PUNCTUATION_SET)
        if weak_count <= 0:
            return list(positions)
        weak_ratio = weak_count / max(word_count, 1)
        # V3.2.0+dev.20260205.03: 输出弱标点密度细节，便于定位密集逗号来源
        self._logger.debug(
            "评分层 slow_raw 弱标点密度: word_count={} weak_count={} weak_ratio={:.2f} max_weak_ratio={:.2f}",
            word_count,
            weak_count,
            weak_ratio,
            max_weak_ratio,
        )
        if (
            (word_count >= 6 and weak_ratio >= max_weak_ratio)
            or (word_count >= 4 and weak_count >= max(2, word_count - 1))
        ):
            self._logger.debug(
                "评分层 slow_raw 弱标点过滤触发: word_count={} weak_count={} weak_ratio={:.2f}",
                word_count,
                weak_count,
                weak_ratio,
            )
            return [pos for pos in positions if pos.punctuation not in _WEAK_PUNCTUATION_SET]
        return list(positions)

    @staticmethod
    def _estimate_token_count(text: str) -> int:
        """估算 token 数（用于弱标点密度门控）。"""
        if not text:
            return 0
        if any(ch.isspace() for ch in text):
            return len([t for t in str(text).split() if t])
        return len([ch for ch in str(text) if not ch.isspace()])

    def _postprocess_candidates(
        self,
        *,
        raw_text: str,
        clean_text: str,
        raw_to_clean: Optional[Sequence[Optional[int]]],
        clean_to_raw: Optional[Sequence[int]],
        words: Sequence[WordTimestampLike],
        language: str,
        mode: str,
        candidates: Sequence[PuncPosition],
    ) -> tuple[List[PuncPosition], Optional[PunctuationPostprocessResult]]:
        if not candidates and not raw_text:
            return [], None
        if words:
            config = get_postprocess_config(mode)
            post = postprocess_punctuation(
                raw_text=raw_text,
                clean_text=clean_text,
                raw_to_clean=raw_to_clean,
                clean_to_raw=clean_to_raw,
                words=words,
                language=language,
                mode=mode,
                candidates=candidates,
                config=config,
            )
            return list(post.final_positions), post
        filtered = self._filter_candidates_simple(candidates, language, mode)
        return filtered, None

    @staticmethod
    def _filter_candidates_simple(
        candidates: Sequence[PuncPosition],
        language: str,
        mode: str,
    ) -> List[PuncPosition]:
        if not candidates:
            return []
        config = get_postprocess_config(mode)
        allowed = config.allowed_punct_zh if language in {"zh", "yue"} else config.allowed_punct_en
        allowed_set = set(allowed)
        filtered = [
            position
            for position in candidates
            if position.punctuation in allowed_set and position.confidence >= config.candidate_min_conf
        ]
        filtered.sort(key=lambda item: item.char_index)
        return filtered

    @staticmethod
    def _resolve_source_label(
        config: PunctuationConfig,
        candidates: Sequence[PuncPosition],
        positions: Sequence[PuncPosition],
    ) -> str:
        if not positions and not candidates:
            return f"{config.source_preference}_empty"
        return config.source_preference

    @staticmethod
    def _build_confidence_stats(
        positions: Sequence[PuncPosition],
        model_result: Optional[PunctuationResult],
    ) -> Dict[str, Any]:
        if not positions:
            return {
                "count": 0.0,
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "model_confidence": float(model_result.confidence if model_result else 0.0),
                "model_id": str(model_result.model_id if model_result else ""),
                "processing_time_ms": float(model_result.processing_time_ms if model_result else 0.0),
            }
        confidences = [float(pos.confidence) for pos in positions]
        return {
            "count": float(len(positions)),
            "avg_confidence": sum(confidences) / max(len(confidences), 1),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "model_confidence": float(model_result.confidence if model_result else 0.0),
            "model_id": str(model_result.model_id if model_result else ""),
            "processing_time_ms": float(model_result.processing_time_ms if model_result else 0.0),
        }

    @staticmethod
    def _is_cjk_language(language: str) -> bool:
        lang = (language or "auto").lower()
        return lang.startswith(("zh", "yue", "ja", "jp", "ko"))

    @staticmethod
    def build_punctuated_text(clean_text: str, positions: Sequence[PuncPosition]) -> str:
        """辅助：按位置插入标点（用于调试/兼容输出）。"""
        return apply_punctuation(clean_text, positions)


