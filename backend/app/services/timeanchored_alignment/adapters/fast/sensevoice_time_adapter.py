"""SenseVoice 时间基底适配器。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from app.services.sensevoice_onnx_service import CTCDecoder
from app.services.token_merge_service import merge_tokens
from app.services.timeanchored_alignment.contracts import (
    AcousticCandidate,
    TimeBasePackage,
    TimeBaseQuality,
    TimeBaseUnit,
)


class SenseVoiceTimeAdapter:
    """将 SenseVoice 声学输出适配为 TimeBasePackage。"""

    def __init__(
        self,
        *,
        vocab: Optional[Dict[int, str]] = None,
        blank_id: int = 0,
        top_k: int = 3,
        ambiguity_margin: float = 0.12,
        low_conf_threshold: float = 0.75,
    ) -> None:
        self._decoder = CTCDecoder(vocab=vocab, blank_id=blank_id) if vocab else None
        self._blank_id = int(blank_id)
        self._top_k = max(1, min(int(top_k), 3))
        self._ambiguity_margin = float(ambiguity_margin)
        self._low_conf_threshold = float(low_conf_threshold)

    @property
    def can_decode_ctc(self) -> bool:
        return self._decoder is not None

    def build_time_base(
        self,
        *,
        ctc_logits: Optional[np.ndarray],
        language: str,
        frame_stride: float = 0.06,
        encoder_out_lens: Optional[int] = None,
        compact_acoustic_trace: Optional[Dict[str, Any]] = None,
        raw_tokens: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> TimeBasePackage:
        """构建 TimeBasePackage。"""
        normalized_language = str(language or "auto").lower()
        normalized_stride = float(frame_stride or 0.06)

        if ctc_logits is not None and self._decoder is not None:
            try:
                package = self._build_from_logits(
                    ctc_logits=ctc_logits,
                    language=normalized_language,
                    frame_stride=normalized_stride,
                    encoder_out_lens=encoder_out_lens,
                )
                if package.raw_units or package.word_units:
                    return self._attach_observation_metadata(
                        package=package,
                        compact_acoustic_trace=compact_acoustic_trace or {},
                        encoder_out_lens=encoder_out_lens,
                    )
            except Exception:
                # logits 异常（空帧/NaN/解码失败）时回退 compact_trace，避免整 chunk 丢失。
                pass

        package = self._build_from_compact_trace(
            language=normalized_language,
            frame_stride=normalized_stride,
            compact_acoustic_trace=compact_acoustic_trace or {},
            raw_tokens=list(raw_tokens or []),
        )
        return self._attach_observation_metadata(
            package=package,
            compact_acoustic_trace=compact_acoustic_trace or {},
            encoder_out_lens=encoder_out_lens,
        )

    def _build_from_logits(
        self,
        *,
        ctc_logits: np.ndarray,
        language: str,
        frame_stride: float,
        encoder_out_lens: Optional[int],
    ) -> TimeBasePackage:
        if self._decoder is None:
            raise ValueError("未提供 vocab，无法从 ctc_logits 解码")

        logits = np.asarray(ctc_logits)
        if logits.ndim == 3:
            logits = logits[0]
        if logits.ndim != 2:
            raise ValueError(f"ctc_logits 维度错误: {logits.shape}")
        logits = np.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
        if encoder_out_lens is not None:
            logits = logits[: max(0, int(encoder_out_lens)), :]
        if logits.size == 0:
            raise ValueError("ctc_logits 为空")

        text, decoded_tokens, _confidence, _lang_info = self._decoder.decode(logits, time_stride=frame_stride)
        _ = text  # 保留变量，方便后续调试扩展

        probs = CTCDecoder._softmax(logits)
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        token_ids = np.argmax(probs, axis=-1)
        max_probs = np.nan_to_num(np.max(probs, axis=-1), nan=0.0, posinf=1.0, neginf=0.0)

        enriched_raw = self._attach_top_candidates(
            decoded_tokens=list(decoded_tokens),
            probs=probs,
            frame_stride=frame_stride,
        )

        merge_result = merge_tokens(enriched_raw, language=language)
        raw_units = tuple(self._raw_unit_from_token(token, token_type="raw") for token in enriched_raw)
        word_units = tuple(self._raw_unit_from_token(token, token_type="word") for token in merge_result.words)

        low_prob_ratio = float(np.mean(max_probs < self._low_conf_threshold)) if max_probs.size else 0.0
        avg_max_prob = float(np.mean(max_probs)) if max_probs.size else 0.0
        avg_max_prob = float(max(0.0, min(1.0, avg_max_prob)))
        blank_ratio = float(np.mean(token_ids == self._blank_id)) if token_ids.size else 0.0
        blank_ratio = float(max(0.0, min(1.0, blank_ratio)))
        low_prob_ratio = float(max(0.0, min(1.0, low_prob_ratio)))
        quality = TimeBaseQuality(
            blank_ratio=blank_ratio,
            avg_max_prob=avg_max_prob,
            low_prob_ratio=low_prob_ratio,
            unit_count=len(raw_units),
            word_count=len(word_units),
        )

        return TimeBasePackage(
            raw_units=raw_units,
            word_units=word_units,
            quality=quality,
            language=language,
            frame_stride=frame_stride,
            source="sensevoice",
            metadata={
                "decoder": "ctc",
                "top_k": self._top_k,
                "unit_source": "ctc_logits",
            },
        )

    def _build_from_compact_trace(
        self,
        *,
        language: str,
        frame_stride: float,
        compact_acoustic_trace: Dict[str, Any],
        raw_tokens: Sequence[Dict[str, Any]],
    ) -> TimeBasePackage:
        source_tokens = list(raw_tokens or compact_acoustic_trace.get("raw_tokens") or [])
        merge_result = merge_tokens(source_tokens, language=language)

        raw_units = tuple(self._raw_unit_from_token(token, token_type="raw") for token in source_tokens)
        word_units = tuple(self._raw_unit_from_token(token, token_type="word") for token in merge_result.words)

        token_confidences = [
            self._sanitize_probability(item.get("confidence", 0.0), default=0.0)
            for item in source_tokens
        ]
        avg_conf = float(np.mean(token_confidences)) if token_confidences else 0.0
        low_prob_ratio = (
            float(np.mean([confidence < self._low_conf_threshold for confidence in token_confidences]))
            if token_confidences
            else 0.0
        )
        quality = TimeBaseQuality(
            blank_ratio=self._sanitize_probability(
                compact_acoustic_trace.get("blank_ratio"),
                default=0.0,
            ),
            avg_max_prob=self._sanitize_probability(
                compact_acoustic_trace.get("avg_max_prob"),
                default=avg_conf,
            ),
            low_prob_ratio=self._sanitize_probability(
                compact_acoustic_trace.get("low_prob_ratio"),
                default=low_prob_ratio,
            ),
            unit_count=len(raw_units),
            word_count=len(word_units),
        )

        return TimeBasePackage(
            raw_units=raw_units,
            word_units=word_units,
            quality=quality,
            language=language,
            frame_stride=frame_stride,
            source="sensevoice",
            metadata={
                "decoder": "compact_trace",
                "top_k": self._sanitize_top_k(compact_acoustic_trace.get("top_k"), default=self._top_k),
                "unit_source": "raw_tokens",
            },
        )

    def _attach_top_candidates(
        self,
        *,
        decoded_tokens: list[Dict[str, Any]],
        probs: np.ndarray,
        frame_stride: float,
    ) -> list[Dict[str, Any]]:
        enriched: list[Dict[str, Any]] = []
        frame_count = probs.shape[0]
        for token in decoded_tokens:
            item = dict(token)
            start = float(item.get("start", 0.0) or 0.0)
            end = float(item.get("end", start) or start)
            start_idx = max(0, min(frame_count - 1, int(np.floor(start / frame_stride)))) if frame_count > 0 else 0
            end_idx = max(start_idx + 1, int(np.ceil(end / frame_stride))) if frame_count > 0 else 1
            end_idx = min(frame_count, end_idx)
            token_probs = probs[start_idx:end_idx] if frame_count > 0 else np.empty((0, 0), dtype=np.float32)

            if token_probs.size == 0:
                enriched.append(item)
                continue

            avg_probs = np.mean(token_probs, axis=0)
            top_indices = np.argsort(avg_probs)[::-1][: self._top_k]
            candidates = tuple(
                AcousticCandidate(
                    text=self._decoder.vocab.get(int(index), "<unk>") if self._decoder is not None else "<unk>",
                    score=self._sanitize_probability(avg_probs[int(index)], default=0.0),
                    token_id=int(index),
                )
                for index in top_indices
            )

            should_keep = False
            if len(candidates) >= 2:
                margin = float(candidates[0].score - candidates[1].score)
                should_keep = margin <= self._ambiguity_margin
            if float(item.get("confidence", 1.0) or 1.0) < self._low_conf_threshold:
                should_keep = True

            if should_keep:
                item["top_candidates"] = [
                    {"text": cand.text, "score": cand.score, "token_id": cand.token_id}
                    for cand in candidates
                ]
            enriched.append(item)

        return enriched

    @staticmethod
    def _raw_unit_from_token(token: Dict[str, Any], *, token_type: str) -> TimeBaseUnit:
        candidates_raw = token.get("top_candidates") or []
        candidates: Tuple[AcousticCandidate, ...] = tuple(
            AcousticCandidate(
                text=str(item.get("text", "") or ""),
                score=SenseVoiceTimeAdapter._sanitize_probability(item.get("score", 0.0), default=0.0),
                token_id=int(item.get("token_id")) if item.get("token_id") is not None else None,
            )
            for item in candidates_raw
            if isinstance(item, dict)
        )
        return TimeBaseUnit(
            text=str(token.get("word", "") or ""),
            start=float(token.get("start", 0.0) or 0.0),
            end=float(token.get("end", token.get("start", 0.0)) or 0.0),
            confidence=SenseVoiceTimeAdapter._sanitize_probability(token.get("confidence", 0.0), default=0.0),
            token_type=token_type,
            top_candidates=candidates,
            source="sensevoice",
        )

    @staticmethod
    def _sanitize_probability(value: Any, *, default: float = 0.0) -> float:
        candidate = default if value is None else value
        try:
            numeric = float(candidate)
        except (TypeError, ValueError):
            numeric = float(default)
        if not np.isfinite(numeric):
            numeric = float(default)
        return float(max(0.0, min(1.0, numeric)))

    @staticmethod
    def _sanitize_top_k(value: Any, *, default: int) -> int:
        candidate = default if value is None else value
        try:
            numeric = int(float(candidate))
        except (TypeError, ValueError):
            numeric = int(default)
        return max(1, min(numeric, 3))

    @staticmethod
    def _attach_observation_metadata(
        *,
        package: TimeBasePackage,
        compact_acoustic_trace: Dict[str, Any],
        encoder_out_lens: Optional[int],
    ) -> TimeBasePackage:
        metadata = dict(getattr(package, "metadata", {}) or {})
        if encoder_out_lens is not None:
            metadata["encoder_out_lens"] = int(encoder_out_lens)
        blank_track = compact_acoustic_trace.get("blank_track")
        if blank_track is not None:
            metadata["blank_track"] = [float(item) for item in list(blank_track)]
        sparse_logits = compact_acoustic_trace.get("sparse_logits")
        if sparse_logits is not None:
            metadata["sparse_logits"] = list(sparse_logits)
        if metadata == dict(getattr(package, "metadata", {}) or {}):
            return package
        return replace(package, metadata=metadata)
