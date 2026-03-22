"""SenseVoice 时间基底适配器。"""

from __future__ import annotations

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
            return self._build_from_logits(
                ctc_logits=ctc_logits,
                language=normalized_language,
                frame_stride=normalized_stride,
                encoder_out_lens=encoder_out_lens,
            )

        return self._build_from_compact_trace(
            language=normalized_language,
            frame_stride=normalized_stride,
            compact_acoustic_trace=compact_acoustic_trace or {},
            raw_tokens=list(raw_tokens or []),
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
        if encoder_out_lens is not None:
            logits = logits[: max(0, int(encoder_out_lens)), :]

        text, decoded_tokens, _confidence, _lang_info = self._decoder.decode(logits, time_stride=frame_stride)
        _ = text  # 保留变量，方便后续调试扩展

        probs = CTCDecoder._softmax(logits)
        token_ids = np.argmax(probs, axis=-1)
        max_probs = np.max(probs, axis=-1)

        enriched_raw = self._attach_top_candidates(
            decoded_tokens=list(decoded_tokens),
            probs=probs,
            frame_stride=frame_stride,
        )

        merge_result = merge_tokens(enriched_raw, language=language)
        raw_units = tuple(self._raw_unit_from_token(token, token_type="raw") for token in enriched_raw)
        word_units = tuple(self._raw_unit_from_token(token, token_type="word") for token in merge_result.words)

        low_prob_ratio = float(np.mean(max_probs < self._low_conf_threshold)) if max_probs.size else 0.0
        quality = TimeBaseQuality(
            blank_ratio=float(np.mean(token_ids == self._blank_id)) if token_ids.size else 0.0,
            avg_max_prob=float(np.mean(max_probs)) if max_probs.size else 0.0,
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

        avg_conf = float(np.mean([float(item.get("confidence", 0.0)) for item in source_tokens])) if source_tokens else 0.0
        low_prob_ratio = (
            float(np.mean([float(item.get("confidence", 0.0)) < self._low_conf_threshold for item in source_tokens]))
            if source_tokens
            else 0.0
        )
        quality = TimeBaseQuality(
            blank_ratio=float(compact_acoustic_trace.get("blank_ratio", 0.0) or 0.0),
            avg_max_prob=float(compact_acoustic_trace.get("avg_max_prob", avg_conf) or avg_conf),
            low_prob_ratio=float(compact_acoustic_trace.get("low_prob_ratio", low_prob_ratio) or low_prob_ratio),
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
                "top_k": int(compact_acoustic_trace.get("top_k", self._top_k) or self._top_k),
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
                    score=float(avg_probs[int(index)]),
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
                score=float(item.get("score", 0.0) or 0.0),
                token_id=int(item.get("token_id")) if item.get("token_id") is not None else None,
            )
            for item in candidates_raw
            if isinstance(item, dict)
        )
        return TimeBaseUnit(
            text=str(token.get("word", "") or ""),
            start=float(token.get("start", 0.0) or 0.0),
            end=float(token.get("end", token.get("start", 0.0)) or 0.0),
            confidence=float(token.get("confidence", 0.0) or 0.0),
            token_type=token_type,
            top_candidates=candidates,
            source="sensevoice",
        )
