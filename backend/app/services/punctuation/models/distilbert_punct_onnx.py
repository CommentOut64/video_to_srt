"""
DistilBERT 英文标点 ONNX 适配器。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tokenizers import Tokenizer

from app.services.punctuation.base import PuncPosition
from app.services.punctuation.models.onnx_base import OnnxPunctuationAdapter


class DistilBertPunctOnnxAdapter(OnnxPunctuationAdapter):
    """英文标点模型适配器（基于 DistilBERT TokenClassification ONNX）。"""

    def __init__(self, model_id: Optional[str] = None) -> None:
        super().__init__(
            model_id=model_id or "punct-distilbert-en",
            default_punctuation=".",
            terminal_punctuations=".!?",
        )
        self._tokenizer: Optional[Tokenizer] = None
        self._label_map: Dict[int, str] = {}
        self._max_length: int = 512

    def _predict_with_session(self, session, text: str) -> List[PuncPosition]:
        if not text:
            return []
        tokenizer = self._ensure_tokenizer()
        encoded = tokenizer.encode(text)
        if len(encoded.ids) > self._max_length:
            return self._predict_long_text(session, tokenizer, text)
        return self._predict_with_encoding(session, tokenizer, text, encoded, offset=0)

    def _predict_long_text(
        self,
        session,
        tokenizer: Tokenizer,
        text: str,
    ) -> List[PuncPosition]:
        words = list(re.finditer(r"\S+", text))
        if not words:
            return []

        positions: List[PuncPosition] = []
        chunk_start = words[0].start()
        last_end = chunk_start

        for idx, match in enumerate(words):
            candidate_end = match.end()
            candidate_text = text[chunk_start:candidate_end]
            encoded = tokenizer.encode(candidate_text)
            if len(encoded.ids) > self._max_length and last_end > chunk_start:
                chunk_text = text[chunk_start:last_end]
                chunk_encoded = tokenizer.encode(chunk_text)
                positions.extend(
                    self._predict_with_encoding(
                        session, tokenizer, chunk_text, chunk_encoded, offset=chunk_start
                    )
                )
                chunk_start = match.start()
            last_end = candidate_end

            if idx == len(words) - 1:
                chunk_text = text[chunk_start:candidate_end]
                chunk_encoded = tokenizer.encode(chunk_text)
                positions.extend(
                    self._predict_with_encoding(
                        session, tokenizer, chunk_text, chunk_encoded, offset=chunk_start
                    )
                )
        return positions

    def _predict_with_encoding(
        self,
        session,
        tokenizer: Tokenizer,
        text: str,
        encoded,
        offset: int,
    ) -> List[PuncPosition]:
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        logits = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )[0][0]

        word_ids = encoded.word_ids
        offsets = encoded.offsets
        word_to_tokens: Dict[int, List[int]] = {}
        for token_index, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            word_to_tokens.setdefault(word_id, []).append(token_index)

        positions: List[PuncPosition] = []
        for token_indices in word_to_tokens.values():
            token_idx = token_indices[-1]
            label_id = int(np.argmax(logits[token_idx]))
            punct = self._label_map.get(label_id, "")
            if not punct:
                continue
            start, end = offsets[token_idx]
            if end <= 0:
                continue
            confidence = self._softmax(logits[token_idx])[label_id]
            positions.append(
                PuncPosition(
                    char_index=offset + end - 1,
                    punctuation=punct,
                    confidence=float(confidence),
                )
            )
        return positions

    def _ensure_tokenizer(self) -> Tokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        model_dir = self._model_dir
        if model_dir is None:
            raise RuntimeError("模型目录未就绪")
        tokenizer_path = model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise RuntimeError(f"未找到 tokenizer.json: {tokenizer_path}")
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._label_map = self._load_label_map(model_dir)
        return self._tokenizer

    def _load_label_map(self, model_dir: Path) -> Dict[int, str]:
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return {0: "?", 1: "!", 2: ",", 3: "", 4: "."}
        data = json.loads(config_path.read_text(encoding="utf-8"))
        id2label = data.get("id2label", {})
        label_map: Dict[int, str] = {}
        for key, label in id2label.items():
            label_text = str(label).upper()
            if "COMMA" in label_text:
                punct = ","
            elif "PERIOD" in label_text:
                punct = "."
            elif "QUESTION" in label_text:
                punct = "?"
            elif "EXLAMATION" in label_text or "EXCLAMATION" in label_text:
                punct = "!"
            else:
                punct = ""
            try:
                label_map[int(key)] = punct
            except ValueError:
                continue
        return label_map

    @staticmethod
    def _softmax(row: np.ndarray) -> np.ndarray:
        row = row.astype(np.float32)
        row = row - np.max(row)
        exp_row = np.exp(row)
        return exp_row / np.sum(exp_row)
