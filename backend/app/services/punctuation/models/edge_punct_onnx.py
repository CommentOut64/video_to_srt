"""
Edge-Punct-Casing ONNX 适配器（真实模型推理 + 兜底）。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.services.punctuation.base import PuncPosition
from app.services.punctuation.models.onnx_base import OnnxPunctuationAdapter


class EdgePunctOnnxAdapter(OnnxPunctuationAdapter):
    """英文 Edge-Punct-Casing 模型适配器。"""

    def __init__(self, model_id: Optional[str] = None) -> None:
        super().__init__(
            model_id=model_id or "punct-edge-punct-en",
            default_punctuation=".",
            terminal_punctuations=".!?",
        )
        self._token_to_id: Optional[Dict[str, int]] = None
        self._token_scores: Optional[Dict[str, float]] = None
        self._unk_id: int = 0
        self._unk_score: float = -20.0
        self._max_token_len: int = 0
        self._spm = None
        # 对齐官方标签顺序：1->逗号，2->句号，3->问号
        self._punct_map = {1: ",", 2: ".", 3: "?"}

    def _predict_with_session(self, session, text: str) -> List[PuncPosition]:
        if not text:
            return []
        tokens, valid_ids, word_end_indices = self._tokenize(text)
        if not tokens:
            return []
        token_ids = np.array([tokens], dtype=np.int32)
        valid_ids_arr = np.array([valid_ids], dtype=np.int32)
        label_len = int(sum(valid_ids))
        label_lens = np.array([label_len], dtype=np.int32)

        outputs = session.run(
            None,
            {
                "token_ids": token_ids,
                "valid_ids": valid_ids_arr,
                "label_lens": label_lens,
            },
        )
        punct_logits = outputs[1]  # active_punct_logits
        positions: List[PuncPosition] = []
        for idx, row in enumerate(punct_logits):
            label_id = int(np.argmax(row))
            if label_id == 0:
                continue
            punct = self._punct_map.get(label_id)
            if not punct:
                continue
            if idx >= len(word_end_indices):
                break
            confidence = self._softmax(row)[label_id]
            positions.append(
                PuncPosition(
                    char_index=word_end_indices[idx],
                    punctuation=punct,
                    confidence=float(confidence),
                )
            )
        return positions

    def _tokenize(self, text: str) -> Tuple[List[int], List[int], List[int]]:
        spm = self._load_spm()
        if spm is not None:
            return self._tokenize_with_spm(text, spm)
        token_to_id = self._load_vocab()
        tokens: List[int] = []
        valid_ids: List[int] = []
        word_end_indices: List[int] = []

        for match in re.finditer(r"\S+", text):
            word = match.group(0)
            word_tokens = self._tokenize_word(word, token_to_id)
            if not word_tokens:
                word_tokens = [self._unk_id]
            for idx, tok in enumerate(word_tokens):
                tokens.append(tok)
                valid_ids.append(1 if idx == 0 else 0)
            word_end_indices.append(match.end() - 1)

        # 追加 BOS/EOS，提高与训练时输入格式的兼容性
        bos_id = token_to_id.get("<s>")
        eos_id = token_to_id.get("</s>")
        if bos_id is not None and eos_id is not None:
            tokens = [bos_id] + tokens + [eos_id]
            valid_ids = [1] + valid_ids + [1]

        return tokens, valid_ids, word_end_indices

    def _tokenize_with_spm(self, text: str, spm) -> Tuple[List[int], List[int], List[int]]:
        tokens: List[int] = []
        valid_ids: List[int] = []
        word_end_indices: List[int] = []
        unk_id = spm.unk_id()

        for match in re.finditer(r"\S+", text):
            word = match.group(0)
            word_tokens = spm.encode(word, out_type=int)
            if not word_tokens:
                word_tokens = [unk_id]
            for idx, tok in enumerate(word_tokens):
                tokens.append(int(tok))
                valid_ids.append(1 if idx == 0 else 0)
            word_end_indices.append(match.end() - 1)

        bos_id = spm.bos_id()
        eos_id = spm.eos_id()
        if bos_id != -1 and eos_id != -1:
            tokens = [bos_id] + tokens + [eos_id]
            valid_ids = [1] + valid_ids + [1]

        return tokens, valid_ids, word_end_indices

    def _tokenize_word(self, word: str, token_to_id: Dict[str, int]) -> List[int]:
        if not word:
            return []
        sequence = "▁" + word
        length = len(sequence)
        neg_inf = -1.0e18
        scores = self._token_scores or {}

        best = [neg_inf] * (length + 1)
        back: List[Optional[Tuple[int, str]]] = [None] * (length + 1)
        best[0] = 0.0

        for idx in range(length):
            if best[idx] <= neg_inf / 2:
                continue
            max_len = min(self._max_token_len, length - idx)
            for size in range(1, max_len + 1):
                sub = sequence[idx : idx + size]
                score = scores.get(sub)
                if score is None:
                    continue
                candidate = best[idx] + score
                if candidate > best[idx + size]:
                    best[idx + size] = candidate
                    back[idx + size] = (idx, sub)
            # 保底使用 <unk> 逐字符吞掉，避免分词失败
            unk_candidate = best[idx] + self._unk_score
            if unk_candidate > best[idx + 1]:
                best[idx + 1] = unk_candidate
                back[idx + 1] = (idx, "<unk>")

        if best[length] <= neg_inf / 2:
            return [self._unk_id]

        pieces: List[int] = []
        cursor = length
        while cursor > 0:
            node = back[cursor]
            if node is None:
                return [self._unk_id]
            prev, token = node
            pieces.append(token_to_id.get(token, self._unk_id))
            cursor = prev
        pieces.reverse()
        return pieces

    def _load_vocab(self) -> Dict[str, int]:
        if self._token_to_id is not None:
            return self._token_to_id
        model_dir = self._model_dir
        if model_dir is None:
            raise RuntimeError("模型目录未就绪")
        vocab_path = Path(model_dir) / "bpe.vocab"
        if not vocab_path.exists():
            vocab_path = Path(model_dir) / "vocab.txt"
        tokens: List[str] = []
        scores: Dict[str, float] = {}
        with vocab_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                token = parts[0].strip()
                tokens.append(token)
                if len(parts) > 1:
                    try:
                        scores[token] = float(parts[1])
                    except ValueError:
                        scores[token] = 0.0
                else:
                    scores[token] = 0.0
        self._token_to_id = {token: idx for idx, token in enumerate(tokens)}
        self._token_scores = scores
        self._unk_id = self._token_to_id.get("<unk>", 0)
        if scores:
            self._unk_score = min(scores.values()) - 1.0
        self._max_token_len = max((len(token) for token in tokens), default=1)
        return self._token_to_id

    def _load_spm(self):
        if self._spm is not None:
            return self._spm
        model_dir = self._model_dir
        if model_dir is None:
            return None
        spm_path = Path(model_dir) / "bpe.model"
        if not spm_path.exists():
            return None
        try:
            import sentencepiece as spm  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"SentencePiece 未安装，无法加载 {spm_path}") from exc
        processor = spm.SentencePieceProcessor()
        processor.load(str(spm_path))
        self._spm = processor
        return self._spm

    @staticmethod
    def _softmax(row: np.ndarray) -> np.ndarray:
        row = row.astype(np.float32)
        row = row - np.max(row)
        exp_row = np.exp(row)
        return exp_row / np.sum(exp_row)
