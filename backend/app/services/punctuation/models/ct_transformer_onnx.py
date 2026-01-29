"""
CT-Transformer ONNX 适配器（真实模型推理 + 兜底）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.services.punctuation.base import PuncPosition
from app.services.punctuation.models.onnx_base import OnnxPunctuationAdapter


class CTTransformerOnnxAdapter(OnnxPunctuationAdapter):
    """中文 CT-Transformer 模型适配器。"""

    def __init__(self, model_id: Optional[str] = None) -> None:
        super().__init__(
            model_id=model_id or "punct-ct-transformer-zh",
            default_punctuation="。",
            terminal_punctuations="。！？",
        )
        self._token_to_id: Optional[Dict[str, int]] = None
        self._punc_list: Optional[List[str]] = None
        self._unk_id: int = 0

    def _predict_with_session(self, session, text: str) -> List[PuncPosition]:
        if not text:
            return []
        token_ids = self._encode(text)
        if not token_ids:
            return []
        inputs = np.array([token_ids], dtype=np.int32)
        lengths = np.array([len(token_ids)], dtype=np.int32)
        outputs = session.run(
            None,
            {
                "inputs": inputs,
                "text_lengths": lengths,
            },
        )
        logits = outputs[0][0]  # shape: (len, 6)
        punc_list = self._punc_list or []
        positions: List[PuncPosition] = []
        for idx, row in enumerate(logits):
            if idx >= len(text):
                break
            if text[idx].isspace():
                continue
            label_id = int(np.argmax(row))
            if label_id >= len(punc_list):
                continue
            label = punc_list[label_id]
            if label in {"_", "<unk>"}:
                continue
            confidence = self._softmax(row)[label_id]
            positions.append(PuncPosition(char_index=idx, punctuation=label, confidence=float(confidence)))
        return positions

    def _encode(self, text: str) -> List[int]:
        token_to_id = self._load_tokenizer()
        return [token_to_id.get(char, self._unk_id) for char in text]

    def _load_tokenizer(self) -> Dict[str, int]:
        if self._token_to_id is not None:
            return self._token_to_id
        model_dir = self._model_dir
        if model_dir is None:
            raise RuntimeError("模型目录未就绪")
        tokens_path = Path(model_dir) / "tokens.json"
        config_json_path = Path(model_dir) / "config.json"
        config_yaml_path = Path(model_dir) / "config.yaml"
        tokens = json.loads(tokens_path.read_text(encoding="utf-8"))
        self._token_to_id = {token: idx for idx, token in enumerate(tokens)}
        self._unk_id = self._token_to_id.get("<unk>", 0)
        if config_json_path.exists():
            config = json.loads(config_json_path.read_text(encoding="utf-8"))
        elif config_yaml_path.exists():
            import yaml

            config = yaml.safe_load(config_yaml_path.read_text(encoding="utf-8")) or {}
        else:
            raise FileNotFoundError(f"未找到 config.json/config.yaml: {model_dir}")
        self._punc_list = config.get("model_conf", {}).get("punc_list", [])
        return self._token_to_id

    @staticmethod
    def _softmax(row: np.ndarray) -> np.ndarray:
        row = row.astype(np.float32)
        row = row - np.max(row)
        exp_row = np.exp(row)
        return exp_row / np.sum(exp_row)
