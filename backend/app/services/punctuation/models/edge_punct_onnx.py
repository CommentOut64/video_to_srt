"""
英文 Edge-Punct-Casing ONNX 适配器。
"""

from __future__ import annotations

import logging

from app.services.punctuation.models.onnx_base import OnnxPunctuationAdapter


class EdgePunctOnnxAdapter(OnnxPunctuationAdapter):
    """Edge-Punct-Casing 英文标点模型。"""

    def __init__(self, model_id: str = "punct-edge-punct-en", logger: logging.Logger | None = None) -> None:
        super().__init__(
            model_id=model_id,
            terminal_punctuations=[".", "!", "?"],
            default_punctuation=".",
            logger=logger,
        )
