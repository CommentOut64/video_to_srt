"""
中文 CT-Transformer ONNX 适配器。
"""

from __future__ import annotations

import logging

from app.services.punctuation.models.onnx_base import OnnxPunctuationAdapter


class CTTransformerOnnxAdapter(OnnxPunctuationAdapter):
    """CT-Transformer 中文标点模型。"""

    def __init__(self, model_id: str = "punct-ct-transformer-zh", logger: logging.Logger | None = None) -> None:
        super().__init__(
            model_id=model_id,
            terminal_punctuations=["。", "！", "？", "!", "?"],
            default_punctuation="。",
            logger=logger,
        )
