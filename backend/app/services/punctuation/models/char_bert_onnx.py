"""
字符级 BERT ONNX 适配器（轻量兜底实现）。
"""
from typing import Optional

from app.services.punctuation.models.onnx_base import OnnxPunctuationAdapter


class CharBertOnnxAdapter(OnnxPunctuationAdapter):
    """日文字符级 BERT 模型适配器。"""

    def __init__(self, model_id: Optional[str] = None) -> None:
        super().__init__(
            model_id=model_id or "punct-char-bert-ja",
            default_punctuation="。",
            terminal_punctuations="。！？",
        )
