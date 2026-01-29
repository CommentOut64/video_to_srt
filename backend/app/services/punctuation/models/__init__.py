"""
标点模型适配器集合（当前为轻量兜底实现）。
"""
from app.services.punctuation.models.onnx_base import OnnxPunctuationAdapter
from app.services.punctuation.models.ct_transformer_onnx import CTTransformerOnnxAdapter
from app.services.punctuation.models.edge_punct_onnx import EdgePunctOnnxAdapter
from app.services.punctuation.models.distilbert_punct_onnx import DistilBertPunctOnnxAdapter
from app.services.punctuation.models.char_bert_onnx import CharBertOnnxAdapter
from app.services.punctuation.models.punct_cap_seg_onnx import PunctCapSegOnnxAdapter
from app.services.punctuation.models.wetext_processor import WeTextProcessor

# 兼容旧命名（保留导出）
CTTransformerONNX = CTTransformerOnnxAdapter
EdgePunctONNX = EdgePunctOnnxAdapter
DistilBertPunctONNX = DistilBertPunctOnnxAdapter
CharBertONNX = CharBertOnnxAdapter
PunctCapSegONNX = PunctCapSegOnnxAdapter

__all__ = [
    "OnnxPunctuationAdapter",
    "CTTransformerOnnxAdapter",
    "EdgePunctOnnxAdapter",
    "DistilBertPunctOnnxAdapter",
    "CharBertOnnxAdapter",
    "PunctCapSegOnnxAdapter",
    "CTTransformerONNX",
    "EdgePunctONNX",
    "DistilBertPunctONNX",
    "CharBertONNX",
    "PunctCapSegONNX",
    "WeTextProcessor",
]
