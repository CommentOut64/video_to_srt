"""标点模型适配器集合。"""

from app.services.punctuation.models.ct_transformer_onnx import CTTransformerOnnxAdapter  # noqa: F401
from app.services.punctuation.models.edge_punct_onnx import EdgePunctOnnxAdapter  # noqa: F401
from app.services.punctuation.models.char_bert_onnx import CharBertOnnxAdapter  # noqa: F401
from app.services.punctuation.models.wetext_processor import WeTextProcessor  # noqa: F401
