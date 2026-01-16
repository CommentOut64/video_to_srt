"""Loader 实现包。"""

from app.core.asr.loaders.onnx_loader import OnnxLoader  # noqa: F401
from app.core.asr.loaders.torch_loader import TorchLoader  # noqa: F401
from app.core.asr.loaders.ct2_loader import CT2Loader  # noqa: F401
from app.core.asr.loaders.external_loader import ExternalLoader  # noqa: F401
