"""
同音检索服务模块。

V3.2.0+dev.20260210.02
"""

from .db import GlobalTermRule, HomophoneDb, IndexState, PostingRecord
from .runtime import HomophoneRuntime, get_homophone_runtime, get_homophone_service, index_chunk_async
from .service import HomophoneMatch, HomophoneService
from .tokenizers import TokenReading

__all__ = [
    "GlobalTermRule",
    "HomophoneDb",
    "HomophoneRuntime",
    "HomophoneMatch",
    "HomophoneService",
    "IndexState",
    "PostingRecord",
    "TokenReading",
    "get_homophone_runtime",
    "get_homophone_service",
    "index_chunk_async",
]
