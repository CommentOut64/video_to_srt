"""
ONNX 标点模型基类（轻量兜底实现）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.model_manager_v2 import get_model_manager_v2
from app.services.punctuation.base import PuncPosition, PunctuationModelAdapter


class OnnxPunctuationAdapter(PunctuationModelAdapter):
    """ONNX 标点模型适配器（策略模式的底层适配层）。"""

    def __init__(
        self,
        model_id: str,
        *,
        default_punctuation: str,
        terminal_punctuations: str,
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self._loaded = False
        self._model_path: Optional[str] = None
        self._model_dir: Optional[Path] = None
        self._session = None
        self._last_error: Optional[str] = None
        self._default_punctuation = default_punctuation
        self._terminal_punctuations = set(terminal_punctuations)
        # 标点模型默认走 CPU，避免因 CUDA 环境不完整导致加载失败
        self._device = device or "cpu"

    def load(self, model_path: str) -> None:
        self._model_path = model_path
        self._ensure_session()

    def predict(self, text: str) -> List[PuncPosition]:
        if not text:
            return []
        session = self._ensure_session()
        if session is None:
            return self._fallback_predict(text)
        try:
            return self._predict_with_session(session, text)
        except Exception as exc:
            self._last_error = str(exc)
            return self._fallback_predict(text)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "loaded": self._loaded,
            "model_path": self._model_path,
            "model_dir": str(self._model_dir) if self._model_dir else None,
            "last_error": self._last_error,
            "adapter": self.__class__.__name__,
        }

    def unload(self) -> None:
        self._loaded = False
        self._model_path = None
        self._model_dir = None
        self._session = None

    def _ensure_session(self):
        if self._session is not None:
            return self._session
        try:
            manager = get_model_manager_v2()
            session = manager.acquire(self.model_id, device=self._device)
            spec = manager.registry.get(self.model_id)
            self._model_dir = Path(manager.downloader.ensure_local(spec))
            self._session = session
            self._loaded = True
            return self._session
        except Exception as exc:
            self._last_error = str(exc)
            return None

    def _predict_with_session(self, session, text: str) -> List[PuncPosition]:
        """由子类实现具体推理逻辑。"""
        raise NotImplementedError

    def _fallback_predict(self, text: str) -> List[PuncPosition]:
        trimmed = text.rstrip()
        if not trimmed:
            return []
        if trimmed[-1] in self._terminal_punctuations:
            return []
        return [
            PuncPosition(
                char_index=len(trimmed) - 1,
                punctuation=self._default_punctuation,
                confidence=0.3,
            )
        ]
