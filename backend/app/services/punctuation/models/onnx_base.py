"""
ONNX 标点模型适配器基类。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from app.services.model_manager_v2 import get_model_manager_v2
from app.services.punctuation.base import PuncPosition, PunctuationModelAdapter, PunctuationModelError


class OnnxPunctuationAdapter(PunctuationModelAdapter):
    """ONNX 标点模型适配器通用实现。"""

    def __init__(
        self,
        model_id: str,
        terminal_punctuations: Sequence[str],
        default_punctuation: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(model_id)
        self._terminal_punctuations = set(terminal_punctuations)
        self._default_punctuation = default_punctuation
        self._logger = logger or logging.getLogger(__name__)
        self._session: Optional[Any] = None

    def load(self, model_path: str) -> None:
        """通过 ModelManagerV2 加载 ONNX Session。"""
        try:
            manager = get_model_manager_v2()
            manager.ensure_available(self.model_id)
            self._session = manager.acquire(self.model_id)
        except Exception as exc:
            raise PunctuationModelError(f"ONNX 模型加载失败: {self.model_id} -> {exc}") from exc

    def predict(self, text: str) -> List[PuncPosition]:
        """优先尝试真实推理，失败时退回启发式。"""
        if not text:
            return []

        if self._session:
            predicted = self._predict_with_session(text, self._session)
            if predicted is not None:
                return predicted

        return self._fallback_predict(text)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "loaded": self._session is not None,
            "adapter": self.__class__.__name__,
        }

    def unload(self) -> None:
        self._session = None

    def _predict_with_session(self, text: str, session: Any) -> Optional[List[PuncPosition]]:
        """真实推理入口，默认返回 None 触发启发式策略。"""
        return None

    def _fallback_predict(self, text: str) -> List[PuncPosition]:
        """启发式兜底：句末补默认标点。"""
        stripped = text.rstrip()
        if not stripped:
            return []
        last_char_index = len(stripped) - 1
        if stripped[-1] in self._terminal_punctuations:
            return []
        return [
            PuncPosition(
                char_index=last_char_index,
                punctuation=self._default_punctuation,
                confidence=0.3,
            )
        ]
