"""
PunctCapSeg ONNX 适配器（PCS-47Lang）。
V3.2.0+dev.20260129.09
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.model_manager_v2 import get_model_manager_v2
from app.services.punctuation.base import PuncPosition, PunctuationModelAdapter


class PunctCapSegOnnxAdapter(PunctuationModelAdapter):
    """多语标点适配器（适配器模式：封装第三方模型 API 与统一标点位置输出）。"""

    def __init__(
        self,
        model_id: Optional[str] = None,
        *,
        default_punctuation: str = "。",
        terminal_punctuations: str = "。！？.!?",
        punctuation_chars: Optional[str] = None,
    ) -> None:
        self.model_id = model_id or "punct-pcs-47lang"
        self._default_punctuation = default_punctuation
        self._terminal_punctuations = set(terminal_punctuations)
        self._punctuation_chars = set(punctuation_chars or "，。、！？,.;:!?")
        self._model = None
        self._model_dir: Optional[Path] = None
        self._last_error: Optional[str] = None
        self._is_loaded = False
        self._logger = logging.getLogger(__name__)
        self._info_logged = False

    def load(self, model_path: str) -> None:
        self._model_dir = Path(model_path)
        self._model = self._build_model(self._model_dir)
        self._is_loaded = self._model is not None

    def predict(self, text: str) -> List[PuncPosition]:
        if not text:
            return []
        model = self._ensure_model()
        if model is None:
            return self._fallback_predict(text)
        self._log_model_info_once()
        try:
            predicted = self._infer_text(model, text)
            if not predicted:
                return []
            positions = self._extract_positions(text, predicted)
            if positions:
                return positions
            return self._fallback_predict(text)
        except Exception as exc:
            self._last_error = str(exc)
            return self._fallback_predict(text)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "loaded": self._is_loaded,
            "model_dir": str(self._model_dir) if self._model_dir else None,
            "last_error": self._last_error,
            "adapter": self.__class__.__name__,
        }

    def unload(self) -> None:
        self._model = None
        self._model_dir = None
        self._is_loaded = False

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            manager = get_model_manager_v2()
            model_dir = manager.ensure_available(self.model_id)
        except Exception as exc:
            self._last_error = str(exc)
            return None
        self.load(model_dir)
        return self._model

    def _log_model_info_once(self) -> None:
        if self._info_logged:
            return
        info = self.get_model_info()
        model_dir = info.get("model_dir")
        if model_dir:
            directory = Path(model_dir)
            files = {
                "onnx": (directory / "punct_cap_seg_47lang_int8.onnx").exists(),
                "config": (directory / "config.yaml").exists(),
                "spe": (directory / "spe_unigram_64k_lowercase_47lang.model").exists(),
            }
        else:
            files = {"onnx": False, "config": False, "spe": False}
        self._logger.info("PCS-47Lang 模型信息: %s 文件=%s", info, files)
        self._info_logged = True

    def _build_model(self, model_dir: Path):
        try:
            from punctuators.models import PunctCapSegModelONNX
            from punctuators.models.punc_cap_seg_model import PunctCapSegConfigONNX
        except Exception as exc:
            self._last_error = f"缺少 punctuators 依赖: {exc}"
            return None
        try:
            if not model_dir.exists():
                self._last_error = f"模型目录不存在: {model_dir}"
                return None
            cfg = PunctCapSegConfigONNX(
                spe_filename="spe_unigram_64k_lowercase_47lang.model",
                model_filename="punct_cap_seg_47lang_int8.onnx",
                config_filename="config.yaml",
                directory=str(model_dir),
            )
            return PunctCapSegModelONNX(cfg=cfg, ort_providers=["CPUExecutionProvider"])
        except Exception as exc:
            self._last_error = f"加载 PunctCapSeg 模型失败: {exc}"
            return None

    @staticmethod
    def _infer_text(model, text: str) -> str:
        outputs = model.infer([text])
        if not outputs:
            return ""
        first = outputs[0]
        if isinstance(first, list):
            if not first:
                return ""
            return first[0] if isinstance(first[0], str) else ""
        if isinstance(first, str):
            return first
        return ""

    def _extract_positions(self, text: str, predicted: str) -> List[PuncPosition]:
        positions: List[PuncPosition] = []
        src_index = 0
        last_src_index = -1

        for char in predicted:
            if src_index < len(text) and self._char_equal(text[src_index], char):
                last_src_index = src_index
                src_index += 1
                continue
            if char.isspace():
                continue
            if char in self._punctuation_chars:
                insert_index = last_src_index if last_src_index >= 0 else 0
                positions.append(
                    PuncPosition(
                        char_index=insert_index,
                        punctuation=char,
                        confidence=1.0,
                    )
                )
                continue
            if src_index < len(text) and self._char_equal_casefold(text[src_index], char):
                last_src_index = src_index
                src_index += 1
        return positions

    @staticmethod
    def _char_equal(left: str, right: str) -> bool:
        return left == right

    @staticmethod
    def _char_equal_casefold(left: str, right: str) -> bool:
        return left.casefold() == right.casefold()

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
