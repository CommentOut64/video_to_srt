"""
转录参数构建器

负责统一构建 Whisper 推理参数（运行参数 + 局部覆盖）。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.job_models import JobState


class TranscribeParamBuilder:
    """转录参数构建器（Whisper 参数）"""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def build_whisper_params(
        self,
        job: "JobState",
        overrides: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """构建 Whisper 推理参数（运行参数 + 局部覆盖）。"""
        from app.config.model_config import get_whisper_suppress_tokens
        from app.services.model_manager_v2 import get_model_manager_v2
        from app.services.model_runtime_config_service import (
            get_model_runtime_config_service,
        )
        from app.services.runtime_param_resolver import get_runtime_group
        from app.services.whisper_service import get_whisper_service

        transcription = getattr(job.settings, "transcription", None)
        model_name = getattr(transcription, "whisper_model", "medium")
        whisper_service = get_whisper_service()
        override_keys = {
            key for key, value in (overrides or {}).items() if value is not None
        }
        sources: Dict[str, str] = {}

        try:
            model_id = whisper_service.resolve_model_id(model_name)
            manager = get_model_manager_v2()
            spec = manager.registry.get(model_id)
            runtime_data = (
                get_model_runtime_config_service().get_effective_runtime_for_model(spec)
            )
            runtime = runtime_data.get("effective", {})
            sources = runtime_data.get("sources", {})
        except Exception as exc:
            self.logger.debug("Whisper 运行参数回退分组: %s", exc)
            runtime = get_runtime_group("whisper")

        params: Dict[str, Any] = {
            "language": runtime.get("language"),
            "initial_prompt": runtime.get("initial_prompt"),
            "word_timestamps": runtime.get("word_timestamps"),
            "beam_size": runtime.get("beam_size"),
            "vad_filter": runtime.get("vad_filter"),
            "vad_parameters": runtime.get("vad_parameters"),
            "temperature": runtime.get("temperature"),
            "condition_on_previous_text": runtime.get("condition_on_previous_text"),
            "suppress_tokens": runtime.get("suppress_tokens"),
            "repetition_penalty": runtime.get("repetition_penalty"),
            "no_repeat_ngram_size": runtime.get("no_repeat_ngram_size"),
        }

        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    params[key] = value

        language = params.get("language")
        if language is None or language == "auto" or language == "":
            params["language"] = None

        if params.get("word_timestamps") is None:
            params["word_timestamps"] = False
        if params.get("beam_size") is None:
            params["beam_size"] = 5
        if params.get("vad_filter") is None:
            params["vad_filter"] = True
        if params.get("temperature") is None:
            params["temperature"] = 0.0
        if params.get("condition_on_previous_text") is None:
            params["condition_on_previous_text"] = True
        if params.get("repetition_penalty") is None:
            params["repetition_penalty"] = 1.0
        if params.get("no_repeat_ngram_size") is None:
            params["no_repeat_ngram_size"] = 0

        if context == "patch":
            if not sources:
                if "word_timestamps" not in override_keys:
                    params["word_timestamps"] = False
            else:
                if (
                    "word_timestamps" not in override_keys
                    and sources.get("word_timestamps") == "default"
                ):
                    params["word_timestamps"] = False
            # V3.2.0+dev.20260206.04: patch 场景不再强制关闭 condition_on_previous_text，
            # 统一遵循运行参数或显式覆盖，优先保证英文标点连续性。

        if params.get("suppress_tokens") is None:
            suppress_tokens = get_whisper_suppress_tokens(model_name)
            params["suppress_tokens"] = suppress_tokens if suppress_tokens else None

        return params


_param_builder: Optional[TranscribeParamBuilder] = None


def get_transcribe_param_builder(
    logger: Optional[logging.Logger] = None,
) -> TranscribeParamBuilder:
    """获取参数构建器（单例）"""
    global _param_builder
    if _param_builder is None:
        _param_builder = TranscribeParamBuilder(logger=logger)
    elif logger is not None:
        _param_builder.logger = logger
    return _param_builder
