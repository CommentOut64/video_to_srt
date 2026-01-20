"""
WhisperExecutor - Whisper 推理执行器

Phase 3 实现 - 2025-12-10

封装 WhisperService，提供统一的执行器接口。
"""

import logging
from typing import Optional, Dict, Any, Union
import numpy as np

from app.services.whisper_service import WhisperService


class WhisperExecutor:
    """
    Whisper 推理执行器
    
    封装 WhisperService，提供统一的执行器接口。
    """
    
    def __init__(
        self,
        service: Optional[WhisperService] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化 Whisper 执行器
        
        Args:
            service: Whisper 服务实例
            logger: 日志记录器
        """
        self.service = service or WhisperService()
        self.logger = logger or logging.getLogger(__name__)

    def _resolve_chunk_param_overrides(self) -> Dict[str, Any]:
        """
        解析 Chunk 场景的默认覆盖参数。

        仅当运行参数来源为 default 时才应用覆盖，
        以便用户显式配置时优先使用统一管理参数。
        """
        from app.services.model_runtime_config_service import get_model_runtime_config_service
        from app.services.model_manager_v2 import get_model_manager_v2

        try:
            model_id = self.service.resolve_model_id()
            manager = get_model_manager_v2()
            spec = manager.registry.get(model_id)
            runtime_data = get_model_runtime_config_service().get_effective_runtime_for_model(spec)
            sources = runtime_data.get("sources", {})
        except Exception as exc:
            self.logger.debug("Whisper Chunk 默认参数回退: %s", exc)
            return {
                "vad_filter": False,
                "condition_on_previous_text": False,
            }

        overrides: Dict[str, Any] = {}
        if sources.get("vad_filter") == "default":
            overrides["vad_filter"] = False
        if sources.get("condition_on_previous_text") == "default":
            overrides["condition_on_previous_text"] = False
        return overrides
    
    async def execute(
        self,
        audio: Union[str, np.ndarray],
        start_time: float,
        end_time: float,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        执行 Whisper 推理

        Args:
            audio: 音频数组（已切片的 Chunk 音频，不需要再次切片）
            start_time: Chunk 起始时间（秒）- 仅用于日志，不用于切片
            end_time: Chunk 结束时间（秒）- 仅用于日志，不用于切片
            language: 语言代码（zh/en/auto）
            initial_prompt: 初始提示词（用于引导识别）
            repetition_penalty: 重复惩罚系数，>1 抑制重复（None 表示使用运行参数默认值）
            no_repeat_ngram_size: 禁止重复的 N-gram 大小（None 表示使用运行参数默认值）

        Returns:
            Dict: 推理结果
                - text: 识别文本
                - confidence: 置信度估算
                - language: 检测到的语言

        Note:
            传入的 audio 应该是已经切片好的 Chunk 音频，
            不会再进行二次切片。
        """
        # 自动加载模型（如果未加载）
        if not self.is_loaded():
            self.logger.info('Whisper 模型未加载，正在加载...')
            self.service.load_model()
            self.logger.info('Whisper 模型加载完成')

        duration = end_time - start_time
        self.logger.debug(
            f'执行 Whisper 推理: start={start_time:.2f}s, end={end_time:.2f}s, '
            f'duration={duration:.2f}s, '
            f'prompt={initial_prompt[:50] if initial_prompt else None}'
        )

        # 直接调用 transcribe，不进行二次切片
        # 传入的 audio 已经是切片后的 Chunk 音频
        overrides = self._resolve_chunk_param_overrides()
        result = self.service.transcribe(
            audio=audio,
            language=language,
            initial_prompt=initial_prompt,
            repetition_penalty=repetition_penalty,  # 重复惩罚
            no_repeat_ngram_size=no_repeat_ngram_size,  # N-gram 重复抑制
            **overrides
        )
        
        # 估算置信度
        confidence = self.service.estimate_confidence(result)

        # 提取文本
        text = result.get('text', '').strip()

        self.logger.debug(f'Whisper 推理完成: text={text}, confidence={confidence:.2f}')
        
        return {
            'text': text,
            'confidence': confidence,
            'language': result.get('language', language),
            'raw_result': result
        }
    
    def is_loaded(self) -> bool:
        """
        检查模型是否已加载
        
        Returns:
            bool: 是否已加载
        """
        return self.service.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict: 模型信息
        """
        return {
            'model_name': self.service.config.model_name if self.service.config else 'unknown',
            'model_type': 'Faster-Whisper',
            'is_loaded': self.is_loaded(),
            'supported_languages': ['zh', 'en', 'ja', 'ko', 'auto']
        }
