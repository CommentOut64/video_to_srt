"""
SenseVoiceExecutor - SenseVoice 推理执行器

Phase 3 实现 - 2025-12-10

封装 SenseVoiceONNXService，提供统一的执行器接口。
"""

import logging
from typing import Optional, Dict, Any
import numpy as np

from app.services.sensevoice_onnx_service import SenseVoiceONNXService
from app.models.sensevoice_models import SentenceSegment


class SenseVoiceExecutor:
    """
    SenseVoice 推理执行器
    
    封装 SenseVoiceONNXService，提供统一的执行器接口。
    """
    
    def __init__(
        self,
        service: Optional[SenseVoiceONNXService] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化 SenseVoice 执行器
        
        Args:
            service: SenseVoice ONNX 服务实例
            logger: 日志记录器
        """
        self.service = service or SenseVoiceONNXService()
        self.logger = logger or logging.getLogger(__name__)
    
    async def execute(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        use_itn: bool = True
    ) -> Dict[str, Any]:
        """
        执行 SenseVoice 推理

        Args:
            audio_array: 音频数组
            sample_rate: 采样率
            language: 语言代码（zh/en/auto）
            use_itn: 是否使用 ITN（逆文本归一化）

        Returns:
            Dict: 推理结果
                - text: 识别文本
                - text_clean: 清理后的文本
                - words: 字级时间戳列表
                - confidence: 平均置信度
                - language: 检测到的语言
                - emotion: 情感标签
                - event: 事件标签
        """
        # 自动加载模型（如果未加载）
        if not self.is_loaded():
            self.logger.info('SenseVoice 模型未加载，正在加载...')
            self.service.load_model()
            self.logger.info('SenseVoice 模型加载完成')

        self.logger.debug(f'执行 SenseVoice 推理: audio_len={len(audio_array)}, sr={sample_rate}')

        # 调用底层服务
        result = self.service.transcribe_audio_array(
            audio_array=audio_array,
            sample_rate=sample_rate,
            language=language,
            use_itn=use_itn
        )

        self.logger.debug(f'SenseVoice 推理完成: text={result.get("text", "")}')

        return result
    
    def is_loaded(self) -> bool:
        """
        检查模型是否已加载

        Returns:
            bool: 是否已加载
        """
        return self.service.session is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict: 模型信息
        """
        return {
            'model_name': 'SenseVoice',
            'model_type': 'ONNX',
            'is_loaded': self.is_loaded(),
            'supported_languages': ['zh', 'en', 'ja', 'ko', 'yue']
        }
