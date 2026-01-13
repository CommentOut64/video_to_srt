"""
置信度到准确率的映射器

V3.1.2+dev.20260111.01: 新增置信度映射机制

核心设计：
1. SenseVoice CTC置信度：原始范围0.1-0.5（中文场景），映射到0.50-0.98
2. Whisper logprob置信度：原始范围-2.5~0，映射到0.15-0.98
3. 映射后的值称为 display_confidence，用于前端显示
4. 原始值 confidence 保持不变，用于内部逻辑（补刀触发、熔断决策）

设计原则：
- 不夸大低质量结果：原始值很低时，映射后仍明确提示需要审核
- 不制造焦虑：正常质量的结果映射到合理区间，避免用户误解
- 上限不为100%：即使最高置信度也只映射到0.98，避免过度承诺
- 保留区分度：映射是单调递增的，高低置信度仍可区分
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ConfidenceMapper:
    """置信度到准确率的映射器"""

    # 映射参数（可调整）
    # SenseVoice: 原始范围 0.1-0.5 映射到 0.50-0.98
    SV_RAW_MIN = 0.1
    SV_RAW_MAX = 0.5
    SV_DISPLAY_MIN = 0.50
    SV_DISPLAY_MAX = 0.98

    # Whisper: logprob 范围 -2.5~0 映射到 0.15-0.98
    WHISPER_LOGPROB_HIGH = -0.2   # 高置信度阈值
    WHISPER_LOGPROB_GOOD = -0.5   # 较好置信度阈值
    WHISPER_LOGPROB_MED = -0.8    # 中等置信度阈值
    WHISPER_LOGPROB_LOW = -1.0    # 低置信度阈值

    @classmethod
    def map_sensevoice(cls, raw_conf: float) -> float:
        """
        SenseVoice CTC置信度映射

        V3.1.2+dev.20260111.02: 修正映射范围
        原始范围：0.3-0.7（中文场景实测值）
        映射目标：0.50-0.98

        映射策略：
        - 0.7+ -> 0.98 (极高置信度，上限)
        - 0.6-0.7 -> 0.90-0.98 (高置信度)
        - 0.5-0.6 -> 0.78-0.90 (中高置信度) <- 大部分数据在这个区间
        - 0.4-0.5 -> 0.65-0.78 (中等置信度)
        - 0.3-0.4 -> 0.50-0.65 (较低置信度)
        - <0.3 -> 0.30-0.50 (极低置信度)

        Args:
            raw_conf: 原始置信度 (0.0-1.0)

        Returns:
            映射后的准确率 (0.0-1.0)
        """
        if raw_conf >= 0.7:
            # 极高置信度，上限0.98
            return 0.98
        elif raw_conf >= 0.6:
            # 0.6-0.7 -> 0.90-0.98
            return 0.90 + (raw_conf - 0.6) * 0.8
        elif raw_conf >= 0.5:
            # 0.5-0.6 -> 0.78-0.90 (大部分数据在这里)
            return 0.78 + (raw_conf - 0.5) * 1.2
        elif raw_conf >= 0.4:
            # 0.4-0.5 -> 0.65-0.78
            return 0.65 + (raw_conf - 0.4) * 1.3
        elif raw_conf >= 0.3:
            # 0.3-0.4 -> 0.50-0.65
            return 0.50 + (raw_conf - 0.3) * 1.5
        else:
            # <0.3 -> 0.30-0.50（极低，但不是0）
            return max(0.30, 0.50 + (raw_conf - 0.3) * 0.67)

    @classmethod
    def map_whisper(cls, avg_logprob: float, no_speech_prob: float = 0.0) -> float:
        """
        Whisper logprob映射

        原始范围：-2.5 ~ 0（avg_logprob，越接近0越好）
        映射目标：0.15-0.98

        Args:
            avg_logprob: 平均对数概率 (负值，越接近0越好)
            no_speech_prob: 静音概率 (0.0-1.0)

        Returns:
            映射后的准确率 (0.0-1.0)
        """
        # 分段线性映射（基于实证分布）
        if avg_logprob >= -0.2:
            # 极高置信度: -0.2~0 -> 0.92-0.98
            conf = 0.92 + (avg_logprob + 0.2) * 0.3
        elif avg_logprob >= -0.5:
            # 高置信度: -0.5~-0.2 -> 0.80-0.92
            conf = 0.80 + (avg_logprob + 0.5) * 0.4
        elif avg_logprob >= -0.8:
            # 中等置信度: -0.8~-0.5 -> 0.65-0.80
            conf = 0.65 + (avg_logprob + 0.8) * 0.5
        elif avg_logprob >= -1.0:
            # 较低置信度: -1.0~-0.8 -> 0.50-0.65
            conf = 0.50 + (avg_logprob + 1.0) * 0.75
        else:
            # 极低置信度: <-1.0 -> 0.15-0.50（不截断为0，保留信息）
            conf = max(0.15, 0.50 + (avg_logprob + 1.0) * 0.35)

        # 静音因子独立处理（不是线性相加，而是乘法衰减）
        if no_speech_prob > 0.3:
            conf *= (1.0 - no_speech_prob * 0.8)

        return round(max(0.0, min(0.98, conf)), 3)

    @classmethod
    def map(cls, raw_conf: float, source: str) -> float:
        """
        统一映射入口

        Args:
            raw_conf: 原始置信度
            source: 来源 ("sensevoice", "whisper", "whisper_patch", "llm", "imported" 等)

        Returns:
            映射后的准确率
        """
        if source in ("whisper", "whisper_patch"):
            # Whisper来源：假设raw_conf已经是经过_estimate_whisper_confidence处理的值
            # 这里需要反推logprob或直接使用
            # 由于raw_conf已经是0-1范围，这里做轻微调整
            return cls._map_whisper_normalized(raw_conf)
        elif source in ("sensevoice", "sensevoice_patched"):
            return cls.map_sensevoice(raw_conf)
        elif source == "imported":
            # 导入的字幕，置信度为1.0，直接返回
            return raw_conf
        else:
            # 未知来源，保守处理
            return cls.map_sensevoice(raw_conf)

    @classmethod
    def _map_whisper_normalized(cls, normalized_conf: float) -> float:
        """
        映射已经归一化的Whisper置信度

        由于_estimate_whisper_confidence已经做了 1.0 + avg_logprob 的处理，
        这里对结果做轻微调整，使其更合理。

        Args:
            normalized_conf: 已归一化的置信度 (0.0-1.0)

        Returns:
            映射后的准确率
        """
        # Whisper的归一化置信度通常在0.3-1.0范围
        # 映射到0.50-0.98，比SenseVoice略宽松
        if normalized_conf >= 0.9:
            return 0.98
        elif normalized_conf >= 0.7:
            # 0.7-0.9 -> 0.85-0.98
            return 0.85 + (normalized_conf - 0.7) * 0.65
        elif normalized_conf >= 0.5:
            # 0.5-0.7 -> 0.70-0.85
            return 0.70 + (normalized_conf - 0.5) * 0.75
        elif normalized_conf >= 0.3:
            # 0.3-0.5 -> 0.55-0.70
            return 0.55 + (normalized_conf - 0.3) * 0.75
        else:
            # <0.3 -> 0.35-0.55
            return max(0.35, 0.55 + (normalized_conf - 0.3) * 0.67)

    @classmethod
    def get_display_thresholds(cls, source: str) -> dict:
        """
        获取前端显示的阈值配置

        根据来源返回不同的警告阈值，因为不同引擎的置信度分布不同。

        Args:
            source: 来源

        Returns:
            dict: {"good": float, "warning": float, "danger": float}
        """
        if source in ("whisper", "whisper_patch"):
            return {
                "good": 0.85,      # >= 0.85 绿色
                "warning": 0.65,   # >= 0.65 黄色
                "danger": 0.0      # < 0.65 红色
            }
        else:
            # SenseVoice 或其他
            return {
                "good": 0.82,      # >= 0.82 绿色
                "warning": 0.68,   # >= 0.68 黄色
                "danger": 0.0      # < 0.68 红色
            }


# 单例访问
_mapper_instance: Optional[ConfidenceMapper] = None


def get_confidence_mapper() -> ConfidenceMapper:
    """获取置信度映射器单例"""
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = ConfidenceMapper()
    return _mapper_instance
