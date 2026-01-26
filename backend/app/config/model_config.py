"""
模型预加载配置文件
"""

import os
from typing import List


class ModelPreloadConfig:
    """保留配置打印，预加载功能已并入 ModelManager V2。"""

    ENABLED = os.getenv("MODEL_PRELOAD_ENABLED", "true").lower() == "true"
    DEFAULT_MODELS = os.getenv("MODEL_PRELOAD_MODELS", "medium").split(",")
    MAX_CACHE_SIZE = int(os.getenv("MODEL_CACHE_SIZE", "3"))
    MEMORY_THRESHOLD = float(os.getenv("MODEL_MEMORY_THRESHOLD", "0.8"))
    PRELOAD_TIMEOUT = int(os.getenv("MODEL_PRELOAD_TIMEOUT", "300"))
    WARMUP_ENABLED = os.getenv("MODEL_WARMUP_ENABLED", "true").lower() == "true"
    MEMORY_CHECK_INTERVAL = int(os.getenv("MEMORY_CHECK_INTERVAL", "60"))  # 秒

    @classmethod
    def print_config(cls):
        """打印当前配置（仅日志用途）。"""
        print("模型管理配置:")
        print(f"  默认模型: {cls.DEFAULT_MODELS}")
        print(f"  最大缓存大小: {cls.MAX_CACHE_SIZE}")
        print(f"  内存阈值: {cls.MEMORY_THRESHOLD}")
        print(f"  预加载超时: {cls.PRELOAD_TIMEOUT}s")
        print(f"  启用预热标记: {cls.WARMUP_ENABLED}")


# 常用模型配置
WHISPER_MODELS = {
    "tiny": {"size": "~39MB", "speed": "~32x", "memory": "~1GB"},
    "base": {"size": "~74MB", "speed": "~16x", "memory": "~1GB"},
    "small": {"size": "~244MB", "speed": "~6x", "memory": "~2GB"},
    "medium": {"size": "~769MB", "speed": "~2x", "memory": "~5GB"},
    "large": {"size": "~1550MB", "speed": "~1x", "memory": "~10GB"},
    "large-v2": {"size": "~1550MB", "speed": "~1x", "memory": "~10GB"},
    "large-v3": {"size": "~1550MB", "speed": "~1x", "memory": "~10GB"}
}

def get_model_info(model_name: str) -> dict:
    """获取模型信息"""
    return WHISPER_MODELS.get(model_name, {"size": "Unknown", "speed": "Unknown", "memory": "Unknown"})

def recommend_models_by_memory(total_memory_gb: float) -> List[str]:
    """根据可用内存推荐模型"""
    if total_memory_gb < 4:
        return ["tiny", "base"]
    elif total_memory_gb < 8:
        return ["tiny", "base", "small"]
    elif total_memory_gb < 16:
        return ["base", "small", "medium"]
    else:
        return ["medium", "large"]


# ========== Whisper 幻觉抑制配置 ==========
# 通过 scripts/extract_hallucination_tokens.py 生成
# 注意: 不同模型的 Token ID 可能不同，需分别配置
#
# 安全封杀原则:
# - 封杀下划线相关 Token (幻觉主要来源)
# - 封杀 YouTube 风格幻觉词的首 Token
# - 不封杀常见标点和单字母 (如 '.', '[', 'C')

WHISPER_SUPPRESS_TOKENS = {
    # Whisper Medium 模型
    # 运行 `python scripts/extract_hallucination_tokens.py --model medium` 获取
    "medium": [
        # 下划线类 (最常见的幻觉来源)
        62,      # '_' 单个下划线
        10852,   # '__' 双下划线
        23757,   # '____' 四下划线

        # 省略号类
        485,     # '...' 省略号
        353,     # '..' 双点

        # YouTube 风格幻觉 (带空格版本更安全)
        27738,   # ' Questions' 带空格的 Questions
        8511,    # ' Subtitles' 带空格的 Subtitles (首 token)
        25653,   # ' Copyright' 带空格的 Copyright (首 token)
        27917,   # 'Thanks for watching' 首 token
        16216,   # 'Please subscribe' 首 token
        2012,    # ' Amara' 带空格的 Amara (首 token)

        # 音乐符号
        3961,    # 音符符号

        # 注意: 以下 Token 被排除，因为可能误杀正常文本
        # 13,    # '.' 单点 - 太常见，不能封杀
        # 34,    # 'C' (Copyright首字母) - 单字母，不能封杀
        # 58,    # '[' 方括号 - 太常见，不能封杀
        # 8547,  # 'Questions' 无空格版本 - 可能误杀正常问句
        # 39582, # 'Subtitles' 无空格版本 - 首 token 是 'Sub'，可能误杀
    ],

    # Whisper Large-v3 模型 (未来扩展)
    # 运行 `python scripts/extract_hallucination_tokens.py --model large-v3` 获取
    "large-v3": [
        # TODO: 运行脚本后填入实际 Token ID
    ],
}


def get_whisper_suppress_tokens(model_name: str) -> List[int]:
    """
    获取指定模型的幻觉抑制 Token ID 列表

    Args:
        model_name: 模型名称 (如 "medium", "large-v3")

    Returns:
        list: Token ID 列表，用于 suppress_tokens 参数
    """
    model_name_lower = model_name.lower()
    for key, tokens in WHISPER_SUPPRESS_TOKENS.items():
        if key in model_name_lower:
            return tokens
    return []
