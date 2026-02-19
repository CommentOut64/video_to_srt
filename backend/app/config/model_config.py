"""
模型预加载配置文件
"""

import os
from typing import Dict, List


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

DEFAULT_WHISPER_SUPPRESS_TOKENS: List[int] = [
    # 下划线类（最常见幻觉来源）
    62,      # '_' 单个下划线
    10852,   # '__' 双下划线
    23757,   # '____' 四下划线

    # 省略号类
    485,     # '...'
    353,     # '..'

    # YouTube 风格幻觉（带空格版本更安全）
    27738,   # ' Questions'
    8511,    # ' Subtitles'
    25653,   # ' Copyright'
    27917,   # 'Thanks for watching'
    16216,   # 'Please subscribe'
    2012,    # ' Amara'

    # 音乐符号
    3961,
]

# Token Profile（可按模型独立扩展，当前所有模型先复用保守默认列表）
WHISPER_SUPPRESS_TOKENS: Dict[str, List[int]] = {
    "default": list(DEFAULT_WHISPER_SUPPRESS_TOKENS),
    "medium": list(DEFAULT_WHISPER_SUPPRESS_TOKENS),
    "large-v3": list(DEFAULT_WHISPER_SUPPRESS_TOKENS),
}

# 所有 Whisper 模型统一接入 suppress_tokens。
# Why: 先保证全量模型有基础抑制，再逐模型精调 Token Profile。
WHISPER_MODEL_TOKEN_PROFILE: Dict[str, str] = {
    "tiny": "default",
    "tiny.en": "default",
    "base": "default",
    "base.en": "default",
    "small": "default",
    "small.en": "default",
    "medium": "medium",
    "medium.en": "default",
    "large": "default",
    "large-v1": "default",
    "large-v2": "default",
    "large-v3": "large-v3",
    "turbo": "default",
    "large-v3-turbo": "default",
    "distil-large-v2": "default",
    "distil-large-v3": "default",
}


def _normalize_whisper_model_name(model_name: str) -> str:
    """
    归一化 Whisper 模型名，兼容 repo_id / 注册表 id / 别名。
    """
    raw = str(model_name or "").strip().lower()
    if not raw:
        return ""

    tail = raw.split("/")[-1]
    if tail.startswith("faster-whisper-"):
        tail = tail[len("faster-whisper-"):]
    elif tail.startswith("faster-distil-whisper-"):
        tail = f"distil-{tail[len('faster-distil-whisper-'):]}"
    elif tail.startswith("whisper-"):
        tail = tail[len("whisper-"):]

    tail = tail.replace("_", "-")

    # 兼容 whisper-medium-en / tiny-en 这类命名
    if tail.endswith("-en") and not tail.startswith("distil-"):
        tail = f"{tail[:-3]}.en"
    return tail


def get_whisper_suppress_tokens(model_name: str) -> List[int]:
    """
    获取指定模型的幻觉抑制 Token ID 列表

    Args:
        model_name: 模型名称 (如 "medium", "large-v3")

    Returns:
        list: Token ID 列表，用于 suppress_tokens 参数
    """
    normalized = _normalize_whisper_model_name(model_name)
    if not normalized:
        return list(WHISPER_SUPPRESS_TOKENS.get("default", []))

    profile = WHISPER_MODEL_TOKEN_PROFILE.get(normalized)
    if profile is None:
        # 向后兼容：允许按片段匹配既有 profile（如自定义变体名称）
        for key in WHISPER_MODEL_TOKEN_PROFILE:
            if key in normalized:
                profile = WHISPER_MODEL_TOKEN_PROFILE[key]
                break

    if profile is None:
        profile = "default"

    return list(WHISPER_SUPPRESS_TOKENS.get(profile, WHISPER_SUPPRESS_TOKENS["default"]))
