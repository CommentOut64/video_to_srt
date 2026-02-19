"""
Prompt Builder - 安全的 Whisper Prompt 构建器
V3.2.0+dev.20260218.01

Phase 4 实现 - 2025-12-10
Phase 4 优化 - 2025-12-11

提供轻量级的关键词提取，避免 initial_prompt 干扰 Whisper 识别。

优化内容：
1. 智能长度控制 - 根据 Prompt 长度限制动态调整
2. 词频统计 - 优先选择高频关键词
3. LRU 缓存 - 避免重复计算
4. 中文支持 - 添加中文关键词提取

更新日志：
- V3.2.0+dev.20260205.04: 使用空格分隔关键词，避免逗号密集导致 Whisper 标点感染
- V3.2.0+dev.20260218.01: 禁用 `Glossary:` 模板，避免提示词回显污染正文
"""

import re
from typing import List, Set, Optional, Dict
from collections import Counter
from functools import lru_cache


class PromptBuilder:
    """
    安全的 Whisper Prompt 构建器

    核心策略：
    1. 从前文中提取专有名词（大写开头的词）
    2. 过滤停用词
    3. 构建纯关键词 prompt（空格分隔，不携带标签模板）
    4. 避免使用完整句子，防止 Whisper 误判
    5. 避免密集逗号，防止标点感染
    """

    # 停用词表（句首常见的大写词）
    STOPWORDS = {
        "The", "And", "But", "This", "That", "There", "Here",
        "When", "What", "Who", "Where", "Why", "How",
        "Then", "Now", "So", "Or", "If", "As", "At",
        "It", "Its", "I", "You", "He", "She", "We", "They"
    }

    def __init__(
        self,
        max_keywords: int = 20,
        min_word_length: int = 2,
        max_prompt_length: int = 200
    ):
        """
        初始化 Prompt 构建器

        Args:
            max_keywords: 最多提取的关键词数量
            min_word_length: 最小词长度
            max_prompt_length: Prompt 最大长度（字符数）
        """
        self.max_keywords = max_keywords
        self.min_word_length = min_word_length
        self.max_prompt_length = max_prompt_length

        # 关键词频率统计（用于优先选择高频词）
        self.keyword_frequency: Counter = Counter()

    def extract_keywords(self, text: str) -> Dict[str, int]:
        """
        从文本中提取关键词（专有名词）并统计词频

        策略：
        1. 提取所有大写开头的词（英文）
        2. 提取中文词（2-4 个字符的中文词组）
        3. 过滤停用词
        4. 过滤短词（长度 <= min_word_length）
        5. 统计词频

        Args:
            text: 输入文本

        Returns:
            Dict[str, int]: 关键词及其频率
        """
        if not text:
            return {}

        keyword_counts = Counter()

        # 1. 提取英文关键词（大写开头的词）
        english_words = re.findall(r"\b[A-Za-z0-9']+\b", text)
        for word in english_words:
            if (word[0].isupper() and
                word not in self.STOPWORDS and
                len(word) > self.min_word_length):
                keyword_counts[word] += 1

        # 2. 提取中文关键词（2-4 个字符的中文词组）
        # 匹配连续的中文字符
        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
        for word in chinese_words:
            # 过滤常见的停用词
            if word not in {'这个', '那个', '什么', '怎么', '为什么', '因为', '所以', '但是', '然后'}:
                keyword_counts[word] += 1

        return dict(keyword_counts)

    def build_prompt(
        self,
        previous_text: Optional[str] = None,
        user_glossary: Optional[List[str]] = None
    ) -> str:
        """
        构建安全的 Whisper Prompt（优化版）

        格式："word1 word2 word3"（纯关键词，避免模板回显污染）

        优化策略：
        1. 提取关键词并统计词频
        2. 优先选择高频关键词
        3. 动态调整关键词数量以满足长度限制
        4. 更新全局词频统计

        Args:
            previous_text: 上一个 Chunk 的定稿文本
            user_glossary: 用户自定义词表

        Returns:
            str: Prompt 字符串
        """
        keyword_counts = Counter()

        # 1. 从前文中提取关键词并统计词频
        if previous_text:
            extracted = self.extract_keywords(previous_text)
            keyword_counts.update(extracted)
            # 更新全局词频统计
            self.keyword_frequency.update(extracted)

        # 2. 添加用户词表（赋予高权重）
        if user_glossary:
            for word in user_glossary:
                keyword_counts[word] = keyword_counts.get(word, 0) + 10  # 用户词表权重更高

        # 3. 按词频排序，优先选择高频词
        sorted_keywords = [
            word for word, count in keyword_counts.most_common()
        ]

        # 4. 动态调整关键词数量以满足长度限制
        selected_keywords = []
        current_length = 0

        for keyword in sorted_keywords[:self.max_keywords]:
            # 计算添加这个关键词后的长度（空格分隔）
            additional_length = len(keyword) + (1 if selected_keywords else 0)
            if current_length + additional_length > self.max_prompt_length:
                break

            selected_keywords.append(keyword)
            current_length += additional_length

        # 5. 构建 Prompt
        if not selected_keywords:
            return ""

        # V3.2.0+dev.20260218.01: 禁用 Glossary 标签模板，改为纯关键词提示。
        return " ".join(selected_keywords)

    def get_keyword_statistics(self) -> Dict[str, int]:
        """
        获取关键词频率统计

        Returns:
            Dict[str, int]: 关键词及其全局频率
        """
        return dict(self.keyword_frequency.most_common(50))

    def reset_statistics(self):
        """重置关键词频率统计"""
        self.keyword_frequency.clear()


# 全局单例
_prompt_builder = None


def get_prompt_builder() -> PromptBuilder:
    """
    获取全局 Prompt 构建器单例

    Returns:
        PromptBuilder: Prompt 构建器实例
    """
    global _prompt_builder
    if _prompt_builder is None:
        _prompt_builder = PromptBuilder()
    return _prompt_builder
