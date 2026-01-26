"""
伪对齐算法（Pseudo-Alignment）

核心原理：
- SenseVoice 确定的时间窗口（start/end）不可变
- 当 Whisper/LLM 替换文本后，新字符均匀分布在原时间窗口内
- 生成的字级时间戳标记为 is_pseudo=True

V3.1.2+dev.20260111.01: 词级置信度改为 None（表示无词级置信度）
- Whisper 复核后，无法提供精确的词级置信度
- confidence=None 表示"无词级置信度"，前端显示为灰色或无高亮
"""
from typing import List, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from ..models.sensevoice_models import WordTimestamp, SentenceSegment, TextSource

logger = logging.getLogger(__name__)


class PseudoAlignment:
    """伪对齐器"""

    @staticmethod
    def apply(
        original_start: float,
        original_end: float,
        new_text: str,
        default_confidence: Optional[float] = None  # V3.1.2: 改为 None，表示无词级置信度
    ) -> List['WordTimestamp']:
        """
        将新文本均匀映射到原时间段内

        Args:
            original_start: 原始起始时间（由 SenseVoice 确定，不可变）
            original_end: 原始结束时间（由 SenseVoice 确定，不可变）
            new_text: 替换后的新文本
            default_confidence: 词级置信度（V3.1.2: 默认 None，表示无词级置信度）

        Returns:
            List[WordTimestamp]: 伪对齐的字级时间戳列表
        """
        # 延迟导入避免循环依赖
        from ..models.sensevoice_models import WordTimestamp

        duration = original_end - original_start

        if duration <= 0:
            logger.warning(f"伪对齐：无效时间窗口 {original_start}-{original_end}")
            return []

        # 智能切分：英文按单词，中文按字符
        # 检测是否包含英文单词（包含连续的ASCII字母）
        has_english = any(c.isascii() and c.isalpha() for c in new_text)

        if has_english:
            # 英文文本：按单词切分（保留空格信息）
            import re
            # 匹配单词（字母+数字）和标点符号
            tokens = re.findall(r'\w+|[^\w\s]', new_text)
            token_count = len(tokens)
        else:
            # 中文文本：按字符切分（过滤空白）
            tokens = [c for c in new_text if not c.isspace()]
            token_count = len(tokens)

        if token_count == 0:
            logger.warning("伪对齐：文本为空，跳过")
            return []

        # 计算每个token的时长
        step = duration / token_count
        result = []

        for i, token in enumerate(tokens):
            w_start = original_start + (i * step)
            w_end = w_start + step

            result.append(WordTimestamp(
                word=token,
                start=round(w_start, 3),
                end=round(w_end, 3),
                confidence=default_confidence,  # V3.1.2: 可能为 None
                is_pseudo=True  # 标记为伪对齐生成
            ))

        logger.debug(
            f"伪对齐完成: {token_count} 个token, "
            f"时间窗口 {original_start:.2f}-{original_end:.2f}s, "
            f"confidence={'N/A' if default_confidence is None else default_confidence}"
        )

        return result

    @staticmethod
    def apply_to_sentence(
        sentence: 'SentenceSegment',
        new_text: str,
        source: 'TextSource'
    ) -> 'SentenceSegment':
        """
        对句子应用伪对齐

        Args:
            sentence: 原始句子段落
            new_text: 替换后的文本
            source: 文本来源

        Returns:
            修改后的 SentenceSegment（原对象被修改）
        """
        # 生成新的字级时间戳
        new_words = PseudoAlignment.apply(
            original_start=sentence.start,
            original_end=sentence.end,
            new_text=new_text
        )

        # 更新句子
        sentence.mark_as_modified(new_text, source)
        sentence.words = new_words

        return sentence

    @staticmethod
    def merge_words_to_sentence(
        words: List['WordTimestamp'],
        separator: str = ""
    ) -> str:
        """
        将字级时间戳合并为句子文本

        Args:
            words: 字级时间戳列表
            separator: 分隔符（中文通常为空，英文为空格）

        Returns:
            合并后的文本
        """
        return separator.join(w.word for w in words)


# 单例访问
_pseudo_alignment_instance = None


def get_pseudo_alignment() -> PseudoAlignment:
    """获取伪对齐器单例"""
    global _pseudo_alignment_instance
    if _pseudo_alignment_instance is None:
        _pseudo_alignment_instance = PseudoAlignment()
    return _pseudo_alignment_instance
