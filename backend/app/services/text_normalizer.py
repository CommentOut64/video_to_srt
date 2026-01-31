"""
SenseVoice 文本后处理器
清洗特殊标签、重复字符，统一标点符号
"""
import re
import logging

logger = logging.getLogger(__name__)


class TextNormalizer:
    """SenseVoice 文本后处理器"""

    # SenseVoice 特殊标签模式（如 <|zh|>, <|HAPPY|>, <|BGM|> 等）
    SPECIAL_TAGS = re.compile(r'<\|.*?\|>')

    # SentencePiece 分词符号（用于标记空格）
    SENTENCEPIECE_SYMBOL = '▁'

    # 异常重复字符（3个及以上相同字符）
    REPEATED_CHARS = re.compile(r'(.)\1{2,}')

    # 多余空白字符
    EXTRA_SPACES = re.compile(r'\s+')

    # 错误的数字格式（如 10,00 应该是 10,000）
    # 匹配：数字 + 逗号 + 2位数字 + 空格或结尾
    MALFORMED_NUMBER = re.compile(r'(\d+),(\d{2})(?=\s|$)')

    # 标点符号映射（全角 -> 半角）
    PUNCTUATION_MAP = {
        '，': ',', '。': '.', '！': '!', '？': '?',
        '；': ';', '：': ':', '"': '"', '"': '"',
        ''': "'", ''': "'", '（': '(', '）': ')',
        '【': '[', '】': ']', '《': '<', '》': '>',
    }
    # V3.2.0+dev.20260131.05: 保护小数点，避免归一化成句号导致误断句
    DECIMAL_DOT_PLACEHOLDER = "__DECIMAL_DOT__"
    DECIMAL_DOT_PATTERN = re.compile(r"(?<=\d)[\.\u3002\uFF0E](?=\d)")

    # Whisper 幻觉检测模式
    # 重复子串检测: 长度>=4 且重复>=3次的子串
    REPEATED_PATTERN = re.compile(r'(.{4,})\1{2,}')

    # 特定幻觉句式（开头匹配）
    HALLUCINATION_PATTERNS = [
        re.compile(r'^Questions?\s+\d+', re.IGNORECASE),           # "Questions 19..."
        re.compile(r'^Subtitles?\s+by', re.IGNORECASE),            # "Subtitles by..."
        re.compile(r'^Copyright\s+', re.IGNORECASE),               # "Copyright 2024..."
        re.compile(r'^Thanks?\s+for\s+watching', re.IGNORECASE),   # "Thanks for watching"
        re.compile(r'^Please\s+subscribe', re.IGNORECASE),         # "Please subscribe"
    ]

    # 块内重复检测配置
    NGRAM_MIN_SIZE = 3          # N-gram 最小长度（词数）
    NGRAM_MAX_SIZE = 5          # N-gram 最大长度（词数）
    REPEAT_THRESHOLD = 2        # 重复阈值（短语重复超过此次数视为异常）
    COMPRESSION_RATIO_THRESHOLD = 0.6  # 压缩比阈值（唯一词比例低于此值视为异常）

    @staticmethod
    def clean(text: str) -> str:
        """
        清洗文本（移除特殊标签和异常重复）

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not text:
            return ""

        # 1. 移除 SenseVoice 特殊标签
        text = TextNormalizer.SPECIAL_TAGS.sub('', text)

        # 2. 替换 SentencePiece 分词符号为空格
        text = text.replace(TextNormalizer.SENTENCEPIECE_SYMBOL, ' ')

        # 3. 处理异常重复字符（保留最多2个）
        text = TextNormalizer.REPEATED_CHARS.sub(r'\1\1', text)

        # 4. 修复错误的数字格式（10,00 -> 10,000）
        text = TextNormalizer._fix_malformed_numbers(text)

        # 5. 规范化空白字符
        text = TextNormalizer.EXTRA_SPACES.sub(' ', text)

        return text.strip()

    @staticmethod
    def _fix_malformed_numbers(text: str) -> str:
        """
        修复错误的数字格式

        常见错误：
        - "10,00" -> "10,000"（千位分隔符后只有2位数字）
        - "15,00" -> "15,000"

        Args:
            text: 输入文本

        Returns:
            修复后的文本
        """
        if not text:
            return ""

        # 修复模式：数字 + 逗号 + 2位数字 -> 数字 + 逗号 + 3位数字（补0）
        # 例如：10,00 -> 10,000
        fixed_text = TextNormalizer.MALFORMED_NUMBER.sub(r'\1,\g<2>0', text)

        return fixed_text

    @classmethod
    def is_whisper_hallucination(cls, text: str) -> bool:
        """
        检测 Whisper 输出是否为幻觉文本

        Args:
            text: Whisper 输出的文本

        Returns:
            bool: True 表示检测到幻觉
        """
        if not text:
            return False

        text = text.strip()

        # 检测1: 重复子串模式
        if cls.REPEATED_PATTERN.search(text):
            return True

        # 检测2: 特定幻觉句式
        for pattern in cls.HALLUCINATION_PATTERNS:
            if pattern.match(text):
                return True

        # 检测3: 纯下划线/符号（清洗后为空或大幅缩减）
        cleaned = cls.clean(text)
        if not cleaned:
            # 清洗后完全为空
            return True
        
        # 如果清洗后损失严重(>60%)且原文或清洗后很短，可能是垃圾字符
        loss_ratio = 1.0 - (len(cleaned) / len(text))
        if loss_ratio > 0.6 and (len(text) <= 10 or len(cleaned) <= 3):
            return True

        return False

    @classmethod
    def detect_intra_block_repetition(cls, text: str) -> tuple[bool, str]:
        """
        检测块内重复（同一条字幕内的循环重复）

        使用两种检测方法：
        1. N-gram 重复检测：检测 3-5 词短语的重复次数
        2. 压缩比检测：计算唯一词比例

        Args:
            text: 待检测的文本

        Returns:
            tuple[bool, str]: (是否检测到重复, 检测原因)
        """
        if not text or len(text.strip()) < 10:
            return False, ""

        text = text.strip()
        words = text.split()

        # 检测1: 词数太少，无法判断
        if len(words) < cls.NGRAM_MIN_SIZE * 2:
            return False, ""

        # 检测2: N-gram 重复检测
        is_ngram_repeated, ngram_reason = cls._detect_ngram_repetition(words)
        if is_ngram_repeated:
            return True, ngram_reason

        # 检测3: 压缩比检测（唯一词比例）
        is_low_compression, compression_reason = cls._detect_low_compression_ratio(words)
        if is_low_compression:
            return True, compression_reason

        return False, ""

    @classmethod
    def _detect_ngram_repetition(cls, words: list) -> tuple[bool, str]:
        """
        检测 N-gram 短语重复

        Args:
            words: 词列表

        Returns:
            tuple[bool, str]: (是否检测到重复, 检测原因)
        """
        # 遍历不同长度的 N-gram
        for n in range(cls.NGRAM_MIN_SIZE, min(cls.NGRAM_MAX_SIZE + 1, len(words) // 2 + 1)):
            ngram_counts = {}

            # 构建 N-gram 并统计
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                ngram_lower = ngram.lower()
                ngram_counts[ngram_lower] = ngram_counts.get(ngram_lower, 0) + 1

            # 检查是否有短语重复超过阈值
            for ngram, count in ngram_counts.items():
                if count > cls.REPEAT_THRESHOLD:
                    return True, f"N-gram 重复: '{ngram}' 出现 {count} 次"

        return False, ""

    @classmethod
    def _detect_low_compression_ratio(cls, words: list) -> tuple[bool, str]:
        """
        检测压缩比（唯一词比例）

        如果唯一词数量占总词数的比例过低，说明存在大量重复

        Args:
            words: 词列表

        Returns:
            tuple[bool, str]: (是否检测到异常, 检测原因)
        """
        if len(words) < 5:
            return False, ""

        # 统计唯一词（忽略大小写）
        unique_words = set(w.lower() for w in words)
        compression_ratio = len(unique_words) / len(words)

        if compression_ratio < cls.COMPRESSION_RATIO_THRESHOLD:
            return True, f"压缩比过低: {compression_ratio:.2f} (唯一词 {len(unique_words)}/{len(words)})"

        return False, ""

    @classmethod
    def truncate_repeated_text(cls, text: str) -> str:
        """
        截断重复文本（找到第一次重复的位置并截断）

        策略：
        1. 找到最长的重复 N-gram
        2. 截断到第一次出现该 N-gram 的结束位置

        Args:
            text: 原始文本

        Returns:
            str: 截断后的文本
        """
        if not text:
            return ""

        words = text.split()
        if len(words) < cls.NGRAM_MIN_SIZE * 2:
            return text

        # 找到重复的 N-gram（从长到短）
        for n in range(min(cls.NGRAM_MAX_SIZE, len(words) // 2), cls.NGRAM_MIN_SIZE - 1, -1):
            for i in range(len(words) - n + 1):
                ngram = words[i:i+n]
                ngram_str = ' '.join(ngram).lower()

                # 查找该 N-gram 的所有出现位置
                occurrences = []
                for j in range(len(words) - n + 1):
                    if ' '.join(words[j:j+n]).lower() == ngram_str:
                        occurrences.append(j)

                # 如果重复超过阈值，截断到第一次出现的结束位置
                if len(occurrences) > cls.REPEAT_THRESHOLD:
                    truncate_pos = occurrences[0] + n
                    truncated_text = ' '.join(words[:truncate_pos])
                    logger.info(
                        f"截断重复文本: N-gram='{ngram_str}' 重复 {len(occurrences)} 次, "
                        f"截断位置={truncate_pos}, 原长度={len(words)}"
                    )
                    return truncated_text

        # 如果没有找到明显的重复 N-gram，但压缩比过低，截断到一半
        unique_words = set(w.lower() for w in words)
        compression_ratio = len(unique_words) / len(words)

        if compression_ratio < cls.COMPRESSION_RATIO_THRESHOLD:
            truncate_pos = len(words) // 2
            truncated_text = ' '.join(words[:truncate_pos])
            logger.info(
                f"压缩比过低，截断到一半: 压缩比={compression_ratio:.2f}, "
                f"截断位置={truncate_pos}"
            )
            return truncated_text

        return text

    @classmethod
    def clean_whisper_output(cls, text: str) -> str:
        """
        清洗 Whisper 输出（比 SenseVoice 清洗更激进）

        检测顺序：
        1. 基础清洗
        2. 幻觉检测
        3. 块内重复检测（新增）

        Args:
            text: Whisper 原始输出

        Returns:
            str: 清洗后的文本，如果是幻觉或重复则返回空字符串或截断后的文本
        """
        if not text:
            return ""

        # 先用基础清洗
        cleaned = cls.clean(text)

        # 检测是否为幻觉
        if cls.is_whisper_hallucination(text):
            logger.warning(f"检测到 Whisper 幻觉，返回空: '{text[:50]}...'")
            return ""

        # 检测块内重复
        is_repeated, reason = cls.detect_intra_block_repetition(cleaned)
        if is_repeated:
            logger.warning(f"检测到块内重复: {reason}, 原文: '{cleaned[:50]}...'")
            # 尝试截断重复部分
            truncated = cls.truncate_repeated_text(cleaned)
            if truncated != cleaned:
                logger.info(f"截断后文本: '{truncated[:50]}...'")
                return truncated
            else:
                # 无法截断，返回空（说明整个句子都是重复）
                logger.warning(f"无法截断重复，返回空")
                return ""

        return cleaned

    @staticmethod
    def normalize_punctuation(text: str, to_fullwidth: bool = True) -> str:
        """
        统一标点符号（全角/半角）

        Args:
            text: 输入文本
            to_fullwidth: True=转全角, False=转半角

        Returns:
            标点统一后的文本
        """
        if not text:
            return ""

        text = TextNormalizer._protect_decimal_dots(text)

        if to_fullwidth:
            # 半角转全角
            mapping = {v: k for k, v in TextNormalizer.PUNCTUATION_MAP.items()}
        else:
            # 全角转半角
            mapping = TextNormalizer.PUNCTUATION_MAP

        for old, new in mapping.items():
            text = text.replace(old, new)

        return TextNormalizer._restore_decimal_dots(text)

    @staticmethod
    def _protect_decimal_dots(text: str) -> str:
        if not text:
            return ""
        return TextNormalizer.DECIMAL_DOT_PATTERN.sub(TextNormalizer.DECIMAL_DOT_PLACEHOLDER, text)

    @staticmethod
    def _restore_decimal_dots(text: str) -> str:
        if not text:
            return ""
        return text.replace(TextNormalizer.DECIMAL_DOT_PLACEHOLDER, ".")

    @staticmethod
    def extract_tags(text: str) -> dict:
        """
        提取 SenseVoice 特殊标签信息

        Args:
            text: 原始文本

        Returns:
            提取的标签信息字典
        """
        tags = {
            "language": None,
            "emotion": None,
            "event": None,
            "raw_tags": []
        }

        if not text:
            return tags

        # 查找所有标签
        found_tags = TextNormalizer.SPECIAL_TAGS.findall(text)
        tags["raw_tags"] = found_tags

        for tag in found_tags:
            # 移除 <| 和 |>
            tag_content = tag[2:-2].lower()

            # 语言标签
            if tag_content in ['zh', 'en', 'ja', 'ko', 'yue', 'auto']:
                tags["language"] = tag_content
            # 情感标签
            elif tag_content in ['happy', 'sad', 'angry', 'neutral', 'fearful', 'disgusted', 'surprised']:
                tags["emotion"] = tag_content
            # 事件标签
            elif tag_content in ['bgm', 'noise', 'speech', 'applause', 'laughter']:
                tags["event"] = tag_content

        return tags

    @staticmethod
    def process(text: str, to_fullwidth: bool = None, extract_info: bool = False, language: str = None) -> dict:
        """
        完整处理流程

        Args:
            text: 原始文本
            to_fullwidth: 是否转换为全角标点（None=自动根据语言判断）
            extract_info: 是否提取标签信息
            language: 语言代码（zh/en/ja等），用于自适应标点归一化

        Returns:
            处理结果字典，包含 text_clean 和可选的 tags
        """
        result = {
            "text_original": text,
            "text_clean": "",
            "tags": None
        }

        if not text:
            return result

        # 提取标签信息（在清洗前）
        if extract_info:
            result["tags"] = TextNormalizer.extract_tags(text)

        # 清洗文本
        clean_text = TextNormalizer.clean(text)

        # 自适应标点归一化：根据语言决定使用全角还是半角标点
        if to_fullwidth is None:
            # 从标签中提取语言（如果没有提取标签，先提取）
            if language is None and result["tags"] is not None:
                language = result["tags"].get("language")

            # 自动判断：中文/日文使用全角，英文使用半角
            if language in ['zh', 'ja', 'ko', 'yue']:
                to_fullwidth = True
            elif language in ['en']:
                to_fullwidth = False
            else:
                # 未知语言，根据文本特征判断
                to_fullwidth = TextNormalizer._detect_language_by_text(clean_text) in ['zh', 'ja']

        # 统一标点
        clean_text = TextNormalizer.normalize_punctuation(clean_text, to_fullwidth)

        result["text_clean"] = clean_text

        return result

    @staticmethod
    def _detect_language_by_text(text: str) -> str:
        """
        根据文本特征简单判断语言（仅用于标点归一化）

        Args:
            text: 文本

        Returns:
            str: 语言代码 zh/en/ja
        """
        if not text:
            return "en"

        # 统计中文字符比例
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text.strip())

        if total_chars == 0:
            return "en"

        chinese_ratio = chinese_chars / total_chars

        # 中文字符超过30%，认为是中文
        if chinese_ratio > 0.3:
            return "zh"
        else:
            return "en"


# 单例实例
_normalizer_instance = None


def get_text_normalizer() -> TextNormalizer:
    """获取文本清洗器单例"""
    global _normalizer_instance
    if _normalizer_instance is None:
        _normalizer_instance = TextNormalizer()
    return _normalizer_instance
