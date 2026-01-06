"""
文本处理工具函数

提供智能文本拼接等功能
V3.1.1+dev.20260106.03: 新增时间戳重叠修复功能
"""
from typing import List, Union, Dict, Any


def smart_join_words(words: List[Union[Dict[str, Any], Any]], word_key: str = "word") -> str:
    """
    智能拼接 words，处理标点符号空格问题

    规则：
    - 标点符号前不加空格
    - 英文单词之间加空格
    - 中文字符之间不加空格

    Args:
        words: 词列表，可以是字典列表或对象列表
        word_key: 如果是字典，指定 word 字段的 key；如果是对象，指定属性名

    Returns:
        拼接后的文本

    Examples:
        >>> words = [{"word": "It"}, {"word": "'"}, {"word": "s"}]
        >>> smart_join_words(words)
        "It's"

        >>> words = [{"word": "Hello"}, {"word": ","}, {"word": "world"}]
        >>> smart_join_words(words)
        "Hello, world"
    """
    if not words:
        return ""

    result = []
    punctuation = set(",.!?;:'\"()[]{}，。！？；：""''（）【】《》、")

    for i, word_obj in enumerate(words):
        # 获取 word 字符串
        if isinstance(word_obj, dict):
            word = word_obj.get(word_key, "")
        else:
            word = getattr(word_obj, word_key, "")

        if not word:
            continue

        if i == 0:
            result.append(word)
        else:
            # 获取前一个 word
            prev_obj = words[i-1]
            if isinstance(prev_obj, dict):
                prev_word = prev_obj.get(word_key, "")
            else:
                prev_word = getattr(prev_obj, word_key, "")

            # 判断是否需要添加空格
            # 标点符号前不加空格
            if word in punctuation:
                result.append(word)
            # 前一个是标点符号，根据情况决定
            elif prev_word in punctuation:
                # 如果前一个是引号、括号等，可能需要空格
                if prev_word in "\"'([{""''（【《":
                    result.append(word)
                else:
                    result.append(" " + word)
            # 中文字符之间不加空格
            elif any('\u4e00' <= c <= '\u9fff' for c in word) or \
                 any('\u4e00' <= c <= '\u9fff' for c in prev_word):
                result.append(word)
            # 英文单词之间加空格
            else:
                result.append(" " + word)

    return "".join(result)


def repair_timestamp_overlaps(segments: List[Dict], gap_ms: float = 1.0) -> List[Dict]:
    """
    V3.1.1+dev.20260106.03: 自动修复时间戳重叠

    策略：如果 前一条.end > 后一条.start，则将 前一条.end 截断为 后一条.start - gap_ms
    原则：下一句的开始时间是神圣不可侵犯的，因为那是人开始说话的点。

    Args:
        segments: 字幕段落列表，每个元素需有 'start' 和 'end' 字段
        gap_ms: 修复后预留的间隔（毫秒），默认 1ms，防止某些播放器闪烁

    Returns:
        修复后的字幕列表（返回新列表，不修改原数据）

    Example:
        >>> segments = [
        ...     {'start': 0.0, 'end': 3.5, 'text': 'Hello'},
        ...     {'start': 3.0, 'end': 5.0, 'text': 'World'}  # 重叠 0.5s
        ... ]
        >>> repaired = repair_timestamp_overlaps(segments)
        >>> repaired[0]['end']  # 截断为 2.999
        2.999
    """
    if not segments:
        return []

    # 按开始时间排序
    sorted_segs = sorted(segments, key=lambda x: x.get('start', 0))

    repaired_segs = []
    gap_sec = gap_ms / 1000.0  # 转换为秒

    for i in range(len(sorted_segs)):
        # 浅拷贝，避免修改原数据
        current = sorted_segs[i].copy()

        # 如果不是最后一条，检查与后面的重叠
        if i < len(sorted_segs) - 1:
            next_seg = sorted_segs[i + 1]
            next_start = next_seg.get('start', 0)
            current_end = current.get('end', 0)

            # 检测重叠
            if current_end > next_start:
                # 计算新的结束时间：下一条开始时间 - 间隔
                new_end = next_start - gap_sec

                # 确保结束时间不早于开始时间
                current_start = current.get('start', 0)
                if new_end < current_start:
                    # 极端情况：重叠太严重，当前字幕完全被覆盖
                    # 策略：至少保留 10ms 的显示时间
                    new_end = current_start + 0.01

                current['end'] = new_end

        repaired_segs.append(current)

    return repaired_segs


def detect_timestamp_overlaps(segments: List[Dict]) -> List[Dict]:
    """
    V3.1.1+dev.20260106.03: 检测时间戳重叠

    Args:
        segments: 字幕段落列表

    Returns:
        重叠信息列表，每个元素包含:
        - first_index: 第一个重叠段落的索引
        - second_index: 第二个重叠段落的索引
        - overlap_duration: 重叠时长（秒）
    """
    if not segments or len(segments) < 2:
        return []

    sorted_segs = sorted(enumerate(segments), key=lambda x: x[1].get('start', 0))
    overlaps = []

    for i in range(len(sorted_segs) - 1):
        current_idx, current = sorted_segs[i]
        next_idx, next_seg = sorted_segs[i + 1]

        current_end = current.get('end', 0)
        next_start = next_seg.get('start', 0)

        # 使用 1ms 容差
        if current_end > next_start + 0.001:
            overlaps.append({
                'first_index': current_idx,
                'second_index': next_idx,
                'overlap_duration': current_end - next_start
            })

    return overlaps


def parse_srt_content(content: str) -> List[Dict]:
    """
    V3.1.1+dev.20260106.03: 解析 SRT 字幕内容为段落列表

    Args:
        content: SRT 文件内容字符串

    Returns:
        字幕段落列表，每个元素包含:
        - index: 序号
        - start: 开始时间（秒）
        - end: 结束时间（秒）
        - text: 字幕文本
    """
    import re

    segments = []
    # 按空行分割字幕块
    blocks = re.split(r'\n\s*\n', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue

        try:
            # 第一行是序号
            index = int(lines[0].strip())

            # 第二行是时间戳
            time_line = lines[1].strip()
            time_match = re.match(
                r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})',
                time_line
            )
            if not time_match:
                continue

            start = _parse_srt_timestamp(time_match.group(1))
            end = _parse_srt_timestamp(time_match.group(2))

            # 剩余行是字幕文本
            text = '\n'.join(lines[2:]).strip()

            segments.append({
                'index': index,
                'start': start,
                'end': end,
                'text': text
            })
        except (ValueError, IndexError):
            continue

    return segments


def _parse_srt_timestamp(timestamp: str) -> float:
    """
    解析 SRT 时间戳为秒数

    Args:
        timestamp: SRT 时间戳 (HH:MM:SS,mmm 或 HH:MM:SS.mmm)

    Returns:
        秒数
    """
    # 统一处理逗号和点号
    timestamp = timestamp.replace(',', '.')
    time_part, ms_part = timestamp.rsplit('.', 1)
    h, m, s = map(int, time_part.split(':'))
    ms = int(ms_part)
    return h * 3600 + m * 60 + s + ms / 1000


def format_srt_timestamp(seconds: float) -> str:
    """
    V3.1.1+dev.20260106.03: 将秒数格式化为 SRT 时间戳

    Args:
        seconds: 秒数

    Returns:
        SRT 时间戳 (HH:MM:SS,mmm)
    """
    if seconds < 0:
        seconds = 0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def segments_to_srt(segments: List[Dict]) -> str:
    """
    V3.1.1+dev.20260106.03: 将字幕段落列表转换为 SRT 格式字符串

    Args:
        segments: 字幕段落列表

    Returns:
        SRT 格式的字符串
    """
    lines = []

    for i, seg in enumerate(segments, 1):
        start = format_srt_timestamp(seg.get('start', 0))
        end = format_srt_timestamp(seg.get('end', 0))
        text = seg.get('text', '')

        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append('')  # 空行分隔

    return '\n'.join(lines)


def repair_srt_overlaps(content: str, gap_ms: float = 1.0) -> tuple:
    """
    V3.1.1+dev.20260106.03: 修复 SRT 内容中的时间戳重叠

    Args:
        content: SRT 文件内容
        gap_ms: 修复后预留的间隔（毫秒）

    Returns:
        tuple: (修复后的 SRT 内容, 修复的重叠数量)
    """
    # 解析 SRT
    segments = parse_srt_content(content)

    if not segments:
        return content, 0

    # 检测重叠
    overlaps = detect_timestamp_overlaps(segments)

    if not overlaps:
        return content, 0

    # 修复重叠
    repaired = repair_timestamp_overlaps(segments, gap_ms)

    # 转回 SRT 格式
    repaired_content = segments_to_srt(repaired)

    return repaired_content, len(overlaps)
