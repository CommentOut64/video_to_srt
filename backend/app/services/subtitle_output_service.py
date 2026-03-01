"""
字幕输出服务 - V3.2.0+dev.20260125.11

统一处理字幕格式化与文件写入，使用策略模式支持多种格式。

核心职责:
1. 从句子列表构建标准化 segments
2. 检测并修复时间戳重叠
3. 输出 SRT/VTT/ASS 格式字幕文件

使用方式:
    service = SubtitleOutputService()
    segments = service.build_segments(sentences, include_translation=True)
    segments, repair_count = service.repair_overlaps(segments)
    service.write_srt(segments, output_path)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from app.services.user_config_service import get_user_config_service
from app.services.subtitle_visibility import filter_hidden_unknown_sentences
from app.utils.text_utils import (
    detect_timestamp_overlaps,
    format_srt_timestamp,
    repair_timestamp_overlaps,
)

if TYPE_CHECKING:
    from app.models.sensevoice_models import SentenceSegment

logger = logging.getLogger(__name__)

# 类型别名
Segment = Dict[str, Any]


# ========== Formatter 策略模式 ==========

class SubtitleFormatter(ABC):
    """字幕格式化器抽象基类"""

    @abstractmethod
    def format(self, segments: List[Segment], **kwargs: Any) -> str:
        """
        将 segments 格式化为字幕文本

        Args:
            segments: 标准化字幕段落列表
            **kwargs: 格式特定参数

        Returns:
            str: 格式化后的字幕文本
        """
        raise NotImplementedError


class SrtFormatter(SubtitleFormatter):
    """
    SRT 格式化器

    使用 text_utils.format_srt_timestamp（截断逻辑）作为唯一时间戳实现。
    """

    def format(self, segments: List[Segment], **kwargs: Any) -> str:
        lines: List[str] = []
        for index, segment in enumerate(segments, 1):
            start = format_srt_timestamp(segment.get("start", 0))
            end = format_srt_timestamp(segment.get("end", 0))
            text = segment.get("text", "")
            lines.append(str(index))
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
        return "\n".join(lines)


class VttFormatter(SubtitleFormatter):
    """
    WebVTT 格式化器

    VTT 与 SRT 类似，但时间戳使用点号分隔毫秒，且需要 WEBVTT 头部。
    """

    def format(self, segments: List[Segment], **kwargs: Any) -> str:
        lines: List[str] = ["WEBVTT", ""]
        for index, segment in enumerate(segments, 1):
            start = self._format_timestamp(segment.get("start", 0))
            end = self._format_timestamp(segment.get("end", 0))
            text = segment.get("text", "")
            lines.append(str(index))
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """VTT 时间戳格式: HH:MM:SS.mmm"""
        if seconds < 0:
            seconds = 0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        # V3.2.0+dev.20260125.11: 使用截断而非四舍五入，与 SRT 保持一致
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class AssFormatter(SubtitleFormatter):
    """
    ASS 格式化器

    委托给 ASSConverter 生成 ASS 格式字幕。
    """

    def format(self, segments: List[Segment], **kwargs: Any) -> str:
        # 延迟导入，避免循环依赖
        from app.utils.ass_converter import ASSConverter

        style_preset = kwargs.get("style_preset", "default")
        title = kwargs.get("title", "Untitled")
        video_width = kwargs.get("video_width", 1920)
        video_height = kwargs.get("video_height", 1080)

        style = ASSConverter.STYLE_PRESETS.get(
            style_preset, ASSConverter.STYLE_PRESETS["default"]
        )

        content_parts: List[str] = [
            ASSConverter.generate_script_info(title, video_width, video_height),
            "\n[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding",
            ASSConverter.generate_style_section(style),
            "\n[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
            "Effect, Text",
        ]

        for segment in segments:
            start = ASSConverter.format_ass_timestamp(segment.get("start", 0))
            end = ASSConverter.format_ass_timestamp(segment.get("end", 0))
            text = segment.get("text", "").replace("\n", "\\N")
            content_parts.append(
                f"Dialogue: 0,{start},{end},{style.name},,0,0,0,,{text}"
            )

        return "\n".join(content_parts)


# ========== 门面服务类 ==========

class SubtitleOutputService:
    """
    字幕输出服务 - 统一处理字幕格式化与文件写入

    设计模式: Facade + Strategy
    - Facade: 提供统一的字幕输出接口
    - Strategy: 通过 Formatter 支持多种格式
    """

    def __init__(self) -> None:
        self._srt_formatter = SrtFormatter()
        self._vtt_formatter = VttFormatter()
        self._ass_formatter = AssFormatter()

    def build_segments(
        self,
        sentences: List["SentenceSegment"],
        include_translation: bool = False,
        apply_offset: bool = True,
        offset_override: float | None = None,
    ) -> List[Segment]:
        """
        从句子列表构建标准化 segments

        Args:
            sentences: SentenceSegment 列表
            include_translation: 是否包含翻译（双语字幕）
            apply_offset: 是否应用全局字幕时间偏移

        Returns:
            List[Segment]: 标准化字幕段落列表
        """
        # V3.2.0+dev.20260130.09: 输出阶段统一应用全局字幕时间偏移
        offset = 0.0
        if apply_offset:
            offset = offset_override if offset_override is not None else get_user_config_service().get_subtitle_time_offset()

        segments: List[Segment] = []
        visible_sentences = filter_hidden_unknown_sentences(sentences)
        for sentence in visible_sentences:
            # 优先使用清洗后的文本
            base_text = getattr(sentence, "text_clean", None) or sentence.text
            translation = getattr(sentence, "translation", None)

            if include_translation and translation:
                text = f"{base_text}\n{translation}"
            else:
                text = base_text

            start = float(sentence.start) + offset
            end = float(sentence.end) + offset
            if start < 0:
                start = 0.0
            if end < start:
                end = start

            segments.append({
                "start": start,
                "end": end,
                "text": text,
            })

        logger.debug(f"[SubtitleOutputService] 构建 {len(segments)} 个 segments")
        return segments

    def repair_overlaps(
        self,
        segments: List[Segment],
        gap_ms: float = 1.0
    ) -> Tuple[List[Segment], int]:
        """
        检测并修复时间戳重叠

        Args:
            segments: 字幕段落列表
            gap_ms: 修复后的最小间隔（毫秒）

        Returns:
            Tuple[List[Segment], int]: (修复后的 segments, 修复数量)
        """
        overlaps = detect_timestamp_overlaps(segments)
        if not overlaps:
            return segments, 0

        repaired = repair_timestamp_overlaps(segments, gap_ms=gap_ms)
        repair_count = len(overlaps)

        logger.info(f"[SubtitleOutputService] 修复了 {repair_count} 处时间戳重叠")
        return repaired, repair_count

    def write_srt(
        self,
        segments: List[Segment],
        output_path: Path,
        auto_repair: bool = True,
    ) -> Path:
        """
        写入 SRT 格式字幕文件

        Args:
            segments: 字幕段落列表
            output_path: 输出文件路径
            auto_repair: 是否自动修复重叠

        Returns:
            Path: 输出文件路径
        """
        output_path = Path(output_path)
        segments_to_write = segments

        if auto_repair:
            segments_to_write, repair_count = self.repair_overlaps(segments_to_write)
            if repair_count > 0:
                logger.debug(f"[SubtitleOutputService] SRT 写入前修复了 {repair_count} 处重叠")

        content = self._srt_formatter.format(segments_to_write)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

        logger.info(f"[SubtitleOutputService] 已写入 SRT: {output_path}")
        return output_path

    def write_vtt(
        self,
        segments: List[Segment],
        output_path: Path,
        auto_repair: bool = True,
    ) -> Path:
        """
        写入 WebVTT 格式字幕文件

        Args:
            segments: 字幕段落列表
            output_path: 输出文件路径
            auto_repair: 是否自动修复重叠

        Returns:
            Path: 输出文件路径
        """
        output_path = Path(output_path)
        segments_to_write = segments

        if auto_repair:
            segments_to_write, repair_count = self.repair_overlaps(segments_to_write)
            if repair_count > 0:
                logger.debug(f"[SubtitleOutputService] VTT 写入前修复了 {repair_count} 处重叠")

        content = self._vtt_formatter.format(segments_to_write)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

        logger.info(f"[SubtitleOutputService] 已写入 VTT: {output_path}")
        return output_path

    def write_ass(
        self,
        segments: List[Segment],
        output_path: Path,
        style_preset: str = "default",
        video_width: int = 1920,
        video_height: int = 1080,
    ) -> Path:
        """
        写入 ASS 格式字幕文件

        Args:
            segments: 字幕段落列表
            output_path: 输出文件路径
            style_preset: 样式预设名称
            video_width: 视频宽度
            video_height: 视频高度

        Returns:
            Path: 输出文件路径
        """
        output_path = Path(output_path)

        # ASS 格式总是自动修复重叠
        segments_to_write, repair_count = self.repair_overlaps(segments)
        if repair_count > 0:
            logger.debug(f"[SubtitleOutputService] ASS 写入前修复了 {repair_count} 处重叠")

        content = self._ass_formatter.format(
            segments_to_write,
            style_preset=style_preset,
            title=output_path.stem or "Untitled",
            video_width=video_width,
            video_height=video_height,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # ASS 使用 UTF-8 BOM 确保兼容性
        output_path.write_text(content, encoding="utf-8-sig")

        logger.info(f"[SubtitleOutputService] 已写入 ASS: {output_path}")
        return output_path


# ========== 单例工厂 ==========

_subtitle_output_service: SubtitleOutputService = None


def get_subtitle_output_service() -> SubtitleOutputService:
    """获取字幕输出服务单例"""
    global _subtitle_output_service
    if _subtitle_output_service is None:
        _subtitle_output_service = SubtitleOutputService()
    return _subtitle_output_service
