"""
字幕文档服务。

设计模式：
- Facade（门面模式）：对 `subtitle_edit_store` 提供统一语义接口。
- Adapter（适配器模式）：在 `sentence_index` 存储模型之上适配 `segment_id`。
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from time import time
from typing import Dict, List, Optional

from app.models.project_models import SubtitleDocMeta, generate_segment_id
from app.services.subtitle_edit_store import (
    add_deletion,
    create_manual_entry,
    get_edit_store_path,
    load_deleted_indices,
    load_edits,
    save_edit,
)
from app.utils.ass_converter import ASSConverter
from app.utils.text_utils import parse_srt_content, segments_to_srt

logger = logging.getLogger(__name__)


@dataclass
class SubtitleSegment:
    """字幕段模型（服务层内部结构）。"""

    segment_id: str
    text: str
    start: float
    end: float
    original_text: Optional[str] = None
    is_modified: bool = False
    is_deleted: bool = False
    legacy_index: Optional[int] = None
    source_type: str = "import"
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "text": self.text,
            "start": float(self.start),
            "end": float(self.end),
            "original_text": self.original_text,
            "is_modified": bool(self.is_modified),
            "is_deleted": bool(self.is_deleted),
            "legacy_index": self.legacy_index,
            "source_type": self.source_type,
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
        }


class SubtitleDocService:
    """字幕文档服务。"""

    SEGMENT_MAP_KEY = "_segment_map"

    def import_segments(
        self,
        project_dir: Path,
        segments: List[dict],
        source_type: str = "import",
    ) -> SubtitleDocMeta:
        """导入字幕并写入 `subtitle_edits.json`。"""
        project_dir.mkdir(parents=True, exist_ok=True)

        edits: Dict[str, dict] = {}
        segment_map: Dict[str, int] = {}
        normalized_segments = self._normalize_segments(segments)

        for index, seg in enumerate(normalized_segments):
            edits[str(index)] = {
                "text": seg.get("text", ""),
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "source": source_type,
                "updated_at": time(),
            }
            segment_map[generate_segment_id()] = index

        payload = {
            "version": "3.2.4",
            "updated_at": time(),
            "edits": edits,
            "deleted_indices": [],
            "next_manual_index": -1,
            self.SEGMENT_MAP_KEY: segment_map,
        }
        self._write_payload(project_dir, payload)

        now = time()
        return SubtitleDocMeta(
            doc_id=project_dir.name,
            project_id=project_dir.name,
            source_type=source_type,  # type: ignore[arg-type]
            segment_count=len(normalized_segments),
            version=1,
            created_at=now,
            updated_at=now,
        )

    def load_segments(self, project_dir: Path) -> List[dict]:
        """加载字幕列表（已应用删除与 segment_id 映射）。"""
        edits = load_edits(project_dir)
        deleted_indices = load_deleted_indices(project_dir)
        segment_map = self._ensure_segment_ids(project_dir, edits)
        index_to_segment = {index: seg_id for seg_id, index in segment_map.items()}

        segments: List[dict] = []
        for index, entry in edits.items():
            if index in deleted_indices:
                continue

            text = str(entry.get("text", "") or "")
            start = float(entry.get("start", 0.0) or 0.0)
            end = float(entry.get("end", start) or start)
            if end < start:
                end = start

            source = str(entry.get("source", "import") or "import")
            if source not in {"transcribe", "import", "legacy", "manual"}:
                source = "import"

            segment = SubtitleSegment(
                segment_id=index_to_segment.get(index, generate_segment_id()),
                text=text,
                start=start,
                end=end,
                original_text=entry.get("original_text"),
                is_modified=bool(
                    entry.get("original_text") is not None
                    or entry.get("is_modified", False)
                ),
                is_deleted=False,
                legacy_index=int(index),
                source_type=source,
                updated_at=float(entry.get("updated_at", time())),
            )
            segments.append(segment.to_dict())

        segments.sort(
            key=lambda item: (
                float(item.get("start", 0.0)),
                float(item.get("end", 0.0)),
                int(item.get("legacy_index", 0) or 0),
            )
        )
        return segments

    def get_segment(self, project_dir: Path, segment_id: str) -> Optional[dict]:
        """按 segment_id 查询字幕段。"""
        normalized_segment_id = str(segment_id or "").strip()
        if not normalized_segment_id:
            return None
        for item in self.load_segments(project_dir):
            if item.get("segment_id") == normalized_segment_id:
                return item
        return None

    def update_segment(
        self,
        project_dir: Path,
        segment_id: str,
        update: dict,
    ) -> bool:
        """更新字幕段。"""
        segment_map = self._load_segment_map(project_dir)
        sentence_index = segment_map.get(segment_id)
        if sentence_index is None:
            return False

        edits = load_edits(project_dir)
        current = edits.get(sentence_index, {})
        original_text = current.get("original_text") or current.get("text")

        normalized_update = {}
        if update.get("text") is not None:
            normalized_update["text"] = str(update.get("text", ""))
        if update.get("start") is not None:
            normalized_update["start"] = float(update.get("start"))
        if update.get("end") is not None:
            normalized_update["end"] = float(update.get("end"))

        if not normalized_update:
            return False

        save_edit(project_dir, sentence_index, normalized_update, original_text)
        self._save_segment_map(project_dir, segment_map)
        return True

    def create_segment(
        self,
        project_dir: Path,
        text: str,
        start: float,
        end: float,
    ) -> dict:
        """新增字幕段。"""
        normalized_text = str(text or "")
        normalized_start = float(start)
        normalized_end = float(end)
        if normalized_end < normalized_start:
            normalized_end = normalized_start

        # 注意：create_manual_entry 会重写 subtitle_edits.json，
        # 因此必须先读取现有映射，再回写以避免丢失旧 segment_id。
        segment_map = self._load_segment_map(project_dir)
        sentence_index, entry = create_manual_entry(
            project_dir,
            normalized_text,
            normalized_start,
            normalized_end,
        )
        segment_id = generate_segment_id()
        segment_map[segment_id] = sentence_index
        self._save_segment_map(project_dir, segment_map)

        return SubtitleSegment(
            segment_id=segment_id,
            text=str(entry.get("text", normalized_text)),
            start=float(entry.get("start", normalized_start)),
            end=float(entry.get("end", normalized_end)),
            original_text=entry.get("original_text"),
            is_modified=True,
            is_deleted=False,
            legacy_index=int(sentence_index),
            source_type=str(entry.get("source", "manual")),
            updated_at=float(entry.get("updated_at", time())),
        ).to_dict()

    def delete_segment(self, project_dir: Path, segment_id: str) -> bool:
        """删除字幕段。"""
        segment_map = self._load_segment_map(project_dir)
        sentence_index = segment_map.get(segment_id)
        if sentence_index is None:
            return False

        add_deletion(project_dir, sentence_index)
        segment_map.pop(segment_id, None)
        self._save_segment_map(project_dir, segment_map)
        return True

    def _load_segment_map(self, project_dir: Path) -> Dict[str, int]:
        payload = self._read_payload(project_dir)
        raw_map = payload.get(self.SEGMENT_MAP_KEY, {})
        result: Dict[str, int] = {}
        if not isinstance(raw_map, dict):
            return result
        for segment_id, sentence_index in raw_map.items():
            try:
                result[str(segment_id)] = int(sentence_index)
            except (TypeError, ValueError):
                continue
        return result

    def _save_segment_map(self, project_dir: Path, seg_map: Dict[str, int]) -> None:
        payload = self._read_payload(project_dir)
        payload[self.SEGMENT_MAP_KEY] = {str(k): int(v) for k, v in seg_map.items()}
        payload["updated_at"] = time()
        self._write_payload(project_dir, payload)

    def _ensure_segment_ids(self, project_dir: Path, edits: Dict[int, dict]) -> Dict[str, int]:
        """确保每条字幕都有 segment_id 映射。"""
        existing_map = self._load_segment_map(project_dir)
        normalized_map: Dict[str, int] = {}
        seen_indices = set()
        is_changed = False

        for segment_id, sentence_index in existing_map.items():
            if sentence_index not in edits:
                is_changed = True
                continue
            if sentence_index in seen_indices:
                is_changed = True
                continue
            normalized_map[segment_id] = sentence_index
            seen_indices.add(sentence_index)

        for sentence_index in edits.keys():
            if sentence_index in seen_indices:
                continue
            normalized_map[generate_segment_id()] = int(sentence_index)
            seen_indices.add(sentence_index)
            is_changed = True

        if is_changed:
            self._save_segment_map(project_dir, normalized_map)
        return normalized_map

    def export_srt(self, project_dir: Path) -> str:
        """导出 SRT 字符串。"""
        segments = self.load_segments(project_dir)
        raw_segments = [
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": str(seg.get("text", "")),
            }
            for seg in segments
        ]
        return segments_to_srt(raw_segments)

    def export_ass(self, project_dir: Path) -> str:
        """导出 ASS 字符串。"""
        segments = self.load_segments(project_dir)
        style = ASSConverter.STYLE_PRESETS["default"]

        content_parts: List[str] = [
            ASSConverter.generate_script_info(title=project_dir.name),
            "\n[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding",
            ASSConverter.generate_style_section(style),
            "\n[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
        for seg in segments:
            start = ASSConverter.format_ass_timestamp(float(seg.get("start", 0.0)))
            end = ASSConverter.format_ass_timestamp(float(seg.get("end", 0.0)))
            text = str(seg.get("text", "")).replace("\n", "\\N")
            content_parts.append(
                f"Dialogue: 0,{start},{end},{style.name},,0,0,0,,{text}"
            )
        return "\n".join(content_parts)

    @staticmethod
    def parse_srt(content: str) -> List[dict]:
        """解析 SRT 内容。"""
        segments = []
        for item in parse_srt_content(content or ""):
            segments.append(
                {
                    "text": str(item.get("text", "")),
                    "start": float(item.get("start", 0.0)),
                    "end": float(item.get("end", 0.0)),
                }
            )
        return segments

    @staticmethod
    def parse_ass(content: str) -> List[dict]:
        """解析 ASS 内容。"""
        result: List[dict] = []
        if not content:
            return result

        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line.startswith("Dialogue:"):
                continue
            payload = line[len("Dialogue:") :].strip()
            parts = payload.split(",", 9)
            if len(parts) < 10:
                continue
            try:
                start = SubtitleDocService._parse_ass_timestamp(parts[1].strip())
                end = SubtitleDocService._parse_ass_timestamp(parts[2].strip())
            except ValueError:
                continue
            text = parts[9].replace("\\N", "\n")
            result.append({"text": text, "start": start, "end": end})

        result.sort(key=lambda item: (item["start"], item["end"]))
        return result

    @staticmethod
    def parse_vtt(content: str) -> List[dict]:
        """解析 VTT 内容。"""
        result: List[dict] = []
        if not content:
            return result

        lines = content.splitlines()
        current_start: Optional[float] = None
        current_end: Optional[float] = None
        text_lines: List[str] = []

        time_pattern = re.compile(
            r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})"
        )

        def flush_segment() -> None:
            if current_start is None or current_end is None:
                return
            if not text_lines:
                return
            result.append(
                {
                    "text": "\n".join(text_lines).strip(),
                    "start": float(current_start),
                    "end": float(current_end),
                }
            )

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                flush_segment()
                current_start = None
                current_end = None
                text_lines = []
                continue
            match = time_pattern.match(line)
            if match:
                current_start = SubtitleDocService._parse_vtt_timestamp(match.group(1))
                current_end = SubtitleDocService._parse_vtt_timestamp(match.group(2))
                text_lines = []
                continue
            if line.upper() == "WEBVTT":
                continue
            if current_start is not None and current_end is not None:
                text_lines.append(raw_line)

        flush_segment()
        result.sort(key=lambda item: (item["start"], item["end"]))
        return result

    @staticmethod
    def _normalize_segments(segments: List[dict]) -> List[dict]:
        normalized: List[dict] = []
        for raw in segments or []:
            if not isinstance(raw, dict):
                continue
            start = float(raw.get("start", 0.0) or 0.0)
            end = float(raw.get("end", start) or start)
            if end < start:
                end = start
            normalized.append(
                {
                    "text": str(raw.get("text", "") or ""),
                    "start": start,
                    "end": end,
                }
            )
        normalized.sort(key=lambda item: (item["start"], item["end"]))
        return normalized

    @staticmethod
    def _parse_ass_timestamp(raw: str) -> float:
        """
        解析 ASS 时间戳（H:MM:SS.cc）。
        """
        pattern = re.compile(r"(\d+):(\d{2}):(\d{2})\.(\d{2})")
        matched = pattern.match(raw)
        if not matched:
            raise ValueError(f"非法 ASS 时间戳: {raw}")
        hours = int(matched.group(1))
        minutes = int(matched.group(2))
        seconds = int(matched.group(3))
        centiseconds = int(matched.group(4))
        return hours * 3600 + minutes * 60 + seconds + centiseconds / 100.0

    @staticmethod
    def _parse_vtt_timestamp(raw: str) -> float:
        pattern = re.compile(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})")
        matched = pattern.match(raw)
        if not matched:
            raise ValueError(f"非法 VTT 时间戳: {raw}")
        hours = int(matched.group(1))
        minutes = int(matched.group(2))
        seconds = int(matched.group(3))
        milliseconds = int(matched.group(4))
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

    def _read_payload(self, project_dir: Path) -> dict:
        path = get_edit_store_path(project_dir)
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            logger.warning("读取 subtitle_edits.json 失败: %s", exc)
            return {}

    def _write_payload(self, project_dir: Path, payload: dict) -> None:
        path = get_edit_store_path(project_dir)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)


_subtitle_doc_service: Optional[SubtitleDocService] = None
_subtitle_doc_service_lock = RLock()


def get_subtitle_doc_service() -> SubtitleDocService:
    """获取 SubtitleDocService 单例。"""
    global _subtitle_doc_service
    if _subtitle_doc_service is not None:
        return _subtitle_doc_service
    with _subtitle_doc_service_lock:
        if _subtitle_doc_service is None:
            _subtitle_doc_service = SubtitleDocService()
    return _subtitle_doc_service
