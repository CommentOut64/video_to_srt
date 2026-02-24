"""
旧任务投影服务。

设计模式：
- Adapter（适配器模式）：把旧 `job_id` 任务适配成新 `project_id` 视图。
- Lazy Migration（懒迁移）：仅在访问时迁移，降低一次性升级风险。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import RLock
from time import time
from typing import Dict, List, Optional, Tuple

from app.core.config import FLAVOR, config
from app.models.project_models import Project, SubtitleDocMeta, generate_legacy_project_id
from app.services.project_service import ProjectService, get_project_service
from app.services.subtitle_doc_service import SubtitleDocService, get_subtitle_doc_service
from app.services.subtitle_edit_store import load_deleted_indices, load_edits
from app.utils.text_utils import parse_srt_content

logger = logging.getLogger(__name__)


class LegacyProjectionService:
    """旧任务投影服务。"""

    MAP_FILE = "_legacy_map.json"

    def __init__(
        self,
        project_service: Optional[ProjectService] = None,
        subtitle_doc_service: Optional[SubtitleDocService] = None,
    ) -> None:
        self._project_service = project_service or get_project_service()
        self._subtitle_doc_service = subtitle_doc_service or get_subtitle_doc_service()
        self._map_path = config.JOBS_DIR / self.MAP_FILE
        self._lock = RLock()
        self._map = self._load_map()

    def resolve(self, job_id: str) -> Tuple[str, bool]:
        """
        解析 `job_id -> project_id`。

        Returns:
            (project_id, is_newly_migrated)
        """
        normalized_job_id = str(job_id or "").strip()
        if not normalized_job_id:
            raise FileNotFoundError("job_id 不能为空")

        with self._lock:
            cached = self._map.get("mappings", {}).get(normalized_job_id)
            if isinstance(cached, dict) and cached.get("project_id"):
                return str(cached["project_id"]), False

        job_dir = config.JOBS_DIR / normalized_job_id
        if not job_dir.exists():
            raise FileNotFoundError(f"任务不存在: {normalized_job_id}")

        existing_project = self._project_service._load_project_meta(job_dir)  # type: ignore[attr-defined]
        if (
            existing_project is not None
            and existing_project.mode == "legacy"
            and existing_project.job_id == normalized_job_id
        ):
            self._record_mapping(normalized_job_id, existing_project.project_id, False)
            return existing_project.project_id, False

        project_id = self._migrate(normalized_job_id)
        return project_id, True

    def _migrate(self, job_id: str) -> str:
        """执行首次迁移。"""
        job_dir = config.JOBS_DIR / job_id
        if not job_dir.exists():
            raise FileNotFoundError(f"任务目录不存在: {job_dir}")

        project_id = generate_legacy_project_id(job_id)
        restored_segments = self._restore_subtitles(job_dir)
        segment_count = len(restored_segments)

        subtitle_file = job_dir / "subtitle_edits.json"
        if not subtitle_file.exists() and restored_segments:
            try:
                self._subtitle_doc_service.import_segments(
                    project_dir=job_dir,
                    segments=restored_segments,
                    source_type="legacy",
                )
            except Exception as exc:
                logger.warning("迁移 legacy 字幕导入失败，继续迁移元信息: %s", exc)

        subtitle_doc = SubtitleDocMeta(
            doc_id=project_id,
            project_id=project_id,
            source_type="legacy",
            segment_count=segment_count,
            version=1,
            created_at=time(),
            updated_at=time(),
        )
        project = Project(
            project_id=project_id,
            title=job_id,
            mode="legacy",
            flavor="lite" if FLAVOR == "lite" else "full",
            job_id=job_id,
            subtitle_doc=subtitle_doc,
            media_assets=self._project_service.scan_media_assets(job_dir),
            dir=str(job_dir),
            created_at=time(),
            updated_at=time(),
        )
        self._project_service.save_project(project, project_dir=job_dir)
        self._record_mapping(job_id, project_id, True)
        return project_id

    def _restore_subtitles(self, job_dir: Path) -> List[dict]:
        """按优先级恢复字幕。"""
        strategy_1 = self._restore_from_edits_and_checkpoint(job_dir)
        if strategy_1:
            return strategy_1

        strategy_2 = self._restore_from_checkpoint(job_dir)
        if strategy_2:
            return strategy_2

        strategy_3 = self._restore_from_srt(job_dir)
        if strategy_3:
            return strategy_3

        return []

    def _restore_from_edits_and_checkpoint(self, job_dir: Path) -> Optional[List[dict]]:
        """
        策略1：`subtitle_edits + checkpoint/transcription_text`。

        说明：
        - 以快照句子为基线，叠加 edits；
        - 再应用 deleted_indices 与手动新增句子。
        """
        snapshot = self._load_checkpoint_or_snapshot(job_dir)
        if snapshot is None:
            return None

        sentences_snapshot = snapshot.get("sentences_snapshot", [])
        if not isinstance(sentences_snapshot, list):
            sentences_snapshot = []

        edits = load_edits(job_dir)
        deleted_indices = load_deleted_indices(job_dir)
        if not sentences_snapshot and not edits:
            return None

        segments: List[dict] = []
        for item in sentences_snapshot:
            if not isinstance(item, dict):
                continue
            raw_index = item.get("_index", item.get("index"))
            try:
                sentence_index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if sentence_index in deleted_indices:
                continue

            edit = edits.get(sentence_index, {})
            text = edit.get("text", item.get("text", ""))
            start = float(edit.get("start", item.get("start", 0.0)) or 0.0)
            end = float(edit.get("end", item.get("end", start)) or start)
            if end < start:
                end = start

            segments.append(
                {
                    "text": str(text),
                    "start": start,
                    "end": end,
                    "source_type": "legacy",
                    "origin_sentence_index": sentence_index,
                }
            )

        # 手动字幕（负索引）
        for sentence_index, edit in edits.items():
            if sentence_index >= 0:
                continue
            if sentence_index in deleted_indices:
                continue
            segments.append(
                {
                    "text": str(edit.get("text", "")),
                    "start": float(edit.get("start", 0.0) or 0.0),
                    "end": float(edit.get("end", 0.0) or 0.0),
                    "source_type": "legacy",
                    "origin_sentence_index": int(sentence_index),
                }
            )

        segments.sort(
            key=lambda item: (
                float(item.get("start", 0.0)),
                float(item.get("end", 0.0)),
                int(item.get("origin_sentence_index", 0) or 0),
            )
        )
        return segments

    def _restore_from_checkpoint(self, job_dir: Path) -> Optional[List[dict]]:
        """策略2：仅 checkpoint/transcription_text 快照。"""
        snapshot = self._load_checkpoint_or_snapshot(job_dir)
        if snapshot is None:
            return None
        sentences_snapshot = snapshot.get("sentences_snapshot", [])
        if not isinstance(sentences_snapshot, list):
            return None

        segments: List[dict] = []
        for item in sentences_snapshot:
            if not isinstance(item, dict):
                continue
            start = float(item.get("start", 0.0) or 0.0)
            end = float(item.get("end", start) or start)
            if end < start:
                end = start
            segments.append(
                {
                    "text": str(item.get("text", "")),
                    "start": start,
                    "end": end,
                    "source_type": "legacy",
                }
            )

        if not segments:
            return None
        segments.sort(key=lambda item: (item["start"], item["end"]))
        return segments

    def _restore_from_srt(self, job_dir: Path) -> Optional[List[dict]]:
        """策略3：从目录内 SRT 恢复。"""
        srt_files = sorted(job_dir.glob("*.srt"))
        if not srt_files:
            return None

        for srt_path in srt_files:
            try:
                content = srt_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    content = srt_path.read_text(encoding="utf-8-sig")
                except Exception:
                    continue
            except Exception:
                continue

            parsed = parse_srt_content(content)
            if not parsed:
                continue

            segments: List[dict] = []
            for item in parsed:
                segments.append(
                    {
                        "text": str(item.get("text", "")),
                        "start": float(item.get("start", 0.0) or 0.0),
                        "end": float(item.get("end", 0.0) or 0.0),
                        "source_type": "legacy",
                    }
                )
            segments.sort(key=lambda row: (row["start"], row["end"]))
            return segments

        return None

    def _load_checkpoint_or_snapshot(self, job_dir: Path) -> Optional[dict]:
        checkpoint_path = job_dir / "checkpoint.json"
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                transcription = data.get("transcription", {})
                if isinstance(transcription, dict):
                    return transcription
            except Exception:
                pass

        snapshot_path = job_dir / "transcription_text.json"
        if snapshot_path.exists():
            try:
                with open(snapshot_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return None

    def _load_map(self) -> dict:
        if not self._map_path.exists():
            return {"version": 1, "mappings": {}}
        try:
            with open(self._map_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            if not isinstance(payload, dict):
                return {"version": 1, "mappings": {}}
            if "mappings" not in payload or not isinstance(payload.get("mappings"), dict):
                payload["mappings"] = {}
            if "version" not in payload:
                payload["version"] = 1
            return payload
        except Exception as exc:
            logger.warning("读取 legacy map 失败，将创建新映射: %s", exc)
            return {"version": 1, "mappings": {}}

    def _save_map(self) -> None:
        self._map_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._map_path.with_suffix(self._map_path.suffix + ".tmp")
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(self._map, file, ensure_ascii=False, indent=2)
        os.replace(temp_path, self._map_path)

    def _record_mapping(self, job_id: str, project_id: str, is_new_migration: bool) -> None:
        with self._lock:
            mappings = self._map.setdefault("mappings", {})
            mappings[job_id] = {
                "project_id": project_id,
                "migrated_at": time(),
                "status": "migrated" if is_new_migration else "existing",
            }
            self._save_map()

    def is_legacy_job(self, job_id: str) -> bool:
        """判断是否为已映射 legacy 任务。"""
        normalized_job_id = str(job_id or "").strip()
        if not normalized_job_id:
            return False
        with self._lock:
            mappings = self._map.get("mappings", {})
            return normalized_job_id in mappings

    def get_all_mappings(self) -> dict:
        """返回映射副本（诊断用途）。"""
        with self._lock:
            return json.loads(json.dumps(self._map))


_legacy_projection_service: Optional[LegacyProjectionService] = None
_legacy_projection_service_lock = RLock()


def get_legacy_projection_service() -> LegacyProjectionService:
    """获取 LegacyProjectionService 单例。"""
    global _legacy_projection_service
    if _legacy_projection_service is not None:
        return _legacy_projection_service
    with _legacy_projection_service_lock:
        if _legacy_projection_service is None:
            _legacy_projection_service = LegacyProjectionService(
                project_service=get_project_service(),
                subtitle_doc_service=get_subtitle_doc_service(),
            )
    return _legacy_projection_service
