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
import re
import shutil
from pathlib import Path
from threading import RLock
from time import sleep, time
from typing import Dict, List, Optional, Tuple

from app.core.config import FLAVOR, config
from app.models.project_models import (
    Project,
    SubtitleDocMeta,
    derive_compat_project_mode,
    generate_legacy_project_id,
    infer_task_mode,
)
from app.services.project_service import ProjectService, get_project_service
from app.services.subtitle_doc_service import SubtitleDocService, get_subtitle_doc_service
from app.services.subtitle_edit_store import load_deleted_indices, load_edits
from app.utils.text_utils import parse_srt_content

logger = logging.getLogger(__name__)


class LegacyProjectionService:
    """旧任务投影服务。"""

    MAP_FILE = "_legacy_map.json"
    _WORKSPACE_PROJECT_PATTERN = re.compile(
        r"^p-\d{8}-\d{6}-(tr|im|lg)-[a-z0-9-]+-[0-9a-z]{4}$"
    )

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
        self._sanitize_workspace_mappings()

    def resolve(self, job_id: str) -> Tuple[str, bool]:
        """
        解析 `job_id -> project_id`。

        Returns:
            (project_id, is_newly_migrated)
        """
        normalized_job_id = str(job_id or "").strip()
        if not normalized_job_id:
            raise FileNotFoundError("job_id 不能为空")

        if self._is_workspace_project_identifier(normalized_job_id):
            with self._lock:
                mappings = self._map.get("mappings", {})
                if normalized_job_id in mappings:
                    mappings.pop(normalized_job_id, None)
                    self._save_map()

        with self._lock:
            cached = self._map.get("mappings", {}).get(normalized_job_id)
            if isinstance(cached, dict) and cached.get("project_id"):
                cached_project_id = str(cached["project_id"])
                cached_project = self._project_service.get_project(cached_project_id)
                if (
                    cached_project is not None
                    and str(cached_project.job_id or "").strip() == normalized_job_id
                ):
                    self._ensure_project_dir_canonical(
                        project=cached_project,
                        legacy_job_id=normalized_job_id,
                    )
                    return cached_project_id, False

                # 历史映射失效：project 已不存在或与 job_id 不匹配，移除后重新解析。
                self._map.get("mappings", {}).pop(normalized_job_id, None)
                self._save_map()

        job_dir = config.JOBS_DIR / normalized_job_id
        if not job_dir.exists():
            raise FileNotFoundError(f"任务不存在: {normalized_job_id}")

        existing_project = self._project_service._load_project_meta(job_dir)  # type: ignore[attr-defined]
        if existing_project is not None and str(existing_project.project_id or "").strip():
            if self._should_preserve_workspace_identity(existing_project, normalized_job_id):
                previous_project_id = str(existing_project.project_id or "").strip()
                if previous_project_id != normalized_job_id:
                    if not str(existing_project.job_id or "").strip() and previous_project_id:
                        existing_project.job_id = previous_project_id
                    existing_project.project_id = normalized_job_id
                    logger.info(
                        "legacy 解析保持 workspace 主身份: workspace=%s old_project_id=%s",
                        normalized_job_id,
                        previous_project_id,
                    )
                self._normalize_project_metadata_for_workspace(existing_project, normalized_job_id)
                existing_project.dir = str(job_dir)
                existing_project.updated_at = time()
                self._project_service.save_project(existing_project, project_dir=job_dir)
                return normalized_job_id, False

            meta_job_id = str(existing_project.job_id or "").strip()
            if not meta_job_id:
                # 补齐历史 project_meta 缺失的 job_id，确保后续 legacy 映射可追踪。
                existing_project.job_id = normalized_job_id
                existing_project.updated_at = time()
                self._project_service.save_project(existing_project, project_dir=job_dir)

            self._ensure_project_dir_canonical(
                project=existing_project,
                legacy_job_id=normalized_job_id,
            )
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
        project_dir = self._promote_legacy_dir(job_dir, project_id)
        restored_segments = self._restore_subtitles(project_dir)
        segment_count = len(restored_segments)

        subtitle_file = project_dir / "subtitle_edits.json"
        if not subtitle_file.exists() and restored_segments:
            try:
                self._subtitle_doc_service.import_segments(
                    project_dir=project_dir,
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
            task_mode="transcribe",
            flavor="lite" if FLAVOR == "lite" else "full",
            job_id=job_id,
            subtitle_doc=subtitle_doc,
            media_assets=self._project_service.scan_media_assets(project_dir),
            dir=str(project_dir),
            created_at=time(),
            updated_at=time(),
        )
        self._project_service.save_project(project, project_dir=project_dir)
        self._record_mapping(job_id, project_id, True)
        return project_id

    def _ensure_project_dir_canonical(self, project: Project, legacy_job_id: str) -> None:
        """
        确保 legacy 任务在解析后立即切换为 project 目录语义。

        规则：
        1. 旧目录 `jobs/{job_id}` 必须迁移到 `jobs/{project_id}`；
        2. 迁移后立即回写 `project_meta.dir`；
        3. 旧目录必须被删除（通过 rename/move 达成）。
        """
        project_dir = Path(project.dir) if project.dir else (config.JOBS_DIR / legacy_job_id)
        if not project_dir.exists():
            legacy_dir = config.JOBS_DIR / legacy_job_id
            canonical_dir = config.JOBS_DIR / project.project_id
            if legacy_dir.exists():
                project_dir = legacy_dir
            elif canonical_dir.exists():
                project_dir = canonical_dir
        if not project_dir.exists():
            return

        target_dir = self._promote_legacy_dir(project_dir, project.project_id)
        if str(target_dir) == str(project_dir):
            return

        project.dir = str(target_dir)
        project.updated_at = time()
        self._project_service.save_project(project, project_dir=target_dir)
        self._record_mapping(legacy_job_id, project.project_id, False)

    def _promote_legacy_dir(self, source_dir: Path, project_id: str) -> Path:
        """
        将 legacy job 目录提升为 project 目录，并删除旧目录入口。

        - 首选同卷 `os.replace`（原子重命名）；
        - 目标目录已存在时，执行覆盖式合并后删除旧目录。
        """
        normalized_project_id = str(project_id or "").strip()
        if not normalized_project_id:
            return source_dir

        target_dir = config.JOBS_DIR / normalized_project_id
        try:
            if source_dir.resolve() == target_dir.resolve():
                return target_dir
        except Exception:
            if source_dir == target_dir:
                return target_dir

        if not source_dir.exists():
            if target_dir.exists():
                return target_dir
            raise FileNotFoundError(f"legacy目录不存在: {source_dir}")

        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if not target_dir.exists():
            self._replace_dir_with_retry(source_dir, target_dir)
            return target_dir

        # 冲突场景：目标目录存在时，按“目标为主 + 源覆盖”策略合并，随后删除源目录。
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
        self._remove_dir_with_retry(source_dir)
        return target_dir

    @staticmethod
    def _is_windows_path_busy_error(exc: BaseException) -> bool:
        if not isinstance(exc, OSError):
            return False
        winerror = getattr(exc, "winerror", None)
        return winerror in {5, 32}

    def _replace_dir_with_retry(
        self,
        source_dir: Path,
        target_dir: Path,
        max_retries: int = 8,
    ) -> None:
        last_exc: Optional[BaseException] = None
        for attempt in range(1, max_retries + 1):
            try:
                os.replace(source_dir, target_dir)
                return
            except (PermissionError, OSError) as exc:
                if not self._is_windows_path_busy_error(exc):
                    raise
                last_exc = exc
                wait_seconds = min(1.6, 0.1 * (2 ** (attempt - 1)))
                logger.warning(
                    "legacy目录提升被占用，准备重试: source=%s, target=%s, attempt=%s/%s, wait=%.2fs, err=%s",
                    source_dir,
                    target_dir,
                    attempt,
                    max_retries,
                    wait_seconds,
                    exc,
                )
                sleep(wait_seconds)

        raise RuntimeError(
            f"legacy目录被占用，无法完成迁移: {source_dir} -> {target_dir}, err={last_exc}"
        ) from last_exc

    def _remove_dir_with_retry(self, source_dir: Path, max_retries: int = 8) -> None:
        last_exc: Optional[BaseException] = None
        for attempt in range(1, max_retries + 1):
            try:
                shutil.rmtree(source_dir, ignore_errors=False)
                return
            except (PermissionError, OSError) as exc:
                if not self._is_windows_path_busy_error(exc):
                    raise
                last_exc = exc
                wait_seconds = min(1.6, 0.1 * (2 ** (attempt - 1)))
                logger.warning(
                    "legacy目录清理被占用，准备重试: dir=%s, attempt=%s/%s, wait=%.2fs, err=%s",
                    source_dir,
                    attempt,
                    max_retries,
                    wait_seconds,
                    exc,
                )
                sleep(wait_seconds)

        raise RuntimeError(f"legacy源目录删除失败（被占用）: {source_dir}, err={last_exc}") from last_exc

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

    def _sanitize_workspace_mappings(self) -> None:
        """
        启动时清理历史 `p-*` 污染映射，避免 workspace 再次被回退为旧标识。
        """
        with self._lock:
            mappings = self._map.setdefault("mappings", {})
            keys_to_remove = [
                key
                for key in list(mappings.keys())
                if self._is_workspace_project_identifier(str(key or ""))
            ]
            if not keys_to_remove:
                return
            for key in keys_to_remove:
                mappings.pop(key, None)
            self._save_map()

    def _record_mapping(self, job_id: str, project_id: str, is_new_migration: bool) -> None:
        if self._is_workspace_project_identifier(job_id):
            return
        with self._lock:
            mappings = self._map.setdefault("mappings", {})
            mappings[job_id] = {
                "project_id": project_id,
                "migrated_at": time(),
                "status": "migrated" if is_new_migration else "existing",
            }
            self._save_map()

    @classmethod
    def _is_workspace_project_identifier(cls, identifier: str) -> bool:
        normalized_identifier = str(identifier or "").strip()
        if not normalized_identifier:
            return False
        return bool(cls._WORKSPACE_PROJECT_PATTERN.fullmatch(normalized_identifier))

    def _should_preserve_workspace_identity(self, project: Project, identifier: str) -> bool:
        if not self._is_workspace_project_identifier(identifier):
            return False
        project_mode = str(getattr(project, "mode", "") or "").strip().lower()
        return project_mode != "legacy"

    @staticmethod
    def _normalize_project_metadata_for_workspace(project: Project, project_id: str) -> None:
        subtitle_doc = getattr(project, "subtitle_doc", None)
        source_type = str(getattr(subtitle_doc, "source_type", "") or "").strip().lower()
        inferred_task_mode = infer_task_mode(
            raw_task_mode=getattr(project, "task_mode", None),
            project_mode=getattr(project, "mode", None),
            subtitle_source_type=source_type,
            project_dir=str(getattr(project, "dir", "") or ""),
        )
        project.task_mode = inferred_task_mode
        project.mode = derive_compat_project_mode(
            task_mode=inferred_task_mode,
            existing_mode=getattr(project, "mode", "normal"),
        )

        if subtitle_doc is not None:
            subtitle_doc.project_id = project_id
            doc_id = str(getattr(subtitle_doc, "doc_id", "") or "").strip()
            if not doc_id or doc_id == str(getattr(project, "job_id", "") or "").strip():
                subtitle_doc.doc_id = project_id
            if inferred_task_mode == "subtitle_edit" and source_type == "transcribe":
                subtitle_doc.source_type = "import"
            elif inferred_task_mode == "transcribe" and source_type == "import":
                subtitle_doc.source_type = "transcribe"
            subtitle_doc.updated_at = time()

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
