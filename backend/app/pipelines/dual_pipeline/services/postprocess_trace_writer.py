"""
后处理追踪落盘服务。

职责：
- 在显式 debug 开关开启时，将各阶段原始数据写入 jobs/{job_id}/debug/postprocess。
- 维护可检索的 manifest，供 debug API/离线排查读取。
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class PostprocessTraceWriter:
    """后处理追踪落盘器。"""

    def __init__(
        self,
        *,
        logger: Any,
        enabled: bool,
        level: str = "summary",
    ) -> None:
        self._logger = logger
        self._enabled = bool(enabled)
        self._level = str(level or "summary").strip().lower()
        if self._level not in {"summary", "full"}:
            self._level = "summary"

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def level(self) -> str:
        return self._level

    def write_stage(
        self,
        *,
        job_dir: Optional[Path],
        chunk_index: int,
        filename: str,
        payload: Any,
        stage: str,
        full_only: bool = False,
    ) -> None:
        """写入单个阶段快照。"""
        if not self._enabled or job_dir is None:
            return
        if full_only and self._level != "full":
            return
        if not filename.endswith(".json"):
            raise ValueError("postprocess trace 仅支持 json 文件")

        trace_dir = self._resolve_chunk_dir(job_dir=job_dir, chunk_index=chunk_index)
        target = trace_dir / filename
        serializable = self._to_jsonable(payload)
        self._atomic_write_json(target, serializable)
        self._update_manifest(
            job_dir=job_dir,
            chunk_index=chunk_index,
            filename=filename,
            stage=stage,
            kind="json",
            path=target.relative_to(job_dir).as_posix(),
        )

    def write_graph_artifact(
        self,
        *,
        job_dir: Optional[Path],
        chunk_index: int,
        filename: str,
        body: str,
        stage: str,
        media_type: str,
    ) -> None:
        """写入图形类产物（svg/html）。"""
        if not self._enabled or job_dir is None:
            return
        trace_dir = self._resolve_chunk_dir(job_dir=job_dir, chunk_index=chunk_index)
        target = trace_dir / filename
        self._atomic_write_text(target, body)
        self._update_manifest(
            job_dir=job_dir,
            chunk_index=chunk_index,
            filename=filename,
            stage=stage,
            kind=media_type,
            path=target.relative_to(job_dir).as_posix(),
        )

    def write_layer_summary(
        self,
        *,
        job_dir: Optional[Path],
        layer_summary: Any,
    ) -> None:
        """写入冻结后的层级 summary。"""
        if not self._enabled or job_dir is None:
            return

        serializable = self._to_jsonable(layer_summary)
        if not isinstance(serializable, dict) or not str(serializable.get("layer", "")).strip():
            raise ValueError("layer_summary 必须包含非空 layer 字段")

        layer = str(serializable["layer"]).strip()
        summary_dir = self._resolve_summary_dir(job_dir=job_dir)
        filename = f"{layer}.summary.json"
        target = summary_dir / filename
        self._atomic_write_json(target, serializable)
        self._update_summary_manifest(
            job_dir=job_dir,
            layer=layer,
            filename=filename,
            path=target.relative_to(job_dir).as_posix(),
        )

    def _resolve_chunk_dir(self, *, job_dir: Path, chunk_index: int) -> Path:
        trace_dir = (
            job_dir
            / "debug"
            / "postprocess"
            / f"chunk_{max(0, int(chunk_index)):04d}"
        )
        trace_dir.mkdir(parents=True, exist_ok=True)
        return trace_dir

    def _resolve_summary_dir(self, *, job_dir: Path) -> Path:
        summary_dir = job_dir / "debug" / "postprocess" / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        return summary_dir

    def _update_manifest(
        self,
        *,
        job_dir: Path,
        chunk_index: int,
        filename: str,
        stage: str,
        kind: str,
        path: str,
    ) -> None:
        manifest_path = job_dir / "debug" / "postprocess" / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = self._load_manifest(manifest_path)
        chunk_key = str(max(0, int(chunk_index)))
        chunk_entry = manifest.setdefault("chunks", {}).setdefault(chunk_key, {})
        file_entries = chunk_entry.setdefault("files", {})
        file_entries[filename] = {
            "filename": filename,
            "stage": stage,
            "kind": kind,
            "path": path,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        chunk_entry["chunk_index"] = max(0, int(chunk_index))
        manifest["updated_at"] = datetime.utcnow().isoformat() + "Z"
        self._atomic_write_json(manifest_path, manifest)

    def _update_summary_manifest(
        self,
        *,
        job_dir: Path,
        layer: str,
        filename: str,
        path: str,
    ) -> None:
        manifest_path = job_dir / "debug" / "postprocess" / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = self._load_manifest(manifest_path)
        manifest.setdefault("summaries", {})[layer] = {
            "filename": filename,
            "layer": layer,
            "kind": "summary",
            "path": path,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        manifest["updated_at"] = datetime.utcnow().isoformat() + "Z"
        self._atomic_write_json(manifest_path, manifest)

    @staticmethod
    def _load_manifest(path: Path) -> Dict[str, Any]:
        if not path.exists():
            now = datetime.utcnow().isoformat() + "Z"
            return {
                "schema_version": "2",
                "created_at": now,
                "updated_at": now,
                "chunks": {},
                "summaries": {},
            }
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError("manifest 非对象")
            payload.setdefault("chunks", {})
            payload.setdefault("summaries", {})
            payload.setdefault("schema_version", "2")
            payload.setdefault("created_at", datetime.utcnow().isoformat() + "Z")
            payload.setdefault("updated_at", datetime.utcnow().isoformat() + "Z")
            return payload
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            now = datetime.utcnow().isoformat() + "Z"
            return {
                "schema_version": "2",
                "created_at": now,
                "updated_at": now,
                "chunks": {},
                "summaries": {},
                "load_error": str(exc),
            }

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        tmp.replace(path)

    @staticmethod
    def _atomic_write_text(path: Path, body: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(body)
        tmp.replace(path)

    @classmethod
    def _to_jsonable(cls, payload: Any) -> Any:
        if payload is None:
            return None
        if isinstance(payload, (str, int, float, bool)):
            return payload
        if isinstance(payload, Path):
            return str(payload)
        if is_dataclass(payload):
            return cls._to_jsonable(asdict(payload))
        if isinstance(payload, dict):
            return {str(k): cls._to_jsonable(v) for k, v in payload.items()}
        if isinstance(payload, (list, tuple, set)):
            return [cls._to_jsonable(item) for item in payload]
        if hasattr(payload, "dict") and callable(getattr(payload, "dict")):
            try:
                return cls._to_jsonable(payload.dict())
            except Exception:
                return str(payload)
        if hasattr(payload, "__dict__"):
            try:
                return cls._to_jsonable(vars(payload))
            except Exception:
                return str(payload)
        return str(payload)
