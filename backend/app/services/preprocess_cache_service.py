"""
PreprocessCacheService - 预处理缓存管理

V3.2.0+dev.20260122.03: 新增预处理缓存体系（阶段零）
V3.2.0+dev.20260123.01: 修复分离缓存原子写入格式识别
V3.2.0+dev.20260123.02: 修复音频原子写入的 fsync 句柄类型问题
V3.2.0+dev.20260123.03: 分离缓存优先恢复与元数据复用
V3.2.0+dev.20260123.04: 分离缓存容错恢复（产物完整即可跳过）
V3.2.0+dev.20260127.05: 增加 LangID/Speaker 缓存与进度恢复
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf

from app.models.circuit_breaker_models import SeparationLevel
from app.models.job_models import PreprocessingConfig
from app.services.audio.chunk_engine import AudioChunk
from app.services.sse_service import get_sse_manager


@dataclass(frozen=True)
class PreprocessCachePaths:
    """预处理缓存路径集合"""

    base_dir: Path
    manifest_path: Path
    vad_dir: Path
    vad_state_path: Path
    vad_chunks_metadata_path: Path
    triage_dir: Path
    triage_results_path: Path
    separation_dir: Path
    separation_stats_path: Path
    separation_full_dir: Path
    separation_full_audio_path: Path
    separation_full_state_path: Path
    langid_dir: Path
    langid_language_map_path: Path
    langid_progress_path: Path
    langid_metadata_path: Path
    langid_report_path: Path
    speaker_dir: Path
    speaker_embeddings_path: Path
    speaker_index_map_path: Path
    speaker_progress_path: Path
    speaker_metadata_path: Path


class PreprocessCacheService:
    """
    预处理缓存管理器

    负责 VAD / 频谱分诊 / 人声分离缓存的写入、校验与恢复。
    """

    VERSION = "3.2.0+dev.20260127.05"
    _ATOMIC_REPLACE_MAX_RETRIES = 8
    _ATOMIC_REPLACE_BASE_DELAY_SECONDS = 0.05
    _ATOMIC_REPLACE_MAX_DELAY_SECONDS = 0.3

    def __init__(self, job_dir: Path, logger: Optional[logging.Logger] = None) -> None:
        self.job_dir = Path(job_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.paths = self._build_paths(self.job_dir)
        self._manifest: Optional[Dict[str, Any]] = None
        self._separation_state: Optional[Dict[str, Any]] = None
        self._separation_mode: Optional[str] = None
        self._separation_total: Optional[int] = None

    @staticmethod
    def _build_paths(job_dir: Path) -> PreprocessCachePaths:
        base_dir = job_dir / "cache_preprocess"
        vad_dir = base_dir / "vad"
        triage_dir = base_dir / "triage"
        separation_dir = base_dir / "separation"
        separation_full_dir = base_dir / "separation_full"
        langid_dir = base_dir / "langid"
        speaker_dir = base_dir / "speaker"
        return PreprocessCachePaths(
            base_dir=base_dir,
            manifest_path=base_dir / "manifest.json",
            vad_dir=vad_dir,
            vad_state_path=vad_dir / "vad_state.json",
            vad_chunks_metadata_path=vad_dir / "chunks_metadata.json",
            triage_dir=triage_dir,
            triage_results_path=triage_dir / "triage_results.json",
            separation_dir=separation_dir,
            separation_stats_path=separation_dir / "separation_stats.json",
            separation_full_dir=separation_full_dir,
            separation_full_audio_path=separation_full_dir / "vocals_full.wav",
            separation_full_state_path=separation_full_dir / "separation_full_state.json",
            langid_dir=langid_dir,
            langid_language_map_path=langid_dir / "language_map.json",
            langid_progress_path=langid_dir / "progress.json",
            langid_metadata_path=langid_dir / "metadata.json",
            langid_report_path=langid_dir / "langid_report.json",
            speaker_dir=speaker_dir,
            speaker_embeddings_path=speaker_dir / "embeddings.npy",
            speaker_index_map_path=speaker_dir / "index_map.json",
            speaker_progress_path=speaker_dir / "progress.json",
            speaker_metadata_path=speaker_dir / "metadata.json",
        )

    def ensure_dirs(self) -> None:
        """确保缓存目录存在"""
        self.paths.base_dir.mkdir(parents=True, exist_ok=True)
        self.paths.vad_dir.mkdir(parents=True, exist_ok=True)
        self.paths.triage_dir.mkdir(parents=True, exist_ok=True)
        self.paths.separation_dir.mkdir(parents=True, exist_ok=True)
        self.paths.separation_full_dir.mkdir(parents=True, exist_ok=True)
        self.paths.langid_dir.mkdir(parents=True, exist_ok=True)
        self.paths.speaker_dir.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> Dict[str, Any]:
        """加载 manifest，如果不存在则创建默认结构"""
        if self._manifest is not None:
            return self._manifest
        if not self.paths.manifest_path.exists():
            self._manifest = {
                "version": self.VERSION,
                "job_id": self.job_dir.name,
                "created_at": self._now_iso(),
                "updated_at": self._now_iso(),
                "stages": {},
            }
            return self._manifest
        try:
            with open(self.paths.manifest_path, "r", encoding="utf-8") as f:
                self._manifest = json.load(f)
        except Exception as exc:
            self.logger.warning("加载预处理 manifest 失败，将重建: %s", exc)
            self._manifest = {
                "version": self.VERSION,
                "job_id": self.job_dir.name,
                "created_at": self._now_iso(),
                "updated_at": self._now_iso(),
                "stages": {},
            }
        return self._manifest

    def save_manifest(self, manifest: Dict[str, Any]) -> None:
        """原子写入 manifest"""
        manifest["updated_at"] = self._now_iso()
        manifest.setdefault("version", self.VERSION)
        manifest.setdefault("job_id", self.job_dir.name)
        self._atomic_write_json(self.paths.manifest_path, manifest)
        self._manifest = manifest

    def update_manifest_stage(
        self,
        stage: str,
        completed: bool,
        state_file: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """更新 manifest 的阶段状态"""
        manifest = self.load_manifest()
        stage_payload = {
            "completed": completed,
            "state_file": state_file,
            "updated_at": self._now_iso(),
        }
        if extra:
            stage_payload.update(extra)
        manifest.setdefault("stages", {})[stage] = stage_payload
        self.save_manifest(manifest)

    def load_vad_chunks(self) -> Optional[List[AudioChunk]]:
        """从 VAD 缓存恢复 AudioChunk 列表"""
        manifest = self.load_manifest()
        stage = manifest.get("stages", {}).get("vad", {})
        if not stage.get("completed", False):
            return None
        if not self.paths.vad_state_path.exists():
            return None
        try:
            with open(self.paths.vad_state_path, "r", encoding="utf-8") as f:
                vad_state = json.load(f)
        except Exception as exc:
            self.logger.warning("加载 VAD 状态失败: %s", exc)
            return None
        if not vad_state.get("completed", False):
            return None
        if not self.paths.vad_chunks_metadata_path.exists():
            return None
        try:
            with open(self.paths.vad_chunks_metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as exc:
            self.logger.warning("加载 VAD chunks_metadata 失败: %s", exc)
            return None
        if not isinstance(metadata, list):
            return None

        chunks: List[AudioChunk] = []
        for meta in metadata:
            index = meta.get("index")
            audio_file = meta.get("file")
            if index is None or not audio_file:
                continue
            audio_path = self.paths.vad_dir / audio_file
            if not audio_path.exists():
                self.logger.warning("VAD 缓存缺失文件: %s", audio_path)
                return None
            try:
                audio, sr = self._read_audio(audio_path)
            except Exception as exc:
                self.logger.warning("读取 VAD 缓存失败: %s", exc)
                return None
            chunk = AudioChunk(
                index=index,
                start=float(meta.get("start_time", 0.0)),
                end=float(meta.get("end_time", 0.0)),
                audio=audio,
                sample_rate=int(meta.get("sample_rate", sr)),
            )
            chunks.append(chunk)

        if not chunks:
            return None
        return chunks

    def load_chunks_metadata(self, require_completed: bool = False) -> Optional[List[Dict[str, Any]]]:
        """读取 chunks_metadata（仅元数据，允许 VAD 音频缺失）"""
        if require_completed:
            if not self.paths.vad_state_path.exists():
                return None
            try:
                with open(self.paths.vad_state_path, "r", encoding="utf-8") as f:
                    vad_state = json.load(f)
                if not vad_state.get("completed", False):
                    return None
            except Exception as exc:
                self.logger.warning("加载 VAD 状态失败: %s", exc)
                return None
        if not self.paths.vad_chunks_metadata_path.exists():
            return None
        try:
            with open(self.paths.vad_chunks_metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as exc:
            self.logger.warning("加载 chunks_metadata 失败: %s", exc)
            return None
        if not isinstance(metadata, list) or not metadata:
            return None
        sanitized: List[Dict[str, Any]] = []
        for item in metadata:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            start_time = item.get("start_time")
            end_time = item.get("end_time")
            if index is None or start_time is None or end_time is None:
                continue
            sanitized.append(
                {
                    "index": int(index),
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "duration": float(item.get("duration", float(end_time) - float(start_time))),
                    "sample_rate": int(item.get("sample_rate", 16000)),
                }
            )
        return sanitized or None

    def load_separation_chunks(self, expected_mode: str) -> Optional[List[AudioChunk]]:
        """优先从分离缓存恢复 AudioChunk（无需 VAD 音频）"""
        metadata = self.load_chunks_metadata(require_completed=False)
        if not metadata:
            return None
        chunks = self._build_chunks_from_metadata(metadata)
        processed_indices, is_completed = self.load_separation_cache(chunks, expected_mode)
        if not is_completed or len(processed_indices) < len(chunks):
            # 分离状态不完整时，尝试仅基于产物文件恢复，避免重跑阶段
            if not self._fill_separation_outputs(chunks, processed_indices):
                return None
            self.logger.info("分离状态不完整但产物齐全，使用分离输出恢复")
        return chunks

    def save_vad_chunks(self, chunks: List[AudioChunk]) -> None:
        """保存 VAD 缓存（chunk 音频 + 元数据）"""
        if not chunks:
            return
        self.ensure_dirs()

        metadata: List[Dict[str, Any]] = []
        for chunk in chunks:
            filename = f"chunk_{chunk.index}.wav"
            audio_path = self.paths.vad_dir / filename
            self._atomic_write_audio(audio_path, chunk.audio, chunk.sample_rate)
            metadata.append(
                {
                    "index": chunk.index,
                    "start_time": chunk.start,
                    "end_time": chunk.end,
                    "duration": chunk.end - chunk.start,
                    "sample_rate": chunk.sample_rate,
                    "file": filename,
                }
            )

        self._atomic_write_json(self.paths.vad_chunks_metadata_path, metadata)
        vad_state = {
            "version": self.VERSION,
            "completed": True,
            "total_chunks": len(metadata),
            "sample_rate": chunks[0].sample_rate,
            "chunks_metadata_file": self.paths.vad_chunks_metadata_path.name,
            "updated_at": self._now_iso(),
        }
        self._atomic_write_json(self.paths.vad_state_path, vad_state)
        self.update_manifest_stage(
            stage="vad",
            completed=True,
            state_file=str(Path("vad") / self.paths.vad_state_path.name),
            extra={"total_chunks": len(metadata)},
        )

    def load_triage_results(
        self,
        total_chunks: int,
        use_dnsmos_triage: bool,
        threshold: float,
        use_smart_probe: bool,
        smart_probe_params: Optional[Dict[str, Any]] = None,
        triage_version: str = "",
        dnsmos_model_hash: str = "",
        dnsmos_threshold_profile_hash: str = "",
    ) -> Optional[Dict[str, Any]]:
        """加载音频预检缓存并校验配置。"""
        if not self.paths.triage_results_path.exists():
            return None
        try:
            with open(self.paths.triage_results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception as exc:
            self.logger.warning("加载分诊缓存失败: %s", exc)
            return None
        if not results.get("completed", False):
            return None
        if results.get("total_chunks") != total_chunks:
            return None
        if results.get("triage_version") != triage_version:
            return None
        if results.get("use_dnsmos_triage") != use_dnsmos_triage:
            return None
        if results.get("dnsmos_model_hash") != dnsmos_model_hash:
            return None
        if results.get("dnsmos_threshold_profile_hash") != dnsmos_threshold_profile_hash:
            return None
        if abs(float(results.get("threshold", threshold)) - threshold) > 1e-6:
            return None
        if results.get("use_smart_probe") != use_smart_probe:
            return None
        cached_chunks = results.get("chunks", [])
        if not isinstance(cached_chunks, list) or len(cached_chunks) != total_chunks:
            return None
        if use_smart_probe:
            if not smart_probe_params:
                return None
            cached_probe = results.get("probe", {})
            if cached_probe.get("probe_sep_ratio_min") != smart_probe_params.get("probe_sep_ratio_min"):
                return None
            if cached_probe.get("probe_min_coverage") != smart_probe_params.get("probe_min_coverage"):
                return None
            if cached_probe.get("max_step_chunks") != smart_probe_params.get("max_step_chunks"):
                return None
        return results

    def apply_triage_results(self, chunks: List[AudioChunk], results: Dict[str, Any]) -> None:
        """将分诊缓存结果应用到 AudioChunk"""
        cache_chunks = results.get("chunks", [])
        if not isinstance(cache_chunks, list):
            return
        chunk_map = {item.get("chunk_index"): item for item in cache_chunks}
        for chunk in chunks:
            cached = chunk_map.get(chunk.index)
            if not cached:
                continue
            chunk.needs_separation = bool(cached.get("need_separation", False))
            chunk.recommended_model = cached.get("recommended_model")

    def save_triage_results(
        self,
        chunks: List[AudioChunk],
        triage_log: Optional[List[Dict[str, Any]]],
        probe_state: Optional[Dict[str, Any]],
        use_dnsmos_triage: bool,
        threshold: float,
        use_smart_probe: bool,
        triage_version: str,
        dnsmos_model_hash: str,
        dnsmos_threshold_profile_hash: str,
    ) -> None:
        """保存音频预检缓存结果。"""
        if not chunks:
            return
        self.ensure_dirs()
        results = {
            "version": self.VERSION,
            "triage_version": triage_version,
            "completed": True,
            "mode": "smart_probe" if use_smart_probe else "standard",
            "total_chunks": len(chunks),
            "use_dnsmos_triage": use_dnsmos_triage,
            # 兼容字段：保持旧键存在，但不再作为匹配主键
            "use_snr_triage": use_dnsmos_triage,
            "dnsmos_model_hash": dnsmos_model_hash,
            "dnsmos_threshold_profile_hash": dnsmos_threshold_profile_hash,
            "threshold": threshold,
            "use_smart_probe": use_smart_probe,
            "probe": probe_state or {},
            "chunks": [
                {
                    "chunk_index": chunk.index,
                    "need_separation": chunk.needs_separation,
                    "recommended_model": chunk.recommended_model,
                }
                for chunk in chunks
            ],
            "triage_log": triage_log or [],
            "updated_at": self._now_iso(),
        }
        self._atomic_write_json(self.paths.triage_results_path, results)
        self.update_manifest_stage(
            stage="triage",
            completed=True,
            state_file=str(Path("triage") / self.paths.triage_results_path.name),
            extra={"total_chunks": len(chunks)},
        )

    def begin_separation(self, mode: str, total_chunks: int) -> None:
        """初始化分离状态"""
        self.ensure_dirs()
        self._separation_mode = mode
        self._separation_total = total_chunks
        if self._separation_state is None:
            state = self._load_separation_state()
            if state is None or state.get("mode") != mode or state.get("total_chunks") != total_chunks:
                state = {
                    "version": self.VERSION,
                    "mode": mode,
                    "total_chunks": total_chunks,
                    "completed": False,
                    "chunks": {},
                    "updated_at": self._now_iso(),
                }
            self._separation_state = state
            self._atomic_write_json(self.paths.separation_stats_path, state)
            self.update_manifest_stage(
                stage="separation",
                completed=bool(state.get("completed", False)),
                state_file=str(Path("separation") / self.paths.separation_stats_path.name),
                extra={"mode": mode},
            )

    def load_separation_cache(
        self,
        chunks: List[AudioChunk],
        expected_mode: str,
    ) -> Tuple[Set[int], bool]:
        """加载分离缓存并更新 AudioChunk"""
        state = self._load_separation_state()
        if not state:
            return set(), False
        if state.get("mode") != expected_mode:
            return set(), False
        if state.get("total_chunks") not in (None, len(chunks)):
            return set(), False

        processed_indices: Set[int] = set()
        chunk_meta = state.get("chunks", {})
        if not isinstance(chunk_meta, dict):
            return set(), False

        for chunk in chunks:
            meta = chunk_meta.get(str(chunk.index))
            if not meta:
                continue
            file_name = meta.get("file")
            if not file_name:
                continue
            audio_path = self.paths.separation_dir / file_name
            if not audio_path.exists():
                continue
            try:
                audio, sr = self._read_audio(audio_path)
            except Exception as exc:
                self.logger.warning("读取分离缓存失败: %s", exc)
                continue
            chunk.audio = audio
            chunk.sample_rate = sr
            chunk.is_separated = bool(meta.get("is_separated", False))
            if "needs_separation" in meta:
                chunk.needs_separation = bool(meta.get("needs_separation"))
            level = meta.get("separation_level")
            if level:
                try:
                    chunk.separation_level = SeparationLevel(level)
                except ValueError:
                    chunk.separation_level = SeparationLevel.NONE
            else:
                chunk.separation_level = SeparationLevel.NONE
            chunk.separation_model = meta.get("separation_model")
            processed_indices.add(chunk.index)

        self._separation_state = state
        return processed_indices, bool(state.get("completed", False))

    def save_separation_chunk(self, chunk: AudioChunk, is_separated: bool, skip_audio_write: bool = False) -> None:
        """保存单个 Chunk 的分离输出"""
        if self._separation_state is None:
            return
        filename = f"chunk_{chunk.index}.wav"
        output_path = self.paths.separation_dir / filename
        if not skip_audio_write:
            self._atomic_write_audio(output_path, chunk.audio, chunk.sample_rate)

        chunk_state = {
            "file": filename,
            "is_separated": is_separated,
            "separation_level": chunk.separation_level.value if chunk.separation_level else SeparationLevel.NONE.value,
            "separation_model": chunk.separation_model,
            "needs_separation": chunk.needs_separation,
        }
        self._separation_state.setdefault("chunks", {})[str(chunk.index)] = chunk_state
        self._separation_state["updated_at"] = self._now_iso()
        self._atomic_write_json(self.paths.separation_stats_path, self._separation_state)
        self._remove_vad_chunk(chunk.index)

    def save_passthrough_chunks(self, chunks: Iterable[AudioChunk], only_unseparated: bool = True) -> None:
        """保存无需分离的 Chunk 输出到分离目录"""
        if self._separation_state is None:
            return
        for chunk in chunks:
            if only_unseparated and chunk.needs_separation:
                continue
            existing = self._separation_state.get("chunks", {}).get(str(chunk.index))
            if existing:
                continue
            output_path = self.paths.separation_dir / f"chunk_{chunk.index}.wav"
            try:
                if not output_path.exists():
                    if not self._try_link_from_vad(chunk.index, output_path):
                        self._atomic_write_audio(output_path, chunk.audio, chunk.sample_rate)
                else:
                    self._atomic_write_audio(output_path, chunk.audio, chunk.sample_rate)
            except Exception as exc:
                self.logger.warning("写入分离输出失败: %s", exc)
                continue
            chunk.separation_level = SeparationLevel.NONE
            chunk.is_separated = False
            chunk.separation_model = None
            self.save_separation_chunk(chunk, is_separated=False, skip_audio_write=True)

    def save_global_separation(
        self,
        chunks: List[AudioChunk],
        separated_path: str,
        separation_model: Optional[str],
    ) -> None:
        """保存整轨分离缓存"""
        if self._separation_state is None:
            return
        if separated_path:
            target = self.paths.separation_full_audio_path
            if not target.exists():
                try:
                    os.link(separated_path, target)
                except OSError:
                    shutil.copyfile(separated_path, target)
        full_state = {
            "version": self.VERSION,
            "completed": True,
            "model": separation_model,
            "updated_at": self._now_iso(),
        }
        self._atomic_write_json(self.paths.separation_full_state_path, full_state)
        self.update_manifest_stage(
            stage="separation_full",
            completed=True,
            state_file=str(Path("separation_full") / self.paths.separation_full_state_path.name),
            extra={"model": separation_model},
        )

        for chunk in chunks:
            if chunk.is_separated:
                if separation_model:
                    chunk.separation_model = separation_model
                if chunk.separation_level == SeparationLevel.NONE:
                    if separation_model == SeparationLevel.MDX_EXTRA.value:
                        chunk.separation_level = SeparationLevel.MDX_EXTRA
                    else:
                        chunk.separation_level = SeparationLevel.HTDEMUCS
                self.save_separation_chunk(chunk, is_separated=True)
            else:
                chunk.separation_level = SeparationLevel.NONE
                chunk.separation_model = None
                self.save_separation_chunk(chunk, is_separated=False)

    def finalize_separation(self, total_chunks: int) -> None:
        """完成分离阶段并更新 manifest"""
        if self._separation_state is None:
            return
        processed = len(self._separation_state.get("chunks", {}))
        is_completed = processed >= total_chunks
        self._separation_state["completed"] = is_completed
        self._separation_state["updated_at"] = self._now_iso()
        self._atomic_write_json(self.paths.separation_stats_path, self._separation_state)
        self.update_manifest_stage(
            stage="separation",
            completed=is_completed,
            state_file=str(Path("separation") / self.paths.separation_stats_path.name),
            extra={"processed": processed, "total_chunks": total_chunks},
        )

    def recover_langid_cache(
        self,
        chunks: Iterable[AudioChunk],
        expected_metadata: Dict[str, Any],
    ) -> Optional[Tuple[Dict[int, Dict[str, Any]], List[int], bool]]:
        """恢复 LangID 缓存，返回 (语言映射, 缺失索引, 是否完成)"""
        if not expected_metadata:
            return None
        if not self.paths.langid_metadata_path.exists():
            return None
        if not self.paths.langid_language_map_path.exists():
            return None

        cached_metadata = self._load_json(self.paths.langid_metadata_path)
        if cached_metadata is None:
            return None
        if not self._match_langid_metadata(cached_metadata, expected_metadata):
            return None

        raw_map = self._load_json(self.paths.langid_language_map_path)
        lang_map = self._normalize_langid_map(raw_map)
        if not lang_map:
            return None

        progress = self._load_json(self.paths.langid_progress_path) or {}
        chunk_indices = {chunk.index for chunk in chunks}
        cached_indices = set(lang_map.keys())
        complete = bool(progress.get("complete", False))
        if not progress:
            complete = cached_indices == chunk_indices
        if complete and cached_indices != chunk_indices:
            return None

        missing = sorted(chunk_indices - cached_indices)
        return lang_map, missing, complete

    def save_langid_cache(
        self,
        language_map: Dict[int, Dict[str, Any]],
        metadata: Dict[str, Any],
        progress: Optional[Dict[str, Any]] = None,
    ) -> None:
        """写入 LangID 缓存（language_map + metadata + progress）"""
        if not language_map:
            return
        self.ensure_dirs()

        sanitized_map = {
            str(index): {
                "language": str(payload.get("language", "auto")),
                "confidence": float(payload.get("confidence", 0.0)),
                "raw_label": payload.get("raw_label"),
                "raw_language": payload.get("raw_language"),
                "raw_confidence": (
                    float(payload.get("raw_confidence"))
                    if payload.get("raw_confidence") is not None
                    else None
                ),
            }
            for index, payload in language_map.items()
        }
        self._atomic_write_json(self.paths.langid_language_map_path, sanitized_map)
        self._atomic_write_json(self.paths.langid_metadata_path, metadata)
        if progress:
            self._atomic_write_json(self.paths.langid_progress_path, progress)

        completed = bool(progress.get("complete", True)) if progress else True
        self.update_manifest_stage(
            stage="langid",
            completed=completed,
            state_file=str(Path("langid") / self.paths.langid_progress_path.name),
            extra={"total_chunks": progress.get("total_chunks") if progress else len(language_map)},
        )

    def save_langid_report(
        self,
        chunks: List[AudioChunk],
        language_map: Dict[int, Dict[str, Any]],
        metadata: Dict[str, Any],
        confidence_threshold: float,
    ) -> None:
        """写入 LangID 详细报告（包含白名单前后标签与置信度）"""
        if not chunks:
            return
        if not language_map:
            return
        self.ensure_dirs()

        report_items: List[Dict[str, Any]] = []
        for chunk in chunks:
            payload = language_map.get(chunk.index, {})
            raw_confidence = payload.get("raw_confidence")
            if raw_confidence is None:
                raw_confidence = payload.get("confidence", 0.0)
            report_items.append(
                {
                    "chunk_index": int(chunk.index),
                    "raw_label": payload.get("raw_label"),
                    "raw_language": payload.get("raw_language"),
                    "language_whitelist": payload.get("language", "auto"),
                    "confidence": float(raw_confidence or 0.0),
                }
            )

        report = {
            "schema_version": "1.0",
            "job_id": self.job_dir.name,
            "generated_at": self._now_iso(),
            "total_chunks": len(chunks),
            "confidence_threshold": float(confidence_threshold),
            "metadata": {
                "mode": metadata.get("mode"),
                "model_id": metadata.get("model_id"),
                "model_repo": metadata.get("model_repo"),
                "model_hash": metadata.get("model_hash"),
                "resolved_device": metadata.get("resolved_device"),
                "whitelist": metadata.get("whitelist", []),
                "logit_bias_score": metadata.get("logit_bias_score"),
            },
            "items": report_items,
        }

        self._atomic_write_json(self.paths.langid_report_path, report)

    @staticmethod
    def build_langid_progress(
        processed_indices: Iterable[int],
        total_chunks: int,
        complete: bool,
    ) -> Dict[str, Any]:
        """构建 LangID 进度快照"""
        processed = sorted({int(idx) for idx in processed_indices})
        last_batch_end = processed[-1] if processed else -1
        return {
            "complete": bool(complete),
            "processed_indices": processed,
            "last_batch_end": last_batch_end,
            "total_chunks": int(total_chunks),
            "updated_at": PreprocessCacheService._now_iso(),
        }

    def recover_speaker_cache(
        self,
        chunks: Iterable[AudioChunk],
        expected_metadata: Dict[str, Any],
    ) -> Optional[Tuple[Dict[int, List[float]], List[int], bool]]:
        """恢复 Speaker 缓存，返回 (embedding映射, 缺失索引, 是否完成)"""
        if not expected_metadata:
            return None
        if not self.paths.speaker_metadata_path.exists():
            return None
        if not self.paths.speaker_index_map_path.exists():
            return None
        if not self.paths.speaker_embeddings_path.exists():
            return None

        cached_metadata = self._load_json(self.paths.speaker_metadata_path)
        if cached_metadata is None:
            return None
        if not self._match_speaker_metadata(cached_metadata, expected_metadata):
            return None

        raw_index_map = self._load_json(self.paths.speaker_index_map_path)
        index_map = self._normalize_index_map(raw_index_map)
        if not index_map:
            return None

        try:
            embeddings = np.load(self.paths.speaker_embeddings_path)
        except Exception as exc:
            self.logger.warning("加载 speaker embeddings 失败: %s", exc)
            return None
        if embeddings.ndim != 2:
            return None

        embedding_map: Dict[int, List[float]] = {}
        for chunk in chunks:
            idx = index_map.get(chunk.index)
            if idx is None or idx >= embeddings.shape[0]:
                continue
            embedding_map[chunk.index] = embeddings[idx].astype(float).tolist()

        progress = self._load_json(self.paths.speaker_progress_path) or {}
        chunk_indices = {chunk.index for chunk in chunks}
        cached_indices = set(embedding_map.keys())
        complete = bool(progress.get("complete", False))
        if not progress:
            complete = cached_indices == chunk_indices
        if complete and cached_indices != chunk_indices:
            return None

        missing = sorted(chunk_indices - cached_indices)
        return embedding_map, missing, complete

    def save_speaker_cache(
        self,
        embedding_map: Dict[int, List[float]],
        metadata: Dict[str, Any],
        progress: Optional[Dict[str, Any]] = None,
    ) -> None:
        """写入 Speaker 缓存（embeddings.npy + index_map + metadata + progress）"""
        if not embedding_map:
            return
        self.ensure_dirs()

        vectors: List[np.ndarray] = []
        index_map: Dict[str, int] = {}
        embedding_dim: Optional[int] = metadata.get("embedding_dim") if metadata else None
        for chunk_index, vector in embedding_map.items():
            array = np.asarray(vector, dtype=np.float32).reshape(-1)
            if embedding_dim is None:
                embedding_dim = int(array.shape[0])
            if int(array.shape[0]) != int(embedding_dim):
                self.logger.warning("Speaker embedding 维度不一致: chunk=%s", chunk_index)
                continue
            index_map[str(chunk_index)] = len(vectors)
            vectors.append(array)

        if not vectors:
            return

        if metadata is None:
            metadata = {}
        if embedding_dim is not None:
            metadata = {**metadata, "embedding_dim": int(embedding_dim)}

        embeddings = np.stack(vectors, axis=0)
        self._atomic_write_numpy(self.paths.speaker_embeddings_path, embeddings)
        self._atomic_write_json(self.paths.speaker_index_map_path, index_map)
        self._atomic_write_json(self.paths.speaker_metadata_path, metadata)
        if progress:
            self._atomic_write_json(self.paths.speaker_progress_path, progress)

        completed = bool(progress.get("complete", True)) if progress else True
        self.update_manifest_stage(
            stage="speaker",
            completed=completed,
            state_file=str(Path("speaker") / self.paths.speaker_progress_path.name),
            extra={"total_chunks": progress.get("total_chunks") if progress else len(index_map)},
        )

    @staticmethod
    def build_speaker_progress(
        processed_indices: Iterable[int],
        total_chunks: int,
        complete: bool,
    ) -> Dict[str, Any]:
        """构建 Speaker 进度快照"""
        processed = sorted({int(idx) for idx in processed_indices})
        last_batch_end = processed[-1] if processed else -1
        return {
            "complete": bool(complete),
            "processed_indices": processed,
            "last_batch_end": last_batch_end,
            "total_chunks": int(total_chunks),
            "updated_at": PreprocessCacheService._now_iso(),
        }

    def reset_separation_cache(self) -> None:
        """清理分离缓存（全局分离暂停回滚）"""
        state = self._separation_state or self._load_separation_state()
        if not state:
            return
        self._clear_directory(self.paths.separation_dir)
        self._clear_directory(self.paths.separation_full_dir)
        state["completed"] = False
        state["chunks"] = {}
        state["updated_at"] = self._now_iso()
        self._separation_state = state
        self._atomic_write_json(self.paths.separation_stats_path, state)
        self.update_manifest_stage(
            stage="separation",
            completed=False,
            state_file=str(Path("separation") / self.paths.separation_stats_path.name),
            extra={"processed": 0, "total_chunks": state.get("total_chunks", 0)},
        )

    def maybe_warn_cache_pressure(self, job_id: str, config: PreprocessingConfig) -> None:
        """当缓存压力过大时提示前端"""
        if not self.paths.base_dir.exists():
            return
        try:
            total, used, free = shutil.disk_usage(self.paths.base_dir)
        except Exception:
            return
        free_gb = free / (1024 ** 3)
        cache_bytes = self._dir_size_bytes(self.paths.base_dir)
        cache_gb = cache_bytes / (1024 ** 3)
        warn_free = free_gb < 2.0
        warn_budget = config.cache_budget_gb > 0 and cache_gb > config.cache_budget_gb
        if not (warn_free or warn_budget):
            return
        sse_manager = get_sse_manager()
        sse_manager.broadcast_sync(
            f"project:{job_id}",
            "signal.cache_gc_recommend",
            {
                "job_id": job_id,
                "free_gb": round(free_gb, 2),
                "cache_gb": round(cache_gb, 2),
                "message": "磁盘空间紧张，建议开启预处理缓存清理",
            },
        )

    def run_gc(self, jobs_root: Path, config: PreprocessingConfig, exclude_job_id: Optional[str] = None) -> None:
        """按配置执行缓存 GC（默认关闭）"""
        if not config.is_preprocess_cache_gc_enabled:
            return
        jobs_root = Path(jobs_root)
        if not jobs_root.exists():
            return

        entries = self._collect_cache_entries(jobs_root, exclude_job_id)
        if not entries:
            return

        now_ts = datetime.utcnow().timestamp()
        if config.ttl_hours > 0:
            ttl_seconds = config.ttl_hours * 3600
            expired = [e for e in entries if now_ts - e["last_touch"] > ttl_seconds]
            for entry in expired:
                self._remove_cache_dir(entry["cache_dir"])
            entries = [e for e in entries if e not in expired]

        budget_bytes = config.cache_budget_gb * (1024 ** 3)
        if budget_bytes > 0:
            entries = sorted(entries, key=lambda e: e["last_touch"])
            total_bytes = sum(e["size_bytes"] for e in entries)
            while entries and total_bytes > budget_bytes:
                entry = entries.pop(0)
                total_bytes -= entry["size_bytes"]
                self._remove_cache_dir(entry["cache_dir"])

        if config.max_tasks > 0 and len(entries) > config.max_tasks:
            entries = sorted(entries, key=lambda e: e["last_touch"])
            remove_count = len(entries) - config.max_tasks
            for i in range(remove_count):
                self._remove_cache_dir(entries[i]["cache_dir"])

    def _collect_cache_entries(self, jobs_root: Path, exclude_job_id: Optional[str]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for job_dir in jobs_root.iterdir():
            if not job_dir.is_dir():
                continue
            if exclude_job_id and job_dir.name == exclude_job_id:
                continue
            cache_dir = job_dir / "cache_preprocess"
            if not cache_dir.exists():
                continue
            manifest = cache_dir / "manifest.json"
            last_touch = cache_dir.stat().st_mtime
            if manifest.exists():
                try:
                    with open(manifest, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                        updated = payload.get("updated_at")
                        if updated:
                            last_touch = datetime.fromisoformat(updated.replace("Z", "")).timestamp()
                except Exception:
                    pass
            size_bytes = self._dir_size_bytes(cache_dir)
            entries.append(
                {
                    "cache_dir": cache_dir,
                    "last_touch": last_touch,
                    "size_bytes": size_bytes,
                }
            )
        return entries

    def _remove_cache_dir(self, path: Path) -> None:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception as exc:
            self.logger.warning("清理缓存目录失败: %s", exc)

    def _clear_directory(self, path: Path) -> None:
        if not path.exists():
            return
        for item in path.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                else:
                    shutil.rmtree(item, ignore_errors=True)
            except Exception as exc:
                self.logger.warning("清理缓存文件失败: %s", exc)

    def _load_separation_state(self) -> Optional[Dict[str, Any]]:
        if not self.paths.separation_stats_path.exists():
            return None
        try:
            with open(self.paths.separation_stats_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            self.logger.warning("加载分离状态失败: %s", exc)
            return None

    def _remove_vad_chunk(self, index: int) -> None:
        vad_path = self.paths.vad_dir / f"chunk_{index}.wav"
        if vad_path.exists():
            try:
                vad_path.unlink()
            except Exception:
                pass

    def _try_link_from_vad(self, index: int, target: Path) -> bool:
        vad_path = self.paths.vad_dir / f"chunk_{index}.wav"
        if not vad_path.exists():
            return False
        try:
            os.link(vad_path, target)
            return True
        except OSError:
            return False

    def _fill_separation_outputs(self, chunks: List[AudioChunk], processed_indices: Set[int]) -> bool:
        """基于分离产物补齐缺失的音频数据（不强制依赖状态文件）"""
        for chunk in chunks:
            if chunk.index in processed_indices:
                continue
            audio_path = self.paths.separation_dir / f"chunk_{chunk.index}.wav"
            if not audio_path.exists():
                return False
            try:
                audio, sr = self._read_audio(audio_path)
            except Exception as exc:
                self.logger.warning("读取分离缓存失败: %s", exc)
                return False
            chunk.audio = audio
            chunk.sample_rate = sr
        return True

    @staticmethod
    def _build_chunks_from_metadata(metadata: List[Dict[str, Any]]) -> List[AudioChunk]:
        """根据元数据构造占位 AudioChunk（音频由后续缓存填充）"""
        chunks: List[AudioChunk] = []
        for item in metadata:
            chunks.append(
                AudioChunk(
                    index=int(item.get("index", len(chunks))),
                    start=float(item.get("start_time", 0.0)),
                    end=float(item.get("end_time", 0.0)),
                    audio=np.zeros(0, dtype=np.float32),
                    sample_rate=int(item.get("sample_rate", 16000)),
                )
            )
        return chunks

    def _load_json(self, path: Path) -> Optional[Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            self.logger.warning("加载缓存文件失败: %s -> %s", path, exc)
            return None

    @staticmethod
    def _float_equal(left: Optional[float], right: Optional[float], tol: float = 1e-6) -> bool:
        if left is None or right is None:
            return left is right
        try:
            return abs(float(left) - float(right)) <= tol
        except (TypeError, ValueError):
            return False

    def _match_langid_metadata(self, cached: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        keys = (
            "schema_version",
            "mode",
            "model_id",
            "model_repo",
            "model_hash",
            "target_duration_seconds",
            "batch_size",
            "resolved_device",
            "torch_version",
            "speechbrain_version",
            "language_code_standard",
            "whitelist",
            "logit_bias_score",
        )
        for key in keys:
            expected_value = expected.get(key)
            cached_value = cached.get(key)
            if key in {"target_duration_seconds", "logit_bias_score"}:
                if not self._float_equal(cached_value, expected_value):
                    return False
                continue
            if expected_value is None:
                continue
            if cached_value != expected_value:
                return False
        return True

    def _match_speaker_metadata(self, cached: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        keys = (
            "schema_version",
            "model_id",
            "model_repo",
            "model_hash",
            "embedding_dim",
            "resolved_device",
            "torch_version",
            "speechbrain_version",
        )
        for key in keys:
            expected_value = expected.get(key)
            cached_value = cached.get(key)
            if expected_value is None:
                continue
            if cached_value != expected_value:
                return False
        return True

    @staticmethod
    def _normalize_langid_map(raw_map: Any) -> Dict[int, Dict[str, Any]]:
        if not isinstance(raw_map, dict):
            return {}
        normalized: Dict[int, Dict[str, Any]] = {}
        for raw_key, payload in raw_map.items():
            try:
                index = int(raw_key)
            except (TypeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            language = payload.get("language")
            if language is None:
                continue
            confidence = payload.get("confidence", 0.0)
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                confidence_value = 0.0
            normalized[index] = {
                "language": str(language),
                "confidence": confidence_value,
                "raw_label": payload.get("raw_label"),
                "raw_language": payload.get("raw_language"),
                "raw_confidence": (
                    float(payload.get("raw_confidence"))
                    if payload.get("raw_confidence") is not None
                    else None
                ),
            }
        return normalized

    @staticmethod
    def _normalize_index_map(raw_map: Any) -> Dict[int, int]:
        if not isinstance(raw_map, dict):
            return {}
        normalized: Dict[int, int] = {}
        for raw_key, raw_value in raw_map.items():
            try:
                index = int(raw_key)
                value = int(raw_value)
            except (TypeError, ValueError):
                continue
            normalized[index] = value
        return normalized

    @staticmethod
    def _now_iso() -> str:
        return datetime.utcnow().isoformat() + "Z"

    @staticmethod
    def _read_audio(path: Path) -> Tuple[np.ndarray, int]:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        return audio, sr

    @staticmethod
    def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        PreprocessCacheService._atomic_replace(temp_path, path)

    @staticmethod
    def _atomic_write_numpy(path: Path, array: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with open(temp_path, "wb") as f:
            np.save(f, array)
            f.flush()
            os.fsync(f.fileno())
        PreprocessCacheService._atomic_replace(temp_path, path)

    @staticmethod
    def _atomic_write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        sf.write(str(temp_path), audio, sample_rate, format="WAV")
        with open(temp_path, "r+b") as f:
            os.fsync(f.fileno())
        PreprocessCacheService._atomic_replace(temp_path, path)

    @staticmethod
    def _is_retryable_atomic_replace_error(exc: OSError) -> bool:
        """判断是否属于 Windows 临时文件锁冲突。"""
        if isinstance(exc, PermissionError):
            return True
        return getattr(exc, "winerror", None) in {5, 32}

    @staticmethod
    def _atomic_replace(temp_path: Path, target_path: Path) -> None:
        """
        原子替换（Windows 友好版）。

        WinError 5/32 在杀软、索引器、并发读写场景下会偶发出现，
        这里短退避重试，避免分离缓存写入因瞬时锁竞争失败。
        """
        delay_seconds = PreprocessCacheService._ATOMIC_REPLACE_BASE_DELAY_SECONDS
        max_retries = PreprocessCacheService._ATOMIC_REPLACE_MAX_RETRIES
        max_delay = PreprocessCacheService._ATOMIC_REPLACE_MAX_DELAY_SECONDS
        for attempt in range(max_retries + 1):
            try:
                os.replace(temp_path, target_path)
                return
            except OSError as exc:
                is_retryable = PreprocessCacheService._is_retryable_atomic_replace_error(exc)
                if not is_retryable or attempt >= max_retries:
                    raise
                time.sleep(delay_seconds)
                delay_seconds = min(delay_seconds * 1.6, max_delay)

    @staticmethod
    def _dir_size_bytes(path: Path) -> int:
        total = 0
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    continue
        return total
