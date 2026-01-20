"""
任务生命周期服务

职责：
1. 任务创建与元信息持久化
2. 任务状态恢复与重启纠偏
3. 断点检查与检查点写入
4. 任务暂停/取消与资源清理
"""
from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple

from app.models.job_models import JobState, JobSettings
from app.services.job_index_service import JobIndexService, get_job_index_service


class JobLifecycleService:
    """
    任务生命周期服务

    统一管理任务元信息、断点与重启恢复。
    """

    _ACTIVE_STATUSES = {
        "processing",
        "queued",
        "pausing",
        "canceling",
        "transcribing",
        "running",
    }

    def __init__(
        self,
        jobs_root: Path,
        logger: Optional[logging.Logger] = None
    ) -> None:
        self.jobs_root = Path(jobs_root)
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)

        self.jobs: Dict[str, JobState] = {}
        self.lock = threading.Lock()

        self.job_index: JobIndexService = get_job_index_service(str(self.jobs_root))
        self.job_index.cleanup_invalid_mappings()

        # 启动时加载已有任务
        self.load_all_jobs_from_disk()

    def load_all_jobs_from_disk(self) -> None:
        """
        启动时扫描并加载所有任务到内存。
        """
        try:
            loaded_count = 0
            for job_dir in self.jobs_root.iterdir():
                if not job_dir.is_dir():
                    continue

                job_id = job_dir.name
                with self.lock:
                    if job_id in self.jobs:
                        continue

                job = self.get_job(job_id)
                if job:
                    loaded_count += 1

            if loaded_count > 0:
                self.logger.info(f"启动时已加载 {loaded_count} 个历史任务到内存")
        except Exception as exc:
            self.logger.error(f"加载历史任务失败: {exc}")

    def create_job(
        self,
        filename: str,
        src_path: str,
        settings: JobSettings,
        job_id: Optional[str] = None
    ) -> JobState:
        """
        创建转录任务
        """
        job_id = job_id or uuid.uuid4().hex
        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        dest_path = job_dir / filename

        # V3.1.1+dev.20260106.01: 使用硬链接替代复制，节省磁盘空间
        if os.path.abspath(src_path) != os.path.abspath(dest_path):
            try:
                os.link(src_path, dest_path)
                self.logger.debug(f"硬链接创建成功: {src_path} -> {dest_path}")
            except (OSError, NotImplementedError) as exc:
                self.logger.warning(f"硬链接创建失败，回退到复制: {exc}")
                try:
                    shutil.copyfile(src_path, dest_path)
                    self.logger.debug(f"文件已复制: {src_path} -> {dest_path}")
                except Exception as copy_err:
                    self.logger.warning(f"文件复制失败: {copy_err}")

        job = JobState(
            job_id=job_id,
            filename=filename,
            dir=str(job_dir),
            input_path=src_path,
            settings=settings,
            status="uploaded",
            phase="pending",
            message="文件已上传"
        )

        with self.lock:
            self.jobs[job_id] = job

        self.job_index.add_mapping(src_path, job_id)
        self.save_job_meta(job)

        self._schedule_waveform_audio_extract(job_id, dest_path)

        self.logger.info(f"任务已创建: {job_id} - {filename}")
        return job

    def save_job_meta(self, job: JobState) -> bool:
        """
        保存任务元信息到 job_meta.json（用于重启后恢复）
        """
        job_dir = Path(job.dir)
        meta_file = job_dir / "job_meta.json"

        try:
            job_dir.mkdir(parents=True, exist_ok=True)
            temp_file = meta_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(job.to_meta_dict(), f, indent=2, ensure_ascii=False)
            temp_file.replace(meta_file)
            self.logger.debug(f"任务元信息已保存: {job.job_id}")
            return True
        except Exception as exc:
            self.logger.error(f"保存任务元信息失败 {job.job_id}: {exc}")
            return False

    def load_job_meta(self, job_id: str) -> Optional[JobState]:
        """
        从 job_meta.json 加载任务元信息
        """
        job_dir = self.jobs_root / job_id
        meta_file = job_dir / "job_meta.json"

        if not meta_file.exists():
            return None

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            job = JobState.from_meta_dict(data)
            job.dir = str(job_dir)

            corrected = self._normalize_orphaned_job(job)
            if corrected:
                self.save_job_meta(job)

            self.logger.debug(f"从 job_meta.json 加载任务: {job_id}")
            return job
        except Exception as exc:
            self.logger.error(f"加载任务元信息失败 {job_id}: {exc}")
            return None

    def get_job(self, job_id: str) -> Optional[JobState]:
        """
        获取任务状态
        """
        with self.lock:
            if job_id in self.jobs:
                return self.jobs[job_id]

        job_dir = self.jobs_root / job_id
        if not job_dir.exists():
            return None

        job = self.load_job_meta(job_id)
        if job:
            with self.lock:
                self.jobs[job_id] = job
            self.logger.info(f"从 job_meta.json 恢复任务: {job_id}")
            return job

        fallback_job = self._load_job_from_disk_legacy(job_id, job_dir)
        if fallback_job:
            with self.lock:
                self.jobs[job_id] = fallback_job
            self.save_job_meta(fallback_job)
            self.logger.info(f"从磁盘恢复任务（旧版兼容）: {job_id}")
        return fallback_job

    def scan_incomplete_jobs(self) -> List[Dict[str, Any]]:
        """
        扫描所有未完成的任务（有checkpoint.json的任务）
        """
        incomplete_jobs: List[Dict[str, Any]] = []

        try:
            for job_dir in self.jobs_root.iterdir():
                if not job_dir.is_dir():
                    continue

                checkpoint_path = job_dir / "checkpoint.json"
                if not checkpoint_path.exists():
                    continue

                try:
                    with open(checkpoint_path, "r", encoding="utf-8") as f:
                        checkpoint_data = json.load(f)

                    job_id = checkpoint_data.get("job_id") or job_dir.name
                    total_segments = checkpoint_data.get("total_segments", 0)
                    processed_indices = checkpoint_data.get("processed_indices", [])
                    processed_count = len(processed_indices)
                    progress = (
                        (processed_count / total_segments) * 100
                        if total_segments > 0 else 0
                    )

                    file_path = self.job_index.get_file_path(job_id)
                    filename = os.path.basename(file_path) if file_path else "未知文件"
                    job_meta = self.load_job_meta(job_id)
                    status = job_meta.status if job_meta else "paused"

                    incomplete_jobs.append({
                        "job_id": job_id,
                        "filename": filename,
                        "file_path": file_path,
                        "progress": round(progress, 2),
                        "processed_segments": processed_count,
                        "total_segments": total_segments,
                        "phase": checkpoint_data.get("phase", "unknown"),
                        "dir": str(job_dir),
                        "status": status,
                    })
                except Exception as exc:
                    self.logger.warning(f"读取检查点失败 {checkpoint_path}: {exc}")
                    continue

            self.logger.info(f"扫描到 {len(incomplete_jobs)} 个未完成任务")
            return incomplete_jobs
        except Exception as exc:
            self.logger.error(f"扫描未完成任务失败: {exc}")
            return []

    def restore_job_from_checkpoint(self, job_id: str) -> Optional[JobState]:
        """
        从检查点恢复任务状态（无 checkpoint 时从头开始）
        """
        job_dir = self.jobs_root / job_id
        if not job_dir.exists():
            return None

        checkpoint = self._load_checkpoint(job_dir)

        try:
            filename = "unknown"
            input_path = None
            for ext in [".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".mp3", ".wav", ".m4a"]:
                matches = list(job_dir.glob(f"*{ext}"))
                if matches:
                    filename = matches[0].name
                    input_path = str(matches[0])
                    break

            if not input_path:
                self.logger.warning(f"无法找到任务 {job_id} 的输入文件")
                return None

            if checkpoint:
                phase = checkpoint.get("phase", "pending")
                total_segments = checkpoint.get("total_segments", 0)
                processed_indices = checkpoint.get("processed_indices", [])
                processed = len(processed_indices)
                progress = round((processed / max(1, total_segments)) * 100, 2)
                message = f"已暂停 ({processed}/{total_segments}段)"
                self.logger.info(f"从检查点恢复任务: {job_id}")
            else:
                phase = "pending"
                total_segments = 0
                processed = 0
                progress = 0
                message = "系统重启，任务已暂停"
                self.logger.info(f"无检查点，任务将从头开始: {job_id}")

            job = JobState(
                job_id=job_id,
                filename=filename,
                dir=str(job_dir),
                input_path=input_path,
                settings=JobSettings(),
                status="paused",
                phase=phase,
                message=message,
                total=total_segments,
                processed=processed,
                progress=progress,
                paused=True,
            )

            with self.lock:
                self.jobs[job_id] = job

            self.save_job_meta(job)
            return job
        except Exception as exc:
            self.logger.error(f"从检查点恢复任务失败: {exc}")
            return None

    def check_file_checkpoint(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        检查文件是否有可用的断点
        """
        job_id = self.job_index.get_job_id(file_path)
        if not job_id:
            return None

        job_dir = self.jobs_root / job_id
        if not job_dir.exists():
            self.job_index.remove_mapping(file_path)
            return None

        checkpoint = self._load_checkpoint(job_dir)
        if not checkpoint:
            return None

        total_segments = checkpoint.get("total_segments", 0)
        processed_indices = checkpoint.get("processed_indices", [])
        processed_count = len(processed_indices)
        progress = (processed_count / total_segments) * 100 if total_segments > 0 else 0

        return {
            "job_id": job_id,
            "progress": round(progress, 2),
            "processed_segments": processed_count,
            "total_segments": total_segments,
            "phase": checkpoint.get("phase", "unknown"),
            "can_resume": True,
        }

    def start_job(self, job_id: str) -> None:
        """
        启动转录任务（兼容入口，不创建线程）
        """
        job = self.get_job(job_id)
        if not job:
            self.logger.warning(f"任务未找到: {job_id}")
            return

        if job.status not in ("uploaded", "failed", "paused", "created"):
            self.logger.warning(f"任务无法启动: {job_id}, 状态: {job.status}")
            return

        job.canceled = False
        job.paused = False
        job.error = None
        self.save_job_meta(job)
        self.logger.warning(f"start_job已废弃，请使用队列服务: {job_id}")

    def pause_job(self, job_id: str) -> bool:
        """
        暂停转录任务（保存断点）
        """
        job = self.get_job(job_id)
        if not job:
            return False

        job.paused = True
        job.status = "paused"
        job.message = "暂停中..."
        self.save_job_meta(job)
        self.logger.info(f"⏸️ 任务暂停请求: {job_id}")
        return True

    def cancel_job(self, job_id: str, delete_data: bool = False) -> Tuple[bool, Optional[str]]:
        """
        取消转录任务
        """
        job = self.get_job(job_id)
        if not job:
            return False, "任务未找到"

        job.canceled = True
        job.message = "取消中..."
        self.save_job_meta(job)
        self.logger.info(f"🛑 任务取消请求: {job_id}, 删除数据: {delete_data}")

        if delete_data:
            try:
                job_dir = Path(job.dir)

                try:
                    from app.services.media_stream_tracker import get_active_streams
                    active_streams = get_active_streams(job_id)
                except Exception:
                    active_streams = 0

                if active_streams > 0:
                    msg = "当前有进程占用，请稍后再试"
                    self.logger.warning(
                        f"[删除任务] 文件被占用，放弃删除: {job_id}, active_streams={active_streams}"
                    )
                    return False, msg

                try:
                    from app.services.media_prep_service import get_media_prep_service
                    media_prep = get_media_prep_service()
                    killed = media_prep.cancel_tasks(job_id)
                    if killed > 0:
                        self.logger.info(f"[删除任务] 已终止 {killed} 个 MediaPrep 子进程: {job_id}")
                        import time
                        time.sleep(0.2)
                except Exception as exc:
                    self.logger.warning(f"[删除任务] 取消 MediaPrep 任务失败: {exc}")

                with self.lock:
                    if job_id in self.jobs:
                        del self.jobs[job_id]
                        self.logger.info(f"已从内存移除任务: {job_id}")

                if job.input_path:
                    self.job_index.remove_mapping(job.input_path)

                if job_dir.exists():
                    success = self._force_remove_directory(
                        job_dir, job_id, max_retries=1, fast_fail=True
                    )
                    if not success:
                        return False, "当前有进程占用，请稍后再试"
                    self.logger.info(f"已删除任务数据: {job_id}")
            except Exception as exc:
                self.logger.error(f"删除任务数据失败: {exc}")
                return False, str(exc)

        return True, None

    # ====== checkpoint ======

    # V3.2.0+dev.20260120.04: 检查点写入迁移到生命周期服务
    def _save_checkpoint(self, job_dir: Path, data: dict, job: JobState) -> None:
        """
        保存检查点数据（原子写入）
        """
        job_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = job_dir / "checkpoint.json"
        temp_path = checkpoint_path.with_suffix(".tmp")

        try:
            if "original_settings" not in data and job and job.settings:
                data["original_settings"] = job.settings.to_dict()
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, checkpoint_path)

            # 同步保存任务元信息（用于重启后恢复任务状态）
            self.save_job_meta(job)
        except Exception as exc:
            self.logger.error(f"保存检查点失败: {exc}", exc_info=True)
            raise

    def _load_checkpoint(self, job_dir: Path) -> Optional[dict]:
        """
        读取检查点数据
        """
        checkpoint_path = job_dir / "checkpoint.json"
        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            self.logger.warning(
                f"检查点文件损坏，将重新开始任务: {checkpoint_path} - {exc}"
            )
            return None

    def _flush_checkpoint_after_split(
        self,
        job: JobState,
        job_dir: Path,
        processing_mode: Any,
        segments: List[dict],
        demucs_state: Optional[dict] = None
    ) -> None:
        """
        分段完成后强制刷新 checkpoint（确保断点续传一致性）
        """
        checkpoint_data: Dict[str, Any] = {
            "job_id": job.job_id,
            "phase": "split_complete",
            "segments": segments,
            "processing_mode": processing_mode.value,
        }
        if demucs_state:
            checkpoint_data["demucs"] = demucs_state

        self._save_checkpoint(job_dir, checkpoint_data, job)

        saved_checkpoint = self._load_checkpoint(job_dir)
        if saved_checkpoint is None:
            raise RuntimeError("checkpoint write verification failed: file not readable")
        if saved_checkpoint.get("phase") != "split_complete":
            raise RuntimeError("checkpoint write verification failed: phase mismatch")
        if len(saved_checkpoint.get("segments", [])) != len(segments):
            raise RuntimeError("checkpoint write verification failed: segments count mismatch")

        self.logger.info(
            f"checkpoint flushed and verified after split (mode: {processing_mode.value}, "
            f"segments: {len(segments)})"
        )

    # ====== internal helpers ======

    def _normalize_orphaned_job(self, job: JobState) -> bool:
        """
        系统重启后纠偏：将运行态任务标记为暂停，避免前端显示“转录中”。
        """
        if job.status not in self._ACTIVE_STATUSES:
            return False

        job.status = "paused"
        job.paused = True
        job.message = "系统重启，任务已暂停"
        return True

    def _load_job_from_disk_legacy(self, job_id: str, job_dir: Path) -> Optional[JobState]:
        """
        旧版兼容：从目录推断任务信息。
        """
        filename = "未知文件"
        input_path = None

        for ext in [".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".mp3", ".wav", ".m4a"]:
            matches = list(job_dir.glob(f"*{ext}"))
            if matches:
                filename = matches[0].name
                input_path = str(matches[0])
                break

        srt_files = list(job_dir.glob("*.srt"))
        is_finished = len(srt_files) > 0

        job = JobState(
            job_id=job_id,
            filename=filename,
            dir=str(job_dir),
            input_path=input_path,
            status="finished" if is_finished else "paused",
            phase="editing" if is_finished else "transcribing",
            progress=100 if is_finished else 0,
            message="已完成" if is_finished else "系统重启，任务已暂停",
            srt_path=str(srt_files[0]) if srt_files else None,
            paused=not is_finished,
        )

        checkpoint_path = job_dir / "checkpoint.json"
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)
                total_segments = checkpoint_data.get("total_segments", 0)
                processed_indices = checkpoint_data.get("processed_indices", [])
                if total_segments > 0:
                    job.progress = min((len(processed_indices) / total_segments) * 100, 100)
                job.phase = checkpoint_data.get("phase", job.phase)
                job.language = checkpoint_data.get("language")
                if "unaligned_results" in checkpoint_data:
                    job.segments = checkpoint_data["unaligned_results"]
            except Exception as exc:
                self.logger.warning(f"读取checkpoint失败 {checkpoint_path}: {exc}")

        return job

    def _schedule_waveform_audio_extract(self, job_id: str, dest_path: Path) -> None:
        """
        异步提取音频供波形图使用。
        """
        audio_path = Path(dest_path).parent / "audio.wav"
        if audio_path.exists():
            self.logger.debug(f"[{job_id}] 音频文件已存在，跳过提取")
            return

        self.logger.info(f"[{job_id}] 启动后台音频提取...")

        def extract_audio_for_waveform() -> None:
            """后台提取音频供波形图使用"""
            try:
                import warnings
                import librosa
                import soundfile as sf
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="PySoundFile failed")
                    warnings.filterwarnings("ignore", message="audioread")
                    audio_array, sr = librosa.load(str(dest_path), sr=16000, mono=True)
                sf.write(str(audio_path), audio_array, sr)
                self.logger.info(f"[{job_id}] 音频提取完成: {audio_path}")
            except Exception as exc:
                self.logger.error(f"[{job_id}] 音频提取失败: {exc}")

        threading.Thread(
            target=extract_audio_for_waveform,
            daemon=True,
            name=f"AudioExtract-{job_id[:8]}"
        ).start()

    def _force_remove_directory(
        self,
        directory: Path,
        job_id: str,
        max_retries: int = 3,
        fast_fail: bool = False
    ) -> bool:
        """
        强制删除目录（处理 Windows 文件占用问题）
        """
        import time
        import stat

        gc.collect()
        time.sleep(0.1)

        attempts = 1 if fast_fail else max_retries
        for attempt in range(attempts):
            try:
                shutil.rmtree(directory)
                self.logger.info(f"[强制删除] 成功删除目录: {job_id}, 尝试次数: {attempt + 1}")
                return True
            except PermissionError as exc:
                if fast_fail or attempt >= attempts - 1:
                    self.logger.warning(f"[强制删除] 删除失败 (快速返回): {exc}")
                    return False
                self.logger.warning(
                    f"[强制删除] 删除失败 (尝试 {attempt + 1}/{max_retries}): {exc}, "
                    f"等待 {0.5 * (attempt + 1)}s 后重试"
                )
                time.sleep(0.5 * (attempt + 1))

        failed_files = []
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                file_path = Path(root) / name
                try:
                    os.chmod(file_path, stat.S_IWRITE)
                    file_path.unlink()
                except Exception as exc:
                    self.logger.warning(f"[强制删除] 无法删除文件: {file_path.name}, {exc}")
                    failed_files.append(str(file_path))

            for name in dirs:
                dir_path = Path(root) / name
                try:
                    dir_path.rmdir()
                except Exception as exc:
                    self.logger.debug(f"[强制删除] 无法删除目录: {dir_path.name}, {exc}")

        try:
            directory.rmdir()
            self.logger.info(f"[强制删除] 逐个删除完成: {job_id}")
            return True
        except Exception as exc:
            if failed_files:
                self.logger.error(
                    f"[强制删除] 部分文件无法删除: {job_id}, "
                    f"失败文件数: {len(failed_files)}, 错误: {exc}"
                )
            else:
                self.logger.warning(f"[强制删除] 根目录删除失败: {job_id}, {exc}")
            return False


_job_lifecycle_service: Optional[JobLifecycleService] = None


def get_job_lifecycle_service(
    jobs_root: Path,
    logger: Optional[logging.Logger] = None
) -> JobLifecycleService:
    """
    获取 JobLifecycleService 单例
    """
    global _job_lifecycle_service
    if _job_lifecycle_service is None:
        _job_lifecycle_service = JobLifecycleService(jobs_root=jobs_root, logger=logger)
    return _job_lifecycle_service
