"""
JobManager - 任务管理器

Phase 4 实现 - 2025-12-11

核心职责：
1. 任务创建、查询、更新、删除
2. 任务状态管理（JobState）
3. 任务元信息持久化（job_meta.json）
4. 任务生命周期管理（启动、暂停、取消）
5. 任务索引和查询

从 transcription_service.py 拆分出来，专注于任务管理功能。
"""

import json
import os
import shutil
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, List
from threading import Lock
from dataclasses import asdict

from app.models.job_models import JobState, JobSettings


class JobManager:
    """
    任务管理器

    负责任务的创建、查询、更新、删除和生命周期管理。
    """

    def __init__(
        self,
        jobs_root: Path,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化任务管理器

        Args:
            jobs_root: 任务根目录
            logger: 日志记录器
        """
        self.jobs_root = Path(jobs_root)
        self.jobs_root.mkdir(parents=True, exist_ok=True)

        self.logger = logger or logging.getLogger(__name__)

        # 任务缓存（内存中的任务状态）
        self.jobs: Dict[str, JobState] = {}
        self.lock = Lock()

        # 加载所有已存在的任务
        self._load_all_jobs_from_disk()

    def _load_all_jobs_from_disk(self):
        """从磁盘加载所有任务的元信息"""
        if not self.jobs_root.exists():
            return

        for job_dir in self.jobs_root.iterdir():
            if not job_dir.is_dir():
                continue

            job_id = job_dir.name
            job = self.load_job_meta(job_id)

            if job:
                with self.lock:
                    self.jobs[job_id] = job
                self.logger.debug(f"已加载任务: {job_id}")

    def create_job(
        self,
        filename: str,
        src_path: str,
        settings: JobSettings,
        job_id: Optional[str] = None
    ) -> JobState:
        """
        创建转录任务

        Args:
            filename: 文件名
            src_path: 源文件路径
            settings: 任务设置
            job_id: 任务ID（可选，不提供则自动生成）

        Returns:
            JobState: 创建的任务状态对象
        """
        job_id = job_id or uuid.uuid4().hex
        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        dest_path = job_dir / filename

        # V3.1.1+dev.20260106.01: 使用硬链接替代复制，节省磁盘空间
        # 硬链接让 input/video.mp4 和 jobs/{job_id}/video.mp4 指向同一数据块
        # 支持多个任务指向同一个视频文件，删除任务时不影响原始文件
        if os.path.abspath(src_path) != os.path.abspath(dest_path):
            try:
                # 优先使用硬链接
                os.link(src_path, dest_path)
                self.logger.debug(f"硬链接创建成功: {src_path} -> {dest_path}")
            except (OSError, NotImplementedError) as e:
                # 硬链接失败时降级到复制（跨文件系统、网络挂载等场景）
                self.logger.warning(f"硬链接创建失败，回退到复制: {e}")
                try:
                    shutil.copyfile(src_path, dest_path)
                    self.logger.debug(f"文件已复制: {src_path} -> {dest_path}")
                except Exception as copy_err:
                    self.logger.warning(f"文件复制失败: {copy_err}")

        # 创建任务状态对象
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

        # 持久化任务元信息（重启后可恢复）
        self.save_job_meta(job)

        self.logger.info(f"任务已创建: {job_id} - {filename}")
        return job

    def get_job(self, job_id: str) -> Optional[JobState]:
        """
        获取任务状态

        Args:
            job_id: 任务ID

        Returns:
            Optional[JobState]: 任务状态对象，不存在则返回 None
        """
        with self.lock:
            return self.jobs.get(job_id)

    def update_job(self, job: JobState) -> bool:
        """
        更新任务状态

        Args:
            job: 任务状态对象

        Returns:
            bool: 更新是否成功
        """
        with self.lock:
            if job.job_id not in self.jobs:
                self.logger.warning(f"任务不存在: {job.job_id}")
                return False

            self.jobs[job.job_id] = job

        # 持久化更新
        return self.save_job_meta(job)

    def delete_job(self, job_id: str, delete_files: bool = True) -> bool:
        """
        删除任务

        Args:
            job_id: 任务ID
            delete_files: 是否删除任务文件

        Returns:
            bool: 删除是否成功
        """
        with self.lock:
            if job_id not in self.jobs:
                self.logger.warning(f"任务不存在: {job_id}")
                return False

            del self.jobs[job_id]

        # 删除任务文件
        if delete_files:
            job_dir = self.jobs_root / job_id
            if job_dir.exists():
                try:
                    shutil.rmtree(job_dir)
                    self.logger.info(f"任务文件已删除: {job_dir}")
                except Exception as e:
                    self.logger.error(f"删除任务文件失败: {e}", exc_info=True)
                    return False

        self.logger.info(f"任务已删除: {job_id}")
        return True

    def list_jobs(
        self,
        status: Optional[str] = None,
        phase: Optional[str] = None
    ) -> List[JobState]:
        """
        列出所有任务

        Args:
            status: 过滤状态（可选）
            phase: 过滤阶段（可选）

        Returns:
            List[JobState]: 任务列表
        """
        with self.lock:
            jobs = list(self.jobs.values())

        # 过滤
        if status:
            jobs = [j for j in jobs if j.status == status]

        if phase:
            jobs = [j for j in jobs if j.phase == phase]

        # 按 job_id 排序（最新的在前）
        # 注意：如果 JobState 有 created_at 属性，可以改用 created_at 排序
        jobs.sort(key=lambda j: j.job_id, reverse=True)

        return jobs

    def save_job_meta(self, job: JobState) -> bool:
        """
        保存任务元信息到 job_meta.json（用于重启后恢复）

        Args:
            job: 任务状态对象

        Returns:
            bool: 保存是否成功
        """
        job_dir = Path(job.dir)
        meta_path = job_dir / "job_meta.json"

        try:
            # 转换为字典
            job_dict = asdict(job)

            # 写入文件
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(job_dict, f, ensure_ascii=False, indent=2)

            self.logger.debug(f"任务元信息已保存: {meta_path}")
            return True

        except Exception as e:
            self.logger.error(f"保存任务元信息失败: {e}", exc_info=True)
            return False

    def load_job_meta(self, job_id: str) -> Optional[JobState]:
        """
        从 job_meta.json 加载任务元信息

        Args:
            job_id: 任务ID

        Returns:
            Optional[JobState]: 任务状态对象，不存在或损坏则返回 None
        """
        job_dir = self.jobs_root / job_id
        meta_path = job_dir / "job_meta.json"

        if not meta_path.exists():
            self.logger.debug(f"任务元信息不存在: {meta_path}")
            return None

        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                job_dict = json.load(f)

            # 重建 JobState 对象
            # 注意：需要处理嵌套的 JobSettings 对象
            if 'settings' in job_dict and isinstance(job_dict['settings'], dict):
                job_dict['settings'] = JobSettings.from_dict(job_dict['settings'])

            job = JobState(**job_dict)

            self.logger.debug(f"任务元信息已加载: {meta_path}")
            return job

        except Exception as e:
            self.logger.warning(f"加载任务元信息失败: {meta_path} - {e}")
            return None

    def scan_incomplete_jobs(self) -> List[Dict]:
        """
        扫描所有未完成的任务（有 checkpoint.json 的任务）

        Returns:
            List[Dict]: 未完成任务的摘要信息列表
        """
        incomplete_jobs = []

        if not self.jobs_root.exists():
            return incomplete_jobs

        for job_dir in self.jobs_root.iterdir():
            if not job_dir.is_dir():
                continue

            checkpoint_path = job_dir / "checkpoint.json"
            if not checkpoint_path.exists():
                continue

            try:
                # 读取 checkpoint 获取任务信息
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)

                job_id = checkpoint_data.get('job_id') or job_dir.name
                total_chunks = checkpoint_data.get('total_chunks', 0)
                processed_chunks = checkpoint_data.get('processed_chunks', 0)

                # 读取 job_meta 获取更多信息
                job = self.load_job_meta(job_id)

                incomplete_jobs.append({
                    'job_id': job_id,
                    'filename': job.filename if job else None,
                    'phase': checkpoint_data.get('phase', 'unknown'),
                    'total_chunks': total_chunks,
                    'processed_chunks': processed_chunks,
                    'progress': (
                        processed_chunks / total_chunks * 100
                        if total_chunks > 0 else 0
                    ),
                    'language': checkpoint_data.get('language'),
                    'status': job.status if job else 'unknown'
                })

            except Exception as e:
                self.logger.warning(f"读取未完成任务失败 {job_dir}: {e}")

        return incomplete_jobs

    def get_job_dir(self, job_id: str) -> Path:
        """
        获取任务目录

        Args:
            job_id: 任务ID

        Returns:
            Path: 任务目录路径
        """
        return self.jobs_root / job_id

    def job_exists(self, job_id: str) -> bool:
        """
        检查任务是否存在

        Args:
            job_id: 任务ID

        Returns:
            bool: 任务是否存在
        """
        with self.lock:
            return job_id in self.jobs

    def get_job_count(self) -> int:
        """
        获取任务总数

        Returns:
            int: 任务总数
        """
        with self.lock:
            return len(self.jobs)

    def get_jobs_by_status(self, status: str) -> List[JobState]:
        """
        按状态获取任务列表

        Args:
            status: 任务状态

        Returns:
            List[JobState]: 任务列表
        """
        return self.list_jobs(status=status)

    def update_job_status(
        self,
        job_id: str,
        status: str,
        message: Optional[str] = None
    ) -> bool:
        """
        更新任务状态

        Args:
            job_id: 任务ID
            status: 新状态
            message: 状态消息（可选）

        Returns:
            bool: 更新是否成功
        """
        job = self.get_job(job_id)

        if not job:
            self.logger.warning(f"任务不存在: {job_id}")
            return False

        job.status = status
        if message:
            job.message = message

        return self.update_job(job)

    def update_job_progress(
        self,
        job_id: str,
        progress: float,
        message: Optional[str] = None
    ) -> bool:
        """
        更新任务进度

        Args:
            job_id: 任务ID
            progress: 进度（0-100）
            message: 进度消息（可选）

        Returns:
            bool: 更新是否成功
        """
        job = self.get_job(job_id)

        if not job:
            self.logger.warning(f"任务不存在: {job_id}")
            return False

        job.progress = progress
        if message:
            job.message = message

        return self.update_job(job)


# 便捷函数
def get_job_manager(
    jobs_root: Path,
    logger: Optional[logging.Logger] = None
) -> JobManager:
    """
    获取 JobManager 实例

    Args:
        jobs_root: 任务根目录
        logger: 日志记录器

    Returns:
        JobManager 实例
    """
    return JobManager(jobs_root=jobs_root, logger=logger)
