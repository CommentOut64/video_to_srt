"""
Project 工作目录命名服务。

设计模式：
- Domain Service（领域服务）：集中承载目录命名规则、冲突重试与可读 slug 归一化。
"""

from __future__ import annotations

import re
import secrets
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Optional


class ProjectNamingService:
    """Project 工作目录命名服务。"""

    _MODE_MAP = {
        "transcribe": "tr",
        "import": "im",
        "legacy": "lg",
        "tr": "tr",
        "im": "im",
        "lg": "lg",
    }
    _RAND_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
    _RAND_LENGTH = 4
    _MAX_RETRY = 16
    _MAX_SLUG_LENGTH = 32
    _NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")
    _DASH_REPEAT_PATTERN = re.compile(r"-{2,}")

    def normalize_mode(self, mode: str) -> str:
        """标准化命名模式。"""
        normalized = str(mode or "").strip().lower()
        if normalized in self._MODE_MAP:
            return self._MODE_MAP[normalized]
        raise ValueError(f"不支持的命名模式: {mode}")

    def normalize_slug(self, text: str) -> str:
        """
        归一化 slug（仅保留 a-z0-9-）。

        规则：
        1. 小写；
        2. 非 [a-z0-9] 统一替换为 -；
        3. 压缩连续 -；
        4. 去除首尾 -；
        5. 截断到 32 字符；
        6. 空值回退 untitled。
        """
        normalized = str(text or "").strip().lower()
        normalized = self._NON_ALNUM_PATTERN.sub("-", normalized)
        normalized = self._DASH_REPEAT_PATTERN.sub("-", normalized).strip("-")
        if len(normalized) > self._MAX_SLUG_LENGTH:
            normalized = normalized[: self._MAX_SLUG_LENGTH].strip("-")
        return normalized or "untitled"

    def build_slug(self, *, title: str = "", source_filename: str = "") -> str:
        """按优先级生成 slug。"""
        normalized_title = self.normalize_slug(title)
        if normalized_title != "untitled":
            return normalized_title

        filename = str(source_filename or "").strip()
        if filename:
            filename_stem = Path(filename).stem
            normalized_filename = self.normalize_slug(filename_stem)
            if normalized_filename != "untitled":
                return normalized_filename

        return "untitled"

    def generate_workspace_dir_name(
        self,
        *,
        jobs_root: Path,
        mode: str,
        title: str = "",
        source_filename: str = "",
        now: Optional[datetime] = None,
    ) -> str:
        """
        生成工作目录名。

        格式：
        p-{YYYYMMDD}-{HHmmss}-{mode}-{slug}-{rand4}
        """
        mode_code = self.normalize_mode(mode)
        slug = self.build_slug(title=title, source_filename=source_filename)
        dt = now or datetime.now()
        date_part = dt.strftime("%Y%m%d")
        time_part = dt.strftime("%H%M%S")

        jobs_root.mkdir(parents=True, exist_ok=True)
        for _ in range(self._MAX_RETRY):
            rand4 = self._generate_rand4()
            candidate = f"p-{date_part}-{time_part}-{mode_code}-{slug}-{rand4}"
            if self._is_name_available(jobs_root=jobs_root, candidate_name=candidate):
                return candidate

        raise RuntimeError("生成 Project 工作目录名失败：命名冲突重试超过上限")

    def _generate_rand4(self) -> str:
        return "".join(secrets.choice(self._RAND_ALPHABET) for _ in range(self._RAND_LENGTH))

    @staticmethod
    def _is_name_available(*, jobs_root: Path, candidate_name: str) -> bool:
        """
        判断目录名是否可用（按 Windows 语义做大小写不敏感比较）。
        """
        candidate_lower = str(candidate_name).lower()
        for item in jobs_root.iterdir():
            if not item.is_dir():
                continue
            if item.name.lower() == candidate_lower:
                return False
        return True


_project_naming_service: Optional[ProjectNamingService] = None
_project_naming_service_lock = RLock()


def get_project_naming_service() -> ProjectNamingService:
    """获取 ProjectNamingService 单例。"""
    global _project_naming_service
    if _project_naming_service is not None:
        return _project_naming_service
    with _project_naming_service_lock:
        if _project_naming_service is None:
            _project_naming_service = ProjectNamingService()
    return _project_naming_service
