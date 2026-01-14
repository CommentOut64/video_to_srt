#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动化构建前端并生成可被 Bootloader 识别的更新包。

运行方式示例:
    python scripts/build_update_package.py --version 3.1.2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "build"


class BuildError(RuntimeError):
    """构建 / 打包失败时抛出的异常"""


def run_command(cmd: Iterable[str], cwd: Path | None = None) -> None:
    """运行外部命令并在失败时给出更友好的提示"""
    cmd = list(cmd)
    # Windows 下通过 shutil.which 获取真实可执行文件（例如 npm.cmd）
    if os.name == "nt":
        resolved = shutil.which(cmd[0])
        if resolved:
            cmd[0] = resolved
    print(f"[CMD] {' '.join(cmd)} (cwd={cwd or PROJECT_ROOT})")
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except FileNotFoundError as exc:
        raise BuildError(f"命令不可用: {cmd[0]}，请确认已安装并加入 PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise BuildError(f"命令执行失败: {' '.join(cmd)} (退出码 {exc.returncode})") from exc


def build_frontend(skip_install: bool) -> None:
    """执行 npm install + npm run build，产出最新 dist 文件"""
    if not FRONTEND_DIR.exists():
        raise BuildError(f"前端目录不存在: {FRONTEND_DIR}")

    if not skip_install:
        print("[Step] 安装前端依赖 (npm install)")
        run_command(["npm", "install"], cwd=FRONTEND_DIR)
    else:
        print("[Step] 跳过 npm install，由参数指定")

    print("[Step] 构建前端 (npm run build)")
    run_command(["npm", "run", "build"], cwd=FRONTEND_DIR)

    dist_dir = FRONTEND_DIR / "dist"
    if not dist_dir.exists():
        raise BuildError("前端构建完成但未找到 dist 目录，请检查 Vite 配置")


def normalize(path: str) -> str:
    """统一路径分隔符"""
    return path.replace("\\", "/")


def load_update_excludes() -> Tuple[Set[str], Set[str]]:
    """加载 update_config.json 的排除配置，若不存在则使用 Bootloader 默认值"""
    default_dirs = {
        "jobs",
        "models",
        "temp",
        "output",
        "input",
        "logs",
        "tools",
        ".venv",
        "node_modules",
        ".git",
        "backend/models",
        "backend/app/assets",
    }
    default_files = {".env", ".env_installed", ".req_hash"}

    config_path = PROJECT_ROOT / "backend" / "update_config.json"
    if not config_path.exists():
        print(f"[Info] 未找到 {config_path}，使用默认排除列表")
        return default_dirs, default_files

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        dirs = set(data.get("exclude_dirs", default_dirs))
        files = set(data.get("exclude_files", default_files))
        return dirs, files
    except Exception as exc:
        print(f"[Warn] 读取 update_config 失败，使用默认排除列表: {exc}")
        return default_dirs, default_files


def is_test_prefixed(name: str) -> bool:
    return name.lower().startswith("test")


def match_dir_rule(rel_norm: str, rule: str) -> bool:
    rule_norm = normalize(rule)
    if "/" in rule_norm:
        return rel_norm == rule_norm or rel_norm.startswith(f"{rule_norm}/")
    segment = rule_norm
    return (
        rel_norm == segment
        or rel_norm.endswith(f"/{segment}")
        or f"/{segment}/" in rel_norm
    )


def match_file_rule(rel_norm: str, rule: str) -> bool:
    rule_norm = normalize(rule)
    if "/" in rule_norm:
        return rel_norm == rule_norm
    filename = rel_norm.split("/")[-1]
    return filename == rule_norm


@dataclass
class IncludeTarget:
    path: Path
    respect_excludes: bool = True


def copy_selected_paths(
    project_root: Path,
    stage_dir: Path,
    include_targets: List[IncludeTarget],
    exclude_dirs: Set[str],
    exclude_files: Set[str],
) -> Tuple[int, int]:
    """
    仅复制指定路径，仍应用更新排除规则

    Returns:
        (copied_files, skipped_items)
    """
    copied_files = 0
    skipped_items = 0

    def is_allowed(rel_norm: str, allow_overrides: Set[str]) -> bool:
        return any(
            rel_norm == allow or rel_norm.startswith(f"{allow}/")
            for allow in allow_overrides
        )

    def dir_excluded(rel_norm: str, allow_overrides: Set[str]) -> bool:
        if is_test_prefixed(rel_norm.split("/")[-1]) and not is_allowed(rel_norm, allow_overrides):
            return True
        for rule in exclude_dirs:
            if match_dir_rule(rel_norm, rule):
                if is_allowed(rel_norm, allow_overrides):
                    continue
                return True
        return False

    def file_excluded(rel_norm: str, allow_overrides: Set[str]) -> bool:
        if is_test_prefixed(rel_norm.split("/")[-1]) and not is_allowed(rel_norm, allow_overrides):
            return True
        for rule in exclude_files:
            if match_file_rule(rel_norm, rule):
                if is_allowed(rel_norm, allow_overrides):
                    continue
                return True
        return False

    def _copy_entry(src: Path, rel_path: str, allow_overrides: Set[str]):
        nonlocal copied_files, skipped_items
        rel_norm = normalize(rel_path) if rel_path else ""
        if src.is_dir():
            if rel_norm and dir_excluded(rel_norm, allow_overrides):
                skipped_items += 1
                return
            if rel_norm:
                (stage_dir / rel_path).mkdir(parents=True, exist_ok=True)

            for child in src.iterdir():
                child_rel = f"{rel_path}/{child.name}" if rel_path else child.name
                _copy_entry(child, child_rel, allow_overrides)
        else:
            if rel_norm and file_excluded(rel_norm, allow_overrides):
                skipped_items += 1
                return
            dest = stage_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied_files += 1

    for target in include_targets:
        src = project_root / target.path
        if not src.exists():
            raise BuildError(f"指定的 include 路径不存在: {target.path}")

        rel = normalize(target.path.as_posix())
        allow = {rel} if not target.respect_excludes else set()
        _copy_entry(src, rel, allow)

    return copied_files, skipped_items


def make_zip(source_dir: Path, zip_path: Path) -> None:
    """将暂存目录打包为 zip，保持顶层文件夹结构"""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for item in source_dir.rglob("*"):
            arcname = item.relative_to(source_dir.parent)
            zf.write(item, arcname)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建前端并生成 Bootloader 可用的更新包"
    )
    parser.add_argument("--version", required=True, help="目标版本号，例如 3.1.2")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="输出目录（默认: build）",
    )
    parser.add_argument(
        "--skip-npm-install",
        action="store_true",
        help="跳过 npm install（依赖已满足时可使用）",
    )
    parser.add_argument(
        "--skip-frontend-build",
        action="store_true",
        help="跳过 npm run build（dist 已就绪时可使用）",
    )
    parser.add_argument(
        "--keep-stage",
        action="store_true",
        help="保留临时构建目录以便调试",
    )
    parser.add_argument(
        "--include-path",
        action="append",
        dest="include_paths",
        help="需要打包的相对路径，可多次指定（默认：backend, frontend/dist）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version = args.version.lstrip("v")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"[Config] 目标版本: {version}")
    print(f"[Config] 输出目录: {output_dir}")
    print("=" * 60)

    if not args.skip_frontend_build:
        build_frontend(skip_install=args.skip_npm_install)
    else:
        print("[Step] 跳过前端构建，由参数指定")

    default_paths = [
        Path("backend"),
        Path("frontend/dist"),
        Path("requirements.txt"),
        Path("requirements.extra.txt"),
    ]
    anchor_exe = Path("AnchorFlux.exe")
    if anchor_exe.exists():
        default_paths.append(anchor_exe)
    helper_exe = Path("update_helper.exe")
    if helper_exe.exists():
        # 默认将更新辅助程序一并打包，确保离线安装也能自我替换
        default_paths.append(helper_exe)

    include_targets: List[IncludeTarget] = [
        IncludeTarget(p, True) for p in default_paths
    ]
    extra_includes = [Path(p) for p in (args.include_paths or [])]
    include_targets.extend(IncludeTarget(p, False) for p in extra_includes)

    print("[Config] 包含路径:")
    for target in include_targets:
        mode = "自动排除" if target.respect_excludes else "强制包含"
        print(f"  - {target.path} ({mode})")

    stage_parent = output_dir / "update_stage"
    stage_dir = stage_parent / f"AnchorFlux_update_v{version}"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    exclude_dirs, exclude_files = load_update_excludes()
    extra_excludes = {
        "build",
        "__pycache__",
        ".pytest_cache",
        ".VSCodeCounter",
        ".claude",
        ".vscode",
        "patch_tools",
        "backend/models/pretrained/brouhaha/.cache",  # 缓存无需打包
    }
    exclude_dirs = {normalize(p) for p in (set(exclude_dirs) | extra_excludes)}
    exclude_files = {normalize(p) for p in exclude_files}

    try:
        stage_rel = stage_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        stage_rel = None
    if stage_rel:
        exclude_dirs.add(normalize(stage_rel.as_posix().split("/")[0]))

    print("[Step] 复制指定路径到临时目录...")
    copied_files, skipped_items = copy_selected_paths(
        PROJECT_ROOT, stage_dir, include_targets, exclude_dirs, exclude_files
    )
    print(f"[Info] 已复制文件 {copied_files} 个，跳过 {skipped_items} 个排除项")

    zip_path = output_dir / f"AnchorFlux_update_v{version}.zip"
    if zip_path.exists():
        zip_path.unlink()

    print(f"[Step] 打包更新包 -> {zip_path}")
    make_zip(stage_dir, zip_path)

    if not args.keep_stage:
        shutil.rmtree(stage_parent, ignore_errors=True)
    else:
        print(f"[Info] 保留临时目录: {stage_dir}")

    print("=" * 60)
    print("[Done] 更新包生成完成，可上传供 Bootloader 下载")
    print(f"[Done] 包路径: {zip_path}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except BuildError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[WARN] 用户中断")
        sys.exit(1)
