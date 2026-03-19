# -*- coding: utf-8 -*-
"""
打包流水线收敛守卫。

目标：
1. Electron 仅保留单一 Shell 构建入口（build:shell）。
2. 打包 manifest 使用 extras 语义，禁止继续使用 syncArgs（uv 历史语义残留）。
3. pyproject 不再保留 PyInstaller 打包口径，避免与当前主链冲突。
4. build-profile.ps1 仅允许 extras 语义，不再保留 syncArgs 兼容分支。
5. electron_packaging 工作流必须执行本守卫脚本，确保发布链受同一门禁约束。
6. 打包依赖必须启用共享缓存：Lite/Full 共用基线缓存，Full 仅增量安装 extras。
7. 必须启用本地复用与失效机制：嵌入式 Python 包缓存 + 依赖指纹驱动的运行时缓存。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _fail(message: str) -> int:
    print(f"[失败] {message}")
    return 1


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]

    electron_package = _load_json(project_root / "electron" / "package.json")
    scripts = electron_package.get("scripts", {})
    if "build:shell" not in scripts:
        return _fail("electron/package.json 缺少 scripts.build:shell")
    if "pack:shell" in scripts:
        return _fail("electron/package.json 禁止保留 scripts.pack:shell（旁路入口）")
    if "build:shell:script" in scripts:
        return _fail("electron/package.json 禁止保留 scripts.build:shell:script（旁路入口）")

    build_shell_script = project_root / "electron" / "scripts" / "build-shell.mjs"
    if build_shell_script.exists():
        return _fail("electron/scripts/build-shell.mjs 已是历史残留，应删除")

    manifest_paths = (
        project_root / "scripts" / "packaging" / "profile-manifests" / "lite-offline.json",
        project_root / "scripts" / "packaging" / "profile-manifests" / "full-offline.json",
        project_root / "scripts" / "packaging" / "profile-manifests" / "full-hybrid.json",
    )
    for manifest_path in manifest_paths:
        payload = _load_json(manifest_path)
        python_cfg = payload.get("python", {})
        if "syncArgs" in python_cfg:
            return _fail(f"{manifest_path.as_posix()} 禁止使用 python.syncArgs，请迁移为 python.extras")
        extras = python_cfg.get("extras")
        if extras is None:
            return _fail(f"{manifest_path.as_posix()} 缺少 python.extras 字段")
        if not isinstance(extras, list):
            return _fail(f"{manifest_path.as_posix()} 的 python.extras 必须是数组")

    pyproject_text = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    if "pyinstaller" in pyproject_text.lower():
        return _fail("pyproject.toml 仍包含 pyinstaller 残留口径")

    build_profile_text = (project_root / "scripts" / "packaging" / "build-profile.ps1").read_text(encoding="utf-8")
    if "Get-ExtrasFromSyncArgs" in build_profile_text or ".syncArgs" in build_profile_text:
        return _fail("build-profile.ps1 仍保留 syncArgs 兼容逻辑，应收敛为 python.extras 唯一路径")
    if "PIP_CACHE_DIR" not in build_profile_text:
        return _fail("build-profile.ps1 缺少 PIP_CACHE_DIR 配置，未启用共享依赖缓存")
    if "--no-cache-dir" in build_profile_text:
        return _fail("build-profile.ps1 仍使用 --no-cache-dir，会强制禁用依赖缓存")
    if "安装 Lite/Full 共用基线依赖: ." not in build_profile_text:
        return _fail("build-profile.ps1 缺少基线依赖安装步骤（Lite/Full 共用）")
    if "安装 Full 增量依赖:" not in build_profile_text:
        return _fail("build-profile.ps1 缺少 Full 增量依赖安装步骤")
    if "function Get-DependencyFingerprint" not in build_profile_text:
        return _fail("build-profile.ps1 缺少依赖指纹函数（无法保证依赖变更失效刷新）")
    if "dist\\cache\\python-embed" not in build_profile_text:
        return _fail("build-profile.ps1 缺少嵌入式 Python 本地缓存目录（dist/cache/python-embed）")
    if "dist\\cache\\python-runtime" not in build_profile_text:
        return _fail("build-profile.ps1 缺少运行时本地缓存目录（dist/cache/python-runtime）")
    if "[cache] 命中 Python 运行时缓存" not in build_profile_text:
        return _fail("build-profile.ps1 缺少运行时缓存命中逻辑")
    if "pyproject.toml" not in build_profile_text:
        return _fail("build-profile.ps1 未将 pyproject.toml 纳入依赖指纹，依赖变更可能不触发刷新")
    if "uv.lock" not in build_profile_text:
        return _fail("build-profile.ps1 未将 uv.lock 纳入依赖指纹，锁文件变更可能不触发刷新")
    for required_key in (
        "embedded_version",
        "extras",
        "dependency_inputs",
        "pyproject_sha256",
        "uv_lock_sha256",
        "pip_index_url",
        "pip_extra_index_url",
    ):
        if required_key not in build_profile_text:
            return _fail(f"build-profile.ps1 依赖指纹缺少关键字段: {required_key}")

    electron_packaging_text = (project_root / ".github" / "workflows" / "electron_packaging.yml").read_text(
        encoding="utf-8"
    )
    if "check_packaging_pipeline_guards.py" not in electron_packaging_text:
        return _fail("electron_packaging.yml 缺少打包守卫步骤（check_packaging_pipeline_guards.py）")

    print("打包流水线收敛守卫通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
