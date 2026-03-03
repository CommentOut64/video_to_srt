"""
Full/Lite 路由注册门禁脚本。

校验目标：
1. Lite 模式：不注册转录创建路由
2. Full 模式：必须注册转录创建路由
3. Full/Lite 模式：都必须注册同音 project 路由
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


TRANSCRIBE_CREATE_ROUTE = "/api/create-task"
HOMOPHONE_FIND_ROUTE = "/api/projects/{project_id}/homophone/find"


def _inject_backend_path() -> None:
    """把 backend 目录注入 sys.path，保证脚本可直接运行。"""
    repo_root = Path(__file__).resolve().parents[2]
    backend_path = repo_root / "backend"
    backend_str = str(backend_path)
    if backend_str not in sys.path:
        sys.path.insert(0, backend_str)


def _collect_route_paths() -> set[str]:
    """导入 app.main 并提取已注册路由路径集合。"""
    main_module = importlib.import_module("app.main")
    app = getattr(main_module, "app", None)
    if app is None:
        raise RuntimeError("app.main 未暴露 FastAPI app")
    return {route.path for route in app.routes if hasattr(route, "path")}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="校验 Full/Lite 路由注册边界")
    parser.add_argument(
        "--flavor",
        required=True,
        choices=("lite", "full"),
        help="待校验的运行 flavor",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    _inject_backend_path()
    os.environ["ANCHORFLUX_FLAVOR"] = args.flavor

    route_paths = _collect_route_paths()
    has_homophone = HOMOPHONE_FIND_ROUTE in route_paths
    has_transcribe = TRANSCRIBE_CREATE_ROUTE in route_paths

    print(f"[check_flavor_routes] flavor={args.flavor}")
    print(f"[check_flavor_routes] has_homophone={has_homophone}")
    print(f"[check_flavor_routes] has_transcribe={has_transcribe}")

    errors: list[str] = []
    if not has_homophone:
        errors.append(f"缺少同音路由: {HOMOPHONE_FIND_ROUTE}")

    if args.flavor == "lite" and has_transcribe:
        errors.append(f"Lite 模式不应注册路由: {TRANSCRIBE_CREATE_ROUTE}")
    if args.flavor == "full" and not has_transcribe:
        errors.append(f"Full 模式缺少路由: {TRANSCRIBE_CREATE_ROUTE}")

    if errors:
        for message in errors:
            print(f"[check_flavor_routes][ERROR] {message}")
        return 1

    print("[check_flavor_routes] 路由校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

