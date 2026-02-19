# -*- coding: utf-8 -*-
"""
阶段A/C/E/G（冻结与约束）质量守卫脚本。

目标：
1. 冻结新增七层日志口径：禁止新增 `layer="L0" ... layer="L7"` 字面量。
2. 冻结新增 logging 直连：禁止新增 `logging.getLogger(...)` 调用。
3. 冻结新增后处理旧入口导入：禁止新增 L4-L7 旧目录直连导入。
4. 冻结新增 textflow 分层子模块直连导入：调用方统一使用 `app.services.textflow` 包级 API。
5. 冻结新增 alignment 包级旧别名导入：禁止新增 `from app.services.alignment import AlignmentProcessor/FactBuilder`。
6. 冻结新增后处理旧层级术语字面量：禁止新增 `L3/L4/L5/L6/L7`（仅冻结新增，历史遗留走基线）。

说明：
- 脚本默认采用“基线对比”模式：只阻止新增，不阻止历史遗留。
- 若需更新基线，执行：
  python ci_tests/quality/check_textflow_layer_guards.py --update-baseline
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable


TARGET_DIRS: tuple[str, ...] = (
    "backend/app/services",
    "backend/app/pipelines",
    "backend/app/engines",
)

RULES: dict[str, re.Pattern[str]] = {
    "frozen_layer_literals.txt": re.compile(r'layer\s*=\s*["\']L[0-7]["\']'),
    "frozen_getlogger_calls.txt": re.compile(r"logging\.getLogger\s*\("),
    "frozen_legacy_postprocess_imports.txt": re.compile(
        r"from\s+app\.services\.(alignment\.alignment_processor|alignment\.fact_builder|"
        r"punctuation\.semantic_injection_processor|segmentation\.segmentation_processor|"
        r"streaming\.output_processor)\s+import"
    ),
    "frozen_textflow_direct_layer_imports.txt": re.compile(
        r"from\s+app\.services\.textflow\.(collection_layer|scoring_layer|decision_layer|output_layer)\s+import"
    ),
    "frozen_alignment_legacy_exports_imports.txt": re.compile(
        r"from\s+app\.services\.alignment\s+import\s+.*\b(AlignmentProcessor|FactBuilder|FactBuilderConfig)\b"
    ),
    "frozen_legacy_layer_terms.txt": re.compile(r"(?<![A-Za-z0-9_])L[3-7](?:\.[0-9]+)?(?![A-Za-z0-9_])"),
}


def _iter_python_files(project_root: Path, target_dirs: Iterable[str]) -> Iterable[Path]:
    for rel_dir in target_dirs:
        abs_dir = project_root / rel_dir
        if not abs_dir.exists():
            continue
        for file_path in abs_dir.rglob("*.py"):
            if file_path.is_file():
                yield file_path


def _collect_matches(project_root: Path, pattern: re.Pattern[str]) -> set[str]:
    records: set[str] = set()
    for file_path in _iter_python_files(project_root, TARGET_DIRS):
        rel_path = file_path.relative_to(project_root).as_posix()
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = file_path.read_text(encoding="utf-8-sig").splitlines()
        for line in lines:
            if pattern.search(line):
                normalized = line.strip()
                records.add(f"{rel_path}|{normalized}")
    return records


def _load_baseline(baseline_file: Path) -> set[str]:
    if not baseline_file.exists():
        raise FileNotFoundError(f"baseline missing: {baseline_file.as_posix()}")
    content = baseline_file.read_text(encoding="utf-8")
    rows = [row.strip() for row in content.splitlines()]
    return {row for row in rows if row and not row.startswith("#")}


def _write_baseline(baseline_file: Path, records: set[str]) -> None:
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(sorted(records)) + ("\n" if records else "")
    baseline_file.write_text(payload, encoding="utf-8")


def _print_new_violations(title: str, baseline_file: Path, new_items: set[str]) -> None:
    print(f"\n[失败] {title}")
    print(f"基线文件: {baseline_file.as_posix()}")
    print("新增违规项（仅展示前 50 条）：")
    for item in sorted(new_items)[:50]:
        print(f"  - {item}")
    if len(new_items) > 50:
        print(f"  ... 其余 {len(new_items) - 50} 条已省略")


def main() -> int:
    # V3.2.0+dev.20260219.01: 兼容 Windows CI (cp1252) 环境，强制 stdout/stderr 使用 UTF-8
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="文本后处理四层架构阶段A/C/E/G冻结守卫")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="项目根目录（默认自动推断）",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("ci_tests/quality/baselines"),
        help="基线目录（相对 project-root）",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="更新基线并退出（用于阶段性收敛）",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    baseline_dir = (project_root / args.baseline_dir).resolve()

    has_failure = False
    for baseline_name, pattern in RULES.items():
        current_records = _collect_matches(project_root, pattern)
        baseline_file = baseline_dir / baseline_name

        if args.update_baseline:
            _write_baseline(baseline_file, current_records)
            print(f"[已更新基线] {baseline_file.as_posix()} ({len(current_records)} 条)")
            continue

        try:
            baseline_records = _load_baseline(baseline_file)
        except FileNotFoundError:
            print(f"[失败] 基线缺失：{baseline_file.as_posix()}")
            print("请先执行：python ci_tests/quality/check_textflow_layer_guards.py --update-baseline")
            has_failure = True
            continue

        new_records = current_records - baseline_records
        if new_records:
            has_failure = True
            if baseline_name == "frozen_layer_literals.txt":
                title = "检测到新增七层日志口径（layer=\"L0..L7\"）"
            elif baseline_name == "frozen_getlogger_calls.txt":
                title = "检测到新增 logging.getLogger(...) 调用"
            elif baseline_name == "frozen_legacy_postprocess_imports.txt":
                title = "检测到新增后处理旧入口导入（应改用 textflow 包级 API）"
            elif baseline_name == "frozen_textflow_direct_layer_imports.txt":
                title = "检测到新增 textflow 分层子模块直连导入（应改用 app.services.textflow）"
            elif baseline_name == "frozen_legacy_layer_terms.txt":
                title = "检测到新增后处理旧层级术语字面量（L3-L7）"
            else:
                title = "检测到新增 alignment 包级旧别名导入（应改用 app.services.textflow）"
            _print_new_violations(title, baseline_file, new_records)

    if args.update_baseline:
        print("\n阶段A/C/E/G基线已更新完成。")
        return 0

    if has_failure:
        print("\n阶段A/C/E/G冻结守卫未通过。请清理新增违规，或在评审后显式更新基线。")
        return 1

    print("阶段A/C/E/G冻结守卫通过：未检测到新增违规。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
