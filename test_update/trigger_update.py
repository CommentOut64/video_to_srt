#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V3.1.1+dev.20260105.01: 触发本地更新测试

此脚本模拟后端 API 的行为，直接写入 update_signal.json 文件。
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def trigger_local_update():
    """触发本地更新"""

    # 使用本地HTTP服务器的URL
    download_url = "http://localhost:8888/mock_update.zip"
    test_version = "3.1.1-test"

    signal_file = PROJECT_ROOT / "update_signal.json"

    signal_data = {
        "version": test_version,
        "download_url": download_url,
        "changelog": "Test update for local testing",
        "triggered_at": datetime.now().isoformat(),
        "current_version": "3.1.1",
        "test_mode": True  # 标记为测试模式
    }

    with open(signal_file, 'w', encoding='utf-8') as f:
        json.dump(signal_data, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Update Signal Written")
    print("=" * 60)
    print()
    print(f"Signal file: {signal_file}")
    print(f"Target version: {test_version}")
    print(f"Download URL: {download_url}")
    print()
    print("IMPORTANT:")
    print("-" * 60)
    print("1. Make sure local server is running:")
    print("   python test_update/local_server.py")
    print()
    print("2. The bootloader will detect this signal when backend exits.")
    print()
    print("3. To test manually without running backend:")
    print("   python bootloader.py --test-update")
    print()


def trigger_direct_update(use_gui: bool = False):
    """
    直接调用bootloader的更新逻辑进行测试

    V3.1.1+dev.20260105.01: 添加 headless 模式支持
    """

    # 首先确保模拟更新包存在
    mock_zip = PROJECT_ROOT / "test_update" / "mock_update.zip"
    if not mock_zip.exists():
        print("[ERROR] Mock update package not found!")
        print("Run this first: python test_update/create_mock_update.py")
        return False

    # 写入信号文件
    trigger_local_update()

    print()
    print("=" * 60)
    print("Running Direct Update Test")
    print(f"Mode: {'GUI' if use_gui else 'Headless'}")
    print("=" * 60)
    print()

    # 导入并调用bootloader的更新逻辑
    sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from bootloader import Bootloader, UpdateManager, BootloaderConfig, BootloaderLogger

        # 创建配置
        config = BootloaderConfig(project_root=PROJECT_ROOT)
        logger = BootloaderLogger(dev_mode=True)

        # 创建更新管理器
        update_manager = UpdateManager(config, logger)

        # 检查更新信号
        signal_data = update_manager.check_update_signal()

        if signal_data:
            print(f"[OK] Signal detected: version={signal_data.get('version')}")
            print()

            # 执行更新（使用 headless 模式，除非明确指定使用 GUI）
            print("Executing update...")
            success = update_manager.execute_update(signal_data, headless=not use_gui)

            if success:
                print()
                print("[SUCCESS] Update completed!")
                update_manager.clear_update_signal()

                # 检查标记文件
                marker_file = PROJECT_ROOT / ".update_test_marker"
                if marker_file.exists():
                    print()
                    print("[VERIFIED] Marker file found - update mechanism works correctly!")
                    with open(marker_file, 'r', encoding='utf-8') as f:
                        print(f"Content: {f.read()}")
                else:
                    print()
                    print("[WARNING] Marker file not found - check update logic")
            else:
                print()
                print("[FAILED] Update failed!")
        else:
            print("[ERROR] No update signal found")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trigger local update test")
    parser.add_argument("--direct", action="store_true",
                       help="Run direct update test (calls bootloader update logic)")
    parser.add_argument("--signal-only", action="store_true",
                       help="Only write signal file, don't execute update")
    parser.add_argument("--gui", action="store_true",
                       help="Use GUI mode instead of headless (default: headless)")

    args = parser.parse_args()

    if args.signal_only:
        trigger_local_update()
    elif args.direct:
        trigger_direct_update(use_gui=args.gui)
    else:
        # 默认：使用 headless 模式直接测试
        trigger_direct_update(use_gui=args.gui)
