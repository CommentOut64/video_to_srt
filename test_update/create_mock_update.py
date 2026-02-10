#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V3.1.1+dev.20260105.01: 创建模拟更新包用于本地测试

此脚本创建一个模拟的更新zip包，用于测试 bootloader 的更新流程。
更新包中包含一个标记文件，用于验证更新是否成功执行。
"""

import os
import sys
import zipfile
import json
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def create_mock_update_package():
    """创建模拟更新包"""

    test_dir = PROJECT_ROOT / "test_update"
    test_dir.mkdir(exist_ok=True)

    mock_version = "3.1.1-test"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建临时目录结构
    temp_dir = test_dir / "mock_update_content"
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    # 创建更新验证标记文件
    marker_file = temp_dir / ".update_test_marker"
    marker_content = {
        "test_version": mock_version,
        "updated_at": datetime.now().isoformat(),
        "test_id": timestamp,
        "message": "This file proves the update mechanism worked correctly."
    }
    with open(marker_file, 'w', encoding='utf-8') as f:
        json.dump(marker_content, f, ensure_ascii=False, indent=2)

    # 创建模拟的版本文件（用于验证）
    version_file = temp_dir / "test_version.txt"
    with open(version_file, 'w', encoding='utf-8') as f:
        f.write(f"Test Update Version: {mock_version}\n")
        f.write(f"Test ID: {timestamp}\n")

    # 打包成 zip
    zip_path = test_dir / "mock_update.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in temp_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(temp_dir)
                zf.write(file_path, arcname)
                print(f"  Added: {arcname}")

    # 清理临时目录
    import shutil
    shutil.rmtree(temp_dir)

    print(f"\n[OK] Mock update package created: {zip_path}")
    print(f"     Size: {zip_path.stat().st_size} bytes")

    return zip_path


def create_local_http_server_script():
    """创建本地HTTP服务器脚本"""

    server_script = PROJECT_ROOT / "test_update" / "local_server.py"

    content = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本地HTTP服务器 - 用于提供模拟更新包下载
端口: 8888
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8888
DIRECTORY = Path(__file__).parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

if __name__ == "__main__":
    os.chdir(DIRECTORY)
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        print(f"Update package URL: http://localhost:{PORT}/mock_update.zip")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\\nServer stopped")
'''

    with open(server_script, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"[OK] Local server script created: {server_script}")
    return server_script


def main():
    print("=" * 60)
    print("Creating Mock Update Package for Local Testing")
    print("=" * 60)
    print()

    # 创建模拟更新包
    zip_path = create_mock_update_package()

    # 创建本地服务器脚本
    create_local_http_server_script()

    print()
    print("=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print()
    print("1. Start local HTTP server:")
    print("   python test_update/local_server.py")
    print()
    print("2. Run the update trigger script:")
    print("   python test_update/trigger_update.py")
    print()
    print("3. Check for the marker file after update:")
    print(f"   {PROJECT_ROOT / '.update_test_marker'}")
    print()


if __name__ == "__main__":
    main()
