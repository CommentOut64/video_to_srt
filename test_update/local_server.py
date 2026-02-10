#!/usr/bin/env python
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
            print("\nServer stopped")
