#!/usr/bin/env python
"""本地预览服务器。

用法：
    python serve.py          # http://127.0.0.1:8000
    python serve.py 8080     # 指定端口

为什么需要它：浏览器不允许 file:// 页面用 fetch 读取本地 JSON，
所以可视化页面必须通过 HTTP 打开。双击 web/standalone.html 也可以，
那份是数据内联的自包含版本。
"""

from __future__ import annotations

import functools
import http.server
import socketserver
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

    if not (WEB / "graph_data.json").exists():
        print("缺少 web/graph_data.json，请先运行： python build_graph.py")
        return 1

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(WEB))
    socketserver.TCPServer.allow_reuse_address = True

    try:
        httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    except OSError as exc:
        print(f"端口 {port} 起不来（{exc}），换个端口试试： python serve.py {port + 1}")
        return 1

    with httpd:
        print(f"\n  知识图谱可视化已启动： http://127.0.0.1:{port}")
        print("  按 Ctrl+C 停止\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
