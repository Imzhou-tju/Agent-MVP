"""Dashboard 启动入口：python -m entry.dashboard

用法：
    python -m entry.dashboard                # 默认 127.0.0.1:8765
    python -m entry.dashboard --port 9000    # 指定端口
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Research 可观测 Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    from clawgent.dashboard import DashboardServer

    server = DashboardServer(host=args.host, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
