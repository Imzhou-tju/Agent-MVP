"""本地只读 Dashboard HTTP 服务（stdlib 实现，无外部依赖）。

只读：不修改 Research State，不写 JSONL；查询异常返回 500 但不影响调研主流程。
启动：python -m entry.dashboard 或 DashboardServer(...).serve_forever()
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ..core import config
from .indexer import AuditIndexer

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def _json(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


class DashboardHandler(BaseHTTPRequestHandler):
    indexer: AuditIndexer | None = None  # 由 server 注入

    def log_message(self, *args: Any) -> None:
        pass  # 静默，避免刷屏

    def _send(self, code: int, data: Any, ctype: str = "application/json; charset=utf-8") -> None:
        body = data if isinstance(data, bytes) else _json(data)
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._route()
        except Exception as e:  # 查询异常不影响调研，只返回 500
            self._send(500, {"error": f"Dashboard 查询异常: {e}"})

    def _route(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        qs = self._query()

        if path == "/" or path == "/index.html":
            return self._serve_static("index.html")
        if path.startswith("/api/runs") and "/" not in path[len("/api/runs"):]:
            return self._api_runs(qs)
        if path.startswith("/api/runs/"):
            run_id = path.split("/api/runs/", 1)[1]
            if run_id == "list":
                return self._api_runs(qs)
            return self._api_run_detail(run_id, qs)
        if path == "/api/refresh":
            return self._api_refresh()
        return self._send(404, {"error": "not found"})

    def _query(self) -> dict:
        q = {}
        if "?" in self.path:
            for kv in self.path.split("?", 1)[1].split("&"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    q[k] = v
        return q

    def _serve_static(self, name: str) -> None:
        fp = os.path.join(_STATIC_DIR, name)
        if not os.path.exists(fp):
            return self._send(404, {"error": "static not found"})
        with open(fp, "r", encoding="utf-8") as f:
            self._send(200, f.read().encode("utf-8"), "text/html; charset=utf-8")

    def _api_runs(self, qs: dict) -> None:
        if self.indexer is None:
            return self._send(500, {"error": "indexer not ready"})
        self.indexer.refresh()
        runs = self.indexer.list_runs(
            status=qs.get("status") or None,
            before=qs.get("before") or None,
            after=qs.get("after") or None,
        )
        self._send(200, {"runs": runs})

    def _api_run_detail(self, run_id: str, qs: dict) -> None:
        if self.indexer is None:
            return self._send(500, {"error": "indexer not ready"})
        self.indexer.refresh()
        detail = self.indexer.run_detail(run_id)
        if detail is None:
            return self._send(404, {"error": f"run {run_id} 不存在"})
        self._send(200, detail)

    def _api_refresh(self) -> None:
        if self.indexer is None:
            return self._send(500, {"error": "indexer not ready"})
        result = self.indexer.refresh()
        self._send(200, result)


class DashboardServer:
    """包装 indexer + HTTP server 的启动入口。"""

    def __init__(self, host: str | None = None, port: int | None = None,
                 log_dir: str | None = None, db_path: str | None = None) -> None:
        self.host = host or config.DASHBOARD_HOST
        self.port = port or config.DASHBOARD_PORT
        self.indexer = AuditIndexer(log_dir=log_dir, db_path=db_path)
        DashboardHandler.indexer = self.indexer
        self.httpd = ThreadingHTTPServer((self.host, self.port), DashboardHandler)

    def serve_forever(self) -> None:
        print(f"[Dashboard] 服务已启动: http://{self.host}:{self.port}")
        print(f"[Dashboard] 审计日志目录: {self.indexer.log_dir}")
        print(f"[Dashboard] 索引库: {self.indexer.db_path}")
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[Dashboard] 已停止")
        finally:
            self.close()

    def close(self) -> None:
        self.indexer.close()
        self.httpd.server_close()


def main() -> None:
    server = DashboardServer()
    server.serve_forever()


if __name__ == "__main__":
    main()
