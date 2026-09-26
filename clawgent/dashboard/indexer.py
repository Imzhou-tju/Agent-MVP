"""审计事件索引器：把 logs/*.jsonl 增量读进 SQLite，供 Dashboard 查询。

设计：
- 只读 JSONL，不写回；SQLite 是派生索引，删除可重建。
- 增量：每个 JSONL 文件记住已消费的行偏移，只读新增行。
- 事件字段以 audit.py 的结构为准（event_type / run_id / thread_id / timestamp /
  node / task_id / status / duration_ms / metadata / error）；兼容旧 logger 的
  `event` / `ts` 字段（旧 llm_input / tool_call / ai_message 事件）。
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Iterable

from ..core import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    TEXT,
    run_id      TEXT,
    thread_id   TEXT,
    event_type  TEXT,
    node        TEXT,
    task_id     TEXT,
    status      TEXT,
    timestamp   TEXT,
    duration_ms INTEGER,
    metadata    TEXT,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id);
CREATE INDEX IF NOT EXISTS idx_events_node ON events(node);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);

CREATE TABLE IF NOT EXISTS offsets (
    file_path TEXT PRIMARY KEY,
    byte_offset INTEGER NOT NULL,
    line_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS run_summary (
    run_id  TEXT PRIMARY KEY,
    summary TEXT
);
"""

# 事件类型归类（用于 Dashboard 分类统计）
LIFECYCLE_EVENTS = {
    "run_start", "run_end", "node_start", "node_end",
    "task_created", "task_ready", "task_started", "task_completed",
    "task_failed", "task_reopened", "task_skipped",
}
RELIABILITY_EVENTS = {"llm_call", "tool_call", "retry", "fallback", "circuit_breaker"}
RAG_EVENTS = {"rag_iteration", "rag_retrieval", "rerank", "multi_hop_iteration"}
DECISION_EVENTS = {"plan_gate", "aggregation", "review_issue", "repair", "judge_decision"}
TRACE_EVENTS = {"evidence_registered", "citation_verified"}


def _normalize_event(raw: dict) -> dict:
    """把 JSONL 行规范成统一事件 dict。兼容 audit.py 与旧 logger 两种字段。"""
    etype = raw.get("event_type") or raw.get("event") or ""
    ts = raw.get("timestamp") or raw.get("ts") or ""
    return {
        "event_id": raw.get("event_id", ""),
        "run_id": raw.get("run_id", ""),
        "thread_id": raw.get("thread_id", ""),
        "event_type": etype,
        "node": raw.get("node", ""),
        "task_id": raw.get("task_id", ""),
        "status": raw.get("status", ""),
        "timestamp": ts,
        "duration_ms": int(raw.get("duration_ms") or 0),
        "metadata": raw.get("metadata") if isinstance(raw.get("metadata"), dict)
                   else (raw.get("metadata") or {}),
        "error": raw.get("error"),
    }


def index_events(conn: sqlite3.Connection, events: Iterable[dict]) -> int:
    """把一批事件写入 events 表。返回写入条数。"""
    n = 0
    for ev in events:
        conn.execute(
            "INSERT INTO events (event_id, run_id, thread_id, event_type, node,"
            " task_id, status, timestamp, duration_ms, metadata, error)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                ev["event_id"], ev["run_id"], ev["thread_id"], ev["event_type"],
                ev["node"], ev["task_id"], ev["status"], ev["timestamp"],
                ev["duration_ms"],
                json.dumps(ev["metadata"], ensure_ascii=False) if ev["metadata"] else "{}",
                json.dumps(ev["error"], ensure_ascii=False) if ev["error"] else None,
            ),
        )
        n += 1
    return n


class AuditIndexer:
    """增量索引 logs/ 目录下的 JSONL 文件。"""

    def __init__(self, log_dir: str | None = None, db_path: str | None = None) -> None:
        self.log_dir = log_dir or config.AUDIT_LOG_DIR
        self.db_path = db_path or config.AUDIT_INDEX_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def _existing_offsets(self) -> dict:
        rows = self._conn.execute("SELECT file_path, byte_offset, line_count FROM offsets").fetchall()
        return {r["file_path"]: (r["byte_offset"], r["line_count"]) for r in rows}

    def _jsonl_files(self) -> list[str]:
        if not os.path.isdir(self.log_dir):
            return []
        return sorted(
            os.path.join(self.log_dir, f)
            for f in os.listdir(self.log_dir) if f.endswith(".jsonl")
        )

    def refresh(self) -> dict:
        """增量扫描所有 JSONL，把新增行写进索引。返回本次新增的 run 集合。"""
        offsets = self._existing_offsets()
        new_runs: set[str] = set()
        total_new = 0

        for path in self._jsonl_files():
            prev_off, prev_count = offsets.get(path, (0, 0))
            size = os.path.getsize(path)
            if size < prev_off:
                # 文件被截断/重建：从头重扫
                prev_off, prev_count = 0, 0
            if size == prev_off:
                continue

            with open(path, "r", encoding="utf-8") as f:
                f.seek(prev_off)
                batch: list[dict] = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = _normalize_event(json.loads(line))
                    except Exception:
                        continue
                    if ev["event_type"]:
                        batch.append(ev)
                        if ev["run_id"]:
                            new_runs.add(ev["run_id"])
                if batch:
                    total_new += index_events(self._conn, batch)
                end_off = f.tell()
                new_count = prev_count + len(batch)

            self._conn.execute(
                "INSERT INTO offsets (file_path, byte_offset, line_count) VALUES (?,?,?)"
                " ON CONFLICT(file_path) DO UPDATE SET byte_offset=excluded.byte_offset,"
                " line_count=excluded.line_count",
                (path, end_off, new_count),
            )

        self._conn.commit()
        self._rebuild_summaries(list(new_runs))
        return {"new_events": total_new, "new_runs": sorted(new_runs)}

    # ------------------------------------------------------------------
    # Run Summary 聚合（复用 audit.summarize 的纯函数，避免重复业务状态）
    # ------------------------------------------------------------------

    def _rebuild_summaries(self, run_ids: list[str]) -> None:
        if not run_ids:
            return
        from ..core.audit import summarize

        for rid in run_ids:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE run_id=? ORDER BY timestamp, seq", (rid,)
            ).fetchall()
            evs = [dict(r) for r in rows]
            for e in evs:
                if e["metadata"]:
                    try:
                        e["metadata"] = json.loads(e["metadata"])
                    except Exception:
                        e["metadata"] = {}
            summary = summarize(evs)
            self._conn.execute(
                "INSERT INTO run_summary (run_id, summary) VALUES (?,?)"
                " ON CONFLICT(run_id) DO UPDATE SET summary=excluded.summary",
                (rid, json.dumps(summary, ensure_ascii=False)),
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # 查询（Dashboard API 用）
    # ------------------------------------------------------------------

    def list_runs(self, status: str | None = None, before: str | None = None,
                  after: str | None = None, limit: int = 200) -> list[dict]:
        """Run 列表，带 summary。支持状态/时间过滤。"""
        sql = "SELECT run_id, summary FROM run_summary"
        conds: list[str] = []
        args: list[Any] = []
        if status:
            conds.append("json_extract(summary, '$.final_status') = ?")
            args.append(status)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        rows = self._conn.execute(sql, args).fetchall()
        out: list[dict] = []
        for r in rows:
            try:
                s = json.loads(r["summary"])
            except Exception:
                continue
            if before and s.get("start_time", "") > before:
                continue
            if after and s.get("end_time", "") < after:
                continue
            out.append({"run_id": r["run_id"], **s})
        out.sort(key=lambda x: x.get("start_time", ""), reverse=True)
        return out[:limit]

    def run_detail(self, run_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT summary FROM run_summary WHERE run_id=?", (run_id,)
        ).fetchone()
        if not row:
            return None
        try:
            s = json.loads(row["summary"])
        except Exception:
            s = {}
        s["events"] = self.run_events(run_id)
        s["dag"] = self.task_dag(run_id)
        s["trace"] = self.evidence_trace(run_id)
        s["rag"] = self.rag_iterations(run_id)
        s["reliability"] = self.reliability(run_id)
        return s

    def run_events(self, run_id: str, node: str | None = None,
                   task_id: str | None = None, event_type: str | None = None,
                   limit: int = 5000) -> list[dict]:
        sql = "SELECT * FROM events WHERE run_id=?"
        args: list[Any] = [run_id]
        if node:
            sql += " AND node=?"
            args.append(node)
        if task_id:
            sql += " AND task_id=?"
            args.append(task_id)
        if event_type:
            sql += " AND event_type=?"
            args.append(event_type)
        sql += " ORDER BY timestamp, seq LIMIT ?"
        args.append(limit)
        rows = self._conn.execute(sql, args).fetchall()
        return [self._row_event(r) for r in rows]

    def _row_event(self, r: sqlite3.Row) -> dict:
        d = dict(r)
        if d.get("metadata"):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except Exception:
                d["metadata"] = {}
        if d.get("error"):
            try:
                d["error"] = json.loads(d["error"])
            except Exception:
                pass
        return d

    def task_dag(self, run_id: str) -> list[dict]:
        """从 task_created 事件的 metadata 里还原任务依赖图。"""
        tasks: dict[str, dict] = {}
        for e in self.run_events(run_id, event_type="task_created"):
            m = e.get("metadata") or {}
            tid = e.get("task_id") or m.get("task_id", "")
            if not tid:
                continue
            tasks[tid] = {
                "task_id": tid,
                "task_type": m.get("task_type", ""),
                "dependencies": m.get("dependencies", []),
                "status": "PENDING",
                "duration_ms": 0,
                "priority": m.get("priority", 0),
            }
        # 叠加生命周期事件得到最终状态与耗时
        for ev in self.run_events(run_id):
            tid = ev.get("task_id", "")
            if not tid or tid not in tasks:
                continue
            etype = ev.get("event_type", "")
            if etype == "task_started":
                tasks[tid]["status"] = "RUNNING"
            elif etype == "task_completed":
                tasks[tid]["status"] = "COMPLETED"
            elif etype == "task_failed":
                tasks[tid]["status"] = "FAILED"
            elif etype == "task_reopened":
                tasks[tid]["status"] = "REOPENED"
            elif etype == "task_skipped":
                tasks[tid]["status"] = "SKIPPED"
            tasks[tid]["duration_ms"] += ev.get("duration_ms", 0)
        return list(tasks.values())

    def evidence_trace(self, run_id: str) -> list[dict]:
        """从 evidence_registered 事件还原 Evidence 登记信息（不读业务库）。"""
        out = []
        for e in self.run_events(run_id, event_type="evidence_registered"):
            m = e.get("metadata") or {}
            out.append({
                "evidence_id": m.get("evidence_id", ""),
                "source_id": m.get("source_id", ""),
                "source_type": m.get("source_type", ""),
                "verification_status": m.get("verification_status", ""),
                "claim_id": m.get("claim_id", ""),
                "task_id": e.get("task_id", ""),
            })
        return out

    def rag_iterations(self, run_id: str) -> dict:
        """RAG 单轮与多跳迭代信息。"""
        rag = [e.get("metadata") or {} for e in self.run_events(run_id, event_type="rag_iteration")]
        mh = [e.get("metadata") or {} for e in self.run_events(run_id, event_type="multi_hop_iteration")]
        return {"rag_iterations": rag, "multi_hop_iterations": mh}

    def reliability(self, run_id: str) -> dict:
        """可靠性统计：retry / fallback / circuit_breaker 次数。"""
        def cnt(t):
            return len([e for e in self.run_events(run_id, event_type=t)])
        return {
            "retries": cnt("retry"),
            "fallbacks": cnt("fallback"),
            "circuit_breakers": cnt("circuit_breaker"),
        }


def build_index(log_dir: str | None = None, db_path: str | None = None) -> AuditIndexer:
    """构建并刷新索引，返回 indexer（调用方用完需 close）。"""
    idx = AuditIndexer(log_dir=log_dir, db_path=db_path)
    idx.refresh()
    return idx
