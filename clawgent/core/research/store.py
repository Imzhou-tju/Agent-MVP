"""调研结果的 SQLite 持久化。

选型：SQLite（WAL 模式）。理由见文件末尾 STORAGE_RATIONALE。

存的是「一次调研」的完整可溯源结构，表之间的对应关系：

    runs 1 ── N tasks              （一次调研的计划 DAG）
    runs 1 ── N sources            （登记过的来源）
    runs 1 ── N evidences          （带原文引文的证据）
    runs 1 ── N claims             （抽出来的声明）
    claims N ── N evidences        （经 claim_evidence 关联表，带 relation / semantic_relation）
    sources 1 ── N evidences       （evidence.source_id → sources.source_id）
    tasks  1 ── N evidences        （evidence.task_id → tasks.task_id）
    tasks  1 ── 1 task_results     （Researcher 返回的结果包）
    runs 1 ── N issues             （Review 产出的结构化问题）
    runs 1 ── N ledger_events      （只追加的过程流水）
    runs 1 ── 1 plan_gates         （Planner 质量闸门的裁决快照）

写入串行化：Researcher 通过 Send 并发执行，多个协程会同时写库。
SQLite 同一时刻只允许一个写事务，并发写会抛 SQLITE_BUSY。这里用一个
可重入锁 + WAL + busy_timeout 保证写入串行、失败可重试，不引入新依赖。

失败策略：所有写操作失败只打印日志，不阻断调研流程。持久化是旁路，
调研主链路（内存 state）不依赖它成功。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any, Iterable

_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "workspace", "research.sqlite3",
)

# 是否启用持久化。关闭时 ResearchStore 所有写操作直接返回，不建库不连库。
PERSIST_ENABLED = os.getenv("RESEARCH_PERSIST_ENABLED", "true").lower() == "true"


def _default_db_path() -> str:
    try:
        from .. import config
        workspace = getattr(config, "WORKSPACE_DIR", "")
        if workspace:
            return os.path.join(workspace, "research.sqlite3")
    except Exception:
        pass
    return _DEFAULT_DB_PATH


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any, default: Any = "") -> str:
    """把 list / dict 序列化成 JSON 文本；其余原样转字符串。"""
    if value in (None, "", [], {}):
        return json.dumps(default, ensure_ascii=False)
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 建表
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    original_query    TEXT NOT NULL,
    research_context  TEXT,
    sop_type          TEXT,
    plan_summary      TEXT,
    verdict           TEXT,
    verdict_reason    TEXT,
    final_report      TEXT,
    status            TEXT DEFAULT 'RUNNING',
    created_at        TEXT,
    updated_at        TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    run_id            TEXT NOT NULL,
    task_id           TEXT NOT NULL,
    objective         TEXT,
    question          TEXT,
    task_type         TEXT,
    dependencies      TEXT,
    status            TEXT,
    priority          INTEGER,
    parent_task_id    TEXT,
    round_added       INTEGER DEFAULT 0,
    expected_evidence TEXT,
    preferred_sources TEXT,
    search_strategy   TEXT,
    success_criteria  TEXT,
    updated_at        TEXT,
    PRIMARY KEY (run_id, task_id)
);

CREATE TABLE IF NOT EXISTS sources (
    run_id            TEXT NOT NULL,
    source_id         TEXT NOT NULL,
    source_type       TEXT,
    title             TEXT,
    authors           TEXT,
    url               TEXT,
    doi               TEXT,
    publication_year  TEXT,
    venue             TEXT,
    provider          TEXT,
    content_type      TEXT,
    quality           TEXT,
    retrieval_method  TEXT,
    retrieved_at      TEXT,
    metadata          TEXT,
    PRIMARY KEY (run_id, source_id)
);

CREATE TABLE IF NOT EXISTS evidences (
    run_id              TEXT NOT NULL,
    evidence_id         TEXT NOT NULL,
    source_id           TEXT,
    task_id             TEXT,
    document_id         TEXT,
    chunk_id            TEXT,
    quote               TEXT,
    interpretation      TEXT,
    locator             TEXT,
    verification_status TEXT,
    verification_reason TEXT,
    retrieval_query     TEXT,
    retrieval_rank      INTEGER,
    source_type         TEXT,
    retrieval_method    TEXT,
    metadata            TEXT,
    PRIMARY KEY (run_id, evidence_id)
);

CREATE TABLE IF NOT EXISTS claims (
    run_id          TEXT NOT NULL,
    claim_id        TEXT NOT NULL,
    task_id         TEXT,
    text            TEXT,
    claim_type      TEXT,
    scope           TEXT,
    conditions      TEXT,
    status          TEXT,
    support_reason  TEXT,
    PRIMARY KEY (run_id, claim_id)
);

CREATE TABLE IF NOT EXISTS claim_evidence (
    run_id            TEXT NOT NULL,
    evidence_id       TEXT NOT NULL,
    claim_id          TEXT NOT NULL,
    relation          TEXT,
    semantic_relation TEXT,
    PRIMARY KEY (run_id, evidence_id, claim_id)
);

CREATE TABLE IF NOT EXISTS task_results (
    run_id             TEXT NOT NULL,
    task_id            TEXT NOT NULL,
    status             TEXT,
    error              TEXT,
    searched_queries   TEXT,
    unresolved_issues  TEXT,
    PRIMARY KEY (run_id, task_id)
);

CREATE TABLE IF NOT EXISTS issues (
    run_id             TEXT NOT NULL,
    issue_id           TEXT NOT NULL,
    issue_type         TEXT,
    target_task_id     TEXT,
    description        TEXT,
    recommended_action TEXT,
    round_no           INTEGER DEFAULT 0,
    PRIMARY KEY (run_id, issue_id)
);

CREATE TABLE IF NOT EXISTS ledger_events (
    run_id     TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    event_type TEXT,
    task_id    TEXT,
    round_no   INTEGER DEFAULT 0,
    payload    TEXT,
    ts         REAL,
    PRIMARY KEY (run_id, seq)
);

CREATE TABLE IF NOT EXISTS plan_gates (
    run_id        TEXT PRIMARY KEY,
    decision      TEXT,
    sop_type      TEXT,
    repair_count  INTEGER DEFAULT 0,
    critic_called INTEGER DEFAULT 0,
    critic_failed INTEGER DEFAULT 0,
    issues        TEXT
);

CREATE INDEX IF NOT EXISTS idx_evidences_source  ON evidences (run_id, source_id);
CREATE INDEX IF NOT EXISTS idx_evidences_task    ON evidences (run_id, task_id);
CREATE INDEX IF NOT EXISTS idx_claims_task       ON claims (run_id, task_id);
CREATE INDEX IF NOT EXISTS idx_claim_ev_claim    ON claim_evidence (run_id, claim_id);
CREATE INDEX IF NOT EXISTS idx_ledger_type       ON ledger_events (run_id, event_type);
CREATE INDEX IF NOT EXISTS idx_tasks_status      ON tasks (run_id, status);
"""


class ResearchStore:
    """调研结果的读写。

    用法：
        store = ResearchStore()               # 默认 workspace/research.sqlite3
        store.upsert_run(run_id, query=...)
        store.upsert_sources(run_id, sources) # 传 state 里的 dict 列表
    """

    def __init__(self, db_path: str | None = None, enabled: bool | None = None):
        self._db_path = db_path or _default_db_path()
        self._enabled = PERSIST_ENABLED if enabled is None else enabled
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._last_error: str = ""

    # ---- 连接 ----

    @property
    def db_path(self) -> str:
        return self._db_path

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _connect(self) -> sqlite3.Connection | None:
        """惰性建连接。返回 None 表示持久化不可用，调用方跳过。"""
        if not self._enabled:
            return None
        if self._conn is not None:
            return self._conn
        try:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            conn = sqlite3.connect(self._db_path, timeout=15.0,
                                   check_same_thread=False)
            # WAL：读不阻塞写、写不阻塞读；写事务仍然串行，由 _lock 保证
            conn.execute("PRAGMA journal_mode=WAL")
            # NORMAL：WAL 下崩溃不会损坏库，只可能丢最后几条未落盘提交
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=15000")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(SCHEMA)
            conn.commit()
            self._conn = conn
            return conn
        except Exception as e:
            self._last_error = f"建库失败: {e}"
            print(f"[ResearchStore] {self._last_error}")
            self._enabled = False
            return None

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def _write(self, sql: str, rows: Iterable[tuple]) -> int:
        """串行化批量写。失败返回 0，不抛异常。"""
        conn = self._connect()
        if conn is None:
            return 0
        rows = list(rows)
        if not rows:
            return 0
        with self._lock:
            try:
                conn.executemany(sql, rows)
                conn.commit()
                return len(rows)
            except Exception as e:
                self._last_error = f"写入失败: {e}"
                print(f"[ResearchStore] {self._last_error}")
                try:
                    conn.rollback()
                except Exception:
                    pass
                return 0

    # ---- 写：run ----

    def upsert_run(self, run_id: str, query: str = "", **fields: Any) -> None:
        if not run_id:
            return
        conn = self._connect()
        if conn is None:
            return
        allowed = ("research_context", "sop_type", "plan_summary", "verdict",
                   "verdict_reason", "final_report", "status")
        cols = ["run_id", "original_query", "updated_at"]
        vals: list[Any] = [run_id, query, _now()]
        for k in allowed:
            if k in fields and fields[k] is not None:
                cols.append(k)
                vals.append(fields[k])
        with self._lock:
            try:
                conn.execute(
                    "INSERT INTO runs (run_id, original_query, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(run_id) DO UPDATE SET updated_at=excluded.updated_at",
                    (run_id, query, _now(), _now()),
                )
                if len(cols) > 3:
                    sets = ", ".join(f"{c}=excluded.{c}" for c in cols)
                    ph = ", ".join("?" * len(cols))
                    sql = (f"INSERT INTO runs ({', '.join(cols)}) VALUES ({ph}) "
                           f"ON CONFLICT(run_id) DO UPDATE SET {sets}")
                    conn.execute(sql, tuple(vals))
                conn.commit()
            except Exception as e:
                self._last_error = f"写 run 失败: {e}"
                print(f"[ResearchStore] {self._last_error}")

    # ---- 写：DAG 任务 ----

    def upsert_tasks(self, run_id: str, tasks: Iterable[dict]) -> int:
        rows = []
        for t in tasks or []:
            if not isinstance(t, dict):
                continue
            tid = str(t.get("task_id", ""))
            if not tid or not run_id:
                continue
            rows.append((
                run_id, tid,
                t.get("objective", ""), t.get("question", ""), t.get("task_type", ""),
                _json(t.get("dependencies", []), []),
                t.get("status", ""), _as_int(t.get("priority", 1), 1),
                t.get("parent_task_id", ""), _as_int(t.get("round_added", 0)),
                t.get("expected_evidence", ""), _json(t.get("preferred_sources", []), []),
                t.get("search_strategy", ""), _json(t.get("success_criteria", {}), {}),
                _now(),
            ))
        sql = (
            "INSERT INTO tasks (run_id, task_id, objective, question, task_type,"
            " dependencies, status, priority, parent_task_id, round_added,"
            " expected_evidence, preferred_sources, search_strategy, success_criteria,"
            " updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(run_id, task_id) DO UPDATE SET "
            "objective=excluded.objective, question=excluded.question,"
            " task_type=excluded.task_type, dependencies=excluded.dependencies,"
            " status=excluded.status, priority=excluded.priority,"
            " parent_task_id=excluded.parent_task_id, round_added=excluded.round_added,"
            " expected_evidence=excluded.expected_evidence,"
            " preferred_sources=excluded.preferred_sources,"
            " search_strategy=excluded.search_strategy,"
            " success_criteria=excluded.success_criteria, updated_at=excluded.updated_at"
        )
        return self._write(sql, rows)

    # ---- 写：来源 ----

    def upsert_sources(self, run_id: str, sources: Iterable[dict]) -> int:
        rows = []
        for s in sources or []:
            if not isinstance(s, dict):
                continue
            sid = str(s.get("source_id", ""))
            if not sid or not run_id:
                continue
            rows.append((
                run_id, sid, s.get("source_type", ""), s.get("title", ""),
                s.get("authors", ""), s.get("url", ""), s.get("doi", ""),
                str(s.get("publication_year", "")), s.get("venue", ""),
                s.get("provider", ""), s.get("content_type", ""), s.get("quality", ""),
                s.get("retrieval_method", ""), s.get("retrieved_at", ""),
                _json(s.get("metadata", {}), {}),
            ))
        sql = (
            "INSERT INTO sources (run_id, source_id, source_type, title, authors, url,"
            " doi, publication_year, venue, provider, content_type, quality,"
            " retrieval_method, retrieved_at, metadata) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(run_id, source_id) DO UPDATE SET "
            "source_type=excluded.source_type, title=excluded.title,"
            " authors=excluded.authors, url=excluded.url, doi=excluded.doi,"
            " publication_year=excluded.publication_year, venue=excluded.venue,"
            " provider=excluded.provider, content_type=excluded.content_type,"
            " quality=excluded.quality, retrieval_method=excluded.retrieval_method,"
            " retrieved_at=excluded.retrieved_at, metadata=excluded.metadata"
        )
        return self._write(sql, rows)

    # ---- 写：证据 ----

    def upsert_evidences(self, run_id: str, evidences: Iterable[dict]) -> int:
        rows = []
        for e in evidences or []:
            if not isinstance(e, dict):
                continue
            eid = str(e.get("evidence_id", ""))
            if not eid or not run_id:
                continue
            rows.append((
                run_id, eid, e.get("source_id", ""), e.get("task_id", ""),
                e.get("document_id", ""), e.get("chunk_id", ""),
                e.get("quote", ""), e.get("interpretation", ""), e.get("locator", ""),
                e.get("verification_status", ""), e.get("verification_reason", ""),
                e.get("retrieval_query", ""), _as_int(e.get("retrieval_rank", 0)),
                e.get("source_type", ""), e.get("retrieval_method", ""),
                _json(e.get("metadata", {}), {}),
            ))
        sql = (
            "INSERT INTO evidences (run_id, evidence_id, source_id, task_id, document_id,"
            " chunk_id, quote, interpretation, locator, verification_status,"
            " verification_reason, retrieval_query, retrieval_rank, source_type,"
            " retrieval_method, metadata) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(run_id, evidence_id) DO UPDATE SET "
            "source_id=excluded.source_id, task_id=excluded.task_id,"
            " document_id=excluded.document_id, chunk_id=excluded.chunk_id,"
            " quote=excluded.quote, interpretation=excluded.interpretation,"
            " locator=excluded.locator, verification_status=excluded.verification_status,"
            " verification_reason=excluded.verification_reason,"
            " retrieval_query=excluded.retrieval_query,"
            " retrieval_rank=excluded.retrieval_rank, source_type=excluded.source_type,"
            " retrieval_method=excluded.retrieval_method, metadata=excluded.metadata"
        )
        return self._write(sql, rows)

    # ---- 写：声明 ----

    def upsert_claims(self, run_id: str, claims: Iterable[dict]) -> int:
        rows = []
        for c in claims or []:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("claim_id", ""))
            if not cid or not run_id:
                continue
            rows.append((
                run_id, cid, c.get("task_id", ""), c.get("text", ""),
                c.get("claim_type", ""), c.get("scope", ""), c.get("conditions", ""),
                c.get("status", ""), c.get("support_reason", ""),
            ))
        sql = (
            "INSERT INTO claims (run_id, claim_id, task_id, text, claim_type, scope,"
            " conditions, status, support_reason) VALUES (?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(run_id, claim_id) DO UPDATE SET "
            "task_id=excluded.task_id, text=excluded.text,"
            " claim_type=excluded.claim_type, scope=excluded.scope,"
            " conditions=excluded.conditions, status=excluded.status,"
            " support_reason=excluded.support_reason"
        )
        return self._write(sql, rows)

    # ---- 写：声明-证据关联 ----

    def upsert_relations(self, run_id: str, relations: Iterable[dict]) -> int:
        rows = []
        for r in relations or []:
            if not isinstance(r, dict):
                continue
            eid = str(r.get("source_id", ""))     # ClaimRelation.source_id = evidence_id
            cid = str(r.get("target_id", ""))     # ClaimRelation.target_id = claim_id
            if not eid or not cid or not run_id:
                continue
            rows.append((run_id, eid, cid, r.get("relation", ""),
                         r.get("semantic_relation", "")))
        sql = (
            "INSERT INTO claim_evidence (run_id, evidence_id, claim_id, relation,"
            " semantic_relation) VALUES (?,?,?,?,?) "
            "ON CONFLICT(run_id, evidence_id, claim_id) DO UPDATE SET "
            "relation=excluded.relation, semantic_relation=excluded.semantic_relation"
        )
        return self._write(sql, rows)

    # ---- 写：结果包 ----

    def upsert_task_results(self, run_id: str, packets: Iterable[dict]) -> int:
        rows = []
        for p in packets or []:
            if not isinstance(p, dict):
                continue
            tid = str(p.get("task_id", ""))
            if not tid or not run_id:
                continue
            rows.append((
                run_id, tid, p.get("status", ""), p.get("error", ""),
                _json(p.get("searched_queries", []), []),
                _json(p.get("unresolved_issues", []), []),
            ))
        sql = (
            "INSERT INTO task_results (run_id, task_id, status, error, searched_queries,"
            " unresolved_issues) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(run_id, task_id) DO UPDATE SET "
            "status=excluded.status, error=excluded.error,"
            " searched_queries=excluded.searched_queries,"
            " unresolved_issues=excluded.unresolved_issues"
        )
        return self._write(sql, rows)

    # ---- 写：评审问题 ----

    def upsert_issues(self, run_id: str, issues: Iterable[dict],
                      round_no: int = 0) -> int:
        rows = []
        for i in issues or []:
            if not isinstance(i, dict):
                continue
            iid = str(i.get("issue_id", "") or i.get("id", ""))
            if not iid or not run_id:
                continue
            rows.append((
                run_id, iid, i.get("issue_type", ""), i.get("target_task_id", ""),
                i.get("description", ""), i.get("recommended_action", ""),
                _as_int(i.get("round_no", round_no)),
            ))
        sql = (
            "INSERT INTO issues (run_id, issue_id, issue_type, target_task_id,"
            " description, recommended_action, round_no) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(run_id, issue_id) DO UPDATE SET "
            "issue_type=excluded.issue_type, target_task_id=excluded.target_task_id,"
            " description=excluded.description,"
            " recommended_action=excluded.recommended_action, round_no=excluded.round_no"
        )
        return self._write(sql, rows)

    # ---- 写：过程流水 ----

    def append_ledger_events(self, run_id: str, events: Iterable[dict]) -> int:
        rows = []
        for e in events or []:
            if not isinstance(e, dict):
                continue
            seq = _as_int(e.get("seq", 0))
            if not run_id or seq <= 0:
                continue
            rows.append((
                run_id, seq, e.get("event_type", ""), e.get("task_id", ""),
                _as_int(e.get("round_no", 0)), _json(e.get("payload", {}), {}),
                float(e.get("ts", 0.0) or 0.0),
            ))
        sql = (
            "INSERT INTO ledger_events (run_id, seq, event_type, task_id, round_no,"
            " payload, ts) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(run_id, seq) DO UPDATE SET "
            "event_type=excluded.event_type, task_id=excluded.task_id,"
            " round_no=excluded.round_no, payload=excluded.payload, ts=excluded.ts"
        )
        return self._write(sql, rows)

    # ---- 写：质量闸门快照 ----

    def upsert_plan_gate(self, run_id: str, gate: dict) -> None:
        if not run_id or not isinstance(gate, dict):
            return
        conn = self._connect()
        if conn is None:
            return
        with self._lock:
            try:
                conn.execute(
                    "INSERT INTO plan_gates (run_id, decision, sop_type, repair_count,"
                    " critic_called, critic_failed, issues) VALUES (?,?,?,?,?,?,?) "
                    "ON CONFLICT(run_id) DO UPDATE SET decision=excluded.decision,"
                    " sop_type=excluded.sop_type, repair_count=excluded.repair_count,"
                    " critic_called=excluded.critic_called,"
                    " critic_failed=excluded.critic_failed, issues=excluded.issues",
                    (run_id, gate.get("decision", ""), gate.get("sop_type", ""),
                     _as_int(gate.get("repair_count", 0)),
                     1 if gate.get("critic_called") else 0,
                     1 if gate.get("critic_failed") else 0,
                     _json(gate.get("issues", []), [])),
                )
                conn.commit()
            except Exception as e:
                self._last_error = f"写 plan_gate 失败: {e}"
                print(f"[ResearchStore] {self._last_error}")

    # ---- 一次性写入当前 state 快照 ----

    def persist_state(self, run_id: str, state: dict) -> dict:
        """把 state 里已有的实体整体落库。返回各类写入条数，便于排查。"""
        if not run_id:
            return {}
        stats = {
            "tasks": self.upsert_tasks(run_id, state.get("tasks", [])),
            "sources": self.upsert_sources(run_id, state.get("sources", [])),
            "evidences": self.upsert_evidences(run_id, state.get("evidences", [])),
            "claims": self.upsert_claims(run_id, state.get("claims", [])),
            "relations": self.upsert_relations(run_id, state.get("relations", [])),
            "task_results": self.upsert_task_results(run_id, state.get("task_results", [])),
            "issues": self.upsert_issues(run_id, state.get("issues", []),
                                          _as_int(state.get("round_no", 0))),
            "ledger_events": self.append_ledger_events(run_id, state.get("ledger_events", [])),
        }
        self.upsert_run(run_id, state.get("original_query", ""),
                        research_context=state.get("research_context", ""),
                        sop_type=state.get("sop_type", ""),
                        plan_summary=state.get("plan_summary", ""),
                        verdict=state.get("verdict", ""),
                        verdict_reason=state.get("verdict_reason", ""),
                        final_report=state.get("final_report", ""))
        return stats

    # ---- 读：溯源 ----

    def load_run(self, run_id: str) -> dict | None:
        conn = self._connect()
        if conn is None:
            return None
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
            row = cur.fetchone()
            return dict(row) if row else None
        except Exception as e:
            print(f"[ResearchStore] 读 run 失败: {e}")
            return None

    def load_claim_chain(self, run_id: str, claim_id: str) -> dict:
        """从一条声明回溯到来源与检索式（对应 ledger.build_trace 的内存版）。"""
        conn = self._connect()
        if conn is None:
            return {}
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM claims WHERE run_id=? AND claim_id=?", (run_id, claim_id))
            row = cur.fetchone()
            if not row:
                return {}
            claim = dict(row)
            cur = conn.execute(
                "SELECT e.*, ce.relation, ce.semantic_relation, s.title AS source_title,"
                " s.url AS source_url, t.question AS task_question "
                "FROM claim_evidence ce "
                "JOIN evidences e ON e.run_id=ce.run_id AND e.evidence_id=ce.evidence_id "
                "LEFT JOIN sources s ON s.run_id=e.run_id AND s.source_id=e.source_id "
                "LEFT JOIN tasks t ON t.run_id=e.run_id AND t.task_id=e.task_id "
                "WHERE ce.run_id=? AND ce.claim_id=?", (run_id, claim_id))
            claim["evidence_chain"] = [dict(r) for r in cur.fetchall()]
            return claim
        except Exception as e:
            print(f"[ResearchStore] 溯源查询失败: {e}")
            return {}

    def load_claims(self, run_id: str, status: str | None = None) -> list[dict]:
        conn = self._connect()
        if conn is None:
            return []
        try:
            conn.row_factory = sqlite3.Row
            if status:
                cur = conn.execute(
                    "SELECT * FROM claims WHERE run_id=? AND status=?", (run_id, status))
            else:
                cur = conn.execute("SELECT * FROM claims WHERE run_id=?", (run_id,))
            return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            print(f"[ResearchStore] 读 claims 失败: {e}")
            return []

    def load_ledger(self, run_id: str) -> list[dict]:
        conn = self._connect()
        if conn is None:
            return []
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM ledger_events WHERE run_id=? ORDER BY seq", (run_id,))
            return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            print(f"[ResearchStore] 读 ledger 失败: {e}")
            return []


# 全局单例：各节点共用一条连接，写入由 _lock 串行化
_store: ResearchStore | None = None
_store_lock = threading.Lock()


def get_store(db_path: str | None = None) -> ResearchStore:
    """获取全局 ResearchStore（惰性创建）。"""
    global _store
    with _store_lock:
        if _store is None:
            _store = ResearchStore(db_path=db_path)
        return _store


STORAGE_RATIONALE = """
选型结论：SQLite（WAL 模式），不引入新依赖。

场景特征：
- 单进程本地科研调研 Agent，一次调研产出几十到几百条记录，数据量 MB 级；
- Researcher 通过 LangGraph Send 并发执行，并发写来自同一进程的多个协程，
  不是多机多进程；
- 项目已依赖 sqlite3（rag/reliability.py 的 CircuitBreaker / DeadLetterQueue），
  无 ORM、无数据库服务进程。

对比：
- PostgreSQL / MySQL：需要独立服务进程与部署，本场景没有多机并发写、
  没有跨进程共享、数据量不到需要服务端数据库的规模，引入只为「显得正式」，
  代价是部署与运维。
- SQLite：零部署、单文件、标准库直接支持。WAL 模式下读不阻塞写，
  写事务由本模块的锁串行化，避免 SQLITE_BUSY。

SQLite 的边界（面试要主动说）：
- 同一时刻只允许一个写事务，高并发写会排队；本项目写入点在调研节点，
  每次几十条，排队开销可忽略。
- 单机单文件，不适合多机共享；若后续要做多用户并发调研服务，
  需要换成 PostgreSQL 并把 store.py 的 SQL 层替换掉（上层接口不变）。
"""
