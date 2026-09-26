"""ResearchStore 持久化层的离线测试（不连真实 API，不依赖 langgraph）。

store.py 只依赖标准库，直接用 importlib 加载模块文件，绕过
research/__init__.py（它会触发 graph.py → langgraph，本机未安装）。
"""

import importlib.util
import os
import sqlite3
import sys
import tempfile
import threading
import unittest

_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "clawgent", "core", "research", "store.py",
)


def _load_store():
    spec = importlib.util.spec_from_file_location("_research_store", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_research_store"] = mod
    spec.loader.exec_module(mod)
    return mod


store_mod = _load_store()
ResearchStore = store_mod.ResearchStore


def _source(sid="S1", url="https://example.com/a"):
    return {
        "source_id": sid, "source_type": "academic", "title": "论文 A",
        "authors": "张三", "url": url, "doi": "", "publication_year": "2024",
        "venue": "", "provider": "arxiv", "content_type": "full_text",
        "quality": "high", "retrieval_method": "academic",
        "retrieved_at": "2026-09-26T00:00:00+00:00",
        "metadata": {"search_query": "x-vector"},
    }


def _evidence(eid="E1", sid="S1", tid="t1"):
    return {
        "evidence_id": eid, "source_id": sid, "task_id": tid,
        "document_id": "D1", "chunk_id": "c1", "quote": "x-vector 在 2024 年达到 SOTA",
        "interpretation": "说明效果", "locator": "chunk:c1",
        "verification_status": "VERIFIED", "verification_reason": "quote 完整命中来源原文",
        "retrieval_query": "x-vector 性能", "retrieval_rank": 1,
        "source_type": "academic", "retrieval_method": "academic",
        "metadata": {"rerank_status": "SUCCESS"},
    }


def _claim(cid="C1", tid="t1"):
    return {
        "claim_id": cid, "task_id": tid, "text": "x-vector 效果最好",
        "claim_type": "FACT", "scope": "英文数据集", "conditions": "",
        "status": "SUPPORTED", "support_reason": "1 条证据通过原文校验且语义支持",
    }


def _task(tid="t1"):
    return {
        "task_id": tid, "objective": "查 x-vector 性能", "question": "x-vector 性能如何",
        "task_type": "FACT", "dependencies": [], "status": "COMPLETED",
        "priority": 1, "parent_task_id": "", "round_added": 0,
        "expected_evidence": "论文结论", "preferred_sources": ["academic"],
        "search_strategy": "关键词检索", "success_criteria": {"min_evidence_count": 2},
    }


class TestSchema(unittest.TestCase):
    """建表与 PRAGMA。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = ResearchStore(db_path=os.path.join(self.tmp, "t.sqlite3"))

    def tearDown(self):
        self.store.close()

    def test_tables_created(self):
        conn = self.store._connect()
        self.assertIsNotNone(conn)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")
        names = {r[0] for r in cur.fetchall()}
        for t in ("runs", "tasks", "sources", "evidences", "claims",
                  "claim_evidence", "task_results", "issues", "ledger_events",
                  "plan_gates"):
            self.assertIn(t, names, f"缺表 {t}")

    def test_wal_mode(self):
        conn = self.store._connect()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(str(mode).lower(), "wal")


class TestWriteThenRead(unittest.TestCase):
    """写入后能原样读回，且表之间的对应关系正确。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = ResearchStore(db_path=os.path.join(self.tmp, "t.sqlite3"))
        self.run_id = "R-test-1"

    def tearDown(self):
        self.store.close()

    def test_run_written(self):
        self.store.upsert_run(self.run_id, "x-vector 性能如何", sop_type="FACT")
        row = self.store.load_run(self.run_id)
        self.assertIsNotNone(row)
        self.assertEqual(row["original_query"], "x-vector 性能如何")
        self.assertEqual(row["sop_type"], "FACT")

    def test_task_source_evidence_claim_relation(self):
        self.store.upsert_run(self.run_id, "q")
        self.assertEqual(1, self.store.upsert_tasks(self.run_id, [_task()]))
        self.assertEqual(1, self.store.upsert_sources(self.run_id, [_source()]))
        self.assertEqual(1, self.store.upsert_evidences(self.run_id, [_evidence()]))
        self.assertEqual(1, self.store.upsert_claims(self.run_id, [_claim()]))
        self.assertEqual(1, self.store.upsert_relations(
            self.run_id, [{"source_id": "E1", "target_id": "C1",
                           "relation": "SUPPORTS", "semantic_relation": "SUPPORTS"}]))

        conn = self.store._connect()
        # evidence → source
        self.assertEqual(
            "S1", conn.execute(
                "SELECT source_id FROM evidences WHERE run_id=? AND evidence_id=?",
                (self.run_id, "E1")).fetchone()[0])
        # evidence → task
        self.assertEqual(
            "t1", conn.execute(
                "SELECT task_id FROM evidences WHERE run_id=? AND evidence_id=?",
                (self.run_id, "E1")).fetchone()[0])
        # claim ↔ evidence 关联表
        row = conn.execute(
            "SELECT relation, semantic_relation FROM claim_evidence "
            "WHERE run_id=? AND evidence_id=? AND claim_id=?",
            (self.run_id, "E1", "C1")).fetchone()
        self.assertEqual(("SUPPORTS", "SUPPORTS"), tuple(row))

    def test_upsert_is_idempotent(self):
        """同一 id 重复写入是覆盖，不是追加（对应 state 的按 id 合并语义）。"""
        self.store.upsert_run(self.run_id, "q")
        self.store.upsert_sources(self.run_id, [_source()])
        self.store.upsert_sources(self.run_id, [_source()])
        conn = self.store._connect()
        n = conn.execute(
            "SELECT COUNT(*) FROM sources WHERE run_id=?", (self.run_id,)).fetchone()[0]
        self.assertEqual(1, n)

    def test_claim_chain_trace(self):
        """从一条声明能回溯到证据、来源、任务与检索式。"""
        self.store.upsert_run(self.run_id, "q")
        self.store.upsert_tasks(self.run_id, [_task()])
        self.store.upsert_sources(self.run_id, [_source()])
        self.store.upsert_evidences(self.run_id, [_evidence()])
        self.store.upsert_claims(self.run_id, [_claim()])
        self.store.upsert_relations(
            self.run_id, [{"source_id": "E1", "target_id": "C1",
                           "relation": "SUPPORTS", "semantic_relation": "SUPPORTS"}])

        chain = self.store.load_claim_chain(self.run_id, "C1")
        self.assertEqual("x-vector 效果最好", chain.get("text"))
        steps = chain.get("evidence_chain", [])
        self.assertEqual(1, len(steps))
        self.assertEqual("https://example.com/a", steps[0]["source_url"])
        self.assertEqual("x-vector 性能", steps[0]["retrieval_query"])
        self.assertEqual("x-vector 性能如何", steps[0]["task_question"])

    def test_ledger_events(self):
        self.store.upsert_run(self.run_id, "q")
        events = [{"seq": 1, "event_type": "plan_created", "task_id": "",
                   "round_no": 0, "payload": {"task_count": 3}, "ts": 1.0}]
        self.assertEqual(1, self.store.append_ledger_events(self.run_id, events))
        rows = self.store.load_ledger(self.run_id)
        self.assertEqual(1, len(rows))
        self.assertEqual("plan_created", rows[0]["event_type"])

    def test_plan_gate_snapshot(self):
        self.store.upsert_run(self.run_id, "q")
        self.store.upsert_plan_gate(self.run_id, {
            "decision": "ALLOW_WITH_WARNINGS", "sop_type": "FACT",
            "repair_count": 1, "critic_called": True, "critic_failed": False,
            "issues": [{"issue_type": "MISSING_COVERAGE"}],
        })
        conn = self.store._connect()
        row = conn.execute(
            "SELECT decision, sop_type, repair_count, critic_called FROM plan_gates "
            "WHERE run_id=?", (self.run_id,)).fetchone()
        self.assertEqual(("ALLOW_WITH_WARNINGS", "FACT", 1, 1), tuple(row))


class TestPersistState(unittest.TestCase):
    """persist_state 一次性落库整个 state 快照。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = ResearchStore(db_path=os.path.join(self.tmp, "t.sqlite3"))
        self.run_id = "R-test-2"

    def tearDown(self):
        self.store.close()

    def test_persist_state_stats(self):
        state = {
            "original_query": "比较 A 和 B",
            "sop_type": "COMPARISON",
            "tasks": [_task("t1"), _task("t2")],
            "sources": [_source("S1", "https://a"), _source("S2", "https://b")],
            "evidences": [_evidence("E1", "S1", "t1"), _evidence("E2", "S2", "t2")],
            "claims": [_claim("C1", "t1")],
            "relations": [{"source_id": "E1", "target_id": "C1",
                           "relation": "SUPPORTS"}],
            "task_results": [{"task_id": "t1", "status": "COMPLETED",
                              "searched_queries": ["A vs B"]}],
            "issues": [{"issue_id": "I1", "issue_type": "COVERAGE_GAP",
                        "description": "缺 ablation"}],
            "ledger_events": [{"seq": 1, "event_type": "plan_created", "ts": 1.0}],
            "round_no": 1,
        }
        stats = self.store.persist_state(self.run_id, state)
        self.assertEqual(2, stats["tasks"])
        self.assertEqual(2, stats["sources"])
        self.assertEqual(2, stats["evidences"])
        self.assertEqual(1, stats["claims"])
        self.assertEqual(1, stats["relations"])
        self.assertEqual(1, stats["task_results"])
        self.assertEqual(1, stats["issues"])
        self.assertEqual(1, stats["ledger_events"])

        row = self.store.load_run(self.run_id)
        self.assertEqual("比较 A 和 B", row["original_query"])
        self.assertEqual("COMPARISON", row["sop_type"])

    def test_persist_state_empty_run_id_is_noop(self):
        self.assertEqual({}, self.store.persist_state("", {"tasks": [_task()]}))


class TestConcurrency(unittest.TestCase):
    """Researcher 通过 Send 并发执行，多线程写库不能出错、不能丢数据。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = ResearchStore(db_path=os.path.join(self.tmp, "t.sqlite3"))
        self.run_id = "R-conc"

    def tearDown(self):
        self.store.close()

    def test_concurrent_writes_serialized(self):
        self.store.upsert_run(self.run_id, "q")

        def worker(n: int):
            for i in range(20):
                self.store.upsert_sources(
                    self.run_id, [_source(f"S{n}-{i}", f"https://x/{n}/{i}")])

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        conn = self.store._connect()
        n = conn.execute(
            "SELECT COUNT(*) FROM sources WHERE run_id=?", (self.run_id,)).fetchone()[0]
        self.assertEqual(160, n, "并发写入应全部落库，不丢不重")
        self.assertEqual("", self.store._last_error)

    def test_concurrent_evidence_and_claim(self):
        """不同表的并发写入交叉执行，不触发 database is locked。"""
        self.store.upsert_run(self.run_id, "q")
        errors: list[str] = []

        def w_ev(n: int):
            try:
                for i in range(15):
                    self.store.upsert_evidences(
                        self.run_id, [_evidence(f"E{n}-{i}", "S1", "t1")])
            except Exception as e:      # 写入失败只打印，这里额外收集便于断言
                errors.append(str(e))

        def w_cl(n: int):
            try:
                for i in range(15):
                    self.store.upsert_claims(self.run_id, [_claim(f"C{n}-{i}")])
            except Exception as e:
                errors.append(str(e))

        threads = ([threading.Thread(target=w_ev, args=(n,)) for n in range(4)]
                   + [threading.Thread(target=w_cl, args=(n,)) for n in range(4)])
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual([], errors)
        conn = self.store._connect()
        self.assertEqual(60, conn.execute(
            "SELECT COUNT(*) FROM evidences WHERE run_id=?", (self.run_id,)).fetchone()[0])
        self.assertEqual(60, conn.execute(
            "SELECT COUNT(*) FROM claims WHERE run_id=?", (self.run_id,)).fetchone()[0])


class TestDisabled(unittest.TestCase):
    """持久化关闭时所有写操作是空操作，不建库、不报错。"""

    def test_disabled_store_writes_nothing(self):
        tmp = tempfile.mkdtemp()
        db = os.path.join(tmp, "off.sqlite3")
        store = ResearchStore(db_path=db, enabled=False)
        self.assertIsNone(store._connect())
        self.assertEqual(0, store.upsert_tasks("R1", [_task()]))
        self.assertEqual(0, store.upsert_sources("R1", [_source()]))
        self.assertIsNone(store.load_run("R1"))
        stats = store.persist_state("R1", {"tasks": [_task()]})
        self.assertTrue(all(v == 0 for v in stats.values()), "关闭时各项写入条数应为 0")
        self.assertFalse(os.path.exists(db), "关闭时不应创建数据库文件")


class TestFailureIsolation(unittest.TestCase):
    """写库失败不能抛异常阻断调研。"""

    def test_bad_path_does_not_raise(self):
        # 指向一个不可写的路径：建库失败应被吞掉，enabled 置 False
        store = ResearchStore(db_path="/proc/definitely/not/writable/x.sqlite3")
        try:
            conn = store._connect()
            if conn is None:
                self.assertFalse(store.enabled)
            else:
                self.skipTest("该环境路径可写，跳过")
        finally:
            store.close()

    def test_bad_payload_does_not_raise(self):
        tmp = tempfile.mkdtemp()
        store = ResearchStore(db_path=os.path.join(tmp, "t.sqlite3"))
        # 非 dict 条目应被跳过而不是抛异常
        self.assertEqual(0, store.upsert_tasks("R1", ["not-a-dict", None]))
        self.assertEqual(0, store.upsert_sources("R1", [{"no_source_id": True}]))
        store.close()


if __name__ == "__main__":
    unittest.main()
