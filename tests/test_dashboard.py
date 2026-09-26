"""Dashboard 索引器离线测试（§监控/审计 spec）。

只测索引构建、run 列表过滤、run detail 字段、task DAG 还原、证据溯源组装，
不启动真实 HTTP 服务、不联网。JSONL 直接写入临时目录，模拟 logger 落盘格式。
"""

import json
import os
import sys
import tempfile
import unittest
from types import ModuleType

_dotenv = ModuleType("dotenv")
_dotenv.load_dotenv = lambda *a, **k: None
sys.modules["dotenv"] = _dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.join(_HERE, "..")
sys.path.insert(0, _PROJECT)

from clawgent.dashboard.indexer import AuditIndexer


def _line(**kw):
    """构造一条 JSONL 行（与 logger 落盘格式一致：event / ts / thread_id + 字段）。"""
    return json.dumps(kw, ensure_ascii=False)


class TestAuditIndexer(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "idx.sqlite")
        self._write_jsonl()

    def _write_jsonl(self):
        events = [
            {"event": "run_start", "event_type": "run_start", "run_id": "r1",
             "thread_id": "t1", "status": "SUCCESS", "timestamp": "2026-01-01T00:00:00Z",
             "node": "planner", "metadata": {"query": "问题A"}},
            {"event": "task_created", "event_type": "task_created", "run_id": "r1",
             "thread_id": "t1", "task_id": "t1", "status": "SUCCESS",
             "timestamp": "2026-01-01T00:00:01Z", "metadata": {"task_type": "BACKGROUND",
             "dependencies": [], "priority": 1}},
            {"event": "task_created", "event_type": "task_created", "run_id": "r1",
             "thread_id": "t1", "task_id": "t2", "status": "SUCCESS",
             "timestamp": "2026-01-01T00:00:02Z", "metadata": {"task_type": "COMPARISON",
             "dependencies": ["t1"], "priority": 2}},
            {"event": "task_started", "event_type": "task_started", "run_id": "r1",
             "thread_id": "t1", "task_id": "t1", "timestamp": "2026-01-01T00:00:03Z"},
            {"event": "evidence_registered", "event_type": "evidence_registered", "run_id": "r1",
             "thread_id": "t1", "task_id": "t1", "timestamp": "2026-01-01T00:00:04Z",
             "metadata": {"evidence_id": "e1", "source_id": "s1", "source_type": "web",
                          "verification_status": "VERIFIED", "claim_id": "C1"}},
            {"event": "task_completed", "event_type": "task_completed", "run_id": "r1",
             "thread_id": "t1", "task_id": "t1", "timestamp": "2026-01-01T00:00:05Z"},
            {"event": "judge_decision", "event_type": "judge_decision", "run_id": "r1",
             "thread_id": "t1", "timestamp": "2026-01-01T00:00:06Z",
             "metadata": {"decision": "COMPILE", "reason": "done"}},
            {"event": "run_end", "event_type": "run_end", "run_id": "r1",
             "thread_id": "t1", "status": "SUCCESS", "timestamp": "2026-01-01T00:00:07Z",
             "metadata": {"verdict": "COMPILE"}},
            # 第二个 run，FAILED
            {"event": "run_start", "event_type": "run_start", "run_id": "r2",
             "thread_id": "t2", "status": "SUCCESS", "timestamp": "2026-01-02T00:00:00Z",
             "metadata": {"query": "问题B"}},
            {"event": "run_end", "event_type": "run_end", "run_id": "r2",
             "thread_id": "t2", "status": "FAILED", "timestamp": "2026-01-02T00:00:01Z",
             "metadata": {"verdict": "ABORT"}},
        ]
        with open(os.path.join(self.dir, "t1.jsonl"), "w", encoding="utf-8") as f:
            for e in events:
                f.write(_line(**e) + "\n")

    def test_refresh_and_list_runs(self):
        idx = AuditIndexer(log_dir=self.dir, db_path=self.db)
        r = idx.refresh()
        self.assertIn("r1", r["new_runs"])
        self.assertIn("r2", r["new_runs"])
        runs = idx.list_runs()
        ids = [x["run_id"] for x in runs]
        self.assertIn("r1", ids)
        self.assertIn("r2", ids)
        idx.close()

    def test_list_runs_filter_status(self):
        idx = AuditIndexer(log_dir=self.dir, db_path=self.db)
        idx.refresh()
        failed = idx.list_runs(status="FAILED")
        self.assertTrue(all(x["final_status"] == "FAILED" for x in failed))
        self.assertTrue(any(x["run_id"] == "r2" for x in failed))
        idx.close()

    def test_run_detail_fields(self):
        idx = AuditIndexer(log_dir=self.dir, db_path=self.db)
        idx.refresh()
        d = idx.run_detail("r1")
        self.assertIsNotNone(d)
        self.assertEqual(d["final_status"], "SUCCESS")
        self.assertEqual(d["verdict"], "COMPILE")
        self.assertEqual(d["total_tasks"], 2)
        self.assertEqual(d["completed_tasks"], 1)
        self.assertEqual(d["evidence_count"], 1)
        idx.close()

    def test_task_dag_reconstruction(self):
        idx = AuditIndexer(log_dir=self.dir, db_path=self.db)
        idx.refresh()
        dag = idx.run_detail("r1")["dag"]
        by_id = {t["task_id"]: t for t in dag}
        self.assertIn("t1", by_id)
        self.assertIn("t2", by_id)
        self.assertEqual(by_id["t1"]["status"], "COMPLETED")
        self.assertEqual(by_id["t2"]["dependencies"], ["t1"])
        self.assertEqual(by_id["t2"]["task_type"], "COMPARISON")
        idx.close()

    def test_evidence_trace(self):
        idx = AuditIndexer(log_dir=self.dir, db_path=self.db)
        idx.refresh()
        trace = idx.run_detail("r1")["trace"]
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["evidence_id"], "e1")
        self.assertEqual(trace[0]["verification_status"], "VERIFIED")
        self.assertEqual(trace[0]["claim_id"], "C1")
        idx.close()

    def test_incremental_refresh(self):
        idx = AuditIndexer(log_dir=self.dir, db_path=self.db)
        first = idx.refresh()
        second = idx.refresh()
        self.assertEqual(second["new_events"], 0)
        self.assertEqual(second["new_runs"], [])
        idx.close()

    def test_run_not_found(self):
        idx = AuditIndexer(log_dir=self.dir, db_path=self.db)
        idx.refresh()
        self.assertIsNone(idx.run_detail("ghost"))
        idx.close()

    def test_append_incremental(self):
        """追加新事件到已有 JSONL，增量刷新能读到新 run。"""
        idx = AuditIndexer(log_dir=self.dir, db_path=self.db)
        idx.refresh()
        with open(os.path.join(self.dir, "t1.jsonl"), "a", encoding="utf-8") as f:
            f.write(_line(event="run_start", event_type="run_start", run_id="r3",
                          thread_id="t3", status="SUCCESS",
                          timestamp="2026-01-03T00:00:00Z", metadata={"query": "C"}) + "\n")
            f.write(_line(event="run_end", event_type="run_end", run_id="r3",
                          thread_id="t3", status="SUCCESS",
                          timestamp="2026-01-03T00:00:01Z", metadata={"verdict": "COMPILE"}) + "\n")
        r = idx.refresh()
        self.assertIn("r3", r["new_runs"])
        self.assertIsNotNone(idx.run_detail("r3"))
        idx.close()


if __name__ == "__main__":
    unittest.main()
