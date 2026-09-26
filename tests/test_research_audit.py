"""审计事件层 + Run Summary 的离线测试（§监控/审计 spec）。

只测确定性逻辑：事件字段收敛、Summary 聚合、异常流程、写入失败隔离。
不调模型、不联网、不落真实日志文件（用内存 sink 捕获事件）。
"""

import os
import sys
import unittest
from types import ModuleType

_dotenv = ModuleType("dotenv")
_dotenv.load_dotenv = lambda *a, **k: None
sys.modules["dotenv"] = _dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.join(_HERE, "..")
sys.path.insert(0, _PROJECT)

from clawgent.core import audit
from clawgent.core.audit import (
    AUDIT_SCHEMA_VERSION,
    STATUS_FAILED,
    STATUS_SUCCESS,
    AuditEvent,
    add_sink,
    bind,
    clear_sinks,
    emit,
    remove_sink,
    summarize,
    unbind,
)


class TestAuditEvent(unittest.TestCase):
    def test_event_basic_fields(self):
        ev = AuditEvent(event_type="run_start", run_id="r1")
        self.assertEqual(ev.event_type, "run_start")
        self.assertEqual(ev.run_id, "r1")
        self.assertTrue(ev.event_id)
        self.assertTrue(ev.timestamp)

    def test_to_dict_schema(self):
        d = AuditEvent(event_type="x", run_id="r1", status="SUCCESS").to_dict()
        self.assertEqual(d["schema_version"], AUDIT_SCHEMA_VERSION)
        self.assertIn("event_type", d)
        self.assertIn("run_id", d)
        self.assertIn("timestamp", d)

    def test_from_dict(self):
        d = {"event_type": "run_start", "run_id": "r1", "unknown_field": 1}
        ev = AuditEvent.from_dict(d)
        self.assertEqual(ev.run_id, "r1")
        self.assertFalse(hasattr(ev, "unknown_field"))


class TestEmit(unittest.TestCase):
    def setUp(self):
        self.captured = []
        self._sink = self.captured.append
        clear_sinks()
        add_sink(self._sink)
        bind(run_id="", thread_id="", node="", task_id="")

    def tearDown(self):
        clear_sinks()
        bind(run_id="", thread_id="", node="", task_id="")

    def test_emit_returns_event_dict(self):
        d = emit("run_start", status=STATUS_SUCCESS, run_id="r1")
        self.assertIsNotNone(d)
        self.assertEqual(d["event_type"], "run_start")
        self.assertEqual(d["run_id"], "r1")
        self.assertEqual(len(self.captured), 1)

    def test_metadata_truncation(self):
        d = emit("x", metadata={"big": "a" * 5000})
        self.assertLessEqual(len(d["metadata"]["big"]), audit.MAX_METADATA_VALUE_CHARS + 20)

    def test_error_sanitization(self):
        d = emit("x", error=ValueError("boom" * 100))
        self.assertEqual(d["error"]["type"], "ValueError")
        self.assertLessEqual(len(d["error"]["message"]), audit.MAX_ERROR_CHARS)

    def test_emit_with_context(self):
        token = bind(run_id="r9", thread_id="t9", node="planner")
        d = emit("node_end", status=STATUS_SUCCESS)
        self.assertEqual(d["run_id"], "r9")
        self.assertEqual(d["node"], "planner")
        unbind(token)

    def test_sink_failure_isolated(self):
        def bad_sink(event):
            raise RuntimeError("sink boom")
        clear_sinks()
        add_sink(bad_sink)
        add_sink(self._sink)
        # bad_sink 抛异常不应影响 good_sink 也不应让 emit 抛
        d = emit("run_start", status=STATUS_SUCCESS, run_id="r1")
        self.assertIsNotNone(d)
        self.assertEqual(len(self.captured), 1)
        clear_sinks()


class TestSummarize(unittest.TestCase):
    def _run_events(self):
        evs = []
        evs.append(AuditEvent("run_start", run_id="r1", status="SUCCESS",
                              metadata={"query": "q"}).to_dict())
        evs.append(AuditEvent("task_created", run_id="r1", task_id="t1",
                              metadata={"task_type": "BACKGROUND"}).to_dict())
        evs.append(AuditEvent("task_created", run_id="r1", task_id="t2",
                              metadata={"task_type": "COMPARISON"}).to_dict())
        evs.append(AuditEvent("task_started", run_id="r1", task_id="t1").to_dict())
        evs.append(AuditEvent("task_completed", run_id="r1", task_id="t1").to_dict())
        evs.append(AuditEvent("task_failed", run_id="r1", task_id="t2").to_dict())
        evs.append(AuditEvent("task_reopened", run_id="r1", task_id="t2").to_dict())
        evs.append(AuditEvent("retry", run_id="r1", status="FAILED",
                              metadata={"method": "planner"}).to_dict())
        evs.append(AuditEvent("fallback", run_id="r1", status="DEGRADED",
                              metadata={"method": "planner"}).to_dict())
        evs.append(AuditEvent("circuit_breaker", run_id="r1", status="DEGRADED",
                              metadata={"method": "planner"}).to_dict())
        evs.append(AuditEvent("aggregation", run_id="r1", status="SUCCESS", metadata={
            "source_count": 3, "evidence_count": 5,
            "claim_by_status": {"SUPPORTED": 2, "PARTIALLY_SUPPORTED": 1,
                                "UNSUPPORTED": 1, "CONTRADICTED": 1}}).to_dict())
        evs.append(AuditEvent("review_issue", run_id="r1", status="SUCCESS",
                              metadata={"issue_count": 2}).to_dict())
        evs.append(AuditEvent("repair", run_id="r1", status="SUCCESS",
                              metadata={"added_tasks": 1}).to_dict())
        evs.append(AuditEvent("judge_decision", run_id="r1", status="SUCCESS",
                              metadata={"decision": "COMPILE", "reason": "done"}).to_dict())
        evs.append(AuditEvent("evidence_registered", run_id="r1", task_id="t1",
                              metadata={"evidence_id": "e1"}).to_dict())
        evs.append(AuditEvent("run_end", run_id="r1", status="SUCCESS",
                              metadata={"verdict": "COMPILE"}).to_dict())
        return evs

    def test_summary_counts(self):
        s = summarize(self._run_events())
        self.assertEqual(s["final_status"], "SUCCESS")
        self.assertEqual(s["total_tasks"], 2)
        self.assertEqual(s["completed_tasks"], 1)
        self.assertEqual(s["failed_tasks"], 1)
        self.assertEqual(s["reopened_tasks"], 1)
        self.assertEqual(s["total_retries"], 1)
        self.assertEqual(s["total_fallbacks"], 1)
        self.assertEqual(s["circuit_breaker_events"], 1)
        self.assertEqual(s["source_count"], 3)
        self.assertEqual(s["evidence_count"], 5)
        self.assertEqual(s["supported_claims"], 2)
        self.assertEqual(s["partial_claims"], 1)
        self.assertEqual(s["unsupported_claims"], 1)
        self.assertEqual(s["contradicted_claims"], 1)
        self.assertEqual(s["review_issue_count"], 2)
        self.assertEqual(s["repair_rounds"], 1)
        self.assertEqual(s["verdict"], "COMPILE")

    def test_summary_abnormal_termination(self):
        evs = self._run_events()[:2]  # 只有 run_start + task_created，无 run_end
        s = summarize(evs)
        self.assertIn(s["final_status"], ("RUNNING", "FAILED", "UNKNOWN"))

    def test_summary_failed_node(self):
        # 有 node_end FAILED 且无 run_end → FAILED
        evs = [
            AuditEvent("run_start", run_id="r1", status="SUCCESS").to_dict(),
            AuditEvent("node_end", run_id="r1", status=STATUS_FAILED).to_dict(),
        ]
        s = summarize(evs)
        self.assertEqual(s["final_status"], "FAILED")

    def test_summary_empty(self):
        s = summarize([])
        self.assertEqual(s["total_duration"], 0)
        self.assertEqual(s["final_status"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
