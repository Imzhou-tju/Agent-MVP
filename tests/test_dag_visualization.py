# -*- coding: utf-8 -*-
"""DAG 拓扑节点可视化特性测试（TDD 测试先行，100% 离线打桩）。

严禁真实网络与外部 API 调用。覆盖：
1. 正常依赖图转换 (nodes & edges)
2. 状态跃迁准确性 (COMPLETED / RUNNING / REOPENED)
3. 孤立节点与空数据边界防崩
4. 悬空依赖容错防御
5. HTTP API 路由 /api/runs/<run_id>/dag 响应契约
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
    return json.dumps(kw, ensure_ascii=False)


class TestDAGVisualization(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_idx.sqlite")
        self._prepare_sample_logs()
        self.indexer = AuditIndexer(log_dir=self.temp_dir, db_path=self.db_path)
        self.indexer.refresh()

    def tearDown(self):
        self.indexer.close()

    def _prepare_sample_logs(self):
        # 构造包含完整依赖与状态机跃迁的事件流
        events_r1 = [
            {"event_type": "run_start", "run_id": "r1", "thread_id": "th1", "status": "SUCCESS", "timestamp": "2026-01-01T10:00:00Z"},
            {"event_type": "task_created", "run_id": "r1", "thread_id": "th1", "task_id": "t1", "status": "SUCCESS",
             "metadata": {"task_type": "MECHANISM", "dependencies": [], "priority": 1}},
            {"event_type": "task_created", "run_id": "r1", "thread_id": "th1", "task_id": "t2", "status": "SUCCESS",
             "metadata": {"task_type": "COMPARISON", "dependencies": ["t1"], "priority": 2}},
            {"event_type": "task_created", "run_id": "r1", "thread_id": "th1", "task_id": "t3", "status": "SUCCESS",
             "metadata": {"task_type": "SURVEY", "dependencies": ["t1", "t2"], "priority": 3}},
            # 状态演进
            {"event_type": "task_started", "run_id": "r1", "thread_id": "th1", "task_id": "t1", "duration_ms": 150},
            {"event_type": "task_completed", "run_id": "r1", "thread_id": "th1", "task_id": "t1"},
            {"event_type": "task_started", "run_id": "r1", "thread_id": "th1", "task_id": "t2", "duration_ms": 200},
            {"event_type": "task_reopened", "run_id": "r1", "thread_id": "th1", "task_id": "t2"}, # 级联重置
            {"event_type": "run_end", "run_id": "r1", "thread_id": "th1", "status": "SUCCESS", "timestamp": "2026-01-01T10:00:10Z"}
        ]
        
        # 构造空数据 run
        events_empty = [
            {"event_type": "run_start", "run_id": "r_empty", "thread_id": "th2", "status": "SUCCESS", "timestamp": "2026-01-01T11:00:00Z"},
            {"event_type": "run_end", "run_id": "r_empty", "thread_id": "th2", "status": "SUCCESS", "timestamp": "2026-01-01T11:00:05Z"}
        ]

        # 构造含悬空无效依赖的 run
        events_dangling = [
            {"event_type": "run_start", "run_id": "r_dangling", "thread_id": "th3", "status": "SUCCESS", "timestamp": "2026-01-01T12:00:00Z"},
            {"event_type": "task_created", "run_id": "r_dangling", "thread_id": "th3", "task_id": "t_solo", "status": "SUCCESS",
             "metadata": {"dependencies": ["non_existent_node"]}},
            {"event_type": "run_end", "run_id": "r_dangling", "thread_id": "th3", "status": "SUCCESS", "timestamp": "2026-01-01T12:00:05Z"}
        ]

        with open(os.path.join(self.temp_dir, "th1.jsonl"), "w", encoding="utf-8") as f:
            for e in events_r1:
                f.write(_line(**e) + "\n")

        with open(os.path.join(self.temp_dir, "th2.jsonl"), "w", encoding="utf-8") as f:
            for e in events_empty:
                f.write(_line(**e) + "\n")

        with open(os.path.join(self.temp_dir, "th3.jsonl"), "w", encoding="utf-8") as f:
            for e in events_dangling:
                f.write(_line(**e) + "\n")

    def test_dag_structure_nodes_and_edges(self):
        """测试正常 DAG 是否能正确输出标准 nodes 和 edges。"""
        dag = self.indexer.task_dag_graph("r1")
        self.assertIn("nodes", dag)
        self.assertIn("edges", dag)
        
        node_ids = {n["id"] for n in dag["nodes"]}
        self.assertEqual(node_ids, {"t1", "t2", "t3"})
        
        # 验证边集合: t1 -> t2, t1 -> t3, t2 -> t3
        edges = [(e["from"], e["to"]) for e in dag["edges"]]
        self.assertIn(("t1", "t2"), edges)
        self.assertIn(("t1", "t3"), edges)
        self.assertIn(("t2", "t3"), edges)
        self.assertEqual(len(edges), 3)

    def test_dag_node_states_and_reopen(self):
        """测试状态机跃迁，验证 REOPENED 警报状态是否准确捕捉。"""
        dag = self.indexer.task_dag_graph("r1")
        nodes_by_id = {n["id"]: n for n in dag["nodes"]}
        
        self.assertEqual(nodes_by_id["t1"]["status"], "COMPLETED")
        self.assertEqual(nodes_by_id["t2"]["status"], "REOPENED")
        self.assertEqual(nodes_by_id["t3"]["status"], "PENDING")

    def test_empty_dag_boundary(self):
        """测试边界：无任何任务的 Run 必须安全返回空结构，严禁抛异常。"""
        dag = self.indexer.task_dag_graph("r_empty")
        self.assertEqual(dag["nodes"], [])
        self.assertEqual(dag["edges"], [])
        self.assertEqual(dag["summary"]["total_nodes"], 0)

    def test_dangling_dependency_resilience(self):
        """测试容错红线：悬空无效依赖不能生成死锁或断裂边，应保留实体节点。"""
        dag = self.indexer.task_dag_graph("r_dangling")
        self.assertEqual(len(dag["nodes"]), 1)
        self.assertEqual(dag["nodes"][0]["id"], "t_solo")
        # 悬空边被安全过滤或忽略
        self.assertEqual(dag["edges"], [])

    def test_dag_api_route_handler(self):
        """测试 DashboardHandler 的 /api/runs/<run_id>/dag 路由逻辑。"""
        from clawgent.dashboard.server import DashboardHandler
        DashboardHandler.indexer = self.indexer
        
        # 模拟内部派发
        class MockHandler(DashboardHandler):
            def __init__(self, path):
                self.path = path
                self.sent_code = None
                self.sent_data = None
            def _send(self, code, data, content_type="application/json"):
                self.sent_code = code
                self.sent_data = data

        handler = MockHandler("/api/runs/r1/dag")
        handler._route()
        self.assertEqual(handler.sent_code, 200)
        self.assertIn("nodes", handler.sent_data)
        self.assertIn("edges", handler.sent_data)
        self.assertEqual(len(handler.sent_data["nodes"]), 3)


if __name__ == "__main__":
    unittest.main()
