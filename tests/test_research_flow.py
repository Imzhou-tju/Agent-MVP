"""Research 子图整链路的离线测试。

用假 LLM 与假检索替代真实调用，验证图接线与状态合并：
依赖驱动的分批调度、DAG 局部修补、Judge 终止、引用校验、Ledger 不重复。
不访问网络，不需要 API key。
"""

import asyncio
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clawgent.core.research import nodes as R
from clawgent.core.research.ledger import (
    ResearchLedger, TASK_ADDED, TASK_DISPATCHED,
)

SNIPPET = "实验组准确率达到 92.3%，对照组为 81.5%，差异具有统计学意义（p<0.01）。"

PLANNER_JSON = json.dumps([
    {"task_id": "t1", "objective": "背景", "question": "方法背景",
     "task_type": "FACT", "expected_evidence": "论文结论", "dependencies": []},
    {"task_id": "t2", "objective": "对比", "question": "方法对比",
     "task_type": "COMPARISON", "expected_evidence": "对比数据", "dependencies": ["t1"]},
    {"task_id": "t3", "objective": "风险", "question": "风险与局限",
     "task_type": "LIMITATION", "expected_evidence": "局限说明", "dependencies": []},
], ensure_ascii=False)

RESEARCHER_JSON = json.dumps({"claims": [
    {"text": "实验组准确率 92.3%", "claim_type": "RESULT", "scope": "该论文实验设置下",
     "conditions": "", "source_index": 1, "quote": "实验组准确率达到 92.3%",
     "interpretation": "效果优于对照"},
    {"text": "对照组准确率为 81.5%", "claim_type": "RESULT", "scope": "",
     "conditions": "", "source_index": 1, "quote": "对照组为 81.5%",
     "interpretation": "基线水平"},
]}, ensure_ascii=False)

REVIEW_JSON = json.dumps({"issues": [
    {"issue_type": "COVERAGE_GAP", "description": "缺少落地成本方面的证据",
     "severity": "high", "target_type": "global", "target_id": "",
     "suggested_action": "add_task",
     "suggested_task": {"question": "落地成本", "task_type": "FACT",
                        "expected_evidence": "成本数据",
                        "search_strategy": "成本 部署", "dependencies": []}},
]}, ensure_ascii=False)

REPORT = "## 执行摘要\n实验组准确率为 92.3% [E1]，对照组为 81.5% [E2]，参见 [S1]。"


class FakeResp:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def invoke(self, messages):
        prompt = messages[0].content
        if "科研调研规划师" in prompt:
            return FakeResp(PLANNER_JSON)
        if "研究任务:" in prompt:
            return FakeResp(RESEARCHER_JSON)
        if "研究评审员" in prompt:
            return FakeResp(REVIEW_JSON)
        if "研究报告撰写员" in prompt:
            return FakeResp(REPORT)
        return FakeResp("{}")


async def fake_hybrid_search(query, web_results=4, rag_results=2, academic_results=5):
    return [{
        "url": f"https://example.com/{abs(hash(query)) % 9973}",
        "title": f"文档-{query}",
        "snippet": SNIPPET,
        "search_query": query,
        "source": "web",
    }]


INITIAL = {
    "original_query": "某方法的效果如何",
    "research_context": "",
    "max_revisions": 2,
    "tasks": [], "round_no": 0, "task_results": [], "sources": [],
    "evidences": [], "claims": [], "relations": [], "issues": [],
    "searched_queries": [], "source_texts": {}, "ledger_events": [],
    "repaired_issue_ids": [],
}


class TestResearchFlow(unittest.TestCase):

    def setUp(self):
        self._orig_llm = R._get_llm
        self._orig_search = R.hybrid_search
        R._get_llm = lambda: FakeLLM()
        R.hybrid_search = fake_hybrid_search
        self.addCleanup(self._restore)

    def _restore(self):
        R._get_llm = self._orig_llm
        R.hybrid_search = self._orig_search

    def _run(self):
        from clawgent.core.research.graph import build_research_graph

        async def main():
            g = build_research_graph()
            return await g.ainvoke(dict(INITIAL), config={"recursion_limit": 100})

        return asyncio.run(main())

    def test_dependency_driven_dispatch_order(self):
        result = self._run()
        led = ResearchLedger(result["ledger_events"])
        dispatched = [(e.task_id, e.round_no) for e in led.by_type(TASK_DISPATCHED)]
        # t2 依赖 t1，必须排在 t1 之后；t1/t3 无依赖，首批一起执行
        self.assertEqual(dispatched[0][0], "t1")
        self.assertEqual(dispatched[1][0], "t3")
        self.assertEqual(dispatched[2][0], "t2")
        self.assertIn("t2", [t for t, _ in dispatched])

    def test_repair_adds_task_only_once(self):
        result = self._run()
        led = ResearchLedger(result["ledger_events"])
        added = [(e.task_id, e.round_no) for e in led.by_type(TASK_ADDED)]
        # 评审每轮重复提出同一个 issue，repair 只能补一次任务
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0][1], 1)

    def test_all_tasks_done_and_verdict_compile(self):
        result = self._run()
        self.assertTrue(all(t["status"] == "DONE" for t in result["tasks"]))
        self.assertEqual(result["verdict"], "COMPILE")

    def test_evidence_verified_and_claims_supported(self):
        result = self._run()
        self.assertTrue(result["evidences"])
        for e in result["evidences"]:
            self.assertEqual(e["verification_status"], "VERIFIED")
        for c in result["claims"]:
            self.assertEqual(c["status"], "SUPPORTED")
            self.assertTrue(c["evidence_ids"])

    def test_ledger_has_no_duplicated_events(self):
        result = self._run()
        led = ResearchLedger(result["ledger_events"])
        dispatched = led.by_type(TASK_DISPATCHED)
        # 每个任务只应被调度一次（全量回写会让事件随节点数成倍复制）
        self.assertEqual(len(dispatched), len({e.task_id for e in dispatched}))

    def test_report_carries_citations_and_references(self):
        result = self._run()
        self.assertEqual(result["citation_issues"], [])
        self.assertIn("[S1]", result["final_report"])
        self.assertIn("参考来源", result["final_report"])


if __name__ == "__main__":
    unittest.main()
