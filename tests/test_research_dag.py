"""Research Task DAG / Evidence / Claim / Citation 的确定性逻辑测试。

只覆盖不调用模型的部分：任务调度、来源登记、引文校验、支撑状态推导、引用校验。
模型相关节点（planner/researcher/review/compiler）不在此测试范围内。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clawgent.core.research import dag as dag_mod
from clawgent.core.research import evidence as ev_mod
from clawgent.core.research import claim as claim_mod
from clawgent.core.research import citation as cit_mod
from clawgent.core.research import ledger as ledger_mod
from clawgent.core.research import state as state_mod


class TestTaskDAG(unittest.TestCase):

    def _dag(self):
        return dag_mod.TaskDAG([
            {"task_id": "t1", "question": "背景"},
            {"task_id": "t2", "question": "对比", "dependencies": ["t1"]},
            {"task_id": "t3", "question": "风险"},
        ])

    def test_only_dependency_free_tasks_are_ready(self):
        d = self._dag()
        ready = d.ready_tasks()
        self.assertEqual({t.task_id for t in ready}, {"t1", "t3"})

    def test_downstream_unlocks_after_dependency_done(self):
        d = self._dag()
        d.mark_running(["t1", "t3"])
        d.mark_done(["t1", "t3"])
        ready = d.ready_tasks()
        self.assertEqual({t.task_id for t in ready}, {"t2"})

    def test_add_task_with_missing_dependency_rejected(self):
        d = self._dag()
        ok, reason = d.add_task(dag_mod.ResearchTask(task_id="t9", dependencies=["nope"]))
        self.assertFalse(ok)
        self.assertIn("依赖不存在", reason)
        self.assertFalse(d.has("t9"))

    def test_cycle_rejected(self):
        d = dag_mod.TaskDAG([
            {"task_id": "a"},
            {"task_id": "b", "dependencies": ["a"]},
        ])
        # c 依赖 b，再把 a 的依赖改成 c，形成 a -> c -> b -> a
        ok, _ = d.add_task(dag_mod.ResearchTask(task_id="c", dependencies=["b"]))
        self.assertTrue(ok)
        d.tasks["a"].dependencies = ["c"]
        self.assertTrue(d._has_cycle())

    def test_add_task_rejects_duplicate_id(self):
        d = self._dag()
        ok, reason = d.add_task(dag_mod.ResearchTask(task_id="t1"))
        self.assertFalse(ok)
        self.assertIn("重复", reason)

    def test_reopen_invalidates_downstream(self):
        d = self._dag()
        d.mark_done(["t1", "t2"])
        self.assertTrue(d.reopen("t1", reason="证据不足"))
        self.assertEqual(d.get("t1").status, dag_mod.REOPENED)
        self.assertEqual(d.get("t2").status, dag_mod.REOPENED)
        self.assertEqual(d.get("t1").reopen_count, 1)

    def test_blocked_when_dependency_failed(self):
        d = self._dag()
        d.mark_failed(["t1"])
        d.refresh()
        # t2 依赖 t1，t1 失败后 t2 被卡住；不依赖 t1 的 t3 不受影响
        self.assertEqual(d.get("t2").status, dag_mod.BLOCKED)
        self.assertEqual({t.task_id for t in d.ready_tasks()}, {"t3"})
        self.assertFalse(d.is_complete())


class TestSourceRegistry(unittest.TestCase):

    def test_same_url_converges_to_one_source_id(self):
        reg = ev_mod.SourceRegistry()
        s1 = reg.upsert(ev_mod.Source(source_id="", url="https://a.com/x", title="A"))
        s2 = reg.upsert(ev_mod.Source(source_id="", url="https://a.com/x", title="A full"))
        self.assertEqual(s1.source_id, s2.source_id)
        self.assertEqual(len(reg.to_dicts()), 1)

    def test_empty_url_falls_back_to_title(self):
        reg = ev_mod.SourceRegistry()
        a = reg.upsert(ev_mod.Source(source_id="", title="无链接文档", source_type="local_kb"))
        b = reg.upsert(ev_mod.Source(source_id="", title="无链接文档", source_type="local_kb"))
        self.assertEqual(a.source_id, b.source_id)


class TestEvidenceVerifier(unittest.TestCase):

    def setUp(self):
        self.reg = ev_mod.SourceRegistry()
        self.src = self.reg.upsert(ev_mod.Source(source_id="", url="https://a.com/x"))
        self.text = "实验组准确率达到 92.3%，对照组为 81.5%，差异具有统计学意义。"

    def _ev(self, quote, locator=""):
        return ev_mod.Evidence(
            evidence_id=ev_mod.make_evidence_id(self.src.source_id, quote, locator),
            source_id=self.src.source_id,
            quote=quote,
            locator=locator,
        )

    def test_exact_quote_verified(self):
        v = ev_mod.EvidenceVerifier(self.reg)
        ev = v.verify(self._ev("实验组准确率达到 92.3%，对照组为 81.5%"), self.text)
        self.assertEqual(ev.verification_status, ev_mod.VERIFIED)

    def test_paraphrase_invalid(self):
        v = ev_mod.EvidenceVerifier(self.reg)
        ev = v.verify(self._ev("模型效果提升了大约 50%，远超预期表现"), self.text)
        self.assertEqual(ev.verification_status, ev_mod.INVALID)

    def test_unregistered_source_invalid(self):
        v = ev_mod.EvidenceVerifier(self.reg)
        ev = ev_mod.Evidence(evidence_id="E1", source_id="S999", quote="一段足够长的引用文本")
        self.assertEqual(v.check(ev, self.text)[0], ev_mod.INVALID)

    def test_missing_source_text_unverified(self):
        v = ev_mod.EvidenceVerifier(self.reg)
        ev = v.verify(self._ev("实验组准确率达到 92.3%"), "")
        self.assertEqual(ev.verification_status, ev_mod.UNVERIFIED)

    def test_locator_mismatch_invalid(self):
        v = ev_mod.EvidenceVerifier(self.reg)
        ev = self._ev("实验组准确率达到 92.3%", locator="chunk:c1")
        ev.chunk_id = "c2"
        self.assertEqual(v.check(ev, self.text)[0], ev_mod.INVALID)


class TestClaimGraph(unittest.TestCase):

    def test_status_derived_from_evidence(self):
        graph = claim_mod.ClaimGraph()
        c = claim_mod.Claim(claim_id="C1", text="A 方法优于 B 方法")
        graph.add_claim(c)
        graph.link_evidence("E1", "C1")
        graph.link_evidence("E2", "C1")
        index = {
            "E1": {"verification_status": "VERIFIED"},
            "E2": {"verification_status": "INVALID"},
        }
        graph.recompute_statuses(index)
        self.assertEqual(c.status, claim_mod.PARTIALLY_SUPPORTED)

    def test_all_verified_is_supported(self):
        graph = claim_mod.ClaimGraph()
        c = claim_mod.Claim(claim_id="C1", text="结论")
        graph.add_claim(c)
        graph.link_evidence("E1", "C1")
        graph.recompute_statuses({"E1": {"verification_status": "VERIFIED"}})
        self.assertEqual(c.status, claim_mod.SUPPORTED)

    def test_contradiction_wins(self):
        graph = claim_mod.ClaimGraph()
        c = claim_mod.Claim(claim_id="C1", text="结论")
        graph.add_claim(c)
        graph.link_evidence("E1", "C1")
        graph.link_evidence("E2", "C1", relation=claim_mod.CONTRADICTS)
        graph.recompute_statuses({"E1": {"verification_status": "VERIFIED"},
                                  "E2": {"verification_status": "VERIFIED"}})
        self.assertEqual(c.status, claim_mod.CONTRADICTED)

    def test_claim_id_stable_across_branches(self):
        self.assertEqual(claim_mod.make_claim_id("A 方法优于 B 方法"),
                         claim_mod.make_claim_id("A 方法优于 B 方法"))

    def test_same_claim_merges_evidence_ids(self):
        graph = claim_mod.ClaimGraph()
        a = claim_mod.Claim(claim_id="C1", text="结论", evidence_ids=["E1"])
        b = claim_mod.Claim(claim_id="C1", text="结论", evidence_ids=["E2"])
        graph.add_claim(a)
        graph.add_claim(b)
        self.assertEqual(sorted(graph.get_claim("C1").evidence_ids), ["E1", "E2"])

    def test_relation_to_unknown_claim_rejected(self):
        graph = claim_mod.ClaimGraph()
        ok, reason = graph.link_evidence("E1", "C404")
        self.assertFalse(ok)


class TestCitation(unittest.TestCase):

    def setUp(self):
        self.sources = [{"source_id": "S1", "title": "论文 A", "url": "https://a.com"}]
        self.evidences = [
            {"evidence_id": "E1", "source_id": "S1", "quote": "准确率 92.3%",
             "verification_status": "VERIFIED"},
            {"evidence_id": "E2", "source_id": "S1", "quote": "编造的内容",
             "verification_status": "INVALID"},
        ]
        self.catalog = cit_mod.SourceCatalog(self.sources, self.evidences)

    def test_known_marker_resolves(self):
        result = cit_mod.CitationVerifier(self.catalog).verify("结论 [S1] 与 [E1]。")
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["used_source_ids"], ["S1"])

    def test_unknown_marker_reported(self):
        result = cit_mod.CitationVerifier(self.catalog).verify("结论 [S9]。")
        self.assertEqual(result["issues"][0]["issue_type"], cit_mod.UNKNOWN_REF)

    def test_invalid_evidence_citation_reported(self):
        result = cit_mod.CitationVerifier(self.catalog).verify("结论 [E2]。")
        self.assertEqual(result["issues"][0]["issue_type"], cit_mod.UNVERIFIED_EVIDENCE)

    def test_uncited_claim_reported(self):
        claims = [{"claim_id": "C1", "text": "某结论", "status": "SUPPORTED",
                   "evidence_ids": ["E1"]}]
        result = cit_mod.CitationVerifier(self.catalog).verify("什么都没引用。", claims)
        self.assertEqual(result["issues"][0]["issue_type"], cit_mod.UNCITED_CLAIM)

    def test_references_only_render_used_sources(self):
        catalog = cit_mod.SourceCatalog(
            self.sources + [{"source_id": "S2", "title": "论文 B", "url": "https://b.com"}],
            self.evidences,
        )
        text = catalog.render_references(used_only=True, used_source_ids=["S1"])
        self.assertIn("论文 A", text)
        self.assertNotIn("论文 B", text)


class TestStateReducers(unittest.TestCase):

    def test_sources_merged_by_id(self):
        left = [{"source_id": "S1", "title": "A"}]
        right = [{"source_id": "S1", "url": "https://a"}, {"source_id": "S2", "title": "B"}]
        merged = state_mod.merge_sources(left, right)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["title"], "A")
        self.assertEqual(merged[0]["url"], "https://a")

    def test_claims_union_evidence_ids(self):
        left = [{"claim_id": "C1", "evidence_ids": ["E1"]}]
        right = [{"claim_id": "C1", "evidence_ids": ["E2"]}]
        merged = state_mod.merge_claims(left, right)
        self.assertEqual(sorted(merged[0]["evidence_ids"]), ["E1", "E2"])

    def test_tasks_overwritten_by_task_id(self):
        left = [{"task_id": "t1", "status": "PENDING"}]
        right = [{"task_id": "t1", "status": "DONE"}]
        merged = state_mod.merge_tasks(left, right)
        self.assertEqual(merged[0]["status"], "DONE")

    def test_relations_deduped(self):
        rel = {"source_id": "E1", "target_id": "C1", "relation": "SUPPORTS"}
        merged = state_mod.merge_relations([rel], [dict(rel)])
        self.assertEqual(len(merged), 1)

    def test_str_list_deduped(self):
        merged = state_mod.merge_str_list(["a", "b"], ["b", "c"])
        self.assertEqual(merged, ["a", "b", "c"])

    def test_append_list_keeps_all(self):
        merged = state_mod.append_list([{"seq": 1}], [{"seq": 2}])
        self.assertEqual(len(merged), 2)


class TestLedger(unittest.TestCase):

    def test_events_numbered_by_time(self):
        led = ledger_mod.ResearchLedger()
        led.append(ledger_mod.PLAN_CREATED, task_count=3)
        led.append(ledger_mod.TASK_DISPATCHED, task_id="t1")
        self.assertEqual([e.seq for e in led.events], [1, 2])
        self.assertEqual(led.by_type(ledger_mod.TASK_DISPATCHED)[0].task_id, "t1")
        self.assertEqual(led.summary()["total_events"], 2)

    def test_trace_back_to_retrieval_query(self):
        trace = ledger_mod.build_trace(
            claims=[{"claim_id": "C1", "text": "结论", "evidence_ids": ["E1"]}],
            evidences=[{"evidence_id": "E1", "source_id": "S1", "quote": "原文",
                        "task_id": "t1", "retrieval_query": "某 query",
                        "verification_status": "VERIFIED"}],
            sources=[{"source_id": "S1", "title": "A", "url": "https://a"}],
            tasks=[{"task_id": "t1", "question": "问题"}],
        )
        step = trace["chains"][0]["evidence_chain"][0]
        self.assertEqual(step["source_id"], "S1")
        self.assertEqual(step["retrieval_query"], "某 query")
        self.assertEqual(step["task_question"], "问题")


if __name__ == "__main__":
    unittest.main()
