"""Research Agent 核心闭环补全：14 个确定性场景测试。

只覆盖不调用模型、不依赖网络的纯逻辑：
DAG 调度 / 环检测 / 依赖失败、Evidence 原文定位校验（伪造/空源）、
Claim 支撑状态推导、ConflictDetector（直接冲突/范围不一致）、
Revision 任务模型、CitationBinder 闭环、CitationVerifier（断引/假源）、
未支撑声明检测器、ResearchPlan 往返。

为保证"离线 + 确定性"可跑，优先尝试从正式包导入；
若环境缺少 langgraph（research/__init__ 会触发 graph 导入），
则改用合成包加载纯逻辑子模块，绕开 graph 依赖。
"""

import importlib.util
import os
import sys
import types
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESEARCH = os.path.normpath(os.path.join(_HERE, "..", "clawgent", "core", "research"))


def _load_pure():
    """以合成包 rc0 加载纯逻辑子模块，绕开对 langgraph 的导入。"""
    pkg = "rc0"
    sys.modules[pkg] = types.ModuleType(pkg)
    sys.modules[pkg].__path__ = []

    def load(name, deps=None):
        deps = deps or []
        for d in deps:
            load(d)
        full = f"{pkg}.{name}"
        if full in sys.modules:
            return sys.modules[full]
        spec = importlib.util.spec_from_file_location(full, os.path.join(_RESEARCH, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = pkg
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    return (
        load("dag"),
        load("claim"),
        load("evidence"),
        load("conflict"),
        load("review", deps=["claim", "conflict"]),
        load("citation", deps=["claim"]),
        load("ledger"),
    )


try:
    from clawgent.core.research import dag as dag_mod
    from clawgent.core.research import claim as claim_mod
    from clawgent.core.research import evidence as ev_mod
    from clawgent.core.research import conflict as conflict_mod
    from clawgent.core.research import review as review_mod
    from clawgent.core.research import citation as cit_mod
    from clawgent.core.research import ledger as ledger_mod
except Exception:  # 缺少 langgraph 时退回合成加载
    dag_mod, claim_mod, ev_mod, conflict_mod, review_mod, cit_mod, ledger_mod = _load_pure()


class TestDagScheduling(unittest.TestCase):
    """场景 1：正常依赖调度。"""

    def test_normal_dispatch(self):
        d = dag_mod.TaskDAG()
        d.add_task(dag_mod.ResearchTask("t1"))
        d.add_task(dag_mod.ResearchTask("t2", dependencies=["t1"]))
        self.assertEqual([t.task_id for t in d.ready_tasks()], ["t1"])
        d.mark_done(["t1"])
        self.assertEqual([t.task_id for t in d.ready_tasks()], ["t2"])


class TestDagCycle(unittest.TestCase):
    """场景 2：环检测（新增后产生环被拒绝）。"""

    def test_add_task_rejects_cycle(self):
        d = dag_mod.TaskDAG()
        # 预置 t3 依赖 t1，再新增 t1 依赖 t3 → 闭环
        d.tasks["t3"] = dag_mod.ResearchTask("t3", dependencies=["t1"])
        ok, reason = d.add_task(dag_mod.ResearchTask("t1", dependencies=["t3"]))
        self.assertFalse(ok)
        self.assertIn("环", reason)

    def test_has_cycle_detects_injected_cycle(self):
        d = dag_mod.TaskDAG()
        d.tasks["t1"] = dag_mod.ResearchTask("t1", dependencies=["t3"])
        d.tasks["t2"] = dag_mod.ResearchTask("t2", dependencies=["t1"])
        d.tasks["t3"] = dag_mod.ResearchTask("t3", dependencies=["t2"])
        self.assertTrue(d._has_cycle())


class TestDagDepFail(unittest.TestCase):
    """场景 3：依赖失败导致下游永久 BLOCKED。"""

    def test_failed_dep_blocks_downstream(self):
        d = dag_mod.TaskDAG()
        d.add_task(dag_mod.ResearchTask("t1"))
        d.add_task(dag_mod.ResearchTask("t2", dependencies=["t1"]))
        d.mark_failed(["t1"])
        self.assertEqual(d.ready_tasks(), [])
        self.assertEqual(d.get("t2").status, dag_mod.BLOCKED)


class TestEvidenceForgery(unittest.TestCase):
    """场景 4：证据伪造（quote 不在来源原文中 → INVALID）。"""

    def test_fake_quote_invalid(self):
        reg = ev_mod.SourceRegistry()
        src = reg.upsert(ev_mod.Source(source_id="", source_type="web",
                                       title="T", url="http://x.example/1"))
        text = "The model achieved 95% accuracy on the benchmark dataset."
        ev = ev_mod.Evidence(evidence_id="E1", source_id=src.source_id,
                             quote="一段从未出现在原文里的引文", locator="")
        ver = ev_mod.EvidenceVerifier(reg)
        status, _ = ver.check(ev, text)
        self.assertEqual(status, ev_mod.INVALID)


class TestEmptySource(unittest.TestCase):
    """场景 5：来源原文缺失 → UNVERIFIED（无法定位）。"""

    def test_missing_source_text(self):
        reg = ev_mod.SourceRegistry()
        src = reg.upsert(ev_mod.Source(source_id="", title="T", url="http://x.example/2"))
        ev = ev_mod.Evidence(evidence_id="E2", source_id=src.source_id,
                             quote="achieved 95% accuracy", locator="")
        ver = ev_mod.EvidenceVerifier(reg)
        status, _ = ver.check(ev, "")
        self.assertEqual(status, ev_mod.UNVERIFIED)


class TestClaimSupport(unittest.TestCase):
    """场景 6：无证据 → UNSUPPORTED；存在 CONTRADICTS 关系 → CONTRADICTED。"""

    def test_no_evidence_unsupported(self):
        g = claim_mod.ClaimGraph()
        g.add_claim(claim_mod.Claim("C1", text="无支撑声明", evidence_ids=[]))
        st = g.recompute_statuses({})
        self.assertEqual(st["C1"], claim_mod.UNSUPPORTED)

    def test_contradicted_relation(self):
        g = claim_mod.ClaimGraph()
        g.add_claim(claim_mod.Claim("C1", text="x", evidence_ids=["E1"]))
        g.add_relation(claim_mod.ClaimRelation("C2", "C1", claim_mod.CONTRADICTS))
        st = g.recompute_statuses({"E1": {"verification_status": "VERIFIED"}})
        self.assertEqual(st["C1"], claim_mod.CONTRADICTED)


class TestConflictDetector(unittest.TestCase):
    """场景 7：直接冲突；场景 8：范围不一致（非直接冲突）。"""

    def test_direct_conflict(self):
        a = {"claim_id": "C1", "scope_fields": {"dataset": "D1", "metric": "acc"},
             "polarity": "positive", "value": "2.1%"}
        b = {"claim_id": "C2", "scope_fields": {"dataset": "D1", "metric": "acc"},
             "polarity": "negative", "value": "4.8%"}
        det = conflict_mod.ConflictDetector()
        self.assertEqual(det.detect(a, b, "CONTRADICTS"), conflict_mod.DIRECT_CONFLICT)
        # 同 scope、极性相反且无显式关系 → POTENTIAL_CONFLICT
        self.assertEqual(det.detect(a, b), conflict_mod.POTENTIAL_CONFLICT)

    def test_scope_mismatch(self):
        a = {"claim_id": "C1", "scope_fields": {"dataset": "D1", "metric": "acc"},
             "polarity": "positive"}
        b = {"claim_id": "C2", "scope_fields": {"dataset": "D2", "metric": "acc"},
             "polarity": "positive"}
        det = conflict_mod.ConflictDetector()
        self.assertEqual(det.detect(a, b), conflict_mod.SCOPE_MISMATCH)


class TestRevisionModel(unittest.TestCase):
    """场景 9：修订任务创建（parent_task_id / revision_round）；重开级联下游。"""

    def test_revision_creation_and_reopen(self):
        d = dag_mod.TaskDAG()
        d.add_task(dag_mod.ResearchTask("t1", question="q"))
        ok, _ = d.add_task(dag_mod.ResearchTask(
            "t1-R1", question="r", parent_task_id="t1", revision_round=1))
        self.assertTrue(ok)
        self.assertEqual(d.get("t1-R1").parent_task_id, "t1")
        self.assertEqual(d.get("t1-R1").revision_round, 1)

        d.add_task(dag_mod.ResearchTask("t2", dependencies=["t1"]))
        d.mark_done(["t1", "t2"])
        self.assertEqual(d.get("t2").status, dag_mod.COMPLETED)
        self.assertTrue(d.reopen("t1"))
        self.assertEqual(d.get("t2").status, dag_mod.REOPENED)


class TestDuplicateRevision(unittest.TestCase):
    """场景 10：重复修订任务（相同 task_id）被 DAG 拒绝。"""

    def test_duplicate_revision_rejected(self):
        d = dag_mod.TaskDAG()
        d.add_task(dag_mod.ResearchTask("t1"))
        ok1, _ = d.add_task(dag_mod.ResearchTask(
            "t1-R1", parent_task_id="t1", revision_round=1))
        ok2, reason2 = d.add_task(dag_mod.ResearchTask(
            "t1-R1", parent_task_id="t1", revision_round=1))
        self.assertTrue(ok1)
        self.assertFalse(ok2)
        self.assertIn("重复", reason2)


class TestCitationClosure(unittest.TestCase):
    """场景 11：引用闭环（草稿句子 → Claim → Evidence → Source）。"""

    def test_closure_and_binding(self):
        src = ev_mod.Source(source_id="S1", title="T", url="http://x.example/3")
        ev = ev_mod.Evidence(evidence_id="E1", source_id="S1", quote="q",
                             verification_status=ev_mod.VERIFIED)
        cat = cit_mod.SourceCatalog([src.to_dict()], [ev.to_dict()])
        claims = [{"claim_id": "C1", "status": "SUPPORTED",
                   "evidence_ids": ["E1"], "text": "x"}]
        report = "结论成立 [E1]。"
        ver = cit_mod.CitationVerifier(cat)
        res = ver.verify(report, claims)
        self.assertEqual(res["used_evidence_ids"], ["E1"])
        self.assertEqual(res["used_source_ids"], ["S1"])
        self.assertTrue(all(
            i["issue_type"] not in (cit_mod.UNKNOWN_REF, cit_mod.UNVERIFIED_EVIDENCE)
            for i in res["issues"]))

        binder = cit_mod.CitationBinder(cat, claims)
        binding = binder.bind(report)
        self.assertIn("E1", binding["cited_evidence_ids"])
        self.assertIn("C1", binding["cited_claim_ids"])


class TestBrokenCitation(unittest.TestCase):
    """场景 12：断引（引用不存在的编号 → UNKNOWN_REF）。"""

    def test_unknown_ref(self):
        ver = cit_mod.CitationVerifier(cit_mod.SourceCatalog([], []))
        res = ver.verify("结论 [E9] 不存在。", [])
        self.assertTrue(any(i["issue_type"] == cit_mod.UNKNOWN_REF for i in res["issues"]))


class TestFakeUrl(unittest.TestCase):
    """场景 13：假 URL / 未登记来源（SourceRegistry 去重 + EvidenceVerifier 拦截）。"""

    def test_source_dedup_and_unregistered_source(self):
        reg = ev_mod.SourceRegistry()
        s1 = reg.upsert(ev_mod.Source(source_id="", url="http://fake.example/paper"))
        s2 = reg.upsert(ev_mod.Source(source_id="", url="http://fake.example/paper"))
        self.assertEqual(s1.source_id, s2.source_id)  # 同一 url 去重到同一来源

        ver = ev_mod.EvidenceVerifier(reg)
        ev = ev_mod.Evidence(evidence_id="E9", source_id="S_not_real",
                             quote="x", locator="")
        status, _ = ver.check(ev, "x")
        self.assertEqual(status, ev_mod.INVALID)


class TestNoEvidenceFailure(unittest.TestCase):
    """场景 14：无证据失败（INVALID 证据 → 声明 UNSUPPORTED，不被引用）。"""

    def test_invalid_evidence_unsupported(self):
        g = claim_mod.ClaimGraph()
        g.add_claim(claim_mod.Claim("C1", text="x", evidence_ids=["E1"]))
        st = g.recompute_statuses({"E1": {"verification_status": ev_mod.INVALID}})
        self.assertEqual(st["C1"], claim_mod.UNSUPPORTED)

    def test_failed_task_makes_dag_complete_but_unfinished(self):
        d = dag_mod.TaskDAG()
        d.add_task(dag_mod.ResearchTask("t1", status=dag_mod.FAILED))
        self.assertTrue(d.is_complete())
        self.assertEqual(d.ready_tasks(), [])


class TestUnsupportedClaimDetector(unittest.TestCase):
    """§48 未支撑声明检测器：正文中的 UNSUPPORTED 声明被移除并单列。"""

    def test_detect_and_sanitize(self):
        claims = [
            {"claim_id": "C1", "text": "已支撑的结论", "status": "SUPPORTED"},
            {"claim_id": "C2", "text": "未支撑的危险结论", "status": "UNSUPPORTED"},
        ]
        det = cit_mod.UnsupportedClaimDetector(claims)
        report = ("已支撑的结论正确。\n"
                  "未支撑的危险结论也是正确的。\n"
                  "## 本报告局限\n未支撑的危险结论未被采用。")
        violations, sanitized = det.detect_and_sanitize(report)
        self.assertEqual(len(violations), 1)
        self.assertNotIn("未支撑的危险结论也是正确的", sanitized)  # 正文中被移除
        self.assertIn("本报告局限", sanitized)  # 豁免小节保留


class TestResearchPlan(unittest.TestCase):
    """配套：ResearchPlan 往返（§4.1）。"""

    def test_plan_roundtrip(self):
        plan = dag_mod.ResearchPlan(
            plan_id="P1", objective="o", constraints="c",
            tasks=[dag_mod.ResearchTask("t1")])
        d = plan.to_dict()
        back = dag_mod.ResearchPlan.from_dict(d)
        self.assertEqual(back.plan_id, "P1")
        self.assertEqual([t.task_id for t in back.tasks], ["t1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
