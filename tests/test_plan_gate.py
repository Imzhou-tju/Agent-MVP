"""Plan Gate（Planner 质量闸门）离线测试。

只测试确定性组件（SOP / SOPSelector / PlanValidator / PlanGate），
Critic 用 Fake 实现注入，全程不调用真实 OpenAI / SiliconFlow / Tavily / MCP / Arxiv。

加载方式：research 包的 __init__.py 会触发 langgraph，故用 importlib 直接加载
纯模块文件（evidence / dag / sop / plan_validation），绕过 graph 依赖。
"""

import importlib.util
import os
import sys
import unittest
from types import ModuleType


def _load(name, path, parent=None):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if parent is not None:
        setattr(parent, name.rsplit(".", 1)[-1], mod)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_HERE = os.path.dirname(os.path.abspath(__file__))
_RESEARCH = os.path.join(_HERE, "..", "clawgent", "core", "research")

# 构造假父包 clawgent.core.research（避免触发 __init__.py → langgraph）
_pkg = ModuleType("clawgent.core.research")
_pkg.__path__ = [_RESEARCH]
sys.modules["clawgent.core.research"] = _pkg
# 假父包 clawgent.core
_core = ModuleType("clawgent.core")
_core.__path__ = [os.path.join(_HERE, "..", "clawgent", "core")]
sys.modules["clawgent.core"] = _core
# 假父包 clawgent
_clawgent = ModuleType("clawgent")
_clawgent.__path__ = [os.path.join(_HERE, "..", "clawgent")]
sys.modules["clawgent"] = _clawgent
_clawgent.core = _core
_core.research = _pkg

evidence = _load("clawgent.core.research.evidence",
                 os.path.join(_RESEARCH, "evidence.py"), _pkg)
dag = _load("clawgent.core.research.dag",
            os.path.join(_RESEARCH, "dag.py"), _pkg)
sop = _load("clawgent.core.research.sop",
            os.path.join(_RESEARCH, "sop.py"), _pkg)
pv = _load("clawgent.core.research.plan_validation",
           os.path.join(_RESEARCH, "plan_validation.py"), _pkg)

ResearchTask = dag.ResearchTask
ResearchPlan = dag.ResearchPlan
TaskDAG = dag.TaskDAG
PlanValidator = pv.PlanValidator
PlanGate = pv.PlanGate
PlanIssue = pv.PlanIssue
GATE_ALLOW = pv.GATE_ALLOW
GATE_ALLOW_WITH_WARNINGS = pv.GATE_ALLOW_WITH_WARNINGS
GATE_REJECT = pv.GATE_REJECT
build_sop = sop.build_sop
select_sop = sop.select_sop


def _task(tid, q="", tt="FACT", deps=None, caps=None):
    return ResearchTask(task_id=tid, objective=q, question=q, task_type=tt,
                        dependencies=deps or [], capabilities=caps or [])


def _plan(tasks, required=None):
    return ResearchPlan(plan_id="P1", objective="q", tasks=tasks,
                        required_capabilities=required or [])


class _FakeCritic:
    """可编程 Critic：不调用 LLM，记录调用次数，可指定抛异常或返回 issues。"""

    def __init__(self, issues=None, raise_error=False):
        self.issues = issues or []
        self.raise_error = raise_error
        self.calls = 0

    def critique(self, query, plan):
        self.calls += 1
        if self.raise_error:
            raise RuntimeError("fake critic failure")
        return list(self.issues)


class TestSOPSelector(unittest.TestCase):

    def test_comparison_keyword(self):
        self.assertEqual(select_sop("对比 A 和 B 哪个更好").sop_type, "COMPARISON")

    def test_mechanism_keyword(self):
        self.assertEqual(select_sop("这个机制如何工作").sop_type, "MECHANISM")

    def test_survey_keyword(self):
        self.assertEqual(select_sop("请综述该领域研究现状").sop_type, "SURVEY")

    def test_default_fact(self):
        self.assertEqual(select_sop("什么是 x-vector").sop_type, "FACT")

    def test_selector_no_llm(self):
        # SOPSelector 是纯关键词规则：同样的 query 永远返回同样结果，且无副作用
        a = select_sop("比较 Transformer 与 RNN")
        b = select_sop("比较 Transformer 与 RNN")
        self.assertEqual(a.sop_type, b.sop_type)


class TestSOP(unittest.TestCase):

    def test_comparison_required_dimensions(self):
        s = build_sop("COMPARISON")
        self.assertIn("object_definition", s.required_dimensions)
        self.assertIn("synthesis", s.required_dimensions)

    def test_comparison_does_not_force_ablation_benchmark(self):
        # §8.10：COMPARISON 不得强制要求 Ablation / Benchmark
        s = build_sop("COMPARISON")
        self.assertNotIn("ablation", s.required_dimensions)
        self.assertNotIn("benchmark", s.required_dimensions)
        self.assertNotIn("complexity", s.required_dimensions)
        self.assertNotIn("implementation", s.required_dimensions)

    def test_unknown_type_falls_back_to_fact(self):
        self.assertEqual(build_sop("NONEXISTENT").sop_type, "FACT")


class TestPlanValidator(unittest.TestCase):

    def test_legal_fact_no_issues(self):
        dag_obj = TaskDAG([_task("t1", "x-vector 是什么", "FACT")])
        v = PlanValidator(sop=build_sop("FACT"))
        self.assertEqual(v.validate(dag_obj, _plan(dag_obj.tasks.values())), [])

    def test_comparison_missing_coverage(self):
        # COMPARISON SOP 要求 object_definition 等维度，但任务没声明对应 capability
        dag_obj = TaskDAG([_task("t1", "对比 A B", "COMPARISON")])
        v = PlanValidator(sop=build_sop("COMPARISON"))
        issues = v.validate(dag_obj, _plan(dag_obj.tasks.values()))
        types = {i.issue_type for i in issues}
        self.assertIn(pv.ISSUE_MISSING_COVERAGE, types)

    def test_premature_synthesis(self):
        # SYNTHESIS 没有任何证据型祖先
        dag_obj = TaskDAG([_task("t1", "综合结论", "SYNTHESIS")])
        v = PlanValidator(sop=build_sop("SURVEY"))
        issues = v.validate(dag_obj, _plan(dag_obj.tasks.values()))
        self.assertTrue(any(i.issue_type == pv.ISSUE_PREMATURE_SYNTHESIS
                            and i.severity == "high" for i in issues))

    def test_comparison_with_evidence_ancestor_ok(self):
        # COMPARISON 有 FACT 祖先 → 无 INSUFFICIENT_EVIDENCE
        dag_obj = TaskDAG([
            _task("t1", "A 的事实", "FACT"),
            _task("t2", "B 的事实", "FACT"),
            _task("t3", "对比 A B", "COMPARISON", deps=["t1", "t2"]),
        ])
        v = PlanValidator(sop=build_sop("COMPARISON"))
        issues = v.validate(dag_obj, _plan(dag_obj.tasks.values()))
        self.assertFalse(any(i.issue_type == pv.ISSUE_INSUFFICIENT_EVIDENCE for i in issues))
        self.assertFalse(any(i.issue_type == pv.ISSUE_PREMATURE_SYNTHESIS for i in issues))

    def test_cycle_and_unknown_dep_and_duplicate(self):
        # 结构问题：依赖不存在 / 自依赖
        dag_obj = TaskDAG([
            _task("a", "", "FACT", deps=["nope"]),   # unknown dependency
        ])
        v = PlanValidator()
        issues = v.validate(dag_obj, _plan(dag_obj.tasks.values()))
        types = {i.issue_type for i in issues}
        self.assertIn(pv.ISSUE_MISSING_DEP, types)

        # duplicate id：TaskDAG 构造时后者覆盖前者，无法构造出真正重复。
        # 用 add_task 拒绝验证（DAG 层已拦）。
        d2 = TaskDAG([_task("x", "a")])
        ok, reason = d2.add_task(_task("x", "b"))
        self.assertFalse(ok)
        self.assertIn("重复", reason)

    def test_redundant_task(self):
        dag_obj = TaskDAG([
            _task("t1", "对比 A 与 B 的性能"),
            _task("t2", "对比 A 与 B 的性能"),
        ])
        v = PlanValidator()
        issues = v.validate(dag_obj, _plan(dag_obj.tasks.values()))
        self.assertTrue(any(i.issue_type == pv.ISSUE_REDUNDANT_TASK for i in issues))

    def test_bad_granularity(self):
        dag_obj = TaskDAG([
            _task("t1", "分析性能并且对比方案同时评估成本以及总结结论", "COMPARISON"),
        ])
        v = PlanValidator()
        issues = v.validate(dag_obj, _plan(dag_obj.tasks.values()))
        self.assertTrue(any(i.issue_type == pv.ISSUE_BAD_GRANULARITY for i in issues))


class TestPlanGate(unittest.TestCase):

    def test_legal_fact_allow(self):
        dag_obj = TaskDAG([_task("t1", "x-vector 是什么", "FACT")])
        gate = PlanGate(sop=build_sop("FACT"), critic=_FakeCritic())
        r = gate.run(dag_obj, _plan(dag_obj.tasks.values()), query="x-vector 是什么")
        self.assertEqual(r.decision, GATE_ALLOW)

    def test_legal_comparison_allow(self):
        dag_obj = TaskDAG([
            _task("t1", "A 事实", "FACT"),
            _task("t2", "B 事实", "FACT"),
            _task("t3", "对比 A B", "COMPARISON", deps=["t1", "t2"],
                  caps=["object_definition", "method_or_mechanism",
                        "evaluation_or_comparison", "evidence", "synthesis"]),
        ])
        gate = PlanGate(sop=build_sop("COMPARISON"), critic=_FakeCritic())
        r = gate.run(dag_obj, _plan(dag_obj.tasks.values()), query="对比 A B")
        self.assertEqual(r.decision, GATE_ALLOW)

    def test_repair_success_allow(self):
        # Premature synthesis：repair 补一个 FACT 证据任务并挂为上游依赖，
        # 补完无 HIGH → ALLOW。sop=None 聚焦 repair 行为，不引入 coverage 噪声。
        dag_obj = TaskDAG([_task("t1", "综合结论", "SYNTHESIS")])
        gate = PlanGate(sop=None, critic=_FakeCritic())
        r = gate.run(dag_obj, _plan(dag_obj.tasks.values()), query="综合")
        self.assertEqual(r.decision, GATE_ALLOW)
        self.assertGreater(r.repair_count, 0)
        # repair 后 SYNTHESIS 有了证据型祖先
        t1 = dag_obj.tasks["t1"]
        self.assertTrue(t1.dependencies)

    def test_repair_still_invalid_reject(self):
        # 无法修复的高危问题（依赖不存在），repair 也补不了 → REJECT
        t = _task("t1", "综合", "SYNTHESIS", deps=["ghost"])
        dag_obj = TaskDAG()
        dag_obj.tasks["t1"] = t  # 绕过 add_task 校验，模拟畸形 DAG
        gate = PlanGate(sop=None, critic=_FakeCritic())
        r = gate.run(dag_obj, _plan(dag_obj.tasks.values()), query="综合")
        self.assertEqual(r.decision, GATE_REJECT)

    def test_critic_failure_fallback(self):
        # Critic 抛异常 → 只用 validator 结果，无 HIGH → ALLOW_WITH_WARNINGS
        dag_obj = TaskDAG([_task("t1", "x-vector 是什么", "FACT")])
        gate = PlanGate(sop=build_sop("FACT"),
                        critic=_FakeCritic(raise_error=True))
        r = gate.run(dag_obj, _plan(dag_obj.tasks.values()), query="q")
        self.assertTrue(r.critic_failed)
        self.assertIn(r.decision, (GATE_ALLOW, GATE_ALLOW_WITH_WARNINGS))

    def test_critic_called_at_most_once(self):
        dag_obj = TaskDAG([_task("t1", "x-vector 是什么", "FACT")])
        fake = _FakeCritic()
        gate = PlanGate(sop=build_sop("FACT"), critic=fake)
        gate.run(dag_obj, _plan(dag_obj.tasks.values()), query="q")
        self.assertLessEqual(fake.calls, 1)

    def test_repair_does_not_rebuild_dag(self):
        # repair 只局部补任务，保留原 task_id
        dag_obj = TaskDAG([_task("t1", "综合结论", "SYNTHESIS")])
        gate = PlanGate(sop=None, critic=_FakeCritic())
        gate.run(dag_obj, _plan(dag_obj.tasks.values()), query="综合")
        self.assertIn("t1", dag_obj.tasks)  # 原任务还在
        self.assertTrue(any(tid != "t1" for tid in dag_obj.tasks))  # 只新增

    def test_critic_issues_trigger_repair_once(self):
        # Critic 返回 add_task issue → repair 补一个任务，最终无 HIGH
        dag_obj = TaskDAG([_task("t1", "x-vector 是什么", "FACT")])
        fake = _FakeCritic(issues=[PlanIssue(
            pv.ISSUE_MISSING_TASK, "缺一个维度", severity="high",
            recommended_action="add_task",
            suggested_task={"question": "补充维度", "task_type": "FACT"},
        )])
        gate = PlanGate(sop=build_sop("FACT"), critic=fake)
        r = gate.run(dag_obj, _plan(dag_obj.tasks.values()), query="q")
        self.assertEqual(r.decision, GATE_ALLOW)
        self.assertEqual(fake.calls, 1)


if __name__ == "__main__":
    unittest.main()
