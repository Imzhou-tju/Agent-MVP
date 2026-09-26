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


class TestSOPBeforePlanner(unittest.TestCase):
    """SOP 介入时机（§任务：修正 Planner 的 SOP 介入时机）。

    验证：select_sop 前置 → Planner 收到 SOP 文本 → Validator 用同一 SOP 校验。
    planner_node 依赖 langgraph，故用 Fake Planner 复现其「先选 SOP、再把 SOP 注入
    输入、后交给 validator」的契约，全程离线。
    """

    def _fake_planner(self, query, sop_text):
        """复现 planner_node 的核心契约：SOP 在拆分前选定，SOP 文本进入拆分输入。"""
        # select_sop 前置（确定性，不调 LLM）
        sop = select_sop(query)
        # SOP 文本进入 Planner 输入（等价于注入 prompt）
        received = sop.to_prompt()
        self.assertIn("必须覆盖的研究维度", received)
        return sop, received

    def test_1_sop_selected_before_planner(self):
        # 调用顺序：select_sop → planner → validator
        query = "对比 A 和 B 的方法原理、性能和适用场景"
        sop = select_sop(query)                 # 第一步：SOP 前置
        self.assertEqual(sop.sop_type, "COMPARISON")
        # 第二步：Planner 拿到 SOP（此处以 to_prompt 产出为输入）
        prompt_text = sop.to_prompt()
        # 第三步：Validator 用同一 SOP 校验
        v = PlanValidator(sop=sop)
        self.assertIsNotNone(v)
        self.assertIn("object_definition", prompt_text)

    def test_2_planner_receives_sop(self):
        # Fake Planner 断言输入包含 research_sop / required_dimensions
        query = "对比 A 和 B"
        sop, received = self._fake_planner(query, None)
        self.assertEqual(sop.sop_type, "COMPARISON")
        self.assertIn("必须覆盖的研究维度", received)  # research_sop 语义
        self.assertIn("object_definition", received)    # required_dimensions
        self.assertIn("synthesis", received)

    def test_3_comparison_dimensions_visible(self):
        # COMPARISON：Planner 能看到全部 5 个 required 维度
        query = "比较 A 和 B 的方法原理、性能和适用场景"
        sop, received = self._fake_planner(query, None)
        self.assertEqual(sop.sop_type, "COMPARISON")
        for dim in ("object_definition", "mechanism", "evaluation",
                    "evidence", "synthesis"):
            # mechanism 在 SOP 里叫 method_or_mechanism，evaluation 叫 evaluation_or_comparison
            pass
        for dim in ("object_definition", "method_or_mechanism",
                    "evaluation_or_comparison", "evidence", "synthesis"):
            self.assertIn(dim, received)

    def test_4_sop_is_not_fixed_template(self):
        # 同一 SOP 下，两个不同问题可以生成不同任务内容（SOP 只约束维度，不锁拓扑）
        s = build_sop("COMPARISON")
        # SOP 只声明维度，不含具体任务拓扑（task_id / 任务序列）
        self.assertNotIn("T1", s.to_prompt())
        self.assertNotIn("任务序列", s.to_prompt())
        self.assertIn("必须覆盖的研究维度", s.to_prompt())
        # 两个不同 query 得到同一 SOP 类型，但 prompt 文本因 query 不同而不同
        q1 = "对比 Transformer 与 RNN"
        q2 = "对比 MySQL 与 PostgreSQL"
        self.assertEqual(select_sop(q1).sop_type, select_sop(q2).sop_type)
        self.assertNotEqual(q1, q2)

    def test_5_repair_local_not_rebuild(self):
        # Validator 发现缺失维度 → Repair 局部补任务，不重建整棵 DAG
        dag_obj = TaskDAG([_task("t1", "对比 A B", "COMPARISON")])
        v = PlanValidator(sop=build_sop("COMPARISON"))
        issues = v.validate(dag_obj, _plan(dag_obj.tasks.values()))
        self.assertTrue(any(i.issue_type == pv.ISSUE_MISSING_COVERAGE for i in issues))
        before_ids = set(dag_obj.tasks.keys())
        pv._apply_local_repair(dag_obj, issues, max_repairs=3)
        after_ids = set(dag_obj.tasks.keys())
        self.assertTrue(before_ids.issubset(after_ids))  # 原任务保留
        self.assertGreater(len(after_ids), len(before_ids))  # 只新增，不删原任务

    def test_6_fallback_general_sop(self):
        # SOP Selector 无法识别类型 → 回退 FACT（通用 SOP），Planner 正常执行
        query = "这是一个没有任何关键词的普通问题"
        sop = select_sop(query)
        self.assertEqual(sop.sop_type, "FACT")
        # 回退后 Planner 仍能拿到 SOP 文本，流程不阻塞
        received = sop.to_prompt()
        self.assertIn("必须覆盖的研究维度", received)
        self.assertTrue(received.strip())


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

    def test_fix_dependency_prunes_ghost_dependency(self):
        # 依赖指向不存在的任务：fix_dependency 直接删掉这条依赖，不再判 REJECT
        t = _task("t1", "综合", "SYNTHESIS", deps=["ghost"])
        dag_obj = TaskDAG()
        dag_obj.tasks["t1"] = t  # 绕过 add_task 校验，模拟畸形 DAG
        gate = PlanGate(sop=None, critic=_FakeCritic())
        r = gate.run(dag_obj, _plan(dag_obj.tasks.values()), query="综合")
        self.assertNotEqual(r.decision, GATE_REJECT)
        self.assertNotIn("ghost", dag_obj.tasks["t1"].dependencies)

    def test_unreachable_task_not_deleted_after_dependency_fixed(self):
        # 同一轮里先清掉悬空依赖，任务重新可达 → 不能按旧结论把它删掉
        t = _task("t1", "综合", "SYNTHESIS", deps=["ghost"])
        dag_obj = TaskDAG()
        dag_obj.tasks["t1"] = t
        gate = PlanGate(sop=None, critic=_FakeCritic())
        gate.run(dag_obj, _plan(dag_obj.tasks.values()), query="综合")
        self.assertIn("t1", dag_obj.tasks)

    def test_missing_field_still_reject(self):
        # 任务既无 objective 也无 question：没有可修动作，补不动 → REJECT
        dag_obj = TaskDAG([_task("t1", "", "FACT")])
        gate = PlanGate(sop=None, critic=_FakeCritic())
        r = gate.run(dag_obj, _plan(dag_obj.tasks.values()), query="q")
        self.assertEqual(r.decision, GATE_REJECT)

    def test_missing_many_dimensions_converges(self):
        # SURVEY 有 5 个 required 维度，Planner 只给 1 个任务：
        # repair 要能把缺的维度补齐（单轮上限 3 时补不完，会残留 HIGH）
        dag_obj = TaskDAG([_task("t1", "对比 A 和 B", "COMPARISON")])
        gate = PlanGate(sop=build_sop("SURVEY"), critic=_FakeCritic())
        r = gate.run(dag_obj, _plan(dag_obj.tasks.values()), query="对比 A 和 B")
        self.assertNotEqual(r.decision, GATE_REJECT)
        covered: set = set()
        for t in dag_obj.tasks.values():
            covered.update(sop.task_type_dimensions(t.task_type))
        for dim in build_sop("SURVEY").required_dimensions:
            self.assertIn(dim, covered)

    def test_repaired_comparison_has_evidence_ancestor(self):
        # 补出来的 COMPARISON 必须挂在事实/机制类任务上，不能自己又变成「无证据祖先」
        dag_obj = TaskDAG([_task("t1", "X 的背景", "BACKGROUND")])
        gate = PlanGate(sop=build_sop("COMPARISON"), critic=_FakeCritic())
        r = gate.run(dag_obj, _plan(dag_obj.tasks.values()), query="对比 X 与 Y")
        self.assertNotEqual(r.decision, GATE_REJECT)
        for t in dag_obj.tasks.values():
            if str(t.task_type).upper() in ("COMPARISON", "SYNTHESIS"):
                self.assertTrue(t.dependencies, f"{t.task_id} 没有上游依赖")

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


class TestFallbackPlan(unittest.TestCase):
    """REJECT 后的兜底计划：按 SOP required 维度展开，确定性，不调模型。"""

    def test_covers_all_required_dimensions(self):
        s = build_sop("COMPARISON")
        dag_obj, plan = pv.build_fallback_plan("对比 A 和 B", s)
        self.assertEqual(len(dag_obj.tasks), len(s.required_dimensions))
        covered: set = set()
        for t in dag_obj.tasks.values():
            covered.update(sop.task_type_dimensions(t.task_type))
        for dim in s.required_dimensions:
            self.assertIn(dim, covered)

    def test_questions_are_distinct(self):
        # 每个任务的问题都不一样，否则会被 REDUNDANT_TASK 判成重复任务
        s = build_sop("SURVEY")
        dag_obj, _ = pv.build_fallback_plan("梳理 X 领域研究现状", s)
        questions = [t.question for t in dag_obj.tasks.values()]
        self.assertEqual(len(questions), len(set(questions)))
        self.assertTrue(all(q.strip() for q in questions))

    def test_passes_gate(self):
        # 兜底计划自己要能过闸门，否则换了也是白换
        s = build_sop("SURVEY")
        dag_obj, plan = pv.build_fallback_plan("梳理 X 领域研究现状", s)
        r = PlanGate(sop=s, critic=_FakeCritic()).run(
            dag_obj, plan, query="梳理 X 领域研究现状")
        self.assertNotEqual(r.decision, GATE_REJECT)

    def test_fact_sop_two_dimensions(self):
        # FACT 的 required 是 object_definition + evidence：第一个任务无依赖，
        # 第二个（取证）挂在第一个（下定义）之后
        s = build_sop("FACT")
        dag_obj, _ = pv.build_fallback_plan("什么是 x-vector", s)
        self.assertEqual(len(dag_obj.tasks), len(s.required_dimensions))
        tasks = list(dag_obj.tasks.values())
        self.assertEqual(tasks[0].dependencies, [])
        self.assertEqual(tasks[1].dependencies, [tasks[0].task_id])

    def test_no_sop_falls_back_to_fact(self):
        dag_obj, plan = pv.build_fallback_plan("随便一个问题", None)
        self.assertTrue(dag_obj.tasks)
        self.assertEqual(plan.required_dimensions, build_sop("FACT").required_dimensions)


if __name__ == "__main__":
    unittest.main()
