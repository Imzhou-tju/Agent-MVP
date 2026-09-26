"""Plan Verification & Repair（轻量级计划验证与修补）。

在既有 Planner / DAG 之上补齐「计划验证」这一环，不改 Scheduler / Researcher /
检索 / Evidence 结构，全部复用现有 ResearchPlan / ResearchTask / TaskDAG 对象。

流程（执行前）：
    Planner → PlanSchemaValidator → PlanCoverageValidator → SemanticPlanCritic
            → LocalRepair → ValidatedPlan → Scheduler

流程（执行后）：
    Evidence / Claim → EvidenceCoverageValidator → PASS / GAP → LocalRepair Task

确定性 / LLM 边界：
- PlanSchemaValidator / PlanCoverageValidator / TaskSuccessCriteriaEvaluator /
  EvidenceCoverageValidator 全部是确定性代码，不调用模型。
- SemanticPlanCritic 是唯一的 LLM 组件，只负责发现语义缺口并给出局部修改建议，
  不重新生成完整 DAG。其输出经 parse 后规整成 PlanIssue，异常时静默降级为空。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .dag import ResearchPlan, ResearchTask, TaskDAG, _DEFAULT_TASK_TYPE

# 合法的任务类型（与 dag.TASK_TYPES 对齐；用于 Schema 校验）
LEGAL_TASK_TYPES = (
    "FACT", "MECHANISM", "COMPARISON", "TREND",
    "EVALUATION", "BACKGROUND", "LIMITATION", "METHOD", "SYNTHESIS",
)

# 任务「完成条件」必需字段的合法取值（结构化 success_criteria）
CRITERIA_KEYS = ("required_fields", "required_source_types", "min_evidence_count")

# TaskSuccessCriteriaEvaluator 的三态结果
PASS = "PASS"
PARTIAL = "PARTIAL"
FAIL = "FAIL"

# PlanIssue 的问题类型（Schema / Coverage / Critic 共用）
ISSUE_DUPLICATE_ID = "DUPLICATE_ID"
ISSUE_MISSING_DEP = "MISSING_DEP"
ISSUE_SELF_DEP = "SELF_DEP"
ISSUE_CYCLE = "CYCLE"
ISSUE_ILLEGAL_TYPE = "ILLEGAL_TASK_TYPE"
ISSUE_MISSING_FIELD = "MISSING_FIELD"
ISSUE_UNREACHABLE = "UNREACHABLE_TASK"
ISSUE_BAD_DEP = "BAD_DEPENDENCY"
ISSUE_MISSING_CAPABILITY = "MISSING_CAPABILITY"
ISSUE_MISSING_TASK = "MISSING_TASK"
ISSUE_REDUNDANT_TASK = "REDUNDANT_TASK"
ISSUE_COARSE_TASK = "COARSE_TASK"
ISSUE_COVERAGE_GAP = "COVERAGE_GAP"

# 每个 task_type 应当能提供的 capability（供 Coverage 做「依赖能提供输入」的语义判断）
_TASK_TYPE_CAPABILITY = {
    "FACT": "fact",
    "MECHANISM": "mechanism",
    "COMPARISON": "comparison",
    "TREND": "trend",
    "EVALUATION": "evaluation",
    "BACKGROUND": "background",
    "LIMITATION": "limitation",
    "METHOD": "method",
    "SYNTHESIS": "synthesis",
}


@dataclass
class PlanIssue:
    """计划验证发现的一个问题（确定性 or 语义）。

    字段与 review_node 产出的 issue 保持同构，便于 repair_node 复用：
    issue_type / severity / target_id / recommended_action / suggested_task。
    """

    issue_type: str
    description: str
    severity: str = "medium"               # high / medium / low
    target_type: str = "task"              # task / plan / global
    target_id: str = ""                    # 关联的 task_id（或空）
    recommended_action: str = "none"       # add_task / reopen_task / remove_task / none
    suggested_task: dict = field(default_factory=dict)  # add_task 时的任务规格
    capability: str = ""                   # MISSING_CAPABILITY 时缺的能力

    def to_dict(self) -> dict:
        return {
            "issue_id": _issue_id(self),
            "issue_type": self.issue_type,
            "description": self.description,
            "severity": self.severity,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "target_task_id": self.target_id if self.target_type == "task" else "",
            "recommended_action": self.recommended_action,
            "suggested_action": self.recommended_action,  # 兼容旧字段
            "required_evidence": "",
            "priority": 2 if self.recommended_action in ("add_task", "remove_task") else 1,
            "suggested_task": dict(self.suggested_task),
        }


def _issue_id(i: PlanIssue) -> str:
    # 稳定 id：由 (类型, 目标, 描述) 哈希，与 repair 的去重逻辑一致
    from .evidence import stable_id
    return stable_id("PI", i.issue_type, i.target_id, i.description)


@dataclass
class PlanValidationResult:
    """一轮计划验证的结果汇总。"""

    valid: bool = True
    issues: list[PlanIssue] = field(default_factory=list)
    missing_capabilities: list[str] = field(default_factory=list)
    covered_capabilities: list[str] = field(default_factory=list)
    repair_count: int = 0                  # 本轮 local repair 实际新增/重开/删除的任务数

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "issues": [i.to_dict() for i in self.issues],
            "missing_capabilities": list(self.missing_capabilities),
            "covered_capabilities": list(self.covered_capabilities),
            "repair_count": self.repair_count,
        }


# ---------------------------------------------------------------------------
# 1. PlanSchemaValidator（确定性）
# ---------------------------------------------------------------------------

class PlanSchemaValidator:
    """增强的 DAG 结构校验（纯确定性，不调模型）。

    在 TaskDAG.add_task 的既有校验（唯一 / 自依赖 / 依赖存在 / 环）之上，补：
    - task_type 合法；
    - 必要字段非空（objective 或 question 至少其一）；
    - 所有任务均可从 root 到达（否则永远不会被调度）；
    - dependency 指向的任务确实能提供当前任务所需输入（按 task_type 语义判断）。
    """

    def validate(self, dag: TaskDAG) -> list[PlanIssue]:
        issues: list[PlanIssue] = []
        tasks = list(dag.tasks.values())

        for t in tasks:
            # task_type 合法
            if str(t.task_type or "").upper() not in LEGAL_TASK_TYPES:
                issues.append(PlanIssue(
                    ISSUE_ILLEGAL_TYPE,
                    f"任务 {t.task_id} 的 task_type 非法：{t.task_type!r}",
                    severity="medium", target_id=t.task_id,
                ))
            # 必要字段非空
            if not (t.objective or "").strip() and not (t.question or "").strip():
                issues.append(PlanIssue(
                    ISSUE_MISSING_FIELD,
                    f"任务 {t.task_id} 缺少 objective/question",
                    severity="high", target_id=t.task_id,
                ))
            # 自依赖 / 依赖不存在（add_task 已拦，这里对整图做兜底重检）
            for dep in t.dependencies:
                if dep == t.task_id:
                    issues.append(PlanIssue(
                        ISSUE_SELF_DEP, f"任务 {t.task_id} 自依赖", "high", t.task_id,
                    ))
                elif dep not in dag.tasks:
                    issues.append(PlanIssue(
                        ISSUE_MISSING_DEP, f"任务 {t.task_id} 依赖不存在的 {dep}",
                        "high", t.task_id,
                    ))

        # root 可达性
        reachable = dag.reachable_from_roots()
        for tid in dag.tasks:
            if tid not in reachable:
                issues.append(PlanIssue(
                    ISSUE_UNREACHABLE,
                    f"任务 {tid} 无法从任何 root 到达，永远不会被调度",
                    "high", tid, recommended_action="remove_task",
                ))

        # 依赖「能提供输入」的语义判断：下游类型应能从上边类型拿到可用结论
        for t in tasks:
            for dep in t.dependencies:
                dep_task = dag.tasks.get(dep)
                if dep_task is None:
                    continue
                if not _dependency_meaningful(dep_task, t):
                    issues.append(PlanIssue(
                        ISSUE_BAD_DEP,
                        f"任务 {t.task_id}（{t.task_type}）依赖 {dep}（{dep_task.task_type}），"
                        f"后者类型无法提供前者所需输入",
                        "medium", t.task_id,
                    ))

        return issues


def _dependency_meaningful(upstream: ResearchTask, downstream: ResearchTask) -> bool:
    """判断 upstream 的产出类型能否为 downstream 提供输入。

    规则（§DAG）：COMPARISON / EVALUATION / TREND / LIMITATION 依赖
    FACT / MECHANISM / METHOD / RESULT / BACKGROUND 才有意义；
    反向（FACT 依赖 COMPARISON）视为不合理的依赖。
    """
    from .dag import _DEPENDENCY_RULES
    down_type = str(downstream.task_type or _DEFAULT_TASK_TYPE).upper()
    up_type = str(upstream.task_type or _DEFAULT_TASK_TYPE).upper()
    allowed = _DEPENDENCY_RULES.get(down_type)
    if allowed is None:
        # 下游类型无规则约束：默认认为有意义（不做过度拦截）
        return True
    return up_type in allowed


# ---------------------------------------------------------------------------
# 2. PlanCoverageValidator（确定性）
# ---------------------------------------------------------------------------

class PlanCoverageValidator:
    """集合覆盖判断：required_capabilities - covered_capabilities（确定性）。

    covered_capabilities 来自两部分：
    - 每个任务的 capabilities 字段显式声明；
    - 每个任务的 task_type 自动贡献一个能力（_TASK_TYPE_CAPABILITY）。
    缺失项输出 MISSING_CAPABILITY 问题，并附带一个 suggested_task 供 repair 补任务。
    """

    def validate(self, plan: ResearchPlan) -> PlanValidationResult:
        required = [c for c in (plan.required_capabilities or plan.required_dimensions or [])]
        covered: set[str] = set()
        for t in plan.tasks:
            for c in t.capabilities:
                covered.add(str(c).lower())
            auto = _TASK_TYPE_CAPABILITY.get(str(t.task_type or "").upper())
            if auto:
                covered.add(auto)

        missing = [c for c in required if str(c).lower() not in covered]
        issues: list[PlanIssue] = []
        for c in missing:
            issues.append(PlanIssue(
                ISSUE_MISSING_CAPABILITY,
                f"计划要求覆盖维度/能力 {c!r}，但没有任何任务覆盖它",
                severity="high", target_type="plan",
                recommended_action="add_task",
                capability=c,
                suggested_task={
                    "question": f"补充关于 {c} 的研究",
                    "task_type": _type_for_capability(c),
                    "expected_evidence": f"关于 {c} 的证据",
                    "search_strategy": f"检索 {c} 相关文献与数据",
                    "capabilities": [c],
                    "dependencies": [],
                },
            ))

        return PlanValidationResult(
            valid=(not issues),
            issues=issues,
            missing_capabilities=missing,
            covered_capabilities=sorted(covered),
        )


def _type_for_capability(cap: str) -> str:
    """能力名 → 建议 task_type（用于 repair 补任务时的默认类型）。"""
    c = str(cap).lower()
    table = {
        "method": "METHOD", "dataset": "FACT", "performance": "RESULT",
        "limitation": "LIMITATION", "comparison": "COMPARISON", "trend": "TREND",
        "evaluation": "EVALUATION", "background": "BACKGROUND", "fact": "FACT",
        "mechanism": "MECHANISM",
    }
    return table.get(c, "FACT")


# ---------------------------------------------------------------------------
# 3. TaskSuccessCriteriaEvaluator（确定性）
# ---------------------------------------------------------------------------

class TaskSuccessCriteriaEvaluator:
    """根据结构化 criteria 判断任务结果 PASS / PARTIAL / FAIL（确定性）。

    结构化 criteria（task.criteria）：
        required_fields: [str]          证据/声明里应出现的字段名（如 dataset/metric/result）
        required_source_types: [str]    证据来源应覆盖的类型（如 academic）
        min_evidence_count: int         至少需要的证据条数

    判断规则（逐条打分，全满足 PASS，部分满足 PARTIAL，全不满足 FAIL）：
    - 证据来源类型覆盖：sources 的 source_type 是否命中 required_source_types；
    - 字段出现：claims 的 text / evidence 的 interpretation 里是否出现 required_fields；
    - 证据条数：evidence 数是否 >= min_evidence_count。
    无 criteria（空 dict）时不评判（返回 PASS，视为不启用），保持向后兼容。
    """

    def evaluate(self, task: ResearchTask,
                 evidences: Iterable[dict] | None = None,
                 claims: Iterable[dict] | None = None) -> str:
        criteria = task.criteria or {}
        if not criteria:
            return PASS  # 未声明结构化完成条件 → 不评判

        evs = list(evidences or [])
        clms = list(claims or [])
        checks: list[bool] = []

        # 证据条数
        if "min_evidence_count" in criteria:
            try:
                need = int(criteria["min_evidence_count"])
            except (TypeError, ValueError):
                need = 0
            checks.append(len(evs) >= need if need > 0 else True)

        # 来源类型覆盖
        if criteria.get("required_source_types"):
            want = {str(s).lower() for s in criteria["required_source_types"]}
            have = {str(e.get("source_type", "")).lower() for e in evs}
            checks.append(bool(want & have))

        # 字段出现（在 claim text 或 evidence interpretation 中检索）
        if criteria.get("required_fields"):
            blob = " ".join(
                [str(c.get("text", "")) for c in clms]
                + [str(e.get("interpretation", "")) for e in evs]
            ).lower()
            fields = [str(f).lower() for f in criteria["required_fields"]]
            checks.append(all(f in blob for f in fields))

        if not checks:
            return PASS
        passed = sum(1 for c in checks if c)
        if passed == len(checks):
            return PASS
        if passed == 0:
            return FAIL
        return PARTIAL


# ---------------------------------------------------------------------------
# 4. EvidenceCoverageValidator（确定性）
# ---------------------------------------------------------------------------

class EvidenceCoverageValidator:
    """执行后覆盖校验：Evidence/Claim 是否覆盖 ResearchPlan 的关键维度（确定性）。

    从最终 Claim 的 claim_type + Evidence 的 source_type 归纳「已覆盖能力」，
    与 plan.required_capabilities/required_dimensions 做差集；
    缺失项输出 COVERAGE_GAP，并附带 suggested_task 供 repair 局部补任务
    （而不是重新生成整个 DAG）。
    """

    # claim_type → capability 映射（与 _TASK_TYPE_CAPABILITY 对齐）
    _CLAIM_TYPE_CAPABILITY = {
        "FACT": "fact", "METHOD": "method", "RESULT": "performance",
        "COMPARISON": "comparison", "LIMITATION": "limitation", "TREND": "trend",
        "INTERPRETATION": "evaluation", "HYPOTHESIS": "mechanism",
    }

    def validate(self, plan: ResearchPlan,
                 claims: Iterable[dict] | None = None,
                 evidences: Iterable[dict] | None = None) -> PlanValidationResult:
        required = [c for c in (plan.required_capabilities or plan.required_dimensions or [])]
        if not required:
            return PlanValidationResult(valid=True)

        covered: set[str] = set()
        for c in (claims or []):
            auto = self._CLAIM_TYPE_CAPABILITY.get(str(c.get("claim_type", "")).upper())
            if auto:
                covered.add(auto)
        for e in (evidences or []):
            st = str(e.get("source_type", "")).lower()
            if st:
                covered.add(st)

        missing = [c for c in required if str(c).lower() not in covered]
        issues: list[PlanIssue] = []
        for c in missing:
            issues.append(PlanIssue(
                ISSUE_COVERAGE_GAP,
                f"研究结论缺少维度 {c!r} 的直接证据（如 comparison/performance），"
                f"仅由单项证据堆叠，无法支撑对比类结论",
                severity="high", target_type="plan",
                recommended_action="add_task",
                capability=c,
                suggested_task={
                    "question": f"补充 {c} 维度的直接对比/量化证据",
                    "task_type": _type_for_capability(c),
                    "expected_evidence": f"{c} 维度的直接证据",
                    "search_strategy": f"检索 {c} 的横向对比数据",
                    "capabilities": [c],
                    "dependencies": [],
                },
            ))

        return PlanValidationResult(
            valid=(not issues),
            issues=issues,
            missing_capabilities=missing,
            covered_capabilities=sorted(covered),
        )


# ---------------------------------------------------------------------------
# 5. SemanticPlanCritic（LLM，唯一模型组件）
# ---------------------------------------------------------------------------

class SemanticPlanCritic:
    """独立的 LLM Critic，对 (User Question, ResearchPlan, Task DAG) 做语义检查。

    只负责发现问题并给出局部修改建议，不重新生成完整 DAG。
    重点检查：遗漏关键维度 / 冗余任务 / 不合理依赖 / 过于粗粒度 / 无法支撑回答。
    输出结构化 JSON；解析失败或 LLM 异常时静默降级为空（不阻断流程）。
    """

    def __init__(self, llm=None, reliable_call=None):
        self._llm = llm
        # 传入研究链路统一可靠性包装（重试→升级→熔断→死信→降级）。
        # 缺省从 research.reliability 导入 reliable_llm_call（其签名含 llm 关键字参数）。
        if reliable_call is not None:
            self._call = reliable_call
        else:
            from .reliability import reliable_llm_call as _rel
            self._call = _rel

    def critique(self, query: str, plan: ResearchPlan) -> list[PlanIssue]:
        if self._llm is None:
            return []
        task_text = "\n".join(
            f"- {t.task_id} [{t.task_type}] {t.objective or t.question}"
            + (f"（依赖 {','.join(t.dependencies)}）" if t.dependencies else "")
            for t in plan.tasks
        )
        prompt = (
            "你是研究计划评审员。检查下面的研究计划是否真正覆盖用户问题。\n\n"
            f"用户问题: {query}\n\n"
            f"计划目标: {plan.objective}\n"
            f"计划约束: {plan.constraints or '无'}\n"
            f"要求覆盖的维度/能力: {', '.join(plan.required_capabilities or plan.required_dimensions) or '未声明'}\n\n"
            f"任务 DAG:\n{task_text}\n\n"
            "检查是否存在以下问题：遗漏关键研究维度 / 无意义或冗余任务 / "
            "不合理依赖 / 应拆分但过于粗粒度的任务 / 无法支撑最终回答的任务。\n"
            "只输出 JSON，不要输出其它内容：\n"
            '{"valid": false, "issues": [{"type": "MISSING_TASK", "target": "t3", '
            '"severity": "HIGH", "description": "...", "recommended_action": "ADD_TASK", '
            '"suggested_task": {"question": "...", "task_type": "FACT", '
            '"capabilities": ["method"], "dependencies": []}}]}\n'
            "没有问题时输出 {\"valid\": true, \"issues\": []}。\n"
            "注意：只给局部修改建议，不要重新生成完整 DAG。"
        )
        try:
            raw = self._call("plan_critic", prompt, llm=self._llm, fallback="").strip()
            parsed = _parse_json_loose(raw)
            raw_issues = parsed.get("issues", []) if isinstance(parsed, dict) else []
            if not isinstance(raw_issues, list):
                return []
        except Exception as e:
            print(f"[PlanCritic] 失败: {e}")
            return []

        issues: list[PlanIssue] = []
        for ri in raw_issues:
            if not isinstance(ri, dict):
                continue
            itype = _normalize_critic_type(str(ri.get("type", "")).upper())
            desc = str(ri.get("description", "")).strip()
            if not desc:
                continue
            severity = str(ri.get("severity", "medium")).lower()
            if severity not in ("high", "medium", "low"):
                severity = "medium"
            action = str(ri.get("recommended_action", "none") or "none").lower()
            action = {"add_task": "add_task", "remove_task": "remove_task",
                      "reopen_task": "reopen_task"}.get(action, "none")
            target = str(ri.get("target", "") or "")
            suggested = ri.get("suggested_task", {}) or {}
            if not isinstance(suggested, dict):
                suggested = {}
            issues.append(PlanIssue(
                issue_type=itype,
                description=desc,
                severity=severity,
                target_type="task" if target else "plan",
                target_id=target,
                recommended_action=action,
                suggested_task=suggested,
            ))
        return issues


def _normalize_critic_type(t: str) -> str:
    """把 Critic 的自由类型映射到统一 PlanIssue 类型。"""
    if "MISSING" in t and "TASK" in t:
        return ISSUE_MISSING_TASK
    if "MISSING" in t and "CAPABILITY" in t:
        return ISSUE_MISSING_CAPABILITY
    if "REDUNDANT" in t or "DUPLICATE" in t:
        return ISSUE_REDUNDANT_TASK
    if "COARSE" in t or "SPLIT" in t:
        return ISSUE_COARSE_TASK
    if "DEP" in t:
        return ISSUE_BAD_DEP
    if "COVERAGE" in t or "GAP" in t:
        return ISSUE_COVERAGE_GAP
    return ISSUE_MISSING_TASK


def _parse_json_loose(raw: str) -> Any:
    import json
    import re
    m = re.search(r'(\{.*\}|\[.*\])', raw, re.DOTALL)
    if not m:
        raise ValueError(f"无法提取 JSON: {raw[:200]}")
    return json.loads(m.group(0))


# ---------------------------------------------------------------------------
# 编排：执行前验证 + 修补（§Repair）
# ---------------------------------------------------------------------------

def validate_plan(
    dag: TaskDAG,
    plan: ResearchPlan,
    query: str = "",
    critic: SemanticPlanCritic | None = None,
    max_repairs: int = 3,
) -> PlanValidationResult:
    """执行前验证编排：Schema → Coverage → SemanticCritic → LocalRepair。

    返回 PlanValidationResult（含 issues 与 missing_capabilities）。
    修补动作（add_task / remove_task）就地作用在 dag 上，并统计 repair_count。
    达到 max_repairs 后停止，保留当前结果与未解决问题。
    """
    schema = PlanSchemaValidator()
    coverage = PlanCoverageValidator()

    issues: list[PlanIssue] = []
    issues.extend(schema.validate(dag))

    cov_result = coverage.validate(plan)
    issues.extend(cov_result.issues)
    missing_caps = list(cov_result.missing_capabilities)
    covered_caps = list(cov_result.covered_capabilities)

    if critic is not None:
        issues.extend(critic.critique(query, plan))

    # local repair：对 add_task / remove_task 类问题就地修改 DAG
    repair_count = _apply_local_repair(dag, issues, max_repairs)

    # 修补后重跑 schema 校验（去重，保留仍未解决的）
    residual = schema.validate(dag)
    known = {_issue_id(i) for i in issues}
    for i in residual:
        if _issue_id(i) not in known:
            issues.append(i)

    return PlanValidationResult(
        valid=(not [i for i in issues if i.severity == "high"]),
        issues=issues,
        missing_capabilities=missing_caps,
        covered_capabilities=covered_caps,
        repair_count=repair_count,
    )


def _apply_local_repair(dag: TaskDAG, issues: list[PlanIssue], max_repairs: int) -> int:
    """就地执行 local repair：只处理 add_task / remove_task，不重建整图。

    - add_task：按 suggested_task 构造 ResearchTask 加入 DAG（受 max_repairs 上限）；
    - remove_task：删除无意义/冗余任务（仅当其无下游时，避免破坏依赖）。
    返回实际修改的任务数。
    """
    repaired = 0
    for issue in issues:
        if repaired >= max_repairs:
            break
        action = issue.recommended_action
        if action == "remove_task" and issue.target_id:
            tid = issue.target_id
            if tid in dag.tasks and not _has_downstream(dag, tid):
                del dag.tasks[tid]
                repaired += 1
            continue
        if action != "add_task":
            continue
        spec = issue.suggested_task or {}
        question = str(spec.get("question", "") or issue.description).strip()
        if not question:
            continue
        deps = [str(d) for d in (spec.get("dependencies") or []) if d in dag.tasks]
        # 能力缺失的补任务：若其上游能力已存在，则依赖提供该能力的任务
        if not deps and issue.capability:
            deps = _deps_for_capability(dag, issue.capability)
        task = ResearchTask(
            task_id=_new_task_id(dag, issue.capability or "repair"),
            objective=str(spec.get("expected_evidence", "") or question),
            question=question,
            task_type=str(spec.get("task_type", "FACT")).upper(),
            expected_evidence=str(spec.get("expected_evidence", "")),
            search_strategy=str(spec.get("search_strategy", "")),
            dependencies=deps,
            capabilities=[str(x) for x in (spec.get("capabilities") or []) if x],
            priority=2,
            notes=issue.description,
        )
        ok, _reason = dag.add_task(task)
        if ok:
            repaired += 1
    return repaired


def _has_downstream(dag: TaskDAG, task_id: str) -> bool:
    return any(task_id in t.dependencies for t in dag.tasks.values())


def _deps_for_capability(dag: TaskDAG, capability: str) -> list[str]:
    """为缺失能力补任务时，找出已覆盖该能力或其上游能力的任务作为依赖。"""
    from .dag import _DEPENDENCY_RULES
    want = str(capability).lower()
    # 该能力作为下游类型时，应依赖哪些上游能力
    down_type = _type_for_capability(want)
    upstream_types = _DEPENDENCY_RULES.get(down_type, ())
    deps = []
    for t in dag.tasks.values():
        caps = {str(c).lower() for c in t.capabilities}
        caps.add(_TASK_TYPE_CAPABILITY.get(str(t.task_type).upper(), ""))
        if want in caps:
            deps.append(t.task_id)
        elif any(_TASK_TYPE_CAPABILITY.get(u, "") in caps for u in upstream_types):
            deps.append(t.task_id)
    return deps[:2]  # 最多挂 2 个上游，避免过度耦合


def _new_task_id(dag: TaskDAG, hint: str) -> str:
    import re
    base = "pv-" + re.sub(r"[^a-zA-Z0-9_-]", "_", hint)[:16]
    tid = base
    i = 1
    while tid in dag.tasks:
        i += 1
        tid = f"{base}-{i}"
    return tid
