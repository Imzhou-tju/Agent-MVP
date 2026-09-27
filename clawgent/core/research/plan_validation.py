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

# Plan Gate 专项问题类型（§4）：针对 Planner 拆分质量的确定性判断
ISSUE_MISSING_COVERAGE = "MISSING_COVERAGE"          # 未覆盖用户问题/SOP 要求维度
ISSUE_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # Comparison/Synthesis 无 evidence ancestor
ISSUE_PREMATURE_SYNTHESIS = "PREMATURE_SYNTHESIS"     # 过早进入 Comparison/Synthesis
ISSUE_INSUFFICIENT_DEPENDENCY = "INSUFFICIENT_DEPENDENCY"  # 下游缺少必要依赖
ISSUE_BAD_GRANULARITY = "BAD_GRANULARITY"             # 任务包含过多明显动作

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
    # add_task / reopen_task / remove_task / fix_dependency / none
    # fix_dependency：删掉指向不存在任务的依赖与自依赖（MISSING_DEP / SELF_DEP）
    recommended_action: str = "none"
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
                        ISSUE_SELF_DEP, f"任务 {t.task_id} 自依赖",
                        severity="high", target_id=t.task_id,
                        recommended_action="fix_dependency",
                    ))
                elif dep not in dag.tasks:
                    issues.append(PlanIssue(
                        ISSUE_MISSING_DEP, f"任务 {t.task_id} 依赖不存在的 {dep}",
                        severity="high", target_id=t.task_id,
                        recommended_action="fix_dependency",
                    ))

        # root 可达性
        reachable = dag.reachable_from_roots()
        for tid in dag.tasks:
            if tid not in reachable:
                issues.append(PlanIssue(
                    ISSUE_UNREACHABLE,
                    f"任务 {tid} 无法从任何 root 到达，永远不会被调度",
                    severity="high", target_id=tid, recommended_action="remove_task",
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
    """能力名 → 建议 task_type（用于 repair 补任务时的默认类型）。

    先查 sop.CAPABILITY_TO_TASK_TYPE（SOP 的维度名都在那张表里），查不到再走旧表。
    这张表同时被 _deps_for_capability 用来找上游类型：把 evaluation_or_comparison
    错认成 FACT 会导致补出来的对比任务找不到上游，进而被判成"无证据祖先"。
    """
    from .sop import capability_to_task_type as _sop_type, known_capability

    c = str(cap).lower()
    if known_capability(c):
        return _sop_type(c)
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
            "你是研究计划评审员。只判断：这个计划是否足以合理回答用户问题。\n\n"
            f"用户问题: {query}\n\n"
            f"计划目标: {plan.objective}\n"
            f"计划约束: {plan.constraints or '无'}\n"
            f"要求覆盖的维度/能力: {', '.join(plan.required_capabilities or plan.required_dimensions) or '未声明'}\n\n"
            f"任务 DAG:\n{task_text}\n\n"
            "检查是否存在以下问题：遗漏关键研究维度 / 无意义或冗余任务 / "
            "不合理依赖 / 应拆分但过于粗粒度的任务 / 无法支撑最终回答的任务。\n"
            "只输出 JSON，不要输出其它内容，issue 最多 3 个：\n"
            '{"decision": "REPAIR", "issues": [{"issue_type": "MISSING_RESEARCH_ANGLE", '
            '"severity": "HIGH", "target_task_id": "t3", "description": "...", '
            '"recommended_action": "ADD_TASK"}]}\n'
            "没有问题或计划足以回答时输出 {\"decision\": \"PASS\", \"issues\": []}。\n"
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
        for ri in raw_issues[:3]:  # §5：最多 3 个 issue
            if not isinstance(ri, dict):
                continue
            itype = _normalize_critic_type(str(ri.get("issue_type", ri.get("type", ""))).upper())
            desc = str(ri.get("description", "")).strip()
            if not desc:
                continue
            severity = str(ri.get("severity", "medium")).lower()
            if severity not in ("high", "medium", "low"):
                severity = "medium"
            action = str(ri.get("recommended_action", "none") or "none").lower()
            action = {"add_task": "add_task", "remove_task": "remove_task",
                      "reopen_task": "reopen_task"}.get(action, "none")
            target = str(ri.get("target_task_id", ri.get("target", "")) or "")
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
    max_repairs: int = 8,
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
    """就地执行 local repair：只处理 add_task / remove_task / fix_dependency，不重建整图。

    - add_task：按 suggested_task 构造 ResearchTask 加入 DAG（受 max_repairs 上限）；
    - remove_task：删除无意义/冗余任务（仅当其无下游时，避免破坏依赖）；
    - fix_dependency：删掉指向不存在任务的依赖与自依赖。
    返回实际修改的任务数。
    """
    repaired = 0
    for issue in issues:
        if repaired >= max_repairs:
            break
        action = issue.recommended_action
        if action == "fix_dependency" and issue.target_id:
            task = dag.tasks.get(issue.target_id)
            if task is not None:
                before = list(task.dependencies)
                task.dependencies = [
                    d for d in before if d != issue.target_id and d in dag.tasks
                ]
                if task.dependencies != before:
                    repaired += 1
            continue
        if action == "remove_task" and issue.target_id:
            tid = issue.target_id
            if tid not in dag.tasks or _has_downstream(dag, tid):
                continue
            # UNREACHABLE 是本轮先算出来的：同一轮里 fix_dependency 可能已经把
            # 悬空依赖清掉，任务重新变成可达。按当前图再判一次，避免误删任务。
            if (issue.issue_type == ISSUE_UNREACHABLE
                    and tid in dag.reachable_from_roots()):
                continue
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
            # §6 允许「补 dependency」：若 issue 指向某个目标任务（如过早进入
            # Comparison/Synthesis），把补出来的证据任务挂成它的上游依赖，
            # 使 PREMATURE_SYNTHESIS / INSUFFICIENT_EVIDENCE 能被真正消除。
            if issue.target_id and issue.target_id in dag.tasks:
                target = dag.tasks[issue.target_id]
                if task.task_id not in target.dependencies:
                    target.dependencies.append(task.task_id)
    return repaired


def _has_downstream(dag: TaskDAG, task_id: str) -> bool:
    return any(task_id in t.dependencies for t in dag.tasks.values())


def _deps_for_capability(dag: TaskDAG, capability: str) -> list[str]:
    """为缺失能力补任务时，找出已覆盖该能力或其上游能力的任务作为依赖。

    _DEPENDENCY_RULES 没有为 SYNTHESIS 这类类型声明上游，此时退化成「挂能提供
    事实/证据的祖先」（_EVIDENCE_PROVIDER_TYPES），保证补出来的综合任务不会立刻
    被 _check_evidence_dependency 判成过早进入综合。
    """
    from .dag import _DEPENDENCY_RULES
    want = str(capability).lower()
    # 该能力作为下游类型时，应依赖哪些上游能力
    down_type = _type_for_capability(want)
    upstream_types = _DEPENDENCY_RULES.get(down_type, ())
    if not upstream_types:
        upstream_types = _EVIDENCE_PROVIDER_TYPES
    deps = []
    for t in dag.tasks.values():
        caps = {str(c).lower() for c in t.capabilities}
        caps.add(_TASK_TYPE_CAPABILITY.get(str(t.task_type).upper(), ""))
        if want in caps:
            deps.append(t.task_id)
        elif str(t.task_type).upper() in upstream_types:
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


# ---------------------------------------------------------------------------
# 兜底计划：Plan Gate 判 REJECT 后的确定性替代方案
# ---------------------------------------------------------------------------

def build_fallback_plan(query: str, sop: "ResearchSOP | None" = None,
                        plan_id: str = "") -> tuple[TaskDAG, ResearchPlan]:
    """按 SOP 的 required 维度生成一个最小可执行计划（确定性，不调模型）。

    用在 Plan Gate 判 REJECT 之后：Planner 的图补不动时（比如任务缺
    objective/question、缺的维度补不出来），用这份计划继续调研，而不是带着坏计划往下跑。

    每个 required 维度一个任务，任务类型由 _type_for_capability 决定，
    依赖按 dag._DEPENDENCY_RULES 挂到已建任务上（没有规则可依时挂事实/证据型任务），
    保证 COMPARISON / SYNTHESIS 不会一开始就缺证据祖先。
    """
    from .sop import build_sop, dimension_label

    sop = sop or build_sop("FACT")
    required = list(sop.required_dimensions) or ["object_definition"]

    dag = TaskDAG()
    built: list[ResearchTask] = []
    for i, dim in enumerate(required, 1):
        tt = _type_for_capability(dim)
        label = dimension_label(dim)
        text = f"{query} 的{label}" if query else label
        task = ResearchTask(
            task_id=f"fb{i}",
            objective=text,
            question=text,
            task_type=tt,
            expected_evidence=f"关于{label}的可引用结论",
            search_strategy=f"检索「{query}」在{label}方面的资料",
            dependencies=_fallback_upstream(built, tt),
            capabilities=[str(dim).lower()],
            priority=1,
            notes=f"兜底计划：SOP({sop.sop_type}) 要求覆盖维度 {dim}",
        )
        ok, _reason = dag.add_task(task)
        if ok:
            built.append(task)

    if not dag.tasks:       # 理论上到不了：required 为空已在上面兜住
        dag.add_task(ResearchTask(task_id="fb1", objective=query, question=query))

    plan = ResearchPlan(
        plan_id=plan_id or f"fallback-{sop.sop_type}",
        objective=query,
        tasks=list(dag.tasks.values()),
        required_dimensions=list(required),
    )
    return dag, plan


def _fallback_upstream(built: list[ResearchTask], task_type: str) -> list[str]:
    """兜底计划的依赖挂载：按 _DEPENDENCY_RULES 找已建任务，最多挂 2 个。"""
    from .dag import _DEPENDENCY_RULES

    allowed = _DEPENDENCY_RULES.get(str(task_type).upper()) or _EVIDENCE_PROVIDER_TYPES
    cands = [t.task_id for t in built if str(t.task_type).upper() in allowed]
    if not cands:
        cands = [t.task_id for t in built]
    return cands[-2:]


# ---------------------------------------------------------------------------
# Plan Gate：Planner 质量闸门（§任务：实现 Planner 质量闸门）
# ---------------------------------------------------------------------------
#
# 目标：以尽量低的额外 LLM 成本提高 Plan 可靠性。
# 流程（§7）：
#   User Query → Planner → SOP → PlanValidator(确定性) → PlanCritic(1次)
#     → PASS → Scheduler
#     → FAIL → Repair(1次) → Validator → Scheduler / Reject
#
# 边界（§2）：只改 Planner 阶段；不碰 Researcher / Evidence / Claim / Review /
# Judge / RAG / MCP / Runtime Agent。DAG 调度逻辑复用。

# 门控决策
GATE_ALLOW = "ALLOW"
GATE_ALLOW_WITH_WARNINGS = "ALLOW_WITH_WARNINGS"
GATE_REJECT = "REJECT"

# 「能提供证据 / 事实」的上游任务类型：Comparison / Synthesis 必须有此类祖先
_EVIDENCE_PROVIDER_TYPES = ("FACT", "MECHANISM", "METHOD", "RESULT", "BACKGROUND",
                            "EVALUATION", "TREND")

# 需要证据祖先支撑的任务类型（过早进入即 PREMATURE_SYNTHESIS）
_EVIDENCE_DEPENDENT_TYPES = ("COMPARISON", "SYNTHESIS")

# 动作连接词：objective/question 里出现多个，判为 BAD_GRANULARITY（一个任务塞了过多动作）
_GRANULARITY_CONNECTORS = ("并且", "同时", "以及", "另外", "此外", "还要", "既要", "又要",
                           "和", "与", "并")


@dataclass
class GateResult:
    """Plan Gate 的最终门控结果。"""

    decision: str = GATE_ALLOW                      # ALLOW / ALLOW_WITH_WARNINGS / REJECT
    issues: list[PlanIssue] = field(default_factory=list)
    validator_issues: list[PlanIssue] = field(default_factory=list)
    critic_issues: list[PlanIssue] = field(default_factory=list)
    repair_count: int = 0
    critic_called: bool = False
    critic_failed: bool = False
    sop_type: str = "FACT"

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "issues": [i.to_dict() for i in self.issues],
            "validator_issues": [i.to_dict() for i in self.validator_issues],
            "critic_issues": [i.to_dict() for i in self.critic_issues],
            "repair_count": self.repair_count,
            "critic_called": self.critic_called,
            "critic_failed": self.critic_failed,
            "sop_type": self.sop_type,
        }


def _ancestors(dag: TaskDAG, task_id: str) -> set[str]:
    """返回 task_id 的所有祖先（直接 + 间接上游），不含自身。"""
    result: set[str] = set()
    stack = list(dag.tasks.get(task_id, ResearchTask(task_id="")).dependencies)
    seen: set[str] = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur not in dag.tasks:
            continue
        result.add(cur)
        stack.extend(dag.tasks[cur].dependencies)
    return result


def _has_evidence_ancestor(dag: TaskDAG, task: ResearchTask) -> bool:
    """判断任务是否有「能提供事实/证据」的祖先。"""
    for anc in _ancestors(dag, task.task_id):
        at = dag.tasks.get(anc)
        if at is None:
            continue
        if str(at.task_type or _DEFAULT_TASK_TYPE).upper() in _EVIDENCE_PROVIDER_TYPES:
            return True
    return False


def _task_blob(task: ResearchTask) -> str:
    return " ".join(str(x) for x in (task.objective, task.question) if x).strip()


class PlanValidator:
    """Planner 质量的确定性校验器（纯 Python，不调外部 API，§4）。

    在既有 PlanSchemaValidator / PlanCoverageValidator 之上，补针对「拆分质量」的判断：
    - required_dimensions 覆盖（MISSING_COVERAGE）
    - Comparison / Synthesis 是否有 evidence ancestor（INSUFFICIENT_EVIDENCE）
    - 过早进入 Comparison / Synthesis（PREMATURE_SYNTHESIS）
    - 明显重复任务（REDUNDANT_TASK）
    - 任务塞了过多明显动作（BAD_GRANULARITY）
    - 明显错误依赖（BAD_DEPENDENCY，复用 SchemaValidator）

    不做 0~1 plan score。
    """

    def __init__(self, sop: "ResearchSOP | None" = None):
        self._sop = sop

    def validate(self, dag: TaskDAG, plan: ResearchPlan | None = None) -> list[PlanIssue]:
        issues: list[PlanIssue] = []

        # 1. 结构（复用 SchemaValidator）+ 依赖语义
        issues.extend(PlanSchemaValidator().validate(dag))

        # 2. SOP required 维度覆盖（MISSING_COVERAGE）
        issues.extend(self._check_coverage(dag, plan))

        # 3. Comparison / Synthesis 证据祖先（INSUFFICIENT_EVIDENCE / PREMATURE_SYNTHESIS）
        issues.extend(self._check_io_contract(dag))
        issues.extend(self._check_evidence_dependency(dag))

        # 4. 明显重复任务（REDUNDANT_TASK）
        issues.extend(self._check_redundancy(dag))

        # 5. 任务粒度（BAD_GRANULARITY）
        issues.extend(self._check_granularity(dag))

        return issues

    def _check_coverage(self, dag: TaskDAG, plan: ResearchPlan | None) -> list[PlanIssue]:
        """SOP 要求覆盖的维度，是否被任务覆盖。

        覆盖来源：任务的显式 capabilities + 任务 task_type 自动提供的维度
        （如 BACKGROUND 提供 object_definition，FACT 提供 evidence）。
        """
        required: list[str] = []
        if self._sop is not None:
            required = list(self._sop.required_dimensions)
        if plan is not None:
            # 计划显式声明的维度覆盖（或扩展）SOP 维度
            declared = list(plan.required_capabilities or plan.required_dimensions or [])
            if declared:
                required = declared

        if not required:
            return []

        from .sop import task_type_dimensions

        covered: set[str] = set()
        for t in dag.tasks.values():
            for c in t.capabilities:
                covered.add(str(c).lower())
            covered.update(task_type_dimensions(t.task_type))

        issues: list[PlanIssue] = []
        for r in required:
            rl = str(r).lower()
            if rl in covered:
                continue
            issues.append(PlanIssue(
                ISSUE_MISSING_COVERAGE,
                f"SOP 要求覆盖维度 {r!r}，但没有任务覆盖",
                severity="high", target_type="plan",
                recommended_action="add_task", capability=r,
                suggested_task={
                    "question": f"补充关于 {r} 的研究",
                    "task_type": capability_to_task_type(r),
                    "capabilities": [r],
                    "dependencies": [],
                },
            ))
        return issues

    
    def _check_io_contract(self, dag: TaskDAG) -> list[PlanIssue]:
        issues = []
        for task in dag.tasks.values():
            tt = str(task.task_type).upper()
            if tt in _EVIDENCE_DEPENDENT_TYPES: # SYNTHESIS, COMPARISON
                has_provider = False
                for dep_id in task.dependencies:
                    if dep_id in dag.tasks:
                        dep_tt = str(dag.tasks[dep_id].task_type).upper()
                        if dep_tt in _EVIDENCE_PROVIDER_TYPES:
                            has_provider = True
                            break
                if not has_provider:
                    issues.append(PlanIssue(
                        issue_type=ISSUE_INSUFFICIENT_EVIDENCE,
                        description=f"节点 {task.task_id} ({tt}) 必须直接依赖至少一个提供证据的节点 (如 FACT, MECHANISM)",
                        severity="HIGH",
                        target_id=task.task_id,
                        recommended_action="add_task"
                    ))
        return issues

    def _check_evidence_dependency(self, dag: TaskDAG) -> list[PlanIssue]:
        """Comparison / Synthesis 必须有 evidence ancestor，否则过早。"""
        issues: list[PlanIssue] = []
        for t in dag.tasks.values():
            tt = str(t.task_type or _DEFAULT_TASK_TYPE).upper()
            if tt not in _EVIDENCE_DEPENDENT_TYPES:
                continue
            if _has_evidence_ancestor(dag, t):
                continue
            if tt == "SYNTHESIS":
                issues.append(PlanIssue(
                    ISSUE_PREMATURE_SYNTHESIS,
                    f"任务 {t.task_id} 是 SYNTHESIS，但没有任何证据型祖先，"
                    f"属于过早进入综合",
                    severity="high", target_id=t.task_id,
                    recommended_action="add_task", capability="evidence",
                    suggested_task={
                        "question": f"为 {t.task_id} 补充事实/机制类证据任务",
                        "task_type": "FACT",
                        "capabilities": ["evidence"],
                        "dependencies": [],
                    },
                ))
            else:
                issues.append(PlanIssue(
                    ISSUE_INSUFFICIENT_EVIDENCE,
                    f"任务 {t.task_id} 是 {tt}，但没有任何证据型祖先，"
                    f"对比结论缺乏事实支撑",
                    severity="high", target_id=t.task_id,
                    recommended_action="add_task", capability="evidence",
                    suggested_task={
                        "question": f"为 {t.task_id} 补充对象的事实性证据",
                        "task_type": "FACT",
                        "capabilities": ["evidence"],
                        "dependencies": [],
                    },
                ))
        return issues

    def _check_redundancy(self, dag: TaskDAG) -> list[PlanIssue]:
        """明显重复任务：两个任务 objective+question 高度相似（归一化后相同/互为子串）。"""
        issues: list[PlanIssue] = []
        tasks = list(dag.tasks.values())
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                a = _task_blob(tasks[i]).strip()
                b = _task_blob(tasks[j]).strip()
                if not a or not b:
                    continue
                if a == b or a in b or b in a:
                    # 优先保留 task_id 更小/更靠前的，删除后者
                    victim = tasks[j]
                    issues.append(PlanIssue(
                        ISSUE_REDUNDANT_TASK,
                        f"任务 {victim.task_id} 与 {tasks[i].task_id} 重复",
                        severity="medium", target_id=victim.task_id,
                        recommended_action="remove_task",
                    ))
        return issues

    def _check_granularity(self, dag: TaskDAG) -> list[PlanIssue]:
        """任务粒度：objective/question 里出现 >=3 个动作连接词，判为塞了过多动作。"""
        issues: list[PlanIssue] = []
        for t in dag.tasks.values():
            blob = _task_blob(t)
            if not blob:
                continue
            hits = sum(1 for c in _GRANULARITY_CONNECTORS if c in blob)
            if hits >= 3:
                issues.append(PlanIssue(
                    ISSUE_BAD_GRANULARITY,
                    f"任务 {t.task_id} 目标包含 {hits} 个并列动作，粒度过粗，建议拆分",
                    severity="medium", target_id=t.task_id,
                    recommended_action="none",
                ))
        return issues


# 兼容旧 import：从 sop 导入（延迟，避免打包时循环依赖）
def capability_to_task_type(capability: str) -> str:
    from .sop import capability_to_task_type as _f
    return _f(capability)


class PlanGate:
    """Planner 质量闸门（§7）。

    编排：
        1. PlanValidator 确定性校验；
        2. 有 HIGH 问题 → Repair 一轮并重检，最多 MAX_VALIDATOR_REPAIR_ROUNDS 轮；
        3. PlanCritic 一次（最多 1 次 LLM 调用）；
        4. Critic 给出 issue → Repair 一次；
        5. 终态 validator 校验 → ALLOW / ALLOW_WITH_WARNINGS / REJECT。

    约束：Planner 1 次、Critic 1 次；Repair 是确定性的，不额外调模型。
    Critic 调用失败 → 只用 Validator 结果：无 HIGH → ALLOW_WITH_WARNINGS，有 HIGH → REJECT。
    REJECT 本身不做阻断：由调用方（planner_node）决定是兜底还是停用，见 build_fallback_plan。

    critic / repair 均可注入 Fake 实现（离线测试，§8）。
    """

    def __init__(self, sop: "ResearchSOP | None" = None,
                 critic: "SemanticPlanCritic | None" = None,
                 validator: PlanValidator | None = None):
        self._sop = sop
        self._validator = validator or PlanValidator(sop=sop)
        self._critic = critic if critic is not None else PlanGENVerificationCritic()

    def run(self, dag: TaskDAG, plan: ResearchPlan, query: str = "") -> GateResult:
        result = GateResult()
        if self._sop is not None:
            result.sop_type = self._sop.sop_type

        # 1. 确定性校验：有 HIGH 就 repair 一轮，再校验，最多 MAX_VALIDATOR_REPAIR_ROUNDS 轮。
        # 只做一轮时，SURVEY/COMPARISON 的 5 个 required 维度补不完（单轮上限挡住），
        # 残留的 HIGH 会把整个计划判成 REJECT。
        validator_issues: list[PlanIssue] = []
        for _ in range(MAX_VALIDATOR_REPAIR_ROUNDS):
            validator_issues = self._validator.validate(dag, plan)
            if not _has_high(validator_issues):
                break
            n = _apply_local_repair(dag, validator_issues, MAX_REPAIRS)
            result.repair_count += n
            if n == 0:      # 补不动了（如 recommended_action=none 的 HIGH），再循环也没用
                break
        result.validator_issues = validator_issues

        # 3. Critic（最多 1 次）
        if self._critic is not None:
            result.critic_called = True
            try:
                critic_issues = self._critic.critique(query, plan)
                result.critic_issues = critic_issues
            except Exception:
                critic_issues = []
                result.critic_failed = True
        else:
            critic_issues = []

        # 4. Critic 判 REPAIR → Repair 一次
        if critic_issues:
            result.repair_count += _apply_local_repair(dag, critic_issues, MAX_REPAIRS)

        # 5. 终态校验（repair 后的最终问题集合）
        final_issues = self._validator.validate(dag, plan)
        result.issues = _dedupe_issues(final_issues)

        if _has_high(final_issues):
            result.decision = GATE_REJECT
        elif result.critic_failed and _has_high(validator_issues):
            # Critic 失败 fallback：只用 validator 结果，有 HIGH → REJECT
            result.decision = GATE_REJECT
        elif result.issues:
            result.decision = GATE_ALLOW_WITH_WARNINGS
        else:
            result.decision = GATE_ALLOW
        return result


# 单轮 repair 允许修改的任务数上限。SOP 最多声明 5 个 required 维度，
# 一轮要把缺的维度补齐就得 >= 5，故取 5 并留余量（fix_dependency 也占额度）。
MAX_REPAIRS = 8
# validator → repair 的最大轮数。补完一轮后重检，仍缺就再补一轮。
MAX_VALIDATOR_REPAIR_ROUNDS = 2


def _has_high(issues: list[PlanIssue]) -> bool:
    return any(i.severity == "high" for i in issues)


def _dedupe_issues(issues: list[PlanIssue]) -> list[PlanIssue]:
    seen: set[str] = set()
    out: list[PlanIssue] = []
    for i in issues:
        k = _issue_id(i)
        if k in seen:
            continue
        seen.add(k)
        out.append(i)
    return out


class PlanGENVerificationCritic(SemanticPlanCritic):
    """
    基于 PlanGEN 思想的 Plan Gate V2 验证器:
    1. Constraint Agent: 从 query 提取专属 Checklist
    2. Verification Agent: 模拟空跑 DAG 节点能否回答 Checklist
    """
    def critique(self, query: str, plan: ResearchPlan) -> list[PlanIssue]:
        if self._llm is None or not query:
            return []
            
        import json
        # Phase 1: Constraint Agent (提取 Instance-Specific Checklist)
        constraint_prompt = f"""你是一个 Constraint Agent。
请从以下用户的核心研究问题中，提取出必须被任务计划覆盖的关键维度/条件（Checklist）。
仅输出 JSON 格式的列表，如 ["条件1", "条件2"]。
User Query: {query}
"""
        try:
            raw_checklist = self._call(self._llm, [("system", constraint_prompt)])
            checklist_text = raw_checklist.content.strip()
            if checklist_text.startswith('```json'):
                checklist_text = checklist_text[7:-3]
            checklist = json.loads(checklist_text)
            if not isinstance(checklist, list): checklist = []
        except Exception:
            checklist = []
            
        if not checklist:
            return super().critique(query, plan) # 退化到旧版

        # Phase 2: Verification Agent (Simulated Rollout)
        dag_context = ""
        for t in plan.tasks:
            dag_context += f"Task [{t.task_id}]({t.task_type}): {t.question}\n"
            
        checklist_str = chr(10).join(f"- {c}" for c in checklist)
        verify_prompt = f"""你是一个 Verification Agent。
当前的 DAG 计划包含以下任务：
{dag_context}

请假定这些任务均被完美执行并获取了信息。汇总这些信息能否完全解答以下 Checklist 中的每一项？
Checklist:
{checklist_str}

如果你发现任何一项在 DAG 中找不到对应的任务支撑，请按以下 JSON 格式报告 Issue：
[{{
    "issue_type": "SCOPE_MISMATCH",
    "description": "指出遗漏的具体 Checklist 项",
    "severity": "MEDIUM",
    "recommended_action": "add_task"
}}]
如果完美覆盖，返回空数组 []。
"""
        try:
            raw_issues = self._call(self._llm, [("system", verify_prompt)])
            resp_text = raw_issues.content.strip()
            if resp_text.startswith('```json'):
                resp_text = resp_text[7:-3]
            issues_json = json.loads(resp_text)
            issues = []
            for i in issues_json:
                issues.append(PlanIssue(
                    issue_type=i.get("issue_type", "SCOPE_MISMATCH"),
                    description=i.get("description", "Semantic coverage missing"),
                    severity=i.get("severity", "MEDIUM"),
                    target_id="",
                    recommended_action=i.get("recommended_action", "add_task")
                ))
            return issues
        except Exception:
            return super().critique(query, plan)

