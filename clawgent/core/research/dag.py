"""Research Task DAG：任务模型与确定性调度器。

职责边界：
- Planner 决定研究计划与任务依赖（LLM）
- 本模块只做确定性工作：状态推进、READY 判定、依赖校验、局部修改

DAG 是可变状态：Critic 发现问题后由 DAG Repair 新增或重开局部任务，
不重新生成整棵树。因此任务集合按 task_id 合并（update_by_task_id），
不使用 list append。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

PENDING = "PENDING"
READY = "READY"
RUNNING = "RUNNING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
BLOCKED = "BLOCKED"
SKIPPED = "SKIPPED"
REOPENED = "REOPENED"

# 向后兼容：部分调用方/测试仍使用 DONE 字面量
DONE = COMPLETED

TERMINAL_STATES = (COMPLETED, FAILED, SKIPPED)

# 单轮 DAG Repair 允许新增的任务数上限，防止 Critic 一次扩出大量任务
MAX_REPAIR_TASKS = 3

# 任务类型建议分类（§4）。Planner 至少区分这些类型，决定任务职责与依赖补全。
TASK_TYPES = (
    "FACT", "MECHANISM", "COMPARISON", "TREND",
    "EVALUATION", "BACKGROUND", "LIMITATION", "METHOD", "SYNTHESIS",
)
_DEFAULT_TASK_TYPE = "FACT"

# 依赖补全规则（§5）：仅当能从「类型 + 标题/描述 + 引用」确定存在语义依赖时才补。
# 键为「下游任务类型」，值为「它应当依赖的上游任务类型」。
_DEPENDENCY_RULES = {
    "COMPARISON": ("FACT", "MECHANISM", "METHOD"),
    "EVALUATION": ("FACT", "MECHANISM", "METHOD", "RESULT"),
    "TREND": ("BACKGROUND", "FACT"),
    "LIMITATION": ("METHOD", "EVALUATION", "RESULT"),
}

# 合法状态迁移表（§6）。所有状态变化统一经过 transition_task，
# 杜绝 DONE → RUNNING 这类非法迁移。
_TRANSITIONS = {
    PENDING: {READY, BLOCKED, SKIPPED},
    READY: {RUNNING, SKIPPED, BLOCKED},
    RUNNING: {COMPLETED, FAILED, BLOCKED},
    COMPLETED: {REOPENED},
    FAILED: {REOPENED, PENDING},
    BLOCKED: {READY, PENDING, SKIPPED},
    SKIPPED: {PENDING},
    REOPENED: {READY, BLOCKED},
}


@dataclass
class ResearchTask:
    task_id: str
    objective: str = ""
    question: str = ""
    task_type: str = "FACT"                 # FACT|METHOD|COMPARISON|TREND|SYNTHESIS
    expected_evidence: str = ""
    preferred_sources: list[str] = field(default_factory=list)
    search_strategy: str = ""
    dependencies: list[str] = field(default_factory=list)
    success_criteria: str = ""
    # 结构化完成条件（§Plan Verification）：required_fields / required_source_types /
    # min_evidence_count，供 TaskSuccessCriteriaEvaluator 做确定性 PASS/PARTIAL/FAIL。
    # 与字符串 success_criteria 并存：后者仍给 LLM 读，前者给程序判。
    criteria: dict = field(default_factory=dict)
    # 该任务覆盖的研究维度/能力标签（§Plan Coverage），如 method / dataset / comparison。
    capabilities: list[str] = field(default_factory=list)
    priority: int = 1
    status: str = PENDING
    round_added: int = 0
    reopen_count: int = 0
    # 修订任务（§33）：记录它因哪个任务、第几轮产生
    parent_task_id: str = ""
    revision_round: int = 0
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchTask":
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in (d or {}).items() if k in known}
        kwargs.setdefault("task_id", "")
        t = cls(**kwargs)
        # LLM 可能把 dependencies 输出成 None 或非字符串
        t.dependencies = [str(x) for x in (t.dependencies or []) if x]
        t.preferred_sources = [str(x) for x in (t.preferred_sources or []) if x]
        t.capabilities = [str(x) for x in (t.capabilities or []) if x]
        if not isinstance(t.criteria, dict):
            t.criteria = {}
        return t

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "question": self.question,
            "task_type": self.task_type,
            "expected_evidence": self.expected_evidence,
            "preferred_sources": list(self.preferred_sources),
            "search_strategy": self.search_strategy,
            "dependencies": list(self.dependencies),
            "success_criteria": self.success_criteria,
            "criteria": dict(self.criteria),
            "capabilities": list(self.capabilities),
            "priority": self.priority,
            "status": self.status,
            "round_added": self.round_added,
            "reopen_count": self.reopen_count,
            "parent_task_id": self.parent_task_id,
            "revision_round": self.revision_round,
            "notes": self.notes,
        }


@dataclass
class ResearchPacket:
    """Researcher 的返回单元。

    Researcher 不直接修改全局证据，只返回本 Task 的结果包，
    由 Aggregator 纳入全局研究状态。
    """
    task_id: str
    searched_queries: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)
    unresolved_issues: list[str] = field(default_factory=list)
    status: str = DONE
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "searched_queries": list(self.searched_queries),
            "source_ids": list(self.source_ids),
            "evidence_ids": list(self.evidence_ids),
            "claims": list(self.claims),
            "unresolved_issues": list(self.unresolved_issues),
            "status": self.status,
            "error": self.error,
        }


@dataclass
class ResearchPlan:
    """Planner 的显式产出（§4.1）：研究目标 + 约束 + 任务 DAG。

    ResearchPlan 持有 objective / constraints / source_policy / success_criteria，
    任务以 ResearchTask 列表形式存在；调度器消费的仍是 TaskDAG（由 tasks 构建）。
    """

    plan_id: str = ""
    objective: str = ""
    constraints: str = ""
    source_policy: str = ""
    success_criteria: str = ""
    # 计划必须覆盖的研究维度/能力（§Plan Coverage）。由 Planner 声明或按规则兜底，
    # 供 PlanCoverageValidator 做集合覆盖判断。为空表示不启用覆盖校验（向后兼容）。
    required_dimensions: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    tasks: list[ResearchTask] = field(default_factory=list)
    task_dicts: list[dict] = field(default_factory=list)  # 序列化视图

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "objective": self.objective,
            "constraints": self.constraints,
            "source_policy": self.source_policy,
            "success_criteria": self.success_criteria,
            "required_dimensions": list(self.required_dimensions),
            "required_capabilities": list(self.required_capabilities),
            "tasks": [t.to_dict() for t in self.tasks],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchPlan":
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in (d or {}).items() if k in known}
        tasks = [ResearchTask.from_dict(t) for t in (kwargs.get("tasks") or [])]
        kwargs["tasks"] = tasks
        plan = cls(**kwargs)
        plan.task_dicts = [t.to_dict() for t in tasks]
        # 归一化：维度/能力列表去空去重
        plan.required_dimensions = list(dict.fromkeys(
            str(x) for x in (plan.required_dimensions or []) if x))
        plan.required_capabilities = list(dict.fromkeys(
            str(x) for x in (plan.required_capabilities or []) if x))
        return plan


def _task_text(task: dict) -> str:
    """把任务的可读文本拼成一个串，用于依赖补全时的引用检测。"""
    return " ".join(str(task.get(k, "")) for k in
                    ("title", "description", "question", "objective"))


def normalize_task_dependencies(tasks: list[dict]) -> list[dict]:
    """确定性依赖补全器（§5）。

    只在能从类型 + 文本引用确定存在语义依赖时补依赖，不盲目建边：
    - 按 _DEPENDENCY_RULES 找到「下游任务类型」允许依赖的「上游类型」候选；
    - 候选任务只有在被下游任务的文本显式引用（出现其 task_id 或标题子串）时才补为依赖。

    返回修改后的同一份列表（就地更新 dependencies），不改变其它字段。
    """
    index = {t.get("task_id"): t for t in tasks if t.get("task_id")}
    by_type: dict[str, list[str]] = {}
    for t in tasks:
        tt = str(t.get("task_type", _DEFAULT_TASK_TYPE)).upper()
        by_type.setdefault(tt, []).append(t.get("task_id"))

    for t in tasks:
        tid = t.get("task_id")
        tt = str(t.get("task_type", _DEFAULT_TASK_TYPE)).upper()
        needed = _DEPENDENCY_RULES.get(tt)
        if not needed:
            continue
        deps = set(t.get("dependencies") or [])
        text = _task_text(t)
        for req in needed:
            for cid in by_type.get(req, []):
                if cid == tid or cid in deps:
                    continue
                cand = index.get(cid, {})
                cand_title = str(cand.get("title", ""))
                # 仅当显式引用才补：出现候选 task_id，或候选标题作为子串出现
                if cid in text or (cand_title and cand_title in text):
                    deps.add(cid)
        t["dependencies"] = sorted(deps)
    return tasks


class TaskDAG:
    """任务图：状态推进与依赖判定，全部为确定性逻辑。"""

    def __init__(self, tasks: Iterable[ResearchTask | dict] | None = None):
        self.tasks: dict[str, ResearchTask] = {}
        for t in tasks or []:
            task = t if isinstance(t, ResearchTask) else ResearchTask.from_dict(t)
            if task.task_id:
                self.tasks[task.task_id] = task

    @classmethod
    def from_dicts(cls, dicts: Iterable[dict]) -> "TaskDAG":
        return cls(dicts)

    def to_dicts(self) -> list[dict]:
        return [t.to_dict() for t in self.tasks.values()]

    def get(self, task_id: str) -> ResearchTask | None:
        return self.tasks.get(task_id)

    def has(self, task_id: str) -> bool:
        return task_id in self.tasks

    # ------------------------------------------------------------------
    # 状态推进
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """按依赖完成情况把 PENDING / BLOCKED / REOPENED 推进到 READY 或 BLOCKED。"""
        for task in self.tasks.values():
            if task.status in (RUNNING, COMPLETED, FAILED, SKIPPED):
                continue
            if self._deps_satisfied(task):
                task.status = READY
            else:
                task.status = BLOCKED

    def _deps_satisfied(self, task: ResearchTask) -> bool:
        for dep in task.dependencies:
            dep_task = self.tasks.get(dep)
            if dep_task is None or dep_task.status != COMPLETED:
                return False
        return True

    def ready_tasks(self) -> list[ResearchTask]:
        self.refresh()
        ready = [t for t in self.tasks.values() if t.status == READY]
        ready.sort(key=lambda t: (-int(t.priority or 0), t.task_id))
        return ready

    def mark_running(self, task_ids: Iterable[str]) -> None:
        for tid in task_ids:
            if tid in self.tasks:
                self.transition_task(tid, RUNNING)

    def mark_done(self, task_ids: Iterable[str]) -> None:
        for tid in task_ids:
            if tid in self.tasks:
                self.transition_task(tid, COMPLETED)

    def mark_failed(self, task_ids: Iterable[str]) -> None:
        for tid in task_ids:
            if tid in self.tasks:
                self.transition_task(tid, FAILED)

    def transition_task(self, task_id: str, new_status: str) -> tuple[bool, str]:
        """统一状态迁移入口（§6）。

        拒绝非法迁移（如 COMPLETED → RUNNING）。返回 (是否成功, 原因)。
        所有节点修改任务状态都应经由此方法，保证状态机一致。
        """
        task = self.tasks.get(task_id)
        if task is None:
            return False, "task 不存在"
        cur = task.status
        if cur == new_status:
            return True, ""
        allowed = _TRANSITIONS.get(cur, set())
        if new_status not in allowed:
            return False, f"非法状态迁移: {cur} -> {new_status}"
        task.status = new_status
        return True, ""

    def reopen(self, task_id: str, reason: str = "") -> bool:
        """重开一个已完成的任务，并把它的下游标记为 REOPENED（局部影响）。"""
        task = self.tasks.get(task_id)
        if task is None or task.status in (RUNNING, REOPENED):
            return False
        ok, _ = self.transition_task(task_id, REOPENED)
        if not ok:
            return False
        task.reopen_count += 1
        if reason:
            task.notes = reason
        self._invalidate_downstream(task_id)
        return True

    def downstream(self, task_id: str) -> set[str]:
        """该任务的所有下游任务（含间接）。纯查询，不修改状态。"""
        children: dict[str, list[str]] = {tid: [] for tid in self.tasks}
        for tid, t in self.tasks.items():
            for dep in t.dependencies:
                if dep in children:
                    children[dep].append(tid)

        seen: set[str] = set()
        stack = list(children.get(task_id, []))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(children.get(cur, []))
        return seen

    def _invalidate_downstream(self, task_id: str) -> None:
        """把 task_id 的下游（含间接）中已完成的任务重开为 REOPENED（局部影响）。"""
        for tid in self.downstream(task_id):
            t = self.tasks[tid]
            if t.status == COMPLETED:
                self.transition_task(tid, REOPENED)
                t.reopen_count += 1

    # ------------------------------------------------------------------
    # 结构修改与校验
    # ------------------------------------------------------------------

    def add_task(self, task: ResearchTask) -> tuple[bool, str]:
        """新增任务。返回 (是否成功, 原因)。

        拒绝两种非法结构：依赖指向不存在的任务；新增后产生环。
        """
        if not task.task_id:
            return False, "task_id 为空"
        if task.task_id in self.tasks:
            return False, f"task_id 重复: {task.task_id}"
        if task.task_id in task.dependencies:
            return False, f"自依赖: {task.task_id}"

        for dep in task.dependencies:
            if dep not in self.tasks:
                return False, f"依赖不存在: {dep}"

        self.tasks[task.task_id] = task
        if self._has_cycle():
            del self.tasks[task.task_id]
            return False, "新增后产生环"

        return True, ""

    def _has_cycle(self) -> bool:
        """DFS 三色标记检测环。"""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {tid: WHITE for tid in self.tasks}

        def visit(tid: str) -> bool:
            color[tid] = GRAY
            for dep in self.tasks[tid].dependencies:
                if dep not in color:
                    continue
                if color[dep] == GRAY:
                    return True
                if color[dep] == WHITE and visit(dep):
                    return True
            color[tid] = BLACK
            return False

        for tid in list(self.tasks):
            if color[tid] == WHITE and visit(tid):
                return True
        return False

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def roots(self) -> list[str]:
        """无依赖的任务（入度为 0），即 DAG 的 root。"""
        return [tid for tid, t in self.tasks.items() if not t.dependencies]

    def reachable_from_roots(self) -> set[str]:
        """从所有 root 出发能到达的任务集合（含 root 本身）。

        无法从 root 到达的任务是「悬挂」任务——它永远不会被调度执行。
        供 PlanSchemaValidator 检查 root 可达性。
        """
        reachable: set[str] = set()
        stack = list(self.roots())
        while stack:
            cur = stack.pop()
            if cur in reachable:
                continue
            reachable.add(cur)
            for tid, t in self.tasks.items():
                if cur in t.dependencies and tid not in reachable:
                    stack.append(tid)
        return reachable

    def is_complete(self) -> bool:
        return all(t.status in TERMINAL_STATES for t in self.tasks.values())

    def pending_or_ready(self) -> list[ResearchTask]:
        return [t for t in self.tasks.values() if t.status in (PENDING, READY, BLOCKED, REOPENED)]

    def stats(self) -> dict:
        counts: dict[str, int] = {}
        for t in self.tasks.values():
            counts[t.status] = counts.get(t.status, 0) + 1
        return counts
