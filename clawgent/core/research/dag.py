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
DONE = "DONE"
FAILED = "FAILED"
BLOCKED = "BLOCKED"
REOPENED = "REOPENED"

TERMINAL_STATES = (DONE, FAILED)

# 单轮 DAG Repair 允许新增的任务数上限，防止 Critic 一次扩出大量任务
MAX_REPAIR_TASKS = 3


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
    priority: int = 1
    status: str = PENDING
    round_added: int = 0
    reopen_count: int = 0
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
            "priority": self.priority,
            "status": self.status,
            "round_added": self.round_added,
            "reopen_count": self.reopen_count,
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
            if task.status in (RUNNING, DONE, FAILED):
                continue
            if self._deps_satisfied(task):
                task.status = READY
            else:
                task.status = BLOCKED

    def _deps_satisfied(self, task: ResearchTask) -> bool:
        for dep in task.dependencies:
            dep_task = self.tasks.get(dep)
            if dep_task is None or dep_task.status != DONE:
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
                self.tasks[tid].status = RUNNING

    def mark_done(self, task_ids: Iterable[str]) -> None:
        for tid in task_ids:
            if tid in self.tasks:
                self.tasks[tid].status = DONE

    def mark_failed(self, task_ids: Iterable[str]) -> None:
        for tid in task_ids:
            if tid in self.tasks:
                self.tasks[tid].status = FAILED

    def reopen(self, task_id: str, reason: str = "") -> bool:
        """重开一个已完成的任务，并把它的下游标记为 REOPENED（局部影响）。"""
        task = self.tasks.get(task_id)
        if task is None or task.status in (RUNNING, REOPENED):
            return False
        task.status = REOPENED
        task.reopen_count += 1
        if reason:
            task.notes = reason
        self._invalidate_downstream(task_id)
        return True

    def downstream(self, task_id: str) -> set[str]:
        """该任务的所有下游任务（含间接）。"""
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
        for tid in self.downstream(task_id):
            t = self.tasks[tid]
            if t.status == DONE:
                t.status = REOPENED
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

    def is_complete(self) -> bool:
        return all(t.status in TERMINAL_STATES for t in self.tasks.values())

    def pending_or_ready(self) -> list[ResearchTask]:
        return [t for t in self.tasks.values() if t.status in (PENDING, READY, BLOCKED, REOPENED)]

    def stats(self) -> dict:
        counts: dict[str, int] = {}
        for t in self.tasks.values():
            counts[t.status] = counts.get(t.status, 0) + 1
        return counts
