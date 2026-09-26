"""Research 子图的状态定义与语义化 reducer。

设计要点（与前版的关键差别）：

1. 并发写入字段必须声明 reducer，否则触发 INVALID_CONCURRENT_GRAPH_UPDATE。
   前版统一用 operator.add（列表追加）。追加语义在 Researcher 通过 Send 并发执行时
   会产生重复条目：多个任务引用同一来源时，sources 里会出现同一 URL 的多份登记。
   本版改为按 id 合并的语义化 reducer：
       sources    → 按 source_id 合并（缺字段补齐）
       evidences  → 按 evidence_id 合并
       claims     → 按 claim_id 合并，并合并 evidence_ids
       relations  → 按 (source_id, target_id, relation) 去重
   配合 evidence.stable_id / claim.make_claim_id 用内容哈希生成 id，
   同一来源、同一句话在不同并发分支里会得到同一个 id，从而收敛为一条。

2. tasks 是 DAG，会被 Planner 建立、被 Scheduler 推进状态、被 Repair 局部修改。
   DAG 是可变结构，不能用追加语义；按 task_id 覆盖合并。

3. 删除 confidence_score（0.5 + 0.1*证据数 - 0.2*问题数 的启发式打分，
   与证据是否被校验通过无关）。改为由程序统计的 evidence_support，
   以及每条 Claim 上的 status（见 claim.ClaimGraph.recompute_statuses）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Iterable, TypedDict

# Judge 的裁决
COMPILE = "COMPILE"
COMPILE_WITH_LIMITATIONS = "COMPILE_WITH_LIMITATIONS"
REVISE = "REVISE"
CONTINUE_UNFINISHED = "CONTINUE_UNFINISHED"   # 原始计划任务仍有可执行项，继续循环
ABORT_WITH_LIMITATIONS = "ABORT_WITH_LIMITATIONS"


# ---------------------------------------------------------------------------
# 语义化 reducer
# ---------------------------------------------------------------------------

def _as_list(value: Any) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _merge_dicts(old: dict, new: dict, union_fields: tuple[str, ...] = ()) -> dict:
    """合并两个 dict：new 的非空值覆盖 old，union_fields 取并集。"""
    merged = dict(old)
    for k, v in new.items():
        if k in union_fields:
            cur = merged.get(k) or []
            cur = cur if isinstance(cur, list) else [cur]
            add = v if isinstance(v, list) else [v]
            for item in add:
                if item not in cur:
                    cur.append(item)
            merged[k] = cur
        elif v not in (None, "", [], {}):
            merged[k] = v
        elif k not in merged:
            merged[k] = v
    return merged


def _merge_by_key(left: Any, right: Any, key: str,
                  union_fields: tuple[str, ...] = ()) -> list[dict]:
    out: dict[str, dict] = {}
    for item in _as_list(left) + _as_list(right):
        if not isinstance(item, dict):
            continue
        ident = item.get(key)
        if not ident:
            continue
        if ident in out:
            out[ident] = _merge_dicts(out[ident], item, union_fields)
        else:
            out[ident] = dict(item)
    return list(out.values())


def merge_tasks(left: Any, right: Any) -> list[dict]:
    """DAG 任务：按 task_id 覆盖合并（后写入的状态更新）。"""
    return _merge_by_key(left, right, "task_id")


def merge_sources(left: Any, right: Any) -> list[dict]:
    """来源登记表：按 source_id 合并，只补齐空字段，不覆盖已有值。"""
    return _merge_by_key(left, right, "source_id")


def merge_evidences(left: Any, right: Any) -> list[dict]:
    """证据：按 evidence_id 合并。"""
    return _merge_by_key(left, right, "evidence_id")


def merge_claims(left: Any, right: Any) -> list[dict]:
    """声明：按 claim_id 合并，evidence_ids 取并集。"""
    return _merge_by_key(left, right, "claim_id", union_fields=("evidence_ids",))


def merge_relations(left: Any, right: Any) -> list[dict]:
    """Claim-Evidence 关系：按三元组去重。"""
    out: dict[tuple, dict] = {}
    for item in _as_list(left) + _as_list(right):
        if not isinstance(item, dict):
            continue
        ident = (item.get("source_id", ""), item.get("target_id", ""), item.get("relation", ""))
        if not any(ident):
            continue
        if ident in out:
            out[ident] = _merge_dicts(out[ident], item)
        else:
            out[ident] = dict(item)
    return list(out.values())


def merge_issues(left: Any, right: Any) -> list[dict]:
    """评审问题：按 issue_id 合并。"""
    return _merge_by_key(left, right, "issue_id")


def merge_task_results(left: Any, right: Any) -> list[dict]:
    """ResearchPacket：按 task_id 覆盖（重开的任务产生新包，覆盖旧包）。"""
    return _merge_by_key(left, right, "task_id")


def merge_str_list(left: Any, right: Any) -> list[str]:
    """字符串列表去重（检索式等）。"""
    out: list[str] = []
    left = left if isinstance(left, list) else []
    right = right if isinstance(right, list) else []
    for v in left + right:
        if isinstance(v, str) and v and v not in out:
            out.append(v)
    return out


def append_list(left: Any, right: Any) -> list[Any]:
    """只追加：事件流水等天然 append-only 的字段。"""
    return list(_as_list(left)) + list(_as_list(right))


def merge_dict(left: Any, right: Any) -> dict:
    """dict 合并：右侧覆盖左侧。用于 source_id → 来源正文 的映射。"""
    out = dict(left or {})
    out.update(right or {})
    return out


# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------

@dataclass
class ResearchState:
    """ResearchStateDict 的 dataclass 投影，供阅读与新手指引使用。

    真实运行时状态是下面的 ResearchStateDict（TypedDict，支持 reducer 声明）。
    """

    def __init__(self):
        # 输入
        self.original_query: str = ""
        self.research_context: str = ""

        # Planner：Research Task DAG
        self.tasks: list[dict] = []
        self.plan: dict = field(default_factory=dict)   # ResearchPlan.to_dict()
        self.plan_summary: str = ""
        self.round_no: int = 0
        # 计划验证（§Plan Verification）：执行前 Schema/Coverage/Critic 的结果快照
        self.plan_validation: dict = field(default_factory=dict)
        # 计划质量闸门（§Plan Gate）：SOP 选型 + Validator + Critic + Repair 的门控结果
        self.plan_gate: dict = field(default_factory=dict)
        self.sop_type: str = ""
        # 执行后 Evidence/Claim 覆盖验证的结果快照
        self.coverage_validation: dict = field(default_factory=dict)

        # Researcher（Send 并发）→ 全部走语义化 reducer
        self.task_results: list[dict] = []
        self.sources: list[dict] = []
        self.evidences: list[dict] = []
        self.claims: list[dict] = []
        self.relations: list[dict] = []
        self.searched_queries: list[str] = []
        self.source_texts: dict = field(default_factory=dict)

        # Review
        self.issues: list[dict] = []
        self.review: dict = field(default_factory=dict)  # ResearchReview.to_dict()

        # Judge
        self.verdict: str = ""
        self.verdict_reason: str = ""
        self.evidence_support: dict = field(default_factory=dict)
        # 进展跟踪（§14/§38）：上一轮的声明/来源总数，用于算本轮新增
        self.last_claim_count: int = 0
        self.last_source_count: int = 0
        # 每轮进展日志：new_evidence/claim/source/task + 来源指纹（检测本轮是否真的发现新来源）
        self.progress_log: list[dict] = field(default_factory=list)

        # Compiler
        self.final_report: str = ""
        self.report_sections: list[dict] = []
        self.citation_issues: list[dict] = []
        self.trace: dict = field(default_factory=dict)

        # 过程记录
        self.ledger_events: list[dict] = []

        # 控制
        self.error: str = ""
        self.max_revisions: int = 2


class ResearchStateDict(TypedDict, total=False):
    # ---------------- 输入 ----------------
    original_query: str
    research_context: str

    # ---------------- Planner ----------------
    # DAG 是可变结构：Scheduler 推进状态、Repair 局部增删，故按 task_id 合并
    tasks: Annotated[list[dict], merge_tasks]
    plan: dict                              # ResearchPlan.to_dict()，覆盖合并
    plan_summary: str
    round_no: int
    # 计划验证（§Plan Verification）：执行前与执行后的验证快照
    plan_validation: dict
    # 计划质量闸门（§Plan Gate）：SOP 选型 + Validator + Critic + Repair 门控结果
    plan_gate: dict
    sop_type: str
    coverage_validation: dict

    # ---------------- Researcher（Send 并发写入，必须有 reducer）----------------
    task_results: Annotated[list[dict], merge_task_results]
    sources: Annotated[list[dict], merge_sources]
    evidences: Annotated[list[dict], merge_evidences]
    claims: Annotated[list[dict], merge_claims]
    relations: Annotated[list[dict], merge_relations]
    searched_queries: Annotated[list[str], merge_str_list]
    # 来源正文缓存：source_id → 可用于原文定位的文本（只用于校验，不进报告）
    source_texts: Annotated[dict, merge_dict]

    # ---------------- Review ----------------
    issues: Annotated[list[dict], merge_issues]
    review: dict                           # ResearchReview.to_dict()，覆盖合并
    # 已经被 repair 处理过的 issue_id，避免同一问题每轮都被拿来补任务
    repaired_issue_ids: Annotated[list[str], merge_str_list]

    # ---------------- Judge ----------------
    verdict: str
    verdict_reason: str
    evidence_support: dict
    # 进展检测（§38）：记录上一轮证据总数与连续无新增的轮数
    last_evidence_count: int
    stagnant_rounds: int
    # 进展跟踪（§14）：上一轮声明/来源总数 + 每轮进展日志
    last_claim_count: int
    last_source_count: int
    progress_log: Annotated[list[dict], append_list]

    # ---------------- Compiler ----------------
    final_report: str
    report_sections: list[dict]
    citation_issues: list[dict]
    trace: dict

    # ---------------- 过程记录 ----------------
    ledger_events: Annotated[list[dict], append_list]

    # ---------------- 控制 ----------------
    error: str
    max_revisions: int
