"""Research Review（§29）：把评审结果结构化。

职责边界：
- LLM（review_node）负责提出具体问题（Issue）与修复建议；
- 本模块负责确定性部分：基于 Claim 支撑状态与 ConflictDetector 结果，
  统计 task_coverage / claim_coverage / unsupported_claims / weak_evidence /
  contradictions / scope_mismatches，并装配成 ResearchReview。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .claim import (
    CONTRADICTS,
    PARTIALLY_SUPPORTED,
    SUPPORTED,
    UNSUPPORTED,
)
from .conflict import (
    DIRECT_CONFLICT,
    POTENTIAL_CONFLICT,
    SCOPE_MISMATCH,
    detect_pairwise,
)


@dataclass
class ResearchReview:
    """评审快照（§29）。"""

    task_coverage: float = 0.0          # 已完成任务 / 总任务
    claim_coverage: float = 0.0         # 可用声明 / 总声明
    unsupported_claims: list[str] = field(default_factory=list)
    weak_evidence: list[str] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)      # DIRECT / POTENTIAL
    scope_mismatches: list[dict] = field(default_factory=list)     # SCOPE_MISMATCH
    duplicate_research: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_coverage": self.task_coverage,
            "claim_coverage": self.claim_coverage,
            "unsupported_claims": list(self.unsupported_claims),
            "weak_evidence": list(self.weak_evidence),
            "contradictions": list(self.contradictions),
            "scope_mismatches": list(self.scope_mismatches),
            "duplicate_research": list(self.duplicate_research),
            "recommended_actions": list(self.recommended_actions),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchReview":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


def _relation_of(relations: Iterable[dict]):
    """构造 (a_id, b_id) -> relation 查询闭包，只关心 CONTRADICTS。"""
    table: dict[tuple[str, str], str] = {}
    for r in relations or []:
        if r.get("relation") == CONTRADICTS:
            table[(r.get("source_id", ""), r.get("target_id", ""))] = CONTRADICTS
    return lambda a, b: table.get((a, b))


def build_review(
    claims: list[dict],
    evidences: list[dict],
    tasks: list[dict],
    relations: list[dict],
    duplicate_research: Iterable[str] | None = None,
) -> ResearchReview:
    """基于当前全局状态计算结构化评审（确定性）。"""
    total_tasks = len(tasks)
    done_tasks = sum(1 for t in tasks if t.get("status") in ("COMPLETED", "DONE", "SKIPPED"))
    task_coverage = (done_tasks / total_tasks) if total_tasks else 0.0

    total_claims = len(claims)
    usable = [c for c in claims if c.get("status") in (SUPPORTED, PARTIALLY_SUPPORTED)]
    claim_coverage = (len(usable) / total_claims) if total_claims else 0.0

    unsupported = [c.get("claim_id", "") for c in claims
                   if c.get("status") in (UNSUPPORTED, "CONTRADICTED")]

    weak = [e.get("evidence_id", "") for e in evidences
            if e.get("verification_status") in ("PARTIAL", "UNVERIFIED", "INVALID")]

    # 两两冲突检测：确定性，基于 Claim.scope_fields / polarity / value
    pairs = detect_pairwise(claims, _relation_of(relations))
    contradictions = [p for p in pairs if p["verdict"] in (DIRECT_CONFLICT, POTENTIAL_CONFLICT)]
    scope_mismatches = [p for p in pairs if p["verdict"] == SCOPE_MISMATCH]

    actions: list[str] = []
    if unsupported:
        actions.append(f"补充 {len(unsupported)} 条未支撑声明的证据或调整结论")
    if contradictions:
        actions.append(f"复核 {len(contradictions)} 处结论冲突")
    if scope_mismatches:
        actions.append(f"区分 {len(scope_mismatches)} 处范围不一致的结论")

    return ResearchReview(
        task_coverage=task_coverage,
        claim_coverage=claim_coverage,
        unsupported_claims=unsupported,
        weak_evidence=weak,
        contradictions=contradictions,
        scope_mismatches=scope_mismatches,
        duplicate_research=list(duplicate_research or []),
        recommended_actions=actions,
    )
