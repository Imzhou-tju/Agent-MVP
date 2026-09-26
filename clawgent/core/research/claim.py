"""Claim Layer：声明（Claim）与 Claim-Evidence 关系图。

职责边界：
- Claim 的文本、类型、适用范围由 LLM 产出
- Claim 的支撑状态（supported / partially_supported / ...）由程序按
  其关联 Evidence 的校验状态推导，不由 LLM 自己打分

原实现里的 confidence_score = 0.5 + 0.1*len(evidences) - 0.2*len(high_issues)
是启发式打分，与证据质量无关；这里改为按证据校验结果推导的支撑状态。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable

# Claim 类型
FACT = "FACT"
METHOD = "METHOD"
RESULT = "RESULT"
COMPARISON = "COMPARISON"
LIMITATION = "LIMITATION"
TREND = "TREND"
INTERPRETATION = "INTERPRETATION"
HYPOTHESIS = "HYPOTHESIS"

CLAIM_TYPES = (FACT, METHOD, RESULT, COMPARISON, LIMITATION, TREND, INTERPRETATION, HYPOTHESIS)

# 由 LLM 产出时的兜底映射：无法识别的类型归入 FACT
_FALLBACK_TYPE = FACT

# 支撑状态
SUPPORTED = "SUPPORTED"
PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"
CONTRADICTED = "CONTRADICTED"

# Claim-Evidence 语义关系（由 LLM 判定 quote 是否真的支持 claim，§8）
# 与 Claim-Evidence 的结构关系（SUPPORTS/CONTRADICTS/...）分开：
# 结构关系来自程序推导，语义关系来自模型对「quote→claim」的判断。
SEM_SUPPORTS = "SUPPORTS"
SEM_PARTIALLY_SUPPORTS = "PARTIALLY_SUPPORTS"
SEM_CONTRADICTS = "CONTRADICTS"
SEM_IRRELEVANT = "IRRELEVANT"
SEMANTIC_RELATIONS = (SEM_SUPPORTS, SEM_PARTIALLY_SUPPORTS, SEM_CONTRADICTS, SEM_IRRELEVANT)

# Claim-Evidence / Claim-Claim 关系
SUPPORTS = "SUPPORTS"
CONTRADICTS = "CONTRADICTS"
QUALIFIES = "QUALIFIES"
DERIVED_FROM = "DERIVED_FROM"

RELATIONS = (SUPPORTS, CONTRADICTS, QUALIFIES, DERIVED_FROM)

# 只在 Evidence → Claim 之间允许的关系
_EVIDENCE_RELATIONS = (SUPPORTS, CONTRADICTS, QUALIFIES)
# 只在 Claim → Claim 之间允许的关系
_CLAIM_RELATIONS = (QUALIFIES, CONTRADICTS, DERIVED_FROM)


def normalize_claim_text(text: str) -> str:
    """声明文本归一化，用于去重与稳定 claim_id。"""
    if not text:
        return ""
    return re.sub(r"[\s\W_]+", "", text.lower())


def make_claim_id(text: str, prefix: str = "C") -> str:
    """按文本生成稳定 claim_id：同一句话在不同分支里得到同一个 id。"""
    norm = normalize_claim_text(text)
    if not norm:
        return ""
    return prefix + hashlib.sha1(norm.encode("utf-8")).hexdigest()[:8]


@dataclass
class Claim:
    claim_id: str
    text: str = ""
    claim_type: str = FACT
    scope: str = ""                 # 适用对象/范围的自然语言描述
    conditions: str = ""            # 成立条件，如 "在未做数据增强的前提下"
    # 结构化 scope（§24）：供 ConflictDetector 做确定性比较。
    # 至少包含 dataset / metric / task / protocol 等关键字段时才有可比性。
    scope_fields: dict = field(default_factory=dict)
    # 极性 / 取值（§25-28）：由 LLM 在抽取时给出，用于冲突判定。
    # polarity ∈ {positive, negative, neutral, ""}；value 为可比数值（如 "2.1%"）。
    polarity: str = ""
    value: str = ""
    # 语义关系（§8）：LLM 判定本 claim 与其 evidence 的 quote 之间是否真支持。
    # 缺省为空，程序推导时视作 SUPPORTS（quote 已定位且由抽取配对）。
    semantic_relation: str = ""
    task_id: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    status: str = UNSUPPORTED
    support_reason: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "claim_type": self.claim_type,
            "scope": self.scope,
            "conditions": self.conditions,
            "scope_fields": dict(self.scope_fields),
            "polarity": self.polarity,
            "value": self.value,
            "semantic_relation": self.semantic_relation,
            "task_id": self.task_id,
            "evidence_ids": list(self.evidence_ids),
            "status": self.status,
            "support_reason": self.support_reason,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Claim":
        known = {f for f in cls.__dataclass_fields__}
        c = cls(**{k: v for k, v in (d or {}).items() if k in known})
        c.claim_id = c.claim_id or make_claim_id(c.text)
        c.claim_type = c.claim_type if c.claim_type in CLAIM_TYPES else _FALLBACK_TYPE
        c.evidence_ids = [str(x) for x in (c.evidence_ids or []) if x]
        c.scope_fields = dict(c.scope_fields or {})
        c.semantic_relation = c.semantic_relation if c.semantic_relation in SEMANTIC_RELATIONS else ""
        return c


@dataclass
class ClaimRelation:
    source_id: str        # Evidence.evidence_id 或 Claim.claim_id
    target_id: str        # Claim.claim_id
    relation: str = SUPPORTS
    # 语义关系（§8）：仅当 source 是 Evidence 时由 LLM 判定 quote→claim 的语义。
    semantic_relation: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "semantic_relation": self.semantic_relation,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ClaimRelation":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


class ClaimGraph:
    """声明与证据之间的关系图。

    图的写入分两类：
    - 结构（Claim、关系）来自 LLM 抽取
    - 状态（Claim.status）来自程序推导，见 recompute_statuses
    """

    def __init__(self, claims: Iterable[dict | Claim] | None = None,
                 relations: Iterable[dict | ClaimRelation] | None = None):
        self.claims: dict[str, Claim] = {}
        self.relations: list[ClaimRelation] = []
        for c in claims or []:
            self.add_claim(c if isinstance(c, Claim) else Claim.from_dict(c))
        for r in relations or []:
            self.add_relation(r if isinstance(r, ClaimRelation) else ClaimRelation.from_dict(r))

    # ------------------------------------------------------------------
    # 结构写入
    # ------------------------------------------------------------------

    def add_claim(self, claim: Claim) -> Claim:
        if not claim.claim_id:
            claim.claim_id = make_claim_id(claim.text)
        if not claim.claim_id:
            return claim
        existed = self.claims.get(claim.claim_id)
        if existed is None:
            self.claims[claim.claim_id] = claim
            return claim
        # 同一句话被多个 Task 抽到：合并证据、补全空字段
        for eid in claim.evidence_ids:
            if eid not in existed.evidence_ids:
                existed.evidence_ids.append(eid)
        for f in ("scope", "conditions", "notes", "task_id"):
            if not getattr(existed, f) and getattr(claim, f):
                setattr(existed, f, getattr(claim, f))
        if claim.claim_type in CLAIM_TYPES and existed.claim_type == _FALLBACK_TYPE:
            existed.claim_type = claim.claim_type
        return existed

    def add_relation(self, rel: ClaimRelation) -> tuple[bool, str]:
        """新增关系。返回 (是否成功, 原因)。

        拒绝：目标不是已登记 Claim；关系类型与端点类型不匹配；重复关系。
        """
        if rel.relation not in RELATIONS:
            return False, f"不支持的关系类型: {rel.relation}"
        if rel.target_id not in self.claims:
            return False, f"目标 Claim 不存在: {rel.target_id}"

        is_evidence_endpoint = rel.source_id not in self.claims
        allowed = _EVIDENCE_RELATIONS if is_evidence_endpoint else _CLAIM_RELATIONS
        if rel.relation not in allowed:
            return False, f"{rel.relation} 不适用于该端点类型"

        for r in self.relations:
            if r.source_id == rel.source_id and r.target_id == rel.target_id \
                    and r.relation == rel.relation:
                return False, "关系重复"

        self.relations.append(rel)
        # evidence → claim 的 SUPPORTS 关系同时登记到 Claim.evidence_ids
        if is_evidence_endpoint and rel.relation == SUPPORTS \
                and rel.source_id not in self.claims[rel.target_id].evidence_ids:
            self.claims[rel.target_id].evidence_ids.append(rel.source_id)
        return True, ""

    def link_evidence(self, evidence_id: str, claim_id: str, relation: str = SUPPORTS,
                      note: str = "") -> tuple[bool, str]:
        return self.add_relation(ClaimRelation(evidence_id, claim_id, relation, note))

    def get_claim(self, claim_id: str) -> Claim | None:
        return self.claims.get(claim_id)

    # ------------------------------------------------------------------
    # 状态推导
    # ------------------------------------------------------------------

    def recompute_statuses(
        self,
        evidence_index: dict[str, dict],
        semantic_index: dict[tuple[str, str], str] | None = None,
    ) -> dict[str, str]:
        """确定性推导每条 Claim 的支撑状态（§9）。

        综合三类信息（均确定性，不依赖模型自评）：
        1. Evidence 的 verification_status（来自 EvidenceVerifier）
        2. Evidence→Claim 的语义关系 semantic_relation（来自 ClaimEvidenceSemanticVerifier）
        3. Claim 间 CONTRADICTS 结构关系（来自 ClaimGraph）

        evidence_index: {evidence_id: Evidence.to_dict()}，无对应条目视为缺失。
        semantic_index: {(evidence_id, claim_id): semantic_relation}，缺省视为支持。

        推导规则：
        - 无任何关联证据 → UNSUPPORTED
        - 存在同范围 CONTRADICTS 关系，或某证据 VERIFIED/PARTIAL 且语义 CONTRADICTS → CONTRADICTED
        - 至少 1 条 VERIFIED 且语义为 SUPPORTS/PARTIALLY_SUPPORTS → SUPPORTED
        - 至少 1 条（VERIFIED/PARTIAL）且语义 PARTIALLY_SUPPORTS → PARTIALLY_SUPPORTED
        - 其余（仅有 INVALID/UNVERIFIED 或语义 IRRELEVANT）→ UNSUPPORTED
        """
        semantic_index = semantic_index or {}
        contradicted: set[str] = set()
        for r in self.relations:
            if r.relation == CONTRADICTS:
                contradicted.add(r.target_id)

        for cid, claim in self.claims.items():
            verified_supports = 0
            partial_supports = 0
            contradicts = 0
            for eid in claim.evidence_ids:
                status = self._status_of(eid, evidence_index)
                sem = semantic_index.get((eid, cid), "") or SEM_SUPPORTS
                if status == "VERIFIED" and sem in (SEM_SUPPORTS, SEM_PARTIALLY_SUPPORTS):
                    verified_supports += 1
                elif status in ("VERIFIED", "PARTIAL") and sem == SEM_PARTIALLY_SUPPORTS:
                    partial_supports += 1
                elif status in ("VERIFIED", "PARTIAL") and sem == SEM_CONTRADICTS:
                    contradicts += 1

            if not claim.evidence_ids:
                status = UNSUPPORTED
                reason = "未关联任何证据"
            elif cid in contradicted or contradicts > 0:
                status = CONTRADICTED
                reason = "存在同范围相反证据"
            elif verified_supports > 0:
                status = SUPPORTED
                reason = f"{verified_supports} 条证据通过原文校验且语义支持"
            elif partial_supports > 0:
                status = PARTIALLY_SUPPORTED
                reason = f"{partial_supports} 条证据部分支持（或仅部分匹配原文）"
            else:
                status = UNSUPPORTED
                reason = "无（通过校验且语义支持）的证据"

            claim.status = status
            claim.support_reason = reason
        return {cid: c.status for cid, c in self.claims.items()}

    @staticmethod
    def _status_of(evidence_id: str, evidence_index: dict[str, dict]) -> str:
        ev = evidence_index.get(evidence_id)
        if not ev:
            return "MISSING"
        return ev.get("verification_status", "UNVERIFIED")

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def contradictions(self) -> list[dict]:
        """返回互相冲突的 Claim 对（CONTRADICTS 关系中两端都是 Claim 的）。"""
        out = []
        for r in self.relations:
            if r.relation == CONTRADICTS and r.source_id in self.claims:
                out.append({
                    "claim_a": r.source_id,
                    "claim_b": r.target_id,
                    "note": r.note,
                })
        return out

    def by_status(self, status: str) -> list[Claim]:
        return [c for c in self.claims.values() if c.status == status]

    def usable_claims(self) -> list[Claim]:
        """可用于成文的声明：SUPPORTED 与 PARTIALLY_SUPPORTED。"""
        return [c for c in self.claims.values() if c.status in (SUPPORTED, PARTIALLY_SUPPORTED)]

    def to_dicts(self) -> tuple[list[dict], list[dict]]:
        return (
            [c.to_dict() for c in self.claims.values()],
            [r.to_dict() for r in self.relations],
        )


def recompute_claim_statuses(
    claims: list[dict],
    evidences: list[dict],
    relations: list[dict],
) -> list[dict]:
    """便捷函数（§9 确定性基础设施）：直接对 dict 列表重算 Claim 状态。

    返回更新后的 claims 列表（含 status / support_reason）。语义关系从
    relations 里的 semantic_relation 字段构建索引。
    """
    g = ClaimGraph(claims, relations)
    semantic_index = {
        (r.get("source_id", ""), r.get("target_id", "")): r.get("semantic_relation", "")
        for r in relations
        if r.get("source_id") and r.get("target_id")
    }
    g.recompute_statuses(
        {e.get("evidence_id", ""): e for e in evidences},
        semantic_index,
    )
    return g.to_dicts()[0]
