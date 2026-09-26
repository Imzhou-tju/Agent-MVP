"""Conflict / Scope 检测（§25-28）。

职责边界：
- LLM 负责语义理解：抽取 claim 时给出 scope_fields（dataset/metric/protocol 等）
  与 polarity（positive/negative/neutral），以及对立的语义判断。
- 本模块负责确定性约束：根据结构化字段判断两条 claim 之间是
  NO_CONFLICT / DIRECT_CONFLICT / SCOPE_MISMATCH / POTENTIAL_CONFLICT。

不做简单字符串反转判断（§26）：不扫描 "better"/"worse" 这类词。
只比较显式提供的 scope_fields 与 polarity/value。
"""

from __future__ import annotations

from typing import Any

# 冲突判定结果
NO_CONFLICT = "NO_CONFLICT"
DIRECT_CONFLICT = "DIRECT_CONFLICT"
SCOPE_MISMATCH = "SCOPE_MISMATCH"
POTENTIAL_CONFLICT = "POTENTIAL_CONFLICT"

# 参与 scope 比较的关键字段（§24/§27）。缺少这些字段时两条 claim 不可比。
SCOPE_KEYS = ("dataset", "metric", "task", "protocol", "time_range",
              "population", "experimental_setting")


def _as_dict(claim: Any) -> dict:
    if isinstance(claim, dict):
        return claim
    return getattr(claim, "__dict__", {})


def _scope_fields(claim: Any) -> dict:
    d = _as_dict(claim)
    sf = d.get("scope_fields") or {}
    return dict(sf) if isinstance(sf, dict) else {}


def _polarity(claim: Any) -> str:
    return str((_as_dict(claim).get("polarity") or "")).strip().lower()


def _value(claim: Any) -> str:
    v = _as_dict(claim).get("value")
    return str(v).strip() if v is not None else ""


class ConflictDetector:
    """两条 claim 之间的冲突判定（确定性）。"""

    def detect(
        self,
        claim_a: Any,
        claim_b: Any,
        existing_relation: str | None = None,
    ) -> str:
        """返回四类结果之一。

        existing_relation: 若 ClaimGraph 中已显式存在 CONTRADICTS 关系则传此值，
        直接判定 DIRECT_CONFLICT（已有程序性标记时不再做启发式猜测）。
        """
        sa = _scope_fields(claim_a)
        sb = _scope_fields(claim_b)

        # 1) scope 可比性：两边都提供了值的 key 中，只要有一个不同 → SCOPE_MISMATCH。
        #    例：同方法不同 dataset（§27）→ SCOPE_MISMATCH，不是 DIRECT_CONFLICT。
        common = [k for k in sa if k in sb and sa.get(k) and sb.get(k)]
        if any(str(sa[k]).strip() != str(sb[k]).strip() for k in common):
            return SCOPE_MISMATCH

        # 2) 已有显式反驳关系 → DIRECT_CONFLICT。
        if existing_relation == "CONTRADICTS":
            return DIRECT_CONFLICT

        # 3) scope 可比（关键 key 一致或无冲突）且极性相反 → POTENTIAL_CONFLICT。
        pa, pb = _polarity(claim_a), _polarity(claim_b)
        if pa and pb and pa != pb and pa != "neutral" and pb != "neutral":
            return POTENTIAL_CONFLICT

        # 4) scope 一致且都带有可比数值、但数值不同 → POTENTIAL_CONFLICT。
        #    例：同 dataset/metric/protocol 下 X=2.1% 与 X=4.8%（§28）。
        va, vb = _value(claim_a), _value(claim_b)
        if va and vb and va != vb and common:
            return POTENTIAL_CONFLICT

        return NO_CONFLICT


def detect_pairwise(
    claims: list[Any],
    relation_of: Any = None,
) -> list[dict]:
    """对 claims 列表做两两检测，返回所有非 NO_CONFLICT 的结果。

    relation_of(a_id, b_id) -> str | None：查询已存在的 Claim-Claim 关系。
    """
    out: list[dict] = []
    n = len(claims)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = claims[i], claims[j]
            aid = _as_dict(a).get("claim_id", "")
            bid = _as_dict(b).get("claim_id", "")
            rel = relation_of(aid, bid) if relation_of else None
            verdict = ConflictDetector().detect(a, b, rel)
            if verdict != NO_CONFLICT:
                out.append({
                    "claim_a": aid,
                    "claim_b": bid,
                    "verdict": verdict,
                })
    return out
