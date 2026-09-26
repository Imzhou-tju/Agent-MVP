"""Research Ledger 与 Research Trace：调研过程的记录与回溯。

职责边界：
- Ledger 是只追加的事件流水，记录"做过什么"
- Trace 是按"最终结论 → 声明 → 证据 → 来源 → 任务 → 检索式"回溯出来的路径

两者都是程序生成的，不依赖模型自述。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

# 事件类型
PLAN_CREATED = "plan_created"
# 计划被质量闸门判 REJECT，改用按 SOP 维度展开的兜底计划
PLAN_REJECTED_FALLBACK = "plan_rejected_fallback"
TASK_DISPATCHED = "task_dispatched"
TASK_COMPLETED = "task_completed"
TASK_FAILED = "task_failed"
TASK_ADDED = "task_added"
TASK_REOPENED = "task_reopened"
SOURCE_REGISTERED = "source_registered"
EVIDENCE_EXTRACTED = "evidence_extracted"
EVIDENCE_VERIFIED = "evidence_verified"
CLAIM_ADDED = "claim_added"
ISSUE_FOUND = "issue_found"
REVISION_CREATED = "revision_created"
UNSUPPORTED_CLAIM = "unsupported_claim"
VERDICT = "verdict"
PROGRESS_LOG = "progress_log"
REPORT_COMPILED = "report_compiled"


@dataclass
class LedgerEvent:
    seq: int
    event_type: str
    task_id: str = ""
    round_no: int = 0
    payload: dict = field(default_factory=dict)
    ts: float = 0.0

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "event_type": self.event_type,
            "task_id": self.task_id,
            "round_no": self.round_no,
            "payload": dict(self.payload),
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LedgerEvent":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


class ResearchLedger:
    """只追加的事件流水。并发写入时按 seq 排序后重排。"""

    def __init__(self, events: Iterable[dict | LedgerEvent] | None = None):
        self.events: list[LedgerEvent] = []
        for e in events or []:
            self.events.append(e if isinstance(e, LedgerEvent) else LedgerEvent.from_dict(e))
        self._renumber()
        # 记录载入时的基线：节点只把本节点新增的事件写回 state。
        # 若把全量事件写回，配合 append-only reducer 会让同一条事件每经过一个节点就被复制一份。
        self._base = len(self.events)

    def _renumber(self) -> None:
        self.events.sort(key=lambda e: (e.ts, e.seq))
        for i, e in enumerate(self.events, start=1):
            e.seq = i

    def append(self, event_type: str, task_id: str = "", round_no: int = 0,
               **payload: Any) -> LedgerEvent:
        ev = LedgerEvent(
            seq=len(self.events) + 1,
            event_type=event_type,
            task_id=task_id,
            round_no=round_no,
            payload=payload,
            ts=time.time(),
        )
        self.events.append(ev)
        return ev

    def by_type(self, event_type: str) -> list[LedgerEvent]:
        return [e for e in self.events if e.event_type == event_type]

    def by_task(self, task_id: str) -> list[LedgerEvent]:
        return [e for e in self.events if e.task_id == task_id]

    def to_dicts(self) -> list[dict]:
        return [e.to_dict() for e in self.events]

    def delta(self) -> list[dict]:
        """本实例新增的事件，供节点写回 state（避免全量回写造成重复）。"""
        return [e.to_dict() for e in self.events[self._base:]]

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for e in self.events:
            counts[e.event_type] = counts.get(e.event_type, 0) + 1
        return {"total_events": len(self.events), "by_type": counts}


def build_trace(
    claims: Iterable[dict],
    evidences: Iterable[dict],
    sources: Iterable[dict],
    tasks: Iterable[dict],
    used_claim_ids: Iterable[str] | None = None,
) -> dict:
    """从最终声明回溯到检索式。

    输入均为 dict 列表（state 里的结构），输出可直接写入报告附件或日志。
    """
    ev_index = {e.get("evidence_id", ""): e for e in evidences}
    src_index = {s.get("source_id", ""): s for s in sources}
    task_index = {t.get("task_id", ""): t for t in tasks}
    used = set(used_claim_ids or [])

    chains: list[dict] = []
    for c in claims:
        cid = c.get("claim_id", "")
        if used and cid not in used:
            continue
        evid_ids = c.get("evidence_ids", []) or []
        steps = []
        for eid in evid_ids:
            ev = ev_index.get(eid, {})
            src = src_index.get(ev.get("source_id", ""), {})
            task = task_index.get(ev.get("task_id", ""), {})
            steps.append({
                "evidence_id": eid,
                "quote": (ev.get("quote", "") or "")[:120],
                "verification_status": ev.get("verification_status", ""),
                "source_id": src.get("source_id", ""),
                "source_title": src.get("title", ""),
                "source_url": src.get("url", ""),
                "task_id": task.get("task_id", ""),
                "task_question": task.get("question", ""),
                "retrieval_query": ev.get("retrieval_query", ""),
            })
        chains.append({
            "claim_id": cid,
            "claim_text": c.get("text", ""),
            "claim_type": c.get("claim_type", ""),
            "scope": c.get("scope", ""),
            "status": c.get("status", ""),
            "evidence_chain": steps,
        })

    return {
        "claim_count": len(chains),
        "chains": chains,
    }
