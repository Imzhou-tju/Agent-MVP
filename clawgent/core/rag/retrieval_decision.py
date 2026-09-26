"""多跳检索决策层（§多跳检索决策机制升级）。

职责：把 `search_iterative` 里「下一步检索什么、要不要继续」的判断从
「模型返回一个 next_query 字符串」升级成结构化 `RetrievalDecision`，
并把终止判定做成确定性规则。

确定性边界：
- 本模块不调模型、不依赖 langchain、不做语义相似度判断。
- LLM 只负责产出决策的原始 JSON；字段校验、枚举收敛、截断、
  以及「继续 / 停止」的最终判定全部由这里的规则完成。
- 证据增益只按 evidence_id 做集合去重，不做质量打分、不算置信度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# ---- 检索意图枚举 ----
SEARCH_INTENTS = (
    "FACT", "MECHANISM", "COMPARISON", "RELATION",
    "PERFORMANCE", "CAUSE", "LIMITATION", "OTHER",
)

# ---- 停止原因 ----
STOP_SUFFICIENT = "SUFFICIENT"                    # 模型判定证据已足够
STOP_NO_NEW_SUBQUESTION = "NO_NEW_SUBQUESTION"   # next_query 与已执行过的 query 重复
STOP_NO_EVIDENCE_GAIN = "NO_EVIDENCE_GAIN"       # 本轮没有新 evidence
STOP_REPEATED_GAP = "REPEATED_GAP"               # 连续两轮 gap 相同
STOP_MAX_ITERATIONS = "MAX_ITERATIONS"           # 达到 RAG_MAX_ITERS 硬上限
STOP_INVALID_DECISION = "INVALID_DECISION"       # 决策非法（CONTINUE 但无 next_query / 解析失败）

ACTION_CONTINUE = "CONTINUE"
ACTION_STOP = "STOP"

MAX_EXPECTED_EVIDENCE = 3


def normalize_query(query: str) -> str:
    """query / gap 的归一化：只做去空白与小写，不做语义比较。"""
    return str(query or "").strip().lower()


def evidence_id_of(doc: Any) -> str:
    """一条检索结果的稳定标识：优先 chunk_id，退化到 document_name + 文本前缀。"""
    if not isinstance(doc, dict):
        return ""
    cid = str(doc.get("chunk_id") or "").strip()
    if cid:
        return cid
    name = str(doc.get("document_name") or "").strip()
    text = str(doc.get("text") or "").strip()[:64]
    return f"{name}::{text}" if (name or text) else ""


@dataclass
class RetrievalDecision:
    """一轮检索之后「下一步怎么做」的结构化决策。"""

    action: str = ACTION_STOP                     # CONTINUE / STOP
    gap: str = ""                                 # 当前仍缺什么
    search_intent: str = ""                       # 下一轮检索目标类型（枚举之一）
    next_query: str = ""                          # 下一轮实际检索式
    expected_evidence: list[str] = field(default_factory=list)  # 期望获得的证据描述，最多 3 条
    stop_reason: str = ""                         # STOP 时的原因

    @classmethod
    def from_dict(cls, data: Any) -> "RetrievalDecision":
        """把模型输出的原始 dict 收敛成合法决策。

        规则：
        - action 不是 CONTINUE 一律当 STOP，并忽略 next_query / expected_evidence；
        - search_intent 不在枚举内 → OTHER；
        - expected_evidence 超过 3 条 → 截断；
        - CONTINUE 但 next_query 为空 → 转成 STOP / INVALID_DECISION。
        """
        raw = data if isinstance(data, dict) else {}
        action = str(raw.get("action", "") or "").strip().upper()
        gap = str(raw.get("gap", "") or "").strip()
        intent = str(raw.get("search_intent", "") or "").strip().upper()
        if intent not in SEARCH_INTENTS:
            intent = "OTHER"
        expected = [
            str(x).strip() for x in (raw.get("expected_evidence") or [])
            if str(x).strip()
        ][:MAX_EXPECTED_EVIDENCE]
        next_query = str(raw.get("next_query", "") or "").strip()
        stop_reason = str(raw.get("stop_reason", "") or "").strip().upper()

        if action != ACTION_CONTINUE:
            # STOP：忽略 next_query 与 expected_evidence，避免带着残留字段继续跑
            return cls(action=ACTION_STOP, gap=gap,
                       stop_reason=stop_reason or STOP_SUFFICIENT)
        if not next_query:
            return cls(action=ACTION_STOP, gap=gap, search_intent=intent,
                       stop_reason=STOP_INVALID_DECISION)
        return cls(action=ACTION_CONTINUE, gap=gap, search_intent=intent,
                   next_query=next_query, expected_evidence=expected)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "gap": self.gap,
            "search_intent": self.search_intent,
            "next_query": self.next_query,
            "expected_evidence": list(self.expected_evidence),
            "stop_reason": self.stop_reason,
        }


class IterativeLoopState:
    """多跳循环的状态：已执行的 query、已见过的 evidence、历史 gap。"""

    def __init__(self) -> None:
        self.seen_queries: set[str] = set()
        self.seen_evidence_ids: set[str] = set()
        self.previous_gaps: list[str] = []
        self.repeated_gap_count: int = 0

    def note_query(self, query: str) -> None:
        self.seen_queries.add(normalize_query(query))

    def new_evidence(self, docs: Iterable[Any]) -> list[dict]:
        """返回本轮中未出现过的检索结果（按 evidence_id 去重）。"""
        fresh: list[dict] = []
        for d in docs or []:
            eid = evidence_id_of(d)
            if not eid or eid in self.seen_evidence_ids:
                continue
            self.seen_evidence_ids.add(eid)
            fresh.append(d)
        return fresh

    def register_evidence(self, docs: Iterable[Any]) -> int:
        return len(self.new_evidence(docs))

    def evaluate(self, decision: RetrievalDecision) -> tuple[bool, str]:
        """按 spec 第八节第 10-14 步判定是否继续。

        返回 (是否继续, 停止原因)。继续时停止原因为空串。
        """
        if decision is None or decision.action != ACTION_CONTINUE:
            return False, (decision.stop_reason if decision else STOP_INVALID_DECISION) or STOP_SUFFICIENT

        # 11) next_query 为空 → 非法决策
        if not decision.next_query:
            return False, STOP_INVALID_DECISION

        # 12) 该 query 已执行过 → 不再重复检索
        if normalize_query(decision.next_query) in self.seen_queries:
            return False, STOP_NO_NEW_SUBQUESTION

        # 13) gap 与上一轮相同 → 原地打转
        gap = normalize_query(decision.gap)
        if gap and self.previous_gaps and gap == self.previous_gaps[-1]:
            self.repeated_gap_count += 1
            self.previous_gaps.append(gap)
            if self.repeated_gap_count >= 1:
                return False, STOP_REPEATED_GAP
        else:
            self.repeated_gap_count = 0
            if gap:
                self.previous_gaps.append(gap)

        return True, ""


def run_iterative_loop(
    query: str,
    *,
    max_iters: int,
    retrieve_fn: Callable[[str], list[dict]],
    compress_fn: Callable[[str, list[dict]], str],
    decide_fn: Callable[[str, dict], Any],
    synthesize_fn: Callable[[str, dict], str] | None = None,
) -> dict:
    """多跳循环编排（spec 第八节的 14 步顺序）。

    依赖全部由调用方注入，因此不绑定 LangChain、可离线测试。

    返回字段在原有 answer/findings/iterations/sources 之外，
    额外给出 iteration_decisions 与 stop_reason（调试与可观测用）。
    """
    loop = IterativeLoopState()
    scratchpad: dict = {
        "sub_questions_asked": [],
        "intermediate_findings": [],
        "open_gaps": [],
        "all_sources": [],
        "evidence_state": [],          # 只存 {evidence_id, summary, source_id}
        "iteration_decisions": [],
    }

    current_query = query
    iterations = 0
    stop_reason = ""

    for iteration in range(1, max_iters + 1):
        # 1) 检索（底层双路召回 + rerank + 片段过滤由 retrieve_fn 内部完成）
        docs = retrieve_fn(current_query) or []
        iterations = iteration
        loop.note_query(current_query)
        scratchpad["sub_questions_asked"].append(current_query)

        for d in docs:
            name = d.get("document_name", "")
            if name and name not in scratchpad["all_sources"]:
                scratchpad["all_sources"].append(name)

        # 4) 更新 evidence_state（只记新增证据，不存正文）
        # 5) 本轮新证据数
        fresh = loop.new_evidence(docs)
        new_evidence_count = len(fresh)

        # 3) 结论压缩
        finding = compress_fn(current_query, docs)
        scratchpad["intermediate_findings"].append(finding)
        for d in fresh:
            scratchpad["evidence_state"].append({
                "evidence_id": evidence_id_of(d),
                "summary": finding,
                "source_id": d.get("document_name", ""),
            })

        def _record(action: str, decision: RetrievalDecision | None, reason: str) -> None:
            scratchpad["iteration_decisions"].append({
                "iteration": iteration,
                "action": action,
                "gap": (decision.gap if decision else ""),
                "search_intent": (decision.search_intent if decision else ""),
                "next_query": (decision.next_query if decision else ""),
                "new_evidence_count": new_evidence_count,
                "stop_reason": reason,
            })

        # 6) RAG_MAX_ITERS 是硬上限，优先级最高
        if iteration >= max_iters:
            stop_reason = STOP_MAX_ITERATIONS
            _record(ACTION_STOP, None, stop_reason)
            break

        # 7) 本轮没有新证据 → 不再继续
        if new_evidence_count == 0:
            stop_reason = STOP_NO_EVIDENCE_GAIN
            _record(ACTION_STOP, None, stop_reason)
            break

        # 8/9) 推理并取结构化决策
        raw_decision = decide_fn(query, scratchpad)
        decision = raw_decision if isinstance(raw_decision, RetrievalDecision) \
            else RetrievalDecision.from_dict(raw_decision)

        # 10-14) 终止判定
        cont, reason = loop.evaluate(decision)
        _record(decision.action, decision, reason)
        if not cont:
            stop_reason = reason
            break

        if decision.gap:
            scratchpad["open_gaps"].append(decision.gap)
        current_query = decision.next_query

    answer = synthesize_fn(query, scratchpad) if synthesize_fn else ""

    return {
        "answer": answer,
        "findings": scratchpad["intermediate_findings"],
        "iterations": iterations,
        "sources": scratchpad["all_sources"],
        "evidence_state": scratchpad["evidence_state"],
        "iteration_decisions": scratchpad["iteration_decisions"],
        "stop_reason": stop_reason,
    }
