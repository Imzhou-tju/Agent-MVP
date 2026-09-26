"""Research 子图的节点实现。

节点职责划分（确定性逻辑与模型调用分离）：

| 节点         | 是否调用模型 | 产出                                            |
|--------------|--------------|-------------------------------------------------|
| planner      | 是           | Research Task DAG（任务 + 依赖）                |
| scheduler    | 否           | 按依赖挑出 READY 任务，Send 给 researcher        |
| researcher   | 是           | ResearchPacket：来源登记 + 证据抽取 + 原文校验   |
| aggregator   | 否           | 合并、去重、推导 Claim 支撑状态、统计证据状况    |
| review       | 是           | 结构化问题清单（issue）                          |
| repair       | 否           | 局部修改 DAG：新增任务 / 重开任务                |
| judge        | 否           | 裁决 COMPILE / REVISE / ABORT_WITH_LIMITATIONS   |
| compiler     | 是           | 带引用标记的报告 + 引用校验 + 参考来源渲染       |

原实现里 Planner 直接 fan-out 全部子任务、Critic 只输出 missing_evidence、
Judge 用启发式公式计算 confidence_score，这三处在本版被替换。
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command, Send

from .. import config
from .claim import (
    CLAIM_TYPES,
    Claim,
    ClaimGraph,
    ClaimRelation,
    SUPPORTED,
    PARTIALLY_SUPPORTED,
    SUPPORTS,
    make_claim_id,
)
from .citation import (
    CitationBinder,
    CitationRenderer,
    CitationVerifier,
    SourceCatalog,
    UNKNOWN_REF,
    UNVERIFIED_EVIDENCE,
    UnsupportedClaimDetector,
)
from .dag import MAX_REPAIR_TASKS, ResearchPacket, ResearchPlan, ResearchTask, TaskDAG
from .review import ResearchReview, build_review
from .plan_validation import (
    EvidenceCoverageValidator,
    PlanGate,
    SemanticPlanCritic,
    TaskSuccessCriteriaEvaluator,
)
from .sop import select_sop
from .semantic import ClaimEvidenceSemanticVerifier
from .reliability import reliable_llm_call, ReliableLLM
from .evidence import (
    Evidence,
    Source,
    SourceRegistry,
    EvidenceVerifier,
    FULL_TEXT,
    SEARCH_SNIPPET,
    make_evidence_id,
    stable_id,
)
from .ledger import (
    CLAIM_ADDED,
    EVIDENCE_EXTRACTED,
    EVIDENCE_VERIFIED,
    ISSUE_FOUND,
    PLAN_CREATED,
    PROGRESS_LOG,
    REPORT_COMPILED,
    SOURCE_REGISTERED,
    TASK_ADDED,
    TASK_COMPLETED,
    TASK_DISPATCHED,
    TASK_FAILED,
    TASK_REOPENED,
    REVISION_CREATED,
    UNSUPPORTED_CLAIM,
    VERDICT,
    ResearchLedger,
    build_trace,
)
from .search import hybrid_search
from .state import (
    ABORT_WITH_LIMITATIONS,
    COMPILE,
    COMPILE_WITH_LIMITATIONS,
    CONTINUE_UNFINISHED,
    REVISE,
    ResearchStateDict,
)

# 单条来源正文存进 source_texts 的最大长度
MAX_SOURCE_TEXT_CHARS = 4000

_CLAIM_TYPE_HINT = "FACT|METHOD|RESULT|COMPARISON|LIMITATION|TREND|INTERPRETATION|HYPOTHESIS"


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.RAG_LLM_MODEL,
        api_key=config.RAG_LLM_API_KEY,
        base_url=config.RAG_LLM_BASE_URL,
        temperature=0.3,
    )


def _parse_json(raw: str) -> Any:
    m = re.search(r'(\{.*\}|\[.*\])', raw, re.DOTALL)
    if not m:
        raise ValueError(f"无法从输出中提取 JSON: {raw[:300]}")
    return json.loads(m.group(0))


def _dag(state: ResearchStateDict) -> TaskDAG:
    return TaskDAG.from_dicts(state.get("tasks", []))


def _ledger(state: ResearchStateDict) -> ResearchLedger:
    return ResearchLedger(state.get("ledger_events", []))


def _evidence_index(state: ResearchStateDict) -> dict[str, dict]:
    return {e.get("evidence_id", ""): e for e in state.get("evidences", []) if e.get("evidence_id")}


# ---------------------------------------------------------------------------
# Planner：产出 Research Task DAG
# ---------------------------------------------------------------------------

def planner_node(state: ResearchStateDict) -> Command:
    """把调研问题拆成带依赖关系的任务 DAG。

    与旧版差别：任务不再是一次性全部并发的扁平列表，而是允许声明 dependencies，
    由 scheduler 按依赖分批执行（先做背景调研，再做对比分析）。
    """
    llm = _get_llm()
    query = state.get("original_query", "")
    context = state.get("research_context", "")

    # SOP 前置（§SOP 介入时机）：先选研究 SOP，再让 Planner 依据 SOP 拆任务。
    # select_sop 是关键词规则，确定性、不调 LLM；无法识别类型时回退 FACT，不阻塞。
    sop = select_sop(query)

    prompt = (
        "你正在为科研文献调研生成 ResearchTask DAG。\n\n"
        f"用户问题：\n{query}\n\n"
        f"科研调研 SOP：\n{sop.to_prompt()}\n\n"
        + (f"背景信息：\n{context}\n\n" if context else "")
        + "请根据用户问题，将 SOP 中适用的研究维度实例化为具体研究任务。\n"
        "要求：\n"
        "1. SOP 是规划先验，不是固定 DAG 模板；\n"
        "2. 必须覆盖 SOP 要求的必要研究维度；\n"
        "3. 根据具体问题判断任务粒度，不得机械生成固定任务；\n"
        "4. 根据「后续任务是否依赖前序研究结果」建立 dependency；\n"
        "5. Comparison / Synthesis 类任务原则上应依赖必要的事实或证据任务；\n"
        "6. 不适用的 optional 维度可以省略，但必须有合理依据；\n"
        "7. 任务应能够最终支撑用户问题的回答；\n"
        "8. 保持现有 3-6 个任务的规模约束。\n\n"
        "任务字段说明：\n"
        "- task_id: 形如 t1、t2\n"
        "- objective: 这个任务要回答什么\n"
        "- question: 用于检索的具体问题\n"
        f"- task_type: {_CLAIM_TYPE_HINT} 之一，表示期望产出什么类型的结论\n"
        "- expected_evidence: 期望拿到什么形式的证据（数据/论文结论/官方口径）\n"
        "- preferred_sources: 倾向的来源类型，如 academic / web / local_kb\n"
        "- search_strategy: 检索策略说明（用什么关键词、什么限定条件）\n"
        "- dependencies: 依赖的 task_id 列表，无依赖填 []\n"
        "- success_criteria: 什么算完成\n"
        "- priority: 1(普通) 或 2(高)\n\n"
        "依赖只在确实需要前序结论时声明，能并行的任务不要串起来。\n"
        "只输出 JSON 数组。"
    )

    raw = reliable_llm_call("planner", prompt, llm=llm, fallback="").strip()
    try:
        tasks = _parse_json(raw)
        if not isinstance(tasks, list):
            tasks = []
    except Exception as e:
        print(f"[Planner] 解析失败: {e}")
        tasks = []

    if not tasks:
        tasks = [{
            "task_id": "t1",
            "objective": query,
            "question": query,
            "task_type": "FACT",
            "expected_evidence": "可引用的事实性结论",
            "dependencies": [],
            "success_criteria": "至少 2 条通过原文校验的证据",
            "priority": 1,
        }]

    dag = TaskDAG()
    known_ids: set[str] = set()
    for i, t in enumerate(tasks):
        t = dict(t)
        if not t.get("task_id"):
            t["task_id"] = f"t{i+1}"
        # 依赖必须指向已声明的任务；未知依赖直接丢弃，避免整图失效
        deps = [d for d in (t.get("dependencies") or []) if d in known_ids]
        t["dependencies"] = deps
        task = ResearchTask.from_dict(t)
        ok, reason = dag.add_task(task)
        if ok:
            known_ids.add(task.task_id)
        else:
            print(f"[Planner] 任务被拒绝({reason}): {task.task_id}")

    if not dag.tasks:
        dag.add_task(ResearchTask(task_id="t1", objective=query, question=query))

    plan = ResearchPlan(
        plan_id=stable_id("P", query),
        objective=query,
        constraints=context or "",
        tasks=list(dag.tasks.values()),
    )

    ledger = _ledger(state)
    ledger.append(PLAN_CREATED, task_count=len(dag.tasks),
                  query=query, summary=_plan_summary(dag))

    # ---- 计划质量闸门（§Plan Gate，执行前）----
    # SOP(前置已选) → PlanValidator(确定性) → PlanCritic(1次) → Repair(1次) → 门控裁决。
    # 验证就地修补 DAG，结果写入 plan_gate / plan_validation 供追溯。
    critic = SemanticPlanCritic(llm=llm)
    gate = PlanGate(sop=sop, critic=critic)
    gate_result = gate.run(dag, plan, query=query)
    plan.tasks = list(dag.tasks.values())  # repair 可能就地增删任务，回写 plan

    # 把门控问题转成 issue，交给 repair 复用同一套机制
    plan_validation_issues = [i.to_dict() for i in gate_result.issues
                              if i.recommended_action in ("add_task", "remove_task")]

    return Command(
        update={
            "tasks": dag.to_dicts(),
            "plan": plan.to_dict(),
            "plan_summary": _plan_summary(dag),
            "round_no": 0,
            "plan_validation": gate_result.to_dict(),
            "plan_gate": gate_result.to_dict(),
            "sop_type": sop.sop_type,
            "issues": plan_validation_issues,
            "repaired_issue_ids": [i["issue_id"] for i in plan_validation_issues],
            "ledger_events": ledger.delta(),
        },
        goto="scheduler",
    )


def _plan_summary(dag: TaskDAG) -> str:
    parts = []
    for t in dag.tasks.values():
        dep = f"（依赖 {','.join(t.dependencies)}）" if t.dependencies else ""
        parts.append(f"{t.task_id}:{t.objective or t.question}{dep}")
    return "；".join(parts)


# ---------------------------------------------------------------------------
# Scheduler：按依赖挑 READY 任务（确定性，不调用模型）
# ---------------------------------------------------------------------------

def scheduler_node(state: ResearchStateDict) -> Command:
    """挑出依赖已满足且未执行的任务，Send 给 researcher。

    没有可调度任务时交给 aggregator 汇总。调度判断全部在 dag.TaskDAG 里完成，
    不由模型决定"下一步做什么"。
    """
    dag = _dag(state)
    round_no = state.get("round_no", 0)
    ready = dag.ready_tasks()

    if ready:
        dag.mark_running([t.task_id for t in ready])
        ledger = _ledger(state)
        for t in ready:
            ledger.append(TASK_DISPATCHED, task_id=t.task_id, round_no=round_no,
                          question=t.question)
        sends = [
            Send("researcher", {"task": t.to_dict(), "original_query": state.get("original_query", "")})
            for t in ready
        ]
        return Command(
            update={"tasks": dag.to_dicts(), "ledger_events": ledger.delta()},
            goto=sends,
        )

    # 无 READY：要么全部完成，要么剩余任务被 FAILED 依赖卡住
    return Command(goto="aggregator")


# ---------------------------------------------------------------------------
# Researcher：执行单个 Task，返回 ResearchPacket
# ---------------------------------------------------------------------------

async def researcher_node(state: ResearchStateDict) -> ResearchStateDict:
    """执行一个研究任务：检索 → 登记来源 → 抽取声明与引文 → 校验引文。

    Researcher 不直接修改全局结论，只返回本任务的结果包（ResearchPacket），
    由 aggregator 纳入全局状态。
    """
    task_dict = state.get("task", {})
    task = ResearchTask.from_dict(task_dict)
    original_query = state.get("original_query", "")
    if not task.task_id:
        task.task_id = str(uuid.uuid4())[:8]

    ledger = _ledger(state)
    round_no = state.get("round_no", 0)

    queries = _build_queries(task, original_query)
    results: list[dict] = []
    try:
        gathered = await asyncio.gather(
            *[hybrid_search(q, web_results=4, rag_results=2) for q in queries],
            return_exceptions=True,
        )
        for r in gathered:
            if isinstance(r, list):
                results.extend(r)
    except Exception as e:
        print(f"[Researcher] 检索失败 {task.task_id}: {e}")

    # 按 url/title 去重，保留首次出现
    seen: set[str] = set()
    unique_results: list[dict] = []
    for r in results:
        key = (r.get("url", "") or r.get("title", "")) + r.get("snippet", "")[:40]
        if key and key not in seen:
            seen.add(key)
            unique_results.append(r)

    if not unique_results:
        task.status = "FAILED"
        ledger.append(TASK_FAILED, task_id=task.task_id, round_no=round_no,
                      reason="检索无结果", queries=queries)
        return _packet_update(task, ResearchPacket(
            task_id=task.task_id, searched_queries=queries,
            status="FAILED", error="检索无结果",
        ), [], [], [], [], ledger)

    registry = SourceRegistry()
    source_texts: dict[str, str] = {}
    index_to_source: dict[int, str] = {}
    index_to_result: dict[int, dict] = {}
    for i, r in enumerate(unique_results):
        registered = registry.upsert(Source(
            source_id="",
            source_type=_source_type_of(r),
            title=r.get("title", ""),
            authors=str(r.get("authors", "") or ""),
            url=r.get("url", ""),
            publication_year=str(r.get("year", "") or r.get("publication_year", "") or ""),
            provider=r.get("source", ""),
            # §14/§15 来源元数据：web 检索仅返回摘要 → 标记为 snippet，证据最多 UNVERIFIED
            content_type=SEARCH_SNIPPET if _source_type_of(r) == "web" else FULL_TEXT,
            retrieval_method=_source_type_of(r),
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            metadata={
                "search_query": r.get("search_query", ""),
                # rerank_status 透传：SUCCESS / FALLBACK（RAG 通道才有，其余为空）
                "rerank_status": str(r.get("rerank_status", "")),
            },
        ))
        sid = registered.source_id
        text = (r.get("snippet", "") or "")[:MAX_SOURCE_TEXT_CHARS]
        if sid not in source_texts or len(text) > len(source_texts[sid]):
            source_texts[sid] = text
        index_to_source[i + 1] = sid
        index_to_result[i + 1] = r
        ledger.append(SOURCE_REGISTERED, task_id=task.task_id, source_id=sid,
                      url=registered.url, source_type=registered.source_type)

    sources_out = registry.to_dicts()

    llm = _get_llm()
    results_text = "\n\n".join(
        f"[{i+1}] 标题: {r.get('title','')}\n来源: {r.get('url','')}\n"
        f"正文: {(r.get('snippet','') or '')[:600]}"
        for i, r in enumerate(unique_results[:8])
    )
    prompt = (
        f"研究任务: {task.question or task.objective}\n"
        f"期望证据: {task.expected_evidence or '可引用的事实性结论'}\n"
        f"任务类型: {task.task_type}\n\n"
        f"以下是检索到的原始资料:\n{results_text}\n\n"
        "从资料中抽取 3-6 条声明（claim），每条声明配一段支撑它的原文引文。\n"
        "硬性要求：\n"
        "1. quote 必须是上面正文里逐字出现的一段话，不要改写、不要翻译、不要拼接\n"
        "2. 资料里没有的内容不要写；不确定时把 claim 写成 weaker 的表述\n"
        "3. 每条 claim 填写 scope（适用范围）和 conditions（成立条件），没有就留空\n"
        "4. source_index 必须是上面方括号里的编号\n\n"
        "只输出 JSON:\n"
        '{"claims":[{"text":"...","claim_type":"FACT","scope":"...","conditions":"...",'
        '"source_index":1,"quote":"...","interpretation":"..."}]}'
    )

    try:
        raw = reliable_llm_call("researcher", prompt, llm=llm, fallback="").strip()
        parsed = _parse_json(raw)
        raw_claims = parsed.get("claims", []) if isinstance(parsed, dict) else (parsed or [])
        if not isinstance(raw_claims, list):
            raw_claims = []
    except Exception as e:
        print(f"[Researcher] 抽取失败 {task.task_id}: {e}")
        raw_claims = []

    verifier = EvidenceVerifier(registry)
    chunk_ids = {r.get("chunk_id", "") for r in unique_results if r.get("chunk_id")}

    evidences_out: list[dict] = []
    claims_out: list[dict] = []
    relations_out: list[dict] = []
    evidence_ids: list[str] = []

    for rc in raw_claims:
        text = str(rc.get("text", "")).strip()
        quote = str(rc.get("quote", "")).strip()
        if not text or not quote:
            continue
        try:
            src_index = int(rc.get("source_index", 1) or 1)
        except (TypeError, ValueError):
            continue
        source_id = index_to_source.get(src_index, "")
        src_obj = registry.get(source_id)
        if not src_obj:
            continue

        matched = index_to_result.get(src_index, {})
        locator = ""
        chunk_id = ""
        if matched.get("chunk_id"):
            chunk_id = str(matched["chunk_id"])
            locator = f"chunk:{chunk_id}"

        ev = Evidence(
            evidence_id=make_evidence_id(source_id, quote, locator),
            source_id=source_id,
            document_id=stable_id("D", src_obj.url),
            chunk_id=chunk_id,
            quote=quote,
            interpretation=str(rc.get("interpretation", "")).strip(),
            locator=locator,
            retrieval_query=str(rc.get("query", "")) or (queries[0] if queries else ""),
            retrieval_rank=src_index,
            source_type=src_obj.source_type,
            task_id=task.task_id,
            retrieval_method=str(matched.get("source", "")),
            metadata={
                # rerank_status 透传：SUCCESS=远程重排生效；FALLBACK=退回向量分
                "rerank_status": str(matched.get("rerank_status", "")),
            },
        )
        verifier.verify(ev, source_texts.get(source_id, ""), chunk_ids or None)
        evidences_out.append(ev.to_dict())
        evidence_ids.append(ev.evidence_id)
        ledger.append(EVIDENCE_EXTRACTED, task_id=task.task_id, evidence_id=ev.evidence_id,
                      source_id=source_id)
        ledger.append(EVIDENCE_VERIFIED, task_id=task.task_id, evidence_id=ev.evidence_id,
                      status=ev.verification_status, reason=ev.verification_reason)

        claim_type = str(rc.get("claim_type", "FACT")).upper()
        if claim_type not in CLAIM_TYPES:
            claim_type = "FACT"
        claim = Claim(
            claim_id=make_claim_id(text),
            text=text,
            claim_type=claim_type,
            scope=str(rc.get("scope", "")).strip(),
            conditions=str(rc.get("conditions", "")).strip(),
            task_id=task.task_id,
            evidence_ids=[ev.evidence_id],
        )
        claims_out.append(claim.to_dict())
        relations_out.append(ClaimRelation(
            source_id=ev.evidence_id, target_id=claim.claim_id, relation=SUPPORTS,
        ).to_dict())
        ledger.append(CLAIM_ADDED, task_id=task.task_id, claim_id=claim.claim_id,
                      claim_type=claim_type)

    # §8 Claim-Evidence 语义关系判定：quote 是否真的支持 claim。
    # 由模型判定，失败时整体回退到 SUPPORTS（recompute_statuses 缺省也是 SUPPORTS）。
    if claims_out and evidences_out:
        try:
            semantic_verifier = ClaimEvidenceSemanticVerifier(ReliableLLM(llm, "semantic", ""))
            semantic_index = semantic_verifier.verify(evidences_out, claims_out)
            for rel in relations_out:
                key = (rel.get("source_id", ""), rel.get("target_id", ""))
                if key in semantic_index:
                    rel["semantic_relation"] = semantic_index[key]
        except Exception as e:
            print(f"[Researcher] 语义关系判定异常，回退 SUPPORTS: {e}")

    packet = ResearchPacket(
        task_id=task.task_id,
        searched_queries=queries,
        source_ids=[s["source_id"] for s in sources_out],
        evidence_ids=evidence_ids,
        claims=[c["claim_id"] for c in claims_out],
        unresolved_issues=[] if evidence_ids else ["未能抽取出通过校验的证据"],
        status="COMPLETED" if evidence_ids else "FAILED",
        error="" if evidence_ids else "无可用证据",
    )
    task.status = packet.status
    ledger.append(TASK_COMPLETED if packet.status == "COMPLETED" else TASK_FAILED,
                  task_id=task.task_id, round_no=round_no,
                  evidence_count=len(evidence_ids))

    return _packet_update(task, packet, sources_out, evidences_out, claims_out,
                          relations_out, ledger, source_texts)


def _build_queries(task: ResearchTask, original_query: str) -> list[str]:
    """检索式：任务 question 优先，其次 strategy，最后原始问题。"""
    queries: list[str] = []
    for q in (task.question, task.objective):
        if q and q not in queries:
            queries.append(q)
    if not queries:
        queries.append(original_query)
    return queries[:3]


def _source_type_of(result: dict) -> str:
    src = str(result.get("source", ""))
    if src == "rag":
        return "local_kb"
    if src.startswith("academic") or src in ("arxiv", "semantic_scholar", "pubmed"):
        return "academic"
    return "web"


def _packet_update(
    task: ResearchTask,
    packet: ResearchPacket,
    sources: list[dict],
    evidences: list[dict],
    claims: list[dict],
    relations: list[dict],
    ledger: ResearchLedger,
    source_texts: dict[str, str] | None = None,
) -> dict:
    update: dict = {
        "tasks": [task.to_dict()],
        "task_results": [packet.to_dict()],
        "sources": sources,
        "evidences": evidences,
        "claims": claims,
        "relations": relations,
        "searched_queries": packet.searched_queries,
        "ledger_events": ledger.delta(),
    }
    if source_texts:
        update["source_texts"] = source_texts
    return update


# ---------------------------------------------------------------------------
# Aggregator：合并、推导支撑状态、统计证据状况（确定性）
# ---------------------------------------------------------------------------

def aggregator_node(state: ResearchStateDict) -> ResearchStateDict:
    """合并各 ResearchPacket，并按证据校验结果推导每条 Claim 的支撑状态。

    这里是"证据 → 结论"的唯一推导点：Claim.status 不来自模型自评，
    而来自它所关联 Evidence 的 verification_status。
    """
    relations_in = state.get("relations", [])
    graph = ClaimGraph(state.get("claims", []), relations_in)
    # §8 从 relations 的 semantic_relation 构建语义索引，供确定性状态推导使用
    semantic_index = {
        (r.get("source_id", ""), r.get("target_id", "")): r.get("semantic_relation", "")
        for r in relations_in
        if r.get("source_id") and r.get("target_id")
    }
    graph.recompute_statuses(_evidence_index(state), semantic_index)

    claims, relations = graph.to_dicts()
    support = _support_stats(state.get("evidences", []), claims)

    ledger = _ledger(state)
    return {
        "claims": claims,
        "relations": relations,
        "evidence_support": support,
        "ledger_events": ledger.delta(),
    }


def _support_stats(evidences: list[dict], claims: list[dict]) -> dict:
    ev_counts: dict[str, int] = {}
    for e in evidences:
        s = e.get("verification_status", "UNVERIFIED")
        ev_counts[s] = ev_counts.get(s, 0) + 1

    claim_counts: dict[str, int] = {}
    for c in claims:
        s = c.get("status", "UNSUPPORTED")
        claim_counts[s] = claim_counts.get(s, 0) + 1

    return {
        "evidence_total": len(evidences),
        "evidence_by_status": ev_counts,
        "claim_total": len(claims),
        "claim_by_status": claim_counts,
        "usable_claims": claim_counts.get(SUPPORTED, 0) + claim_counts.get(PARTIALLY_SUPPORTED, 0),
    }


# ---------------------------------------------------------------------------
# Review：结构化评审（原 Critic）
# ---------------------------------------------------------------------------

# 与规范 §30 对齐的问题类型
_ISSUE_TYPES = (
    "MISSING_EVIDENCE",        # 声明缺少证据
    "UNVERIFIED_EVIDENCE",     # 证据未通过原文校验
    "UNSUPPORTED_CLAIM",       # 声明无可用证据
    "WEAK_SOURCE",             # 来源质量不足
    "CONTRADICTORY_EVIDENCE",  # 证据之间存在矛盾
    "SCOPE_MISMATCH",          # 结论适用范围不一致
    "INCOMPLETE_TASK",         # 任务未完成
    "DUPLICATE_RESEARCH",      # 重复研究
    "COVERAGE_GAP",            # 研究问题有未被覆盖的方面
    "LOGIC_GAP",               # 从证据到结论的推理跳跃
)


def review_node(state: ResearchStateDict) -> ResearchStateDict:
    """评审当前研究状态，输出结构化问题清单（§29）。

    与旧版差别：
    - 确定性部分（冲突/范围不一致/未支撑声明）由程序基于 Claim 支撑状态与
      ConflictDetector 结果计算，不依赖模型判断；
    - 每个问题都带 target、recommended_action、required_evidence、priority，
      供 repair 节点做局部修改（§31）。
    """
    llm = _get_llm()
    query = state.get("original_query", "")
    claims = [c for c in state.get("claims", []) if c.get("status") in (SUPPORTED, PARTIALLY_SUPPORTED)]
    evidences = state.get("evidences", [])
    tasks = state.get("tasks", [])
    ledger = _ledger(state)

    # 确定性评审（§29）：基于支撑状态与 ConflictDetector
    review = build_review(claims, evidences, tasks, state.get("relations", []))

    # 执行后覆盖验证（§Evidence Coverage）：Evidence/Claim 是否覆盖 plan 的关键维度。
    # 缺失维度输出 COVERAGE_GAP issue（recommended_action=add_task），
    # 交给 repair_node 局部补任务，而不是重建 DAG。
    coverage_result = None
    plan_dict = state.get("plan", {}) or {}
    plan = ResearchPlan.from_dict(plan_dict) if plan_dict else None
    if plan is not None:
        coverage_result = EvidenceCoverageValidator().validate(
            plan, claims=state.get("claims", []), evidences=evidences,
        )

    claims_text = "\n".join(
        f"- [{c.get('claim_id','')}]({c.get('status','')}) {c.get('text','')}"
        + (f"｜适用范围：{c['scope']}" if c.get("scope") else "")
        + f"｜证据：{','.join(c.get('evidence_ids', []))}"
        for c in claims[:25]
    )
    evidence_text = "\n".join(
        f"- {e.get('evidence_id','')} [{e.get('verification_status','')}] "
        f"{(e.get('quote','') or '')[:160]}（来源 {e.get('source_id','')}）"
        for e in evidences[:25]
    )
    task_text = "\n".join(f"- {t.get('task_id','')} ({t.get('status','')}) {t.get('question','')}"
                          for t in tasks)

    prompt = (
        f"你是研究评审员。原始调研问题: {query}\n\n"
        f"已完成的任务:\n{task_text}\n\n"
        f"当前声明:\n{claims_text}\n\n"
        f"当前证据（含原文校验状态）:\n{evidence_text}\n\n"
        "找出仍然存在的问题。问题类型只能是:\n"
        + "\n".join(f"- {t}" for t in _ISSUE_TYPES)
        + "\n\n每个问题给出:\n"
        "- issue_type: 上面的类型之一\n"
        "- description: 具体说明\n"
        "- severity: high / medium / low\n"
        "- target_type: claim / task / global\n"
        "- target_id: 对应的 claim_id 或 task_id，global 填空字符串\n"
        "- recommended_action: add_task / reopen_task / none\n"
        "- required_evidence: 解决这个问题需要拿到什么证据（一句话）\n"
        "- priority: 1（普通）或 2（高）\n"
        "- suggested_task: 当 recommended_action=add_task 时给出 "
        "{question, task_type, expected_evidence, search_strategy, dependencies}\n\n"
        "证据充分且无明显问题时输出 {\"issues\":[]}。只输出 JSON:\n"
        '{"issues":[{"issue_type":"COVERAGE_GAP","description":"...","severity":"high",'
        '"target_type":"global","target_id":"","recommended_action":"add_task",'
        '"required_evidence":"原始论文实验数据","priority":2,'
        '"suggested_task":{"question":"...","task_type":"FACT","expected_evidence":"...",'
        '"search_strategy":"...","dependencies":[]}}]}'
    )

    try:
        raw = reliable_llm_call("review", prompt, llm=llm, fallback="").strip()
        parsed = _parse_json(raw)
        raw_issues = parsed.get("issues", []) if isinstance(parsed, dict) else (parsed or [])
        if not isinstance(raw_issues, list):
            raw_issues = []
    except Exception as e:
        print(f"[Review] 解析失败: {e}")
        raw_issues = []

    issues: list[dict] = []
    for ri in raw_issues:
        if not isinstance(ri, dict):
            continue
        issue_type = str(ri.get("issue_type", "")).upper()
        if issue_type not in _ISSUE_TYPES:
            continue
        severity = str(ri.get("severity", "low")).lower()
        if severity not in ("high", "medium", "low"):
            severity = "low"
        description = str(ri.get("description", "")).strip()
        if not description:
            continue
        target_id = str(ri.get("target_id", "") or "")
        target_type = str(ri.get("target_type", "global") or "global")
        recommended_action = str(ri.get("recommended_action", ri.get("suggested_action", "none")) or "none")
        priority = int(ri.get("priority") or (2 if recommended_action == "add_task" else 1))
        issues.append({
            "issue_id": stable_id("I", issue_type, target_id, description),
            "issue_type": issue_type,
            "description": description,
            "severity": severity,
            "target_type": target_type,
            "target_id": target_id,
            "target_claim_id": target_id if target_type == "claim" else "",
            "target_task_id": target_id if target_type == "task" else "",
            "recommended_action": recommended_action,
            "suggested_action": recommended_action,  # 兼容旧字段
            "required_evidence": str(ri.get("required_evidence", "") or "").strip(),
            "priority": priority,
            "suggested_task": ri.get("suggested_task", {}) or {},
        })
        ledger.append(ISSUE_FOUND, issue_type=issue_type, severity=severity,
                      target_id=target_id)

    # 确定性发现的冲突 / 范围不一致也写入 issues（recommended_action=none，不触发补任务）
    for c in review.contradictions:
        issues.append(_conflict_issue(c, "CONTRADICTORY_EVIDENCE", "high"))
    for s in review.scope_mismatches:
        issues.append(_conflict_issue(s, "SCOPE_MISMATCH", "medium"))

    # 执行后覆盖验证的 COVERAGE_GAP 也并入 issues（recommended_action=add_task，触发补任务）
    coverage_dict: dict = {}
    if coverage_result is not None:
        coverage_dict = coverage_result.to_dict()
        for gap in coverage_result.issues:
            issues.append(gap.to_dict())
            ledger.append(ISSUE_FOUND, issue_type=gap.issue_type,
                          severity=gap.severity, target_id=gap.target_id)

    return {
        "issues": issues,
        "review": review.to_dict(),
        "coverage_validation": coverage_dict,
        "ledger_events": ledger.delta(),
    }


def _conflict_issue(pair: dict, issue_type: str, severity: str) -> dict:
    """把 ConflictDetector 的结果转成 Issue（确定性，不触发补任务）。"""
    a = pair.get("claim_a", "")
    b = pair.get("claim_b", "")
    desc = f"声明 {a} 与 {b} 存在{('冲突' if issue_type == 'CONTRADICTORY_EVIDENCE' else '适用范围不一致')}（{pair.get('verdict','')}）"
    return {
        "issue_id": stable_id("I", issue_type, a, b),
        "issue_type": issue_type,
        "description": desc,
        "severity": severity,
        "target_type": "claim",
        "target_id": a,
        "target_claim_id": a,
        "target_task_id": "",
        "recommended_action": "none",
        "suggested_action": "none",
        "required_evidence": "",
        "priority": 1,
        "suggested_task": {},
    }


# ---------------------------------------------------------------------------
# Repair：局部修改 DAG（确定性）
# ---------------------------------------------------------------------------

def repair_node(state: ResearchStateDict) -> ResearchStateDict:
    """按 Review 的问题清单局部修改 DAG，不重新生成整棵树。

    只接受两种动作：新增任务（add_task）、重开任务（reopen_task）。
    新增任务数受 MAX_REPAIR_TASKS 限制；非法结构（依赖不存在、成环）由 TaskDAG 拒绝。
    """
    dag = _dag(state)
    round_no = int(state.get("round_no", 0)) + 1
    ledger = _ledger(state)

    repaired = set(state.get("repaired_issue_ids", []) or [])
    # 同一个 issue 只处理一次：issue_id 由 (类型, 目标, 描述) 哈希得到，
    # 评审在下一轮重复提出同一个问题时不会重复补任务（§34）。
    # 动作集合扩展：add_task / reopen_task / remove_task（计划验证会产出 remove_task，
    # 用于删除冗余或不可达任务）。
    pending = [i for i in state.get("issues", [])
               if i.get("severity") in ("high", "medium")
               and (i.get("recommended_action") or i.get("suggested_action"))
                   in ("add_task", "reopen_task", "remove_task")
               and i.get("issue_id", "") not in repaired]

    added = 0
    acted: list[str] = []
    for issue in pending:
        action = issue.get("recommended_action") or issue.get("suggested_action") or "none"
        if action == "reopen_task":
            target = issue.get("target_id", "")
            if target and dag.reopen(target, reason=issue.get("description", "")):
                ledger.append(TASK_REOPENED, task_id=target, round_no=round_no,
                              reason=issue.get("description", ""))
                acted.append(issue.get("issue_id", ""))
            continue

        if action == "remove_task":
            # 删除冗余/不可达任务：仅当该任务无下游时删除，避免破坏依赖链
            target = issue.get("target_id", "")
            if target and dag.has(target) and not any(
                    target in t.dependencies for t in dag.tasks.values()):
                del dag.tasks[target]
                ledger.append(TASK_FAILED, task_id=target, round_no=round_no,
                              reason=f"计划验证删除：{issue.get('description','')}")
                acted.append(issue.get("issue_id", ""))
            continue

        if added >= MAX_REPAIR_TASKS:
            break
        spec = issue.get("suggested_task") or {}
        question = str(spec.get("question", "") or issue.get("description", "")).strip()
        if not question:
            continue
        deps = [str(d) for d in (spec.get("dependencies") or []) if d in dag.tasks]
        # 修订任务命名：指向父任务则 {parent}-R{round}，否则 R{round}-{n}（§33）
        parent = issue.get("target_task_id") or issue.get("target_id", "")
        if parent and dag.has(parent):
            base_id = f"{parent}-R{round_no}"
        else:
            base_id = f"R{round_no}-{added+1}"
        task_id = base_id
        suffix = 1
        while task_id in dag.tasks:  # 同名修订避免覆盖
            suffix += 1
            task_id = f"{base_id}-{suffix}"
        task = ResearchTask(
            task_id=task_id,
            objective=str(spec.get("expected_evidence", "") or question),
            question=question,
            task_type=str(spec.get("task_type", "FACT")).upper(),
            expected_evidence=str(spec.get("expected_evidence", "")),
            search_strategy=str(spec.get("search_strategy", "")),
            dependencies=deps,
            capabilities=[str(c) for c in (spec.get("capabilities") or []) if c],
            criteria=spec.get("criteria", {}) or {},
            priority=2,
            round_added=round_no,
            parent_task_id=parent,
            revision_round=round_no,
            notes=issue.get("description", ""),
        )
        ok, reason = dag.add_task(task)
        if ok:
            added += 1
            acted.append(issue.get("issue_id", ""))
            ledger.append(TASK_ADDED, task_id=task.task_id, round_no=round_no,
                          question=question, reason=issue.get("description", ""))
            ledger.append(REVISION_CREATED, task_id=task.task_id, round_no=round_no,
                          parent_task_id=parent,
                          action=issue.get("required_evidence", "") or question)
        else:
            print(f"[Repair] 新增任务被拒绝({reason}): {task.task_id}")

    return {
        "tasks": dag.to_dicts(),
        "round_no": round_no,
        "repaired_issue_ids": acted,
        "ledger_events": ledger.delta(),
    }


# ---------------------------------------------------------------------------
# Judge：裁决（确定性）
# ---------------------------------------------------------------------------

def judge_node(state: ResearchStateDict) -> Command:
    """决定进入下一轮还是成文（§35-38）。

    裁决不依赖模型打分，只看：
    1. 是否还有可执行的任务（DAG 里存在 READY）
    2. 是否达到最大轮次
    3. 是否存在可用声明（SUPPORTED / PARTIALLY_SUPPORTED）
    4. 进展检测：连续两轮 new_evidence_count == 0 → 终止（§38）
    5. 原始计划任务 vs 修订任务区分（§14）：仅剩修订任务时不再为"补全"空转
    """
    dag = _dag(state)
    round_no = int(state.get("round_no", 0))
    max_revisions = int(state.get("max_revisions", 2))
    support = state.get("evidence_support", {}) or {}
    ledger = _ledger(state)

    # 进展检测（§38）：本轮新增证据数 = 当前证据总数 - 上轮记录值
    ev_total = int(support.get("evidence_total", len(state.get("evidences", []))))
    last = int(state.get("last_evidence_count", 0))
    new_evidence = max(0, ev_total - last)
    stagnant = int(state.get("stagnant_rounds", 0)) + (0 if new_evidence > 0 else 1)

    # 原始计划任务 vs 修订任务区分（§14）：
    #   原始任务 = 初始规划产生（round_added==0 且无父任务）；
    #   修订任务 = repair 在评审后补的（有 parent_task_id 或 round_added>0）。
    unfinished_original: list[str] = []
    unfinished_revision: list[str] = []
    for t in dag.tasks.values():
        if t.status == "COMPLETED":
            continue
        if (t.parent_task_id or int(getattr(t, "round_added", 0) or 0) > 0):
            unfinished_revision.append(t.task_id)
        else:
            unfinished_original.append(t.task_id)

    # 本轮进展日志（§14）：声明/来源新增 + 本轮新增任务 + 来源指纹
    claim_total = int(support.get("claim_total", len(state.get("claims", []))))
    source_total = len(state.get("sources", []))
    new_claims = max(0, claim_total - int(state.get("last_claim_count", 0)))
    new_sources = max(0, source_total - int(state.get("last_source_count", 0)))
    new_tasks = [t.task_id for t in dag.tasks.values()
                 if int(getattr(t, "round_added", 0) or 0) == round_no]
    source_ids = sorted(s.get("source_id", "") for s in state.get("sources", []))
    source_fingerprint = stable_id("FP", *source_ids) if source_ids else ""
    progress_entry = {
        "round": round_no,
        "new_evidence": new_evidence,
        "new_claims": new_claims,
        "new_sources": new_sources,
        "new_tasks": new_tasks,
        "source_fingerprint": source_fingerprint,
        "stagnant_rounds": stagnant,
        "unfinished_original": len(unfinished_original),
        "unfinished_revision": len(unfinished_revision),
    }

    # 连续两轮无新证据：强制终止，避免"搜索→没结果→再搜索"空转
    if stagnant >= 2:
        usable = int(support.get("usable_claims", 0))
        if usable == 0:
            verdict = ABORT_WITH_LIMITATIONS
            reason = f"连续 {stagnant} 轮无新证据且无可引用声明（证据不足）"
        else:
            verdict = COMPILE_WITH_LIMITATIONS
            reason = f"连续 {stagnant} 轮无新证据，按现有 {usable} 条可用声明成文"
        ledger.append(VERDICT, round_no=round_no, verdict=verdict, reason=reason)
        ledger.append(PROGRESS_LOG, round_no=round_no, payload=progress_entry)
        return Command(
            update={"verdict": verdict, "verdict_reason": reason,
                    "last_evidence_count": ev_total, "stagnant_rounds": stagnant,
                    "last_claim_count": claim_total, "last_source_count": source_total,
                    "progress_log": [progress_entry], "ledger_events": ledger.delta()},
            goto="compiler",
        )

    ready = dag.ready_tasks()
    if ready and round_no < max_revisions:
        # 区分本轮可继续的是原始计划任务还是修订任务（§14）
        ready_original = [t.task_id for t in ready
                          if not (t.parent_task_id or int(getattr(t, "round_added", 0) or 0) > 0)]
        if ready_original:
            verdict = CONTINUE_UNFINISHED
            reason = (f"存在 {len(ready)} 个可执行任务（其中 {len(ready_original)} 个为原始计划任务），"
                      f"继续调研以补全未完成部分")
        else:
            verdict = REVISE
            reason = f"存在 {len(ready)} 个可执行修订任务（revision），继续局部补查"
        ledger.append(VERDICT, round_no=round_no, verdict=verdict, reason=reason)
        ledger.append(PROGRESS_LOG, round_no=round_no, payload=progress_entry)
        return Command(
            update={"verdict": verdict, "verdict_reason": reason,
                    "tasks": dag.to_dicts(),
                    "last_evidence_count": ev_total, "stagnant_rounds": stagnant,
                    "last_claim_count": claim_total, "last_source_count": source_total,
                    "progress_log": [progress_entry], "ledger_events": ledger.delta()},
            goto="scheduler",
        )

    # 无可执行任务：装车前清点未完成任务，区分原始/修订（§14）
    usable = int(support.get("usable_claims", 0))
    if usable == 0:
        verdict = ABORT_WITH_LIMITATIONS
        reason = "无通过校验的可用声明"
    elif unfinished_original:
        # 原始计划任务未做完：明确告知报告只会覆盖已完成部分，不假装全覆盖
        verdict = COMPILE_WITH_LIMITATIONS
        reason = (f"可用声明 {usable} 条；原始计划未完成任务 {len(unfinished_original)} 个"
                  f"（{','.join(unfinished_original)}），报告只覆盖已完成部分")
    elif unfinished_revision:
        # 原始任务都做完了，只剩修订任务的补查没全部跑完：直接成文，不空转
        verdict = COMPILE
        reason = (f"可用声明 {usable} 条，原始计划全部完成；"
                  f"剩余 {len(unfinished_revision)} 个修订任务未完成，按现有证据成文")
    else:
        verdict = COMPILE
        reason = f"可用声明 {usable} 条，任务全部完成"

    ledger.append(VERDICT, round_no=round_no, verdict=verdict, reason=reason)
    ledger.append(PROGRESS_LOG, round_no=round_no, payload=progress_entry)
    return Command(
        update={"verdict": verdict, "verdict_reason": reason,
                "last_evidence_count": ev_total, "stagnant_rounds": stagnant,
                "last_claim_count": claim_total, "last_source_count": source_total,
                "progress_log": [progress_entry], "ledger_events": ledger.delta()},
        goto="compiler",
    )


# ---------------------------------------------------------------------------
# Compiler：成文 + 引用校验 + 参考来源渲染
# ---------------------------------------------------------------------------

# 报告固定七段式结构（§15）：由程序规定标题，模型只能往里填内容
REPORT_SECTIONS = (
    "执行摘要",
    "研究背景与问题界定",
    "研究方法与证据来源",
    "主要发现",
    "证据对比与矛盾分析",
    "研究局限与不确定性",
    "结论与建议",
)


def _split_sections(report: str) -> list[dict]:
    """把报告正文按 `## ` 二级标题切分为结构化小节（确定性，便于下游/UI 取用）。"""
    if not report:
        return []
    lines = report.splitlines()
    sections: list[dict] = []
    cur_title = "正文"
    cur_lines: list[str] = []
    for line in lines:
        if line.startswith("## ") and not line.startswith("### "):
            if cur_lines:
                sections.append({"title": cur_title.strip(), "content": "\n".join(cur_lines).strip()})
            cur_title = line[3:].strip()
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_lines:
        sections.append({"title": cur_title.strip(), "content": "\n".join(cur_lines).strip()})
    return sections


def _render_trace_summary(trace: dict) -> str:
    """把可追溯链渲染成程序生成的「可追溯性说明」小节（§15，模型不写这部分）。"""
    chains = trace.get("chains", []) if isinstance(trace, dict) else []
    if not chains:
        return "无可用声明，未形成可追溯链。"
    blocks: list[str] = []
    for ch in chains:
        head = f"- 声明【{ch.get('claim_id','')}】({ch.get('status','')})：{ch.get('claim_text','')}"
        if ch.get("scope"):
            head += f" ｜ 适用范围：{ch['scope']}"
        blocks.append(head)
        for step in ch.get("evidence_chain", []):
            blocks.append(
                f"    - 证据 {step.get('evidence_id','')} [{step.get('verification_status','')}] "
                f"← 来源《{step.get('source_title','')}》"
                f"{('（'+step.get('source_url','')+')') if step.get('source_url') else ''} "
                f"← 任务 {step.get('task_id','')}（检索式：{step.get('retrieval_query','')}）"
            )
    return "\n".join(blocks)


def compiler_node(state: ResearchStateDict) -> ResearchStateDict:
    """成文：模型只能用目录里的编号引用，成文后由 CitationVerifier 校验。

    引用校验不通过时重跑一次并把问题清单反馈给模型，最多重跑 1 次。
    """
    llm = _get_llm()
    query = state.get("original_query", "")
    claims = state.get("claims", [])
    evidences = state.get("evidences", [])
    sources = state.get("sources", [])
    verdict = state.get("verdict", COMPILE)
    support = state.get("evidence_support", {}) or {}
    ledger = _ledger(state)

    catalog = SourceCatalog(sources, evidences)
    all_claims = state.get("claims", [])
    usable = [c for c in claims if c.get("status") in (SUPPORTED, PARTIALLY_SUPPORTED)]
    catalog_text = catalog.render_for_prompt(usable)

    base_prompt = (
        f"你是研究报告撰写员。基于下面的可用来源与声明，撰写一份结构化研究报告。\n\n"
        f"调研问题: {query}\n"
        f"裁决: {verdict}（{state.get('verdict_reason','')}）\n"
        f"证据状况: 证据 {support.get('evidence_total', 0)} 条，"
        f"其中通过原文校验 {support.get('evidence_by_status', {}).get('VERIFIED', 0)} 条；"
        f"可用声明 {support.get('usable_claims', 0)} 条。\n\n"
        f"{catalog_text}\n\n"
        "写作要求:\n"
        "1. 每个关键结论后面必须跟引用标记，形如 [S1] 或 [E3]，只能用上面目录里出现的编号\n"
        "2. 不要编造目录里没有的来源；目录里没有的内容不要写\n"
        "3. 声明带有适用范围或成立条件的，必须在正文里带上\n"
        "4. 报告必须严格包含且仅包含以下 7 个二级标题（## 开头），顺序一致：\n"
        + "\n".join(f"   - ## {s}" for s in REPORT_SECTIONS)
        + "\n   不要把参考来源或可追溯性说明写进正文，这两部分由程序追加\n"
        "5. 不要输出参考来源列表，参考来源由程序追加\n\n"
        "输出完整 Markdown 报告正文。"
    )

    report = reliable_llm_call("compiler", base_prompt, llm=llm, fallback="").strip()

    verifier = CitationVerifier(catalog)
    result = verifier.verify(report, all_claims)
    issues = result["issues"]

    blocking = [i for i in issues if i["issue_type"] in (UNKNOWN_REF, UNVERIFIED_EVIDENCE)]
    if blocking:
        feedback = "\n".join(f"- {i['ref']} {i['detail']}" for i in blocking[:10])
        retry_prompt = (
            base_prompt
            + "\n\n上一版存在以下引用问题，修正后重新输出报告正文：\n"
            + feedback
        )
        report = reliable_llm_call("compiler", retry_prompt, llm=llm, fallback="").strip()
        result = verifier.verify(report, all_claims)
        issues = result["issues"]

    # §44 引用闭环绑定：草稿句子 → Claim ID → Evidence IDs
    binder = CitationBinder(catalog, all_claims)
    binding = binder.bind(report)

    # §48 未支撑声明检测：禁止 UNSUPPORTED/CONTRADICTED 声明作为事实写入正文
    detector = UnsupportedClaimDetector(all_claims)
    unsupported_violations, sanitized = detector.detect_and_sanitize(report)
    if unsupported_violations:
        warn = (
            "\n\n## 未支撑声明警示（不作为事实结论，需人工复核）\n"
            + "\n".join(
                f"- 「{v['text']}」（支撑状态：{v['status']}）" for v in unsupported_violations
            )
        )
        sanitized = sanitized + warn
        ledger.append(UNSUPPORTED_CLAIM, count=len(unsupported_violations),
                      claim_ids=[v["claim_id"] for v in unsupported_violations])
    report = sanitized

    references = catalog.render_references(used_only=True,
                                           used_source_ids=result["used_source_ids"])
    if references:
        report = f"{report}\n\n## 参考来源\n{references}\n"

    # 可追溯性说明：程序生成（§15），把「结论→声明→证据→来源→任务→检索式」回溯链落进报告
    trace = build_trace(usable, evidences, sources, state.get("tasks", []))
    trace_text = _render_trace_summary(trace)
    report = f"{report}\n\n## 可追溯性说明\n{trace_text}\n"

    ledger.append(REPORT_COMPILED, verdict=verdict,
                  citations=len(result["refs"]),
                  citation_issues=len(issues),
                  used_sources=len(result["used_source_ids"]),
                  unsupported_claims=len(unsupported_violations))

    # 局限说明：由程序按任务完成情况与证据校验结果生成，不由模型自己声明
    limitations = ""
    unfinished = [t.get("task_id", "") for t in state.get("tasks", []) if t.get("status") != "COMPLETED"]
    notes: list[str] = []
    if unfinished:
        notes.append(f"未完成任务 {len(unfinished)} 个（{', '.join(unfinished)}），报告未覆盖这些方面")
    invalid = int(support.get("evidence_by_status", {}).get("INVALID", 0))
    if invalid:
        notes.append(f"{invalid} 条证据未通过原文定位校验，未作为引用依据")
    if verdict == ABORT_WITH_LIMITATIONS:
        notes.append(f"可用声明不足（{state.get('verdict_reason','')}），结论需要人工复核")
    if notes:
        limitations = "\n\n## 本报告局限\n" + "\n".join(f"- {n}" for n in notes) + "\n"

    # 把报告正文按二级标题切分为结构化小节，便于下游/UI 取用（含程序追加的参考来源与可追溯性）
    sections = _split_sections(report)

    return {
        "final_report": report + limitations,
        "report_sections": sections,
        "citation_issues": issues,
        "citation_binding": binding,
        "unsupported_violations": unsupported_violations,
        "trace": trace,
        "trace_summary": trace_text,
        "ledger_events": ledger.delta(),
    }
