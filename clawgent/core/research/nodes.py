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
    SourceCatalog,
    CitationVerifier,
    UNKNOWN_REF,
    UNVERIFIED_EVIDENCE,
)
from .dag import MAX_REPAIR_TASKS, ResearchPacket, ResearchTask, TaskDAG
from .evidence import (
    Evidence,
    Source,
    SourceRegistry,
    EvidenceVerifier,
    make_evidence_id,
    stable_id,
)
from .ledger import (
    CLAIM_ADDED,
    EVIDENCE_EXTRACTED,
    EVIDENCE_VERIFIED,
    ISSUE_FOUND,
    PLAN_CREATED,
    REPORT_COMPILED,
    SOURCE_REGISTERED,
    TASK_ADDED,
    TASK_COMPLETED,
    TASK_DISPATCHED,
    TASK_FAILED,
    TASK_REOPENED,
    VERDICT,
    ResearchLedger,
    build_trace,
)
from .search import hybrid_search
from .state import (
    ABORT_WITH_LIMITATIONS,
    COMPILE,
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

    prompt = (
        "你是科研调研规划师。把调研问题拆成 3-6 个研究任务，并标明任务之间的依赖。\n\n"
        f"调研问题: {query}\n"
        + (f"背景信息: {context}\n" if context else "")
        + "\n任务字段说明：\n"
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

    raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
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

    ledger = _ledger(state)
    ledger.append(PLAN_CREATED, task_count=len(dag.tasks),
                  query=query, summary=_plan_summary(dag))

    return Command(
        update={
            "tasks": dag.to_dicts(),
            "plan_summary": _plan_summary(dag),
            "round_no": 0,
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
            metadata={"search_query": r.get("search_query", "")},
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
        raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
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

    packet = ResearchPacket(
        task_id=task.task_id,
        searched_queries=queries,
        source_ids=[s["source_id"] for s in sources_out],
        evidence_ids=evidence_ids,
        claims=[c["claim_id"] for c in claims_out],
        unresolved_issues=[] if evidence_ids else ["未能抽取出通过校验的证据"],
        status="DONE" if evidence_ids else "FAILED",
        error="" if evidence_ids else "无可用证据",
    )
    task.status = packet.status
    ledger.append(TASK_COMPLETED if packet.status == "DONE" else TASK_FAILED,
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
    graph = ClaimGraph(state.get("claims", []), state.get("relations", []))
    graph.recompute_statuses(_evidence_index(state))

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

_ISSUE_TYPES = (
    "MISSING_EVIDENCE",       # 声明缺少证据
    "UNSUPPORTED_CLAIM",      # 声明的证据未通过原文校验
    "CONFLICT",               # 证据之间存在矛盾
    "COVERAGE_GAP",           # 研究问题有未被覆盖的方面
    "SOURCE_QUALITY",         # 来源质量不足
    "OUTDATED_SOURCE",        # 来源过旧
    "LOGIC_GAP",              # 从证据到结论的推理跳跃
)


def review_node(state: ResearchStateDict) -> ResearchStateDict:
    """评审当前研究状态，输出结构化问题清单。

    与旧版差别：旧版 Critic 只输出 missing_evidence / factual_conflict / logic_gap，
    且只有 missing_evidence 会产生补充检索 query；本版每个问题都带
    target（作用于哪条声明/哪个任务）与 suggested_action（补任务 / 重开任务 / 无需动作），
    供 repair 节点做局部修改。
    """
    llm = _get_llm()
    query = state.get("original_query", "")
    claims = [c for c in state.get("claims", []) if c.get("status") in (SUPPORTED, PARTIALLY_SUPPORTED)]
    evidences = state.get("evidences", [])
    tasks = state.get("tasks", [])
    ledger = _ledger(state)

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
        "- suggested_action: add_task / reopen_task / none\n"
        "- suggested_task: 当 suggested_action=add_task 时给出 "
        "{question, task_type, expected_evidence, search_strategy, dependencies}\n\n"
        "证据充分且无明显问题时输出 {\"issues\":[]}。只输出 JSON:\n"
        '{"issues":[{"issue_type":"COVERAGE_GAP","description":"...","severity":"high",'
        '"target_type":"global","target_id":"","suggested_action":"add_task",'
        '"suggested_task":{"question":"...","task_type":"FACT","expected_evidence":"...",'
        '"search_strategy":"...","dependencies":[]}}]}'
    )

    try:
        raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
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
        issues.append({
            "issue_id": stable_id("I", issue_type, target_id, description),
            "issue_type": issue_type,
            "description": description,
            "severity": severity,
            "target_type": str(ri.get("target_type", "global") or "global"),
            "target_id": target_id,
            "suggested_action": str(ri.get("suggested_action", "none") or "none"),
            "suggested_task": ri.get("suggested_task", {}) or {},
        })
        ledger.append(ISSUE_FOUND, issue_type=issue_type, severity=severity,
                      target_id=target_id)

    return {"issues": issues, "ledger_events": ledger.delta()}


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
    # 评审在下一轮重复提出同一个问题时不会重复补任务。
    pending = [i for i in state.get("issues", [])
               if i.get("severity") in ("high", "medium")
               and i.get("suggested_action") in ("add_task", "reopen_task")
               and i.get("issue_id", "") not in repaired]

    added = 0
    acted: list[str] = []
    for issue in pending:
        if issue.get("suggested_action") == "reopen_task":
            target = issue.get("target_id", "")
            if target and dag.reopen(target, reason=issue.get("description", "")):
                ledger.append(TASK_REOPENED, task_id=target, round_no=round_no,
                              reason=issue.get("description", ""))
                acted.append(issue.get("issue_id", ""))
            continue

        if added >= MAX_REPAIR_TASKS:
            break
        spec = issue.get("suggested_task") or {}
        question = str(spec.get("question", "") or issue.get("description", "")).strip()
        if not question:
            continue
        deps = [str(d) for d in (spec.get("dependencies") or []) if d in dag.tasks]
        task = ResearchTask(
            task_id=f"r{round_no}_{added+1}",
            objective=str(spec.get("expected_evidence", "") or question),
            question=question,
            task_type=str(spec.get("task_type", "FACT")).upper(),
            expected_evidence=str(spec.get("expected_evidence", "")),
            search_strategy=str(spec.get("search_strategy", "")),
            dependencies=deps,
            priority=2,
            round_added=round_no,
            notes=issue.get("description", ""),
        )
        ok, reason = dag.add_task(task)
        if ok:
            added += 1
            acted.append(issue.get("issue_id", ""))
            ledger.append(TASK_ADDED, task_id=task.task_id, round_no=round_no,
                          question=question, reason=issue.get("description", ""))
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
    """决定进入下一轮还是成文。

    裁决不依赖模型打分，只看三件事：
    1. 是否还有可执行的任务（DAG 里存在 READY）
    2. 是否达到最大轮次
    3. 是否存在可用声明（SUPPORTED / PARTIALLY_SUPPORTED）
    """
    dag = _dag(state)
    round_no = int(state.get("round_no", 0))
    max_revisions = int(state.get("max_revisions", 2))
    support = state.get("evidence_support", {}) or {}
    ledger = _ledger(state)

    ready = dag.ready_tasks()
    if ready and round_no < max_revisions:
        ledger.append(VERDICT, round_no=round_no, verdict=REVISE,
                      reason=f"存在 {len(ready)} 个可执行任务")
        return Command(
            update={"verdict": REVISE, "verdict_reason": f"存在 {len(ready)} 个可执行任务",
                    "tasks": dag.to_dicts(), "ledger_events": ledger.delta()},
            goto="scheduler",
        )

    # 未完成 = 被依赖卡住、执行失败、或达到轮次上限时还没轮到的任务
    unfinished = [t.task_id for t in dag.tasks.values() if t.status != "DONE"]
    usable = int(support.get("usable_claims", 0))

    if usable == 0:
        verdict = ABORT_WITH_LIMITATIONS
        reason = "无通过校验的可用声明"
    elif unfinished:
        verdict = COMPILE
        reason = (f"可用声明 {usable} 条；未完成任务 {len(unfinished)} 个"
                  f"（{','.join(unfinished)}），报告只覆盖已完成部分")
    else:
        verdict = COMPILE
        reason = f"可用声明 {usable} 条，任务全部完成"

    ledger.append(VERDICT, round_no=round_no, verdict=verdict, reason=reason)
    return Command(
        update={"verdict": verdict, "verdict_reason": reason,
                "ledger_events": ledger.delta()},
        goto="compiler",
    )


# ---------------------------------------------------------------------------
# Compiler：成文 + 引用校验 + 参考来源渲染
# ---------------------------------------------------------------------------

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
        "4. 结构：执行摘要 / 分主题发现 / 矛盾与局限 / 结论与建议\n"
        "5. 不要输出参考来源列表，参考来源由程序追加\n\n"
        "输出完整 Markdown 报告正文。"
    )

    report = llm.invoke([HumanMessage(content=base_prompt)]).content.strip()

    verifier = CitationVerifier(catalog)
    result = verifier.verify(report, usable)
    issues = result["issues"]

    blocking = [i for i in issues if i["issue_type"] in (UNKNOWN_REF, UNVERIFIED_EVIDENCE)]
    if blocking:
        feedback = "\n".join(f"- {i['ref']} {i['detail']}" for i in blocking[:10])
        retry_prompt = (
            base_prompt
            + "\n\n上一版存在以下引用问题，修正后重新输出报告正文：\n"
            + feedback
        )
        report = llm.invoke([HumanMessage(content=retry_prompt)]).content.strip()
        result = verifier.verify(report, usable)
        issues = result["issues"]

    references = catalog.render_references(used_only=True,
                                           used_source_ids=result["used_source_ids"])
    if references:
        report = f"{report}\n\n## 参考来源\n{references}\n"

    trace = build_trace(usable, evidences, sources, state.get("tasks", []))
    ledger.append(REPORT_COMPILED, verdict=verdict,
                  citations=len(result["refs"]),
                  citation_issues=len(issues),
                  used_sources=len(result["used_source_ids"]))

    # 局限说明：由程序按任务完成情况与证据校验结果生成，不由模型自己声明
    limitations = ""
    unfinished = [t.get("task_id", "") for t in state.get("tasks", []) if t.get("status") != "DONE"]
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

    return {
        "final_report": report + limitations,
        "report_sections": [{"title": "完整报告", "content": report,
                             "sources": result["used_source_ids"]}],
        "citation_issues": issues,
        "trace": trace,
        "ledger_events": ledger.delta(),
    }
