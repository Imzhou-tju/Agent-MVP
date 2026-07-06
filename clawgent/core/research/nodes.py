from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command, Send

from .. import config
from .search import hybrid_search
from .state import ResearchStateDict


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


# ---------------------------------------------------------------------------
# Planner：拆分调研任务 → Send API fan-out
# ---------------------------------------------------------------------------

def planner_node(state: ResearchStateDict) -> Command:
    """
    拆分原始问题为 3-6 个可独立检索的子任务，
    通过 Send API 并发分发给 researcher 节点（调研确认的 fan-out 机制）。
    """
    llm = _get_llm()
    query = state.get("original_query", "")
    context = state.get("research_context", "")
    max_rev = state.get("max_revisions", 2)

    prompt = (
        "你是一个专业调研规划师。将下面的调研问题拆分为 3-6 个独立的子调研任务，\n"
        "每个任务应聚焦不同角度（现状、趋势、案例、风险、对比、实施等）。\n\n"
        f"调研问题: {query}\n"
        + (f"背景信息: {context}\n" if context else "")
        + "\n只输出 JSON 数组，每项包含 task_id(string)、question(string)、angle(string)、"
        "search_queries(list[string], 2-3条):\n"
        '[{"task_id":"t1","question":"...","angle":"...","search_queries":["..."]}]'
    )

    raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    try:
        tasks = _parse_json(raw)
        if not isinstance(tasks, list):
            tasks = []
    except Exception as e:
        print(f"[Planner] 解析失败: {e}")
        tasks = [{"task_id": "t1", "question": query, "angle": "综合", "search_queries": [query]}]

    # 确保每个任务有 task_id
    for i, t in enumerate(tasks):
        if not t.get("task_id"):
            t["task_id"] = f"t{i+1}"

    plan_summary = f"共 {len(tasks)} 个子任务: " + "、".join(t.get("angle", "") for t in tasks)

    # Send API fan-out：每个子任务发给 researcher 节点（调研验证的官方推荐方式）
    sends = [Send("researcher", {"sub_task": t, "original_query": query}) for t in tasks]

    return Command(
        update={
            "sub_tasks": tasks,
            "research_plan_summary": plan_summary,
            "revision_count": 0,
            "max_revisions": max_rev,
        },
        goto=sends,
    )


# ---------------------------------------------------------------------------
# Researcher：并发执行（每个 Send 一个独立实例）
# ---------------------------------------------------------------------------

async def researcher_node(state: ResearchStateDict) -> ResearchStateDict:
    """
    接收单个 sub_task，执行混合检索（联网 + RAG），
    提取 claim-级证据。并发写入 evidences（reducer 已声明）。
    """
    sub_task = state.get("sub_task", {})
    original_query = state.get("original_query", "")
    task_id = sub_task.get("task_id", str(uuid.uuid4())[:4])
    search_queries = sub_task.get("search_queries", [sub_task.get("question", original_query)])

    all_results = []
    search_tasks = [hybrid_search(q, web_results=4, rag_results=2) for q in search_queries]
    gathered = await asyncio.gather(*search_tasks, return_exceptions=True)
    for r in gathered:
        if isinstance(r, list):
            all_results.extend(r)

    # 去重
    seen: set[str] = set()
    unique_results = []
    for r in all_results:
        key = r.get("url", "") + r.get("snippet", "")[:50]
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    # LLM 提取 claim 级证据
    llm = _get_llm()
    results_text = "\n\n".join(
        f"[{i+1}] 来源: {r.get('url','')}\n标题: {r.get('title','')}\n内容: {r.get('snippet','')[:400]}"
        for i, r in enumerate(unique_results[:8])
    )
    extract_prompt = (
        f"调研问题: {sub_task.get('question', original_query)}\n\n"
        f"以下是检索到的原始资料:\n{results_text}\n\n"
        "从资料中提取 3-6 条具体的、可验证的事实性声明（claim）。\n"
        "每条 claim 必须标注它来自哪个来源（用来源编号）。\n"
        "只输出 JSON 数组:\n"
        '[{"claim":"...","source_index":1,"relevance":0.9,"raw_text":"原文片段"}]'
    )
    raw = llm.invoke([HumanMessage(content=extract_prompt)]).content.strip()
    try:
        claims = _parse_json(raw)
        if not isinstance(claims, list):
            claims = []
    except Exception:
        claims = []

    evidences = []
    for c in claims:
        src_idx = c.get("source_index", 1) - 1
        src = unique_results[src_idx] if 0 <= src_idx < len(unique_results) else {}
        evidences.append({
            "claim": c.get("claim", ""),
            "raw_text": c.get("raw_text", ""),
            "relevance": float(c.get("relevance", 0.5)),
            "sub_task_id": task_id,
            "source": {
                "url": src.get("url", ""),
                "title": src.get("title", ""),
                "snippet": src.get("snippet", "")[:200],
                "search_query": src.get("search_query", ""),
            },
        })

    return {
        "evidences": evidences,
        "raw_search_results": unique_results,
    }


# ---------------------------------------------------------------------------
# Aggregator：合并去重，建立 claim→evidence 映射
# ---------------------------------------------------------------------------

def aggregator_node(state: ResearchStateDict) -> ResearchStateDict:
    evidences = state.get("evidences", [])
    seen_claims: set[str] = set()
    deduped = []
    claim_map: dict[str, list[dict]] = {}

    for e in evidences:
        claim = e.get("claim", "").strip()
        if not claim:
            continue
        key = claim[:80]
        if key not in seen_claims:
            seen_claims.add(key)
            deduped.append(e)
        if claim not in claim_map:
            claim_map[claim] = []
        claim_map[claim].append(e.get("source", {}))

    return {
        "deduped_evidences": deduped,
        "claim_evidence_map": claim_map,
    }


# ---------------------------------------------------------------------------
# Critic：Red-team，发现证据缺失/事实冲突/逻辑漏洞
# ---------------------------------------------------------------------------

def critic_node(state: ResearchStateDict) -> ResearchStateDict:
    llm = _get_llm()
    query = state.get("original_query", "")
    evidences = state.get("deduped_evidences", [])
    revision_count = state.get("revision_count", 0)

    evidence_text = "\n".join(
        f"- [{e.get('sub_task_id','')}] {e.get('claim','')} (来源: {e.get('source',{}).get('url','')})"
        for e in evidences[:20]
    )

    prompt = (
        f"你是一个批判性评审员。原始调研问题: {query}\n\n"
        f"当前收集的证据:\n{evidence_text}\n\n"
        "请找出:\n"
        "1. 缺失的关键证据（missing_evidence）\n"
        "2. 事实矛盾（factual_conflict）\n"
        "3. 逻辑漏洞（logic_gap）\n\n"
        "如果证据充分、无明显问题，输出空数组。\n"
        "只输出 JSON 数组:\n"
        '[{"issue_type":"missing_evidence","description":"...","severity":"high","related_claims":["..."]}]'
    )
    raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    try:
        notes = _parse_json(raw)
        if not isinstance(notes, list):
            notes = []
    except Exception:
        notes = []

    # 只保留 high/medium 级别问题，low 级别忽略
    significant = [n for n in notes if n.get("severity") in ("high", "medium")]

    # 为有问题的 claim 生成补充检索 query
    revision_queries = []
    for n in significant:
        if n.get("issue_type") == "missing_evidence":
            revision_queries.append(f"{query} {n.get('description', '')}")

    return {
        "critic_notes": significant,
        "revision_queries": revision_queries[:3],  # 最多补3条
    }


# ---------------------------------------------------------------------------
# Revision：根据 Critic 结果补充检索
# ---------------------------------------------------------------------------

async def revision_node(state: ResearchStateDict) -> ResearchStateDict:
    revision_queries = state.get("revision_queries", [])
    if not revision_queries:
        return {"revision_evidences": []}

    llm = _get_llm()
    original_query = state.get("original_query", "")

    all_extra: list[dict] = []
    tasks = [hybrid_search(q, web_results=3, rag_results=2) for q in revision_queries]
    gathered = await asyncio.gather(*tasks, return_exceptions=True)
    for r in gathered:
        if isinstance(r, list):
            all_extra.extend(r)

    if not all_extra:
        return {"revision_evidences": []}

    # 提取补充证据
    results_text = "\n\n".join(
        f"[{i+1}] {r.get('url','')}\n{r.get('snippet','')[:300]}"
        for i, r in enumerate(all_extra[:6])
    )
    prompt = (
        f"原始调研问题: {original_query}\n\n补充检索资料:\n{results_text}\n\n"
        "提取 2-4 条补充 claim，标注来源编号。只输出 JSON 数组:\n"
        '[{"claim":"...","source_index":1,"relevance":0.8,"raw_text":"..."}]'
    )
    raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    try:
        claims = _parse_json(raw)
        if not isinstance(claims, list):
            claims = []
    except Exception:
        claims = []

    extra_evidences = []
    for c in claims:
        src_idx = c.get("source_index", 1) - 1
        src = all_extra[src_idx] if 0 <= src_idx < len(all_extra) else {}
        extra_evidences.append({
            "claim": c.get("claim", ""),
            "raw_text": c.get("raw_text", ""),
            "relevance": float(c.get("relevance", 0.5)),
            "sub_task_id": "revision",
            "source": {
                "url": src.get("url", ""),
                "title": src.get("title", ""),
                "snippet": src.get("snippet", "")[:200],
                "search_query": src.get("search_query", ""),
            },
        })

    return {"revision_evidences": extra_evidences}


# ---------------------------------------------------------------------------
# Judge：裁决当前证据质量，决定是否需要再次 revision 或直接 compile
# 用 Command 路由（调研验证：终止条件用 Command 条件路由，不是 recursion_limit）
# ---------------------------------------------------------------------------

def judge_node(state: ResearchStateDict) -> Command:
    llm = _get_llm()
    query = state.get("original_query", "")
    critic_notes = state.get("critic_notes", [])
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 2)
    deduped = state.get("deduped_evidences", [])
    revision_evs = state.get("revision_evidences", [])

    # 合并所有证据
    all_evidences = deduped + revision_evs
    high_issues = [n for n in critic_notes if n.get("severity") == "high"]

    # 终止条件：超过最大 revision 次数，或无 high 级别问题
    should_compile = (revision_count >= max_revisions) or (len(high_issues) == 0)

    if should_compile:
        confidence = max(0.3, min(1.0, 0.5 + 0.1 * len(all_evidences) - 0.2 * len(high_issues)))
        verdict = "pass" if len(high_issues) == 0 else "partial"
        return Command(
            update={
                "judge_verdict": verdict,
                "confidence_score": round(confidence, 2),
                "deduped_evidences": all_evidences,
            },
            goto="compiler",
        )
    else:
        return Command(
            update={
                "revision_count": revision_count + 1,
                "deduped_evidences": all_evidences,
                "revision_evidences": [],  # 清空，准备下一轮
            },
            goto="critic",
        )


# ---------------------------------------------------------------------------
# Compiler：生成结构化研究报告
# ---------------------------------------------------------------------------

def compiler_node(state: ResearchStateDict) -> ResearchStateDict:
    llm = _get_llm()
    query = state.get("original_query", "")
    evidences = state.get("deduped_evidences", [])
    plan_summary = state.get("research_plan_summary", "")
    confidence = state.get("confidence_score", 0.0)
    verdict = state.get("judge_verdict", "pass")

    # 按 sub_task_id 分组证据
    groups: dict[str, list[dict]] = {}
    for e in evidences:
        tid = e.get("sub_task_id", "general")
        groups.setdefault(tid, []).append(e)

    evidence_block = ""
    for tid, evs in groups.items():
        evidence_block += f"\n## 子任务 {tid}\n"
        for e in evs:
            src = e.get("source", {})
            evidence_block += f"- {e.get('claim','')} [来源: {src.get('url','')}]\n"

    prompt = (
        f"你是一个专业研究报告撰写员。基于以下调研证据，生成一份结构化研究报告。\n\n"
        f"调研问题: {query}\n"
        f"调研计划: {plan_summary}\n"
        f"证据质量: {verdict}（置信度 {confidence:.0%}）\n\n"
        f"收集的证据:\n{evidence_block}\n\n"
        "报告要求:\n"
        "1. 执行摘要（3-5句话）\n"
        "2. 各子方向发现（按角度分节）\n"
        "3. 方案对比或风险评估（如适用）\n"
        "4. 结论与实施建议\n"
        "5. 参考来源列表（URL）\n\n"
        "每个关键结论必须标注来源。输出完整 Markdown 报告。"
    )

    report = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    # 提取来源列表
    sources = list({
        e.get("source", {}).get("url", "")
        for e in evidences
        if e.get("source", {}).get("url", "")
    })

    sections = [{"title": "完整报告", "content": report, "sources": sources}]

    return {
        "final_report": report,
        "report_sections": sections,
    }
