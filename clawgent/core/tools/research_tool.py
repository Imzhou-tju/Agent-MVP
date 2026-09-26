from __future__ import annotations

import asyncio
import uuid

from ..tools.base import clawgent_tool
from ..logger import audit_logger, log_research_event

_research_graph = None


def _get_graph():
    global _research_graph
    if _research_graph is None:
        from ..research.graph import build_research_graph
        _research_graph = build_research_graph()
    return _research_graph


@clawgent_tool
def deep_research(query: str, context: str = "", max_revisions: int = 2, thread_id: str = "") -> str:
    """执行多智能体调研，自动完成任务拆解、联网检索、多角度分析和报告生成。
    适用场景：技术选型、行业调研、企业知识库分析、复杂决策评审。
    支持联网搜索（需配置 TAVILY_API_KEY）和本地知识库混合检索。

    参数:
    query (str): 调研问题或任务描述，支持复杂多跳问题。
    context (str): 可选背景信息，如指定文档范围、行业领域、已知约束等。
    max_revisions (int): 最大评审补充轮次，默认 2，越高越深入但耗时越长。
    thread_id (str): 审计追踪用的会话 ID，缺省自动生成，过程日志写入 logs/<thread_id>.jsonl。

    返回:
    结构化 Markdown 研究报告，包含执行摘要、分主题发现、矛盾与局限、结论建议和参考来源。
    报告中的每个结论带有引用标记，标记在文末参考来源里有对应的来源条目。
    """
    graph = _get_graph()
    if not thread_id:
        thread_id = f"research-{uuid.uuid4().hex[:12]}"
    initial_state = {
        "original_query": query,
        "research_context": context,
        "max_revisions": max_revisions,
        "tasks": [],
        "plan": {},
        "round_no": 0,
        "task_results": [],
        "sources": [],
        "evidences": [],
        "claims": [],
        "relations": [],
        "issues": [],
        "review": {},
        "repaired_issue_ids": [],
        "searched_queries": [],
        "source_texts": {},
        "ledger_events": [],
        "last_evidence_count": 0,
        "last_claim_count": 0,
        "stagnant_rounds": 0,
        "progress_log": [],
    }
    try:
        # 子图是异步图，在同步 tool 里运行
        # recursion_limit 需要覆盖：DAG 分批调度（每批 scheduler+researcher）
        # + 每轮 repair（review+repair+judge+scheduler+researcher）
        result = asyncio.run(
            graph.ainvoke(
                initial_state,
                config={"recursion_limit": 100,
                        "configurable": {"thread_id": thread_id}},
            )
        )
        report = result.get("final_report", "")
        support = result.get("evidence_support", {}) or {}
        verdict = result.get("verdict", "unknown")
        reason = result.get("verdict_reason", "")
        plan = result.get("plan_summary", "")
        tasks = result.get("tasks", [])
        done_tasks = sum(1 for t in tasks if t.get("status") == "COMPLETED")

        # §16 过程留存：把 Research Ledger 事件逐条写入审计日志，并落盘 trace 闭环
        for ev in result.get("ledger_events", []) or []:
            if not isinstance(ev, dict):
                continue
            log_research_event(
                thread_id,
                ev.get("event_type", "unknown"),
                task_id=ev.get("task_id", ""),
                round_no=ev.get("round_no", 0),
                **(ev.get("payload", {}) or {}),
            )
        trace = result.get("trace") or {}
        log_research_event(
            thread_id, "research_trace",
            claim_count=trace.get("claim_count", 0),
            verdict=verdict, reason=reason,
            trace_summary=(result.get("trace_summary", "") or "")[:2000],
        )
        audit_logger.flush()

        ev_by_status = support.get("evidence_by_status", {}) or {}
        header = (
            f"## 调研完成\n"
            f"- 会话：{thread_id}\n"
            f"- 问题：{query}\n"
            f"- 计划：{plan}\n"
            f"- 任务：{done_tasks}/{len(tasks)} 完成（第 {result.get('round_no', 0)} 轮）\n"
            f"- 证据：{support.get('evidence_total', 0)} 条，"
            f"通过原文校验 {ev_by_status.get('VERIFIED', 0)} 条，"
            f"部分匹配 {ev_by_status.get('PARTIAL', 0)} 条，"
            f"未通过 {ev_by_status.get('INVALID', 0)} 条\n"
            f"- 声明：{support.get('claim_total', 0)} 条，可用 {support.get('usable_claims', 0)} 条\n"
            f"- 裁决：{verdict}（{reason}）\n"
        )
        issues = result.get("citation_issues", []) or []
        if issues:
            header += "- 引用校验问题：" + "；".join(
                f"{i.get('ref','')} {i.get('detail','')}" for i in issues[:5]
            ) + "\n"

        header += "\n"
        return header + report if report else header + "（报告生成失败，请检查 LLM 配置）"
    except Exception as e:
        return f"调研执行失败：{e}（请检查 TAVILY_API_KEY 和 RAG_LLM_API_KEY 配置）"
