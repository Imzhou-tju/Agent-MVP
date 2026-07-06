from __future__ import annotations

import asyncio

from ..tools.base import clawgent_tool

_research_graph = None


def _get_graph():
    global _research_graph
    if _research_graph is None:
        from ..research.graph import build_research_graph
        _research_graph = build_research_graph()
    return _research_graph


@clawgent_tool
def deep_research(query: str, context: str = "", max_revisions: int = 2) -> str:
    """执行深度多智能体调研，自动完成任务拆解、联网检索、多角度分析和报告生成。
    适用场景：技术选型、行业调研、企业知识库分析、复杂决策评审。
    支持联网搜索（需配置 TAVILY_API_KEY）和本地知识库混合检索。

    参数:
    query (str): 调研问题或任务描述，支持复杂多跳问题。
    context (str): 可选背景信息，如指定文档范围、行业领域、已知约束等。
    max_revisions (int): 最大评审补充轮次，默认 2，越高越深入但耗时越长。

    返回:
    结构化 Markdown 研究报告，包含执行摘要、各角度发现、结论建议和参考来源。
    """
    graph = _get_graph()
    initial_state = {
        "original_query": query,
        "research_context": context,
        "max_revisions": max_revisions,
        "evidences": [],
        "raw_search_results": [],
        "revision_evidences": [],
        "revision_count": 0,
    }
    try:
        # 子图是异步图，在同步 tool 里运行
        result = asyncio.run(
            graph.ainvoke(
                initial_state,
                config={"recursion_limit": 50},
            )
        )
        report = result.get("final_report", "")
        confidence = result.get("confidence_score", 0.0)
        verdict = result.get("judge_verdict", "unknown")
        plan = result.get("research_plan_summary", "")
        evidence_count = len(result.get("deduped_evidences", []))

        header = (
            f"## 调研完成\n"
            f"- 问题：{query}\n"
            f"- 计划：{plan}\n"
            f"- 证据数：{evidence_count} 条\n"
            f"- 质量评级：{verdict}（置信度 {confidence:.0%}）\n\n"
        )
        return header + report if report else header + "（报告生成失败，请检查 LLM 配置）"
    except Exception as e:
        return f"深度调研执行失败：{e}（请检查 TAVILY_API_KEY 和 RAG_LLM_API_KEY 配置）"
