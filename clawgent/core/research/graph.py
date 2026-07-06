from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from .nodes import (
    aggregator_node,
    compiler_node,
    critic_node,
    judge_node,
    planner_node,
    researcher_node,
    revision_node,
)
from .state import ResearchStateDict


def build_research_graph() -> StateGraph:
    """
    构建 Multi-Agent 调研子图。

    架构决策（基于调研验证结论）：
    - Send API fan-out：Planner → Command(goto=list[Send]) → 并发 Researcher
    - reducer 必须声明：evidences/revision_evidences 字段用 Annotated[list, operator.add]
    - RetryPolicy：挂在 researcher/revision 节点，处理 LLM/网络瞬态失败
    - 终止条件：Judge 用 Command 条件路由（非 recursion_limit，调研否定了该参数）
    - 多 Agent 路径只支持 Tavily（调研已验证）

    图结构：
        START → planner → [Send fan-out] → researcher(×N) → aggregator
              → critic → revision → judge → compiler → END
                           ↑__________________________|（最多 max_revisions 次）
    """
    graph = StateGraph(ResearchStateDict)

    # RetryPolicy：LLM 调用瞬态失败重试（调研验证的官方推荐机制）
    llm_retry = RetryPolicy(max_attempts=3, backoff_factor=0.5)
    net_retry = RetryPolicy(max_attempts=2, backoff_factor=1.0)

    graph.add_node("planner", planner_node, retry=llm_retry)
    # researcher 是异步节点，并发执行（Send fan-out 的 worker）
    graph.add_node("researcher", researcher_node, retry=net_retry)
    graph.add_node("aggregator", aggregator_node, retry=llm_retry)
    graph.add_node("critic", critic_node, retry=llm_retry)
    graph.add_node("revision", revision_node, retry=net_retry)
    # judge 用 Command 路由，不需要 add_edge
    graph.add_node("judge", judge_node, retry=llm_retry)
    graph.add_node("compiler", compiler_node, retry=llm_retry)

    # 固定边
    graph.add_edge(START, "planner")
    # planner → researcher：由 planner_node 返回的 Command(goto=list[Send]) 驱动，无需 add_edge
    # researcher → aggregator：所有 Send worker 完成后汇聚
    graph.add_edge("researcher", "aggregator")
    graph.add_edge("aggregator", "critic")
    graph.add_edge("critic", "revision")
    graph.add_edge("revision", "judge")
    # judge → critic 或 judge → compiler：由 judge_node 返回的 Command 决定，无需 add_conditional_edges
    graph.add_edge("compiler", END)

    return graph.compile()
