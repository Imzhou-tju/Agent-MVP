from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from .nodes import (
    aggregator_node,
    compiler_node,
    judge_node,
    planner_node,
    repair_node,
    researcher_node,
    review_node,
    scheduler_node,
)
from .state import ResearchStateDict


def build_research_graph() -> StateGraph:
    """构建以 Research Task DAG 为调度骨架的调研子图。

    与旧版图结构的差别：

    旧版：planner → Send 全部子任务 → researcher → aggregator → critic → revision
          → judge →（回到 critic）→ compiler
    新版：planner → scheduler ⇄ researcher（按依赖分批）→ aggregator → review
          → repair → judge →（回到 scheduler 做局部补查）→ compiler

    关键变化：
    1. 任务不再一次性全部并发：scheduler 只 Send 依赖已满足的 READY 任务，
       任务完成后回到 scheduler 再放下一批（依赖驱动的分批执行）。
    2. 补检索不再是"critic 生成 query → revision 再搜一遍"的全局重跑，
       而是 review 产出结构化问题 → repair 局部新增/重开 DAG 任务 → 只补这些任务。
    3. Judge 不再用启发式公式计算置信度，只做终止裁决（COMPILE / REVISE /
       ABORT_WITH_LIMITATIONS），裁决依据是"还有没有可执行的任务"与"有没有可用声明"。

    reducer：Researcher 通过 Send 并发执行，所有被并发写入的字段都在 state.py 里
    声明了语义化 reducer（按 id 合并），不再使用 operator.add 追加。

    终止条件由 Judge 的 Command 路由控制，不依赖 recursion_limit 触发。
    """
    graph = StateGraph(ResearchStateDict)

    llm_retry = RetryPolicy(max_attempts=3, backoff_factor=0.5)
    net_retry = RetryPolicy(max_attempts=2, backoff_factor=1.0)

    graph.add_node("planner", planner_node, retry_policy=llm_retry)
    # scheduler 是纯确定性节点，不调用模型，不需要重试
    graph.add_node("scheduler", scheduler_node)
    graph.add_node("researcher", researcher_node, retry_policy=net_retry)
    graph.add_node("aggregator", aggregator_node)
    graph.add_node("review", review_node, retry_policy=llm_retry)
    graph.add_node("repair", repair_node)
    graph.add_node("judge", judge_node)
    graph.add_node("compiler", compiler_node, retry_policy=llm_retry)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "scheduler")
    # scheduler → researcher：由 scheduler_node 的 Command(goto=list[Send]) 驱动
    # researcher → scheduler：一批任务跑完后回到调度器，放下一批依赖已满足的任务
    graph.add_edge("researcher", "scheduler")
    graph.add_edge("aggregator", "review")
    graph.add_edge("review", "repair")
    graph.add_edge("repair", "judge")
    # judge → scheduler（REVISE）或 judge → compiler（COMPILE / ABORT）由 Command 决定
    graph.add_edge("compiler", END)

    return graph.compile()
