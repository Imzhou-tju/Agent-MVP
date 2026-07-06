from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any


@dataclass
class Source:
    url: str
    title: str = ""
    snippet: str = ""
    search_query: str = ""


@dataclass
class Evidence:
    """单条证据，绑定到具体 claim。"""
    claim: str
    source: Source
    raw_text: str = ""
    relevance: float = 0.0
    sub_task_id: str = ""


@dataclass
class SubTask:
    task_id: str
    question: str          # 具体检索问题
    angle: str             # 调研角度描述
    search_queries: list[str] = field(default_factory=list)


@dataclass
class CriticNote:
    issue_type: str        # "missing_evidence" | "factual_conflict" | "logic_gap"
    description: str
    severity: str          # "high" | "medium" | "low"
    related_claims: list[str] = field(default_factory=list)


class ResearchState:
    """
    LangGraph StateGraph 的状态类。
    并发写入的字段必须声明 reducer（Annotated[list, operator.add]），
    否则触发 INVALID_CONCURRENT_GRAPH_UPDATE（调研已验证）。
    """

    def __init__(self):
        # 输入
        self.original_query: str = ""
        self.research_context: str = ""        # 可选：指定文档 / 背景说明

        # Planner 输出
        self.sub_tasks: list[SubTask] = []
        self.research_plan_summary: str = ""

        # Researcher 并发写入（必须 reducer）
        self.evidences: Annotated[list[Evidence], operator.add] = []
        self.raw_search_results: Annotated[list[dict], operator.add] = []

        # Aggregator 输出
        self.deduped_evidences: list[Evidence] = []
        self.claim_evidence_map: dict[str, list[Evidence]] = {}  # claim → evidences

        # Critic 输出
        self.critic_notes: list[CriticNote] = []
        self.revision_queries: list[str] = []  # 补充检索 query

        # Revision 补充写入（reducer）
        self.revision_evidences: Annotated[list[Evidence], operator.add] = []

        # Judge 输出
        self.judge_verdict: str = ""          # "pass" | "needs_revision" | "fail"
        self.confidence_score: float = 0.0
        self.revision_count: int = 0

        # Compiler 输出
        self.final_report: str = ""
        self.report_sections: list[dict] = []  # [{title, content, sources}]

        # 控制
        self.error: str = ""
        self.max_revisions: int = 2


# LangGraph 用 TypedDict 更合适（支持 add_messages reducer 语法）
# 用 TypedDict 重新声明，保留上面 dataclass 作为文档参考

from typing import TypedDict


class ResearchStateDict(TypedDict, total=False):
    # 输入
    original_query: str
    research_context: str

    # Planner
    sub_tasks: list[dict]
    research_plan_summary: str

    # Researcher 并发写入 —— reducer 必须声明
    evidences: Annotated[list[dict], operator.add]
    raw_search_results: Annotated[list[dict], operator.add]

    # Aggregator
    deduped_evidences: list[dict]
    claim_evidence_map: dict[str, list[dict]]

    # Critic
    critic_notes: list[dict]
    revision_queries: list[str]

    # Revision 补充写入 —— reducer 必须声明
    revision_evidences: Annotated[list[dict], operator.add]

    # Judge
    judge_verdict: str
    confidence_score: float
    revision_count: int

    # Compiler
    final_report: str
    report_sections: list[dict]

    # 控制
    error: str
    max_revisions: int
