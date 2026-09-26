"""生成一次完整的审计事件链，用于验证 Dashboard（§监控/审计 spec）。

不调用模型、不联网，直接 emit 一条从 run_start → ... → run_end 的完整链路，
写到 logs/<thread_id>.jsonl。跑完后用 `python -m entry.dashboard` 打开 Dashboard 查看。

用法：
    python scripts/demo_audit_run.py
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from clawgent.core import audit
from clawgent.core.audit import bind, emit
from clawgent.core.logger import audit_logger


def main() -> None:
    run_id = "demo-run-001"
    thread_id = "demo"
    query = "比较 x-vector 与 ECAPA-TDNN 在说话人验证上的方法原理与性能"

    # Planner
    bind(run_id=run_id, thread_id=thread_id, node="planner")
    emit(audit.RUN_START, status="SUCCESS", metadata={"query": query, "run_id": run_id})
    emit(audit.PLAN_GATE, status="SUCCESS", metadata={
        "decision": "ALLOW_WITH_WARNINGS", "repair_count": 1, "sop_type": "COMPARISON"})
    emit(audit.TASK_CREATED, task_id="t1", status="SUCCESS",
         metadata={"task_type": "BACKGROUND", "dependencies": [], "priority": 1})
    emit(audit.TASK_CREATED, task_id="t2", status="SUCCESS",
         metadata={"task_type": "MECHANISM", "dependencies": ["t1"], "priority": 1})
    emit(audit.TASK_CREATED, task_id="t3", status="SUCCESS",
         metadata={"task_type": "COMPARISON", "dependencies": ["t2"], "priority": 2})

    # Scheduler 第 0 轮
    bind(node="scheduler", task_id="")
    emit(audit.TASK_READY, task_id="t1", status="SUCCESS", metadata={"round_no": 0})
    emit(audit.TASK_STARTED, task_id="t1", status="SUCCESS", metadata={"round_no": 0})

    # Researcher t1
    bind(node="researcher", task_id="t1")
    emit(audit.RAG_RETRIEVAL, status="SUCCESS", metadata={
        "query": "x-vector 定义与背景", "academic_result_count": 2,
        "web_result_count": 3, "rag_result_count": 1, "merged_result_count": 5})
    emit(audit.EVIDENCE_REGISTERED, task_id="t1", status="SUCCESS", metadata={
        "evidence_id": "e1", "source_id": "s1", "source_type": "academic",
        "verification_status": "VERIFIED", "claim_id": "C1"})
    emit(audit.TASK_COMPLETED, task_id="t1", status="SUCCESS")

    # Scheduler 第 1 轮
    bind(node="scheduler", task_id="")
    emit(audit.TASK_READY, task_id="t2", status="SUCCESS", metadata={"round_no": 1})
    emit(audit.TASK_STARTED, task_id="t2", status="SUCCESS", metadata={"round_no": 1})

    # Researcher t2（多跳检索）
    bind(node="researcher", task_id="t2")
    emit(audit.MULTI_HOP_ITERATION, status="SUCCESS", metadata={
        "iteration": 1, "query": "ECAPA-TDNN 注意力机制", "new_evidence_count": 2,
        "action": "CONTINUE", "gap": "缺性能对比", "stop_reason": ""})
    emit(audit.MULTI_HOP_ITERATION, status="SUCCESS", metadata={
        "iteration": 2, "query": "x-vector vs ECAPA-TDNN EER", "new_evidence_count": 1,
        "action": "STOP", "gap": "", "stop_reason": "SUFFICIENT"})
    emit(audit.EVIDENCE_REGISTERED, task_id="t2", status="SUCCESS", metadata={
        "evidence_id": "e2", "source_id": "s2", "source_type": "web",
        "verification_status": "VERIFIED", "claim_id": "C2"})
    emit(audit.TASK_COMPLETED, task_id="t2", status="SUCCESS")

    # Aggregator
    bind(node="aggregator", task_id="")
    emit(audit.AGGREGATION, status="SUCCESS", metadata={
        "source_count": 2, "evidence_count": 2,
        "claim_by_status": {"SUPPORTED": 2, "PARTIALLY_SUPPORTED": 0,
                            "UNSUPPORTED": 0, "CONTRADICTED": 0}})

    # Review → Repair → Judge
    bind(node="review", task_id="")
    emit(audit.REVIEW_ISSUE, status="SUCCESS", metadata={"issue_count": 1})
    bind(node="repair", task_id="")
    emit(audit.REPAIR, status="SUCCESS", metadata={"round_no": 1, "added_tasks": 0,
                                                   "reopened_tasks": 0})
    bind(node="judge", task_id="")
    emit(audit.JUDGE_DECISION, status="SUCCESS", metadata={
        "decision": "COMPILE", "reason": "任务全部完成", "round_no": 1,
        "unfinished_tasks": 0, "stagnant_rounds": 0})

    # Compiler
    bind(node="compiler", task_id="")
    emit(audit.CITATION_VERIFIED, status="SUCCESS", metadata={
        "refs": 2, "citation_issues": 0, "used_sources": 2})
    emit(audit.RUN_END, status="SUCCESS", metadata={"verdict": "COMPILE"})

    audit_logger.flush()
    print(f"[Demo] 已写入 logs/{thread_id}.jsonl，run_id={run_id}")
    print("[Demo] 启动 Dashboard 查看: python -m entry.dashboard")


if __name__ == "__main__":
    main()
