# -*- coding: utf-8 -*-
"""
Mock 审计流生成器：为 entry/monitor.py 提供高保真的实时流式事件，方便演示与报告截图。
用法：
    1. 另开终端运行: python entry/monitor.py
    2. 本终端运行: python scripts/mock_stream_monitor.py
"""

import os
import json
import time
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "local_geek_master.jsonl")


def emit_event(event_dict: dict, delay: float = 1.0):
    """向监控日志追加一条事件并留出间隔展示动效"""
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    event_dict.setdefault("ts", now_iso)
    event_dict.setdefault("thread_id", "local_geek_master")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event_dict, ensure_ascii=False) + "\n")
        f.flush()

    time.sleep(delay)


def run_mock_simulation():
    # 确保文件存在，方便 monitor 捕捉
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            pass

    print("[*] 开始向 logs/local_geek_master.jsonl 注入模拟事件流...")

    # 1. 启动状态
    emit_event({
        "event": "system_action",
        "content": "收到多智能体科研研报任务：【大模型长上下文与 SSM/Mamba 混合架构前沿综述】"
    }, delay=1.2)

    # 2. 神经元唤醒
    emit_event({
        "event": "llm_input",
        "message_count": 4
    }, delay=1.0)

    # 3. 规划分解工具调用
    emit_event({
        "event": "tool_call",
        "tool": "research_planner",
        "args": {
            "topic": "Mamba vs Transformer 长上下文效率",
            "sop_mode": "COMPARISON_SOP",
            "max_subtasks": 3
        }
    }, delay=1.5)

    # 4. 规划结果
    emit_event({
        "event": "tool_result",
        "tool": "research_planner",
        "result_summary": "DAG 拓扑构建完成：\n[t1] 检索 Mamba 原论文与线性注意力机制 (PRIORITY=1)\n[t2] 调研 Jamba / Samba 混合架构实证吞吐 (PRIORITY=2, 依赖 t1)\n[t3] 交叉比对 1M+ 上下文外推与 Needle-in-Haystack 评测 (PRIORITY=3, 依赖 t2)"
    }, delay=1.2)

    # 5. 状态流转
    emit_event({
        "event": "system_action",
        "content": "并发调度器生效：Send API 派发 Worker-1 启动学术文献检索"
    }, delay=0.8)

    # 6. 学术检索工具调用
    emit_event({
        "event": "tool_call",
        "tool": "arxiv_mcp_search",
        "args": {
            "query": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
            "max_results": 3,
            "fields": ["title", "abstract", "doi", "published"]
        }
    }, delay=1.6)

    # 7. 学术检索结果回传
    emit_event({
        "event": "tool_result",
        "tool": "arxiv_mcp_search",
        "result_summary": "匹配到 3 篇核心 arXiv 文献：\n1. arXiv:2312.00752: Gu & Dao, 'Mamba: Linear-Time Sequence Modeling...'\n2. arXiv:2403.19887: 'Jamba: A Hybrid Transformer-Mamba Language Model'\n3. arXiv:2404.05892: 'RecurrentGemma: Moving Beyond Transformers for Efficient Inference'"
    }, delay=1.2)

    # 8. RAG 知识库切片提取
    emit_event({
        "event": "tool_call",
        "tool": "rag_hybrid_retriever",
        "args": {
            "query": "Mamba 选择性状态空间 O(N) 吞吐比与显存复杂度",
            "top_k": 5,
            "rerank": True
        }
    }, delay=1.4)

    emit_event({
        "event": "tool_result",
        "tool": "rag_hybrid_retriever",
        "result_summary": "完成 BM25+向量 混合召回与 BGE-Reranker 重排：\n- [Chunk-01] (Score: 0.941) Hardware-aware Scan 硬件感知的并行扫描原理解析\n- [Chunk-02] (Score: 0.892) 5x throughput compared to standard attention at 8k seq_len"
    }, delay=1.5)

    # 9. 证据链对齐与批判校验
    emit_event({
        "event": "system_action",
        "content": "Critic 节点介入：3 条待定断言 (Claims) 均已找到原生 arXiv 溯源证据，核验通过"
    }, delay=1.0)

    # 10. 终局完成
    emit_event({
        "event": "system_action",
        "content": "研报编译成功：生成 4 章节带可溯源引注的科研综合报告，全流程审计归档落盘。"
    }, delay=0.5)

    print("[OK] 模拟事件流注入完毕，请在 monitor.py 窗口查看完整效果。")


if __name__ == "__main__":
    run_mock_simulation()
