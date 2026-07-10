# 多智能体深度调研系统

> `clawgent/core/research/` — LangGraph 子图，7 个节点协作完成从问题拆解到结构化报告的全流程。

---

## 整体架构

用户提出一个复杂调研问题后，系统不直接让 LLM 回答，而是跑一套多角色协作的调研流程：

```
用户问题
    │
    ▼
【Planner】拆题 → 3-6 个子任务（各含 2-3 条检索词）
    │
    │ Send API fan-out
    ├──────────────────────────────────────┐
    ▼              ▼              ▼        ▼
【Researcher】  【Researcher】  【Researcher】...  ← 并发，每个子任务独立实例
  hybrid_search   hybrid_search   hybrid_search
  抽 claim+URL    抽 claim+URL    抽 claim+URL
    │              │              │
    └──────────────┴──────────────┘
                   │ operator.add 自动合并
                   ▼
           【Aggregator】去重，建 claim→URL 映射
                   │
                   ▼
            【Critic】红队挑刺
            找 missing_evidence / factual_conflict / logic_gap
                   │
                   ▼
           【Revision】补充检索（针对 high/medium 级问题）
                   │
                   ▼
             【Judge】裁决
             ┌─────┴─────┐
         未达标          达标
         revision_count+1  │
             │            ▼
         回 Critic    【Compiler】写报告
                          │
                          ▼
                    Markdown 报告（含引用）
```

---

## 节点详解

### Planner（`nodes.py:38`）

将原始问题拆成 3-6 个独立子任务，每个任务对应一个调研角度（现状、趋势、案例、风险、对比等）。

```python
# 输出结构示例
[
  {"task_id": "t1", "question": "Mamba的核心原理是什么？",
   "angle": "技术原理", "search_queries": ["Mamba SSM architecture", "selective state space"]},
  {"task_id": "t2", "question": "与Transformer相比优势在哪？",
   "angle": "对比分析", "search_queries": ["Mamba vs Transformer", "linear attention comparison"]},
  ...
]
```

通过 `Command(goto=[Send("researcher", task), ...])` 实现 fan-out，LangGraph 并发调度所有 Researcher。

---

### Researcher（`nodes.py:92`）

每个子任务对应一个独立 Researcher 实例，做两件事：

**1. 三路混合检索（`search.py: hybrid_search`）**

```
学术 MCP（arXiv / Semantic Scholar / PubMed）  ──┐
Tavily 联网搜索                                ──┼──▶ asyncio.gather，return_exceptions=True
本地知识库（向量 + BM25）                      ──┘
学术结果排前
```

**2. LLM 抽 claim 级证据**

从检索结果里提取 3-6 条具体可验证的事实声明，每条必须绑定来源 URL：

```python
{
    "claim": "Mamba 使用选择性状态空间模型，线性时间复杂度 O(n)",
    "relevance": 0.92,
    "source": {
        "url": "https://arxiv.org/abs/2312.00752",
        "title": "Mamba: Linear-Time Sequence Modeling...",
        "snippet": "原文片段..."
    }
}
```

**并发写入不冲突**：`state.py` 声明 `evidences: Annotated[list[dict], operator.add]`，LangGraph 用 `operator.add` reducer 自动追加合并，多个 Researcher 同时写入不会互相覆盖。

---

### Aggregator（`nodes.py:167`）

对所有 Researcher 输出的 claim 去重（按 claim 前 80 字符），建立 `claim → 来源列表` 的映射表。

---

### Critic（`nodes.py:195`）

扮演"红队审查员"，检查三类问题：

| 问题类型 | 说明 | 示例 |
|---------|------|------|
| `missing_evidence` | 关键角度没有覆盖 | "缺少性能基准测试数据" |
| `factual_conflict` | 两条 claim 互相矛盾 | "A说推理速度快，B说推理速度慢" |
| `logic_gap` | 推理有跳跃 | "从原理直接跳到结论，缺中间步骤" |

只保留 `high / medium` 级问题（`low` 级忽略），针对 `missing_evidence` 类生成补充检索词（最多 3 条）。

---

### Revision（`nodes.py:244`）

拿 Critic 给出的补充检索词，再跑一遍 `hybrid_search`，LLM 从结果里抽 2-4 条新 claim，写入 `revision_evidences`（同样用 `operator.add`）。

如果补查什么都没找到，直接返回空列表——Judge 会识别到"无新增"并终止循环。

---

### Judge（`nodes.py:305`）

**三重终止条件**，满足任一即出报告：

```python
should_compile = (
    len(high_issues) == 0          # ① Critic 没有发现严重问题
    or revision_count >= max_revisions  # ② 已达最大补查轮次（默认 2）
    # ③ 隐含：Revision 返回空 → revision_evidences 为空 → Judge 发现无新增
)
```

不满足则 `revision_count += 1`，`Command(goto="critic")` 回到 Critic 再来一轮。

**置信度计算**：

```python
confidence = max(0.3, min(1.0,
    0.5 + 0.1 * len(all_evidences) - 0.2 * len(high_issues)
))
```

- 每增加一条证据 +0.1
- 每个严重问题 -0.2
- 下限 0.3（不会给出"100% 可信"）

---

### Compiler（`nodes.py:347`）

把所有证据按子任务分组，生成完整 Markdown 报告：

```markdown
## 执行摘要
...（3-5 句话）

## [各角度分节发现]
- 关键结论 [来源: https://arxiv.org/...]
- ...

## 结论与建议
...

## 参考来源
- https://arxiv.org/...
- ...
```

报告头部附置信度（`pass / partial`），`partial` 表示仍有未解决的 high 级问题，用户可自行判断是否需要进一步核查。

---

## 可靠性保障

| 机制 | 位置 | 作用 |
|------|------|------|
| 学术源优先 | `search.py` | arXiv/S2/PubMed 优先于普通网页 |
| claim 强制绑 URL | `nodes.py:140` | 每条结论可溯源，无 URL 则明显留空 |
| Critic 红队 | `nodes.py:195` | 独立视角交叉验证，找矛盾和缺漏 |
| 三重终止 | `nodes.py:319` | 防死循环，最多 2 轮补查 |
| `return_exceptions=True` | `search.py` | 任一检索源失败不影响其他路 |
| 保守置信度 | `nodes.py:322` | 下限 0.3，不虚报可信度 |

---

## 相关文件

```
clawgent/core/research/
├── graph.py      # LangGraph 子图定义，节点连线与重试策略
├── nodes.py      # 7 个节点的具体实现
├── state.py      # ResearchStateDict，含 operator.add reducer 声明
├── search.py     # hybrid_search：学术MCP + Tavily + RAG 三路并发
└── academic.py   # 学术 MCP 客户端（arXiv / Semantic Scholar / PubMed）
```

触发入口：`clawgent/core/tools/research_tool.py` → `deep_research(query)` 工具。
