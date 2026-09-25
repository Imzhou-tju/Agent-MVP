# 多智能体深度调研系统

> `clawgent/core/research/` — LangGraph 子图，8 个节点协作完成从问题拆解到带引用报告的全流程。
>
> 调度骨架是 **Research Task DAG**，核心状态是 **Evidence**，核心数据结构是 **Claim-Evidence 关系**。

---

## 整体架构

用户提出一个复杂调研问题后，系统不直接让 LLM 回答，而是跑一套多角色协作的调研流程：

```
用户问题
    │
    ▼
【Planner】拆题 → 3-6 个研究任务 + 任务间依赖（DAG）
    │
    ▼
【Scheduler】（不调用模型）
 挑出依赖已满足的 READY 任务
    │ Send API fan-out（只发这一批）
    ├──────────────┬──────────────┐
    ▼              ▼              ▼
【Researcher】  【Researcher】  【Researcher】  ← 并发，每个任务独立实例
  检索 → 登记来源 → 抽 claim + quote → 原文定位校验
    │              │              │
    └──────────────┴──────────────┘
                   │ 返回 ResearchPacket，按 id 合并
                   ▼
            回到【Scheduler】放下一批依赖已满足的任务
                   │ 没有 READY 任务
                   ▼
        【Aggregator】（不调用模型）
         按证据校验状态推导每条 Claim 的支撑状态
                   │
                   ▼
             【Review】结构化评审（7 类问题）
                   │
                   ▼
        【Repair】（不调用模型）
         按问题局部新增/重开 DAG 任务，不重建整棵树
                   │
                   ▼
             【Judge】（不调用模型）
        ┌──────────┼──────────────┐
     REVISE      COMPILE      ABORT_WITH_LIMITATIONS
        │            │              │
   回 Scheduler      └──────┬───────┘
                            ▼
                     【Compiler】写报告
                       引用目录 + 引用校验 + 参考来源渲染
                            │
                            ▼
                     Markdown 报告（带 [S1]/[E3] 引用标记）
```

---

## 节点详解

### Planner（`nodes.py`）

把原始问题拆成 3-6 个研究任务，并声明任务之间的依赖。依赖只在确实需要前序结论时才声明。

```python
# 输出结构示例
[
  {"task_id": "t1", "objective": "搞清楚 Mamba 的核心机制",
   "question": "Mamba 选择性状态空间机制原理", "task_type": "FACT",
   "expected_evidence": "论文给出的复杂度结论", "dependencies": []},
  {"task_id": "t2", "objective": "与 Transformer 对比",
   "question": "Mamba 与 Transformer 长序列推理吞吐对比",
   "task_type": "COMPARISON", "dependencies": ["t1"]},
]
```

`task_type` 取值：`FACT | METHOD | RESULT | COMPARISON | LIMITATION | TREND | INTERPRETATION | HYPOTHESIS`。

依赖指向未声明的任务时直接丢弃该依赖，避免整张图失效；新增任务由 `dag.TaskDAG.add_task` 校验重复 id、自依赖、依赖缺失与成环。

---

### Scheduler（`nodes.py`）

不调用模型。按 `dag.TaskDAG` 做确定性调度：

1. `refresh()`：依赖全部 DONE → `READY`；否则 → `BLOCKED`
2. `ready_tasks()`：按 priority 排序后返回 READY 任务
3. 标记为 `RUNNING`，用 `Command(goto=[Send("researcher", ...)])` 只发这一批

任务状态：`PENDING / READY / RUNNING / DONE / FAILED / BLOCKED / REOPENED`。
依赖任务 FAILED 时，下游保持 BLOCKED，不再被调度。

---

### Researcher（`nodes.py`）

每个任务一个独立实例，返回 `ResearchPacket`，不直接改写全局结论。

**1. 三路混合检索（`search.py: hybrid_search`）**

```
学术 MCP（arXiv / Semantic Scholar / PubMed）  ──┐
Tavily 联网搜索                                ──┼──▶ asyncio.gather，return_exceptions=True
本地知识库（向量 + BM25）                      ──┘
学术结果排前
```

**2. 登记来源（`evidence.py: SourceRegistry`）**

每条检索结果登记为 `Source`，`source_id` 由内容哈希生成（`stable_id`），因此并发分支对同一 URL 会得到同一个 id，合并后不会重复登记。去重键优先级：`url > doi > title+source_type`。

**3. 抽取 claim 与 quote，并做原文定位校验（`evidence.py: EvidenceVerifier`）**

模型输出每条 claim 必须携带 `quote`（逐字来自检索正文的引文）。程序按以下规则判定：

| 判定 | 条件 |
|------|------|
| `VERIFIED` | quote 归一化后完整出现在来源正文中 |
| `PARTIAL` | 最长公共片段占 quote 长度的比例 ≥ 0.6 |
| `INVALID` | source_id 未登记 / quote 过短（<8 字符）/ locator 与 chunk_id 不一致 / 无法定位 |
| `UNVERIFIED` | 缺少来源正文，无法判定 |

**它不做什么**：校验只能比对"本次检索拿到的正文"。网页来源拿到的是搜索结果摘要而不是整页正文，`UNVERIFIED`/`INVALID` 只说明"在拿到的这段文本里定位不到"，不等于该说法在原文里不存在。本地知识库来源带 `chunk_id`，可以定位到具体切片（`locator = chunk:xxx`）。

---

### Aggregator（`nodes.py`）

不调用模型。做两件确定性工作：

1. 合并各任务返回的 claim / evidence / relation（按 id 合并，见下文 reducer）
2. `ClaimGraph.recompute_statuses()` 按证据校验结果推导每条 Claim 的支撑状态：

| Claim 状态 | 判定规则 |
|-----------|---------|
| `SUPPORTED` | 至少 1 条 VERIFIED，无 INVALID、无缺失 |
| `PARTIALLY_SUPPORTED` | 至少 1 条 VERIFIED/PARTIAL，但存在 INVALID 或缺失 |
| `CONTRADICTED` | 存在 CONTRADICTS 关系 |
| `UNSUPPORTED` | 无关联证据，或证据全部不可用 |

Claim 的支撑状态不由模型自评，也不再由启发式公式计算。原实现里的
`confidence_score = 0.5 + 0.1*证据数 - 0.2*严重问题数` 已删除——这个数值与证据是否被校验通过无关。

---

### Review（`nodes.py`）

评审当前研究状态，输出结构化问题清单。问题类型：

| 类型 | 说明 |
|------|------|
| `MISSING_EVIDENCE` | 声明缺少证据 |
| `UNSUPPORTED_CLAIM` | 声明的证据未通过原文校验 |
| `CONFLICT` | 证据之间存在矛盾 |
| `COVERAGE_GAP` | 研究问题有未被覆盖的方面 |
| `SOURCE_QUALITY` | 来源质量不足 |
| `OUTDATED_SOURCE` | 来源过旧 |
| `LOGIC_GAP` | 从证据到结论的推理跳跃 |

每个问题带 `severity`、`target_type/target_id`（作用于哪条声明或哪个任务）与 `suggested_action`（`add_task` / `reopen_task` / `none`）。`issue_id` 由 `(类型, 目标, 描述)` 哈希得到，下一轮重复提出同一问题时 id 相同。

---

### Repair（`nodes.py`）

不调用模型。按问题清单局部修改 DAG：

- `add_task`：新增任务，单轮上限 `MAX_REPAIR_TASKS = 3`
- `reopen_task`：`dag.reopen()` 重开任务，并把它的下游标记为 REOPENED（影响局部化）

只处理 `high/medium` 级问题，且同一个 `issue_id` 只处理一次（`repaired_issue_ids`）。非法结构由 `TaskDAG.add_task` 拒绝，不重建整棵任务树。

---

### Judge（`nodes.py`）

不调用模型，只做终止裁决：

| 裁决 | 触发条件 |
|------|---------|
| `REVISE` | DAG 里存在 READY 任务，且未达到 `max_revisions` |
| `COMPILE` | 无可执行任务，且存在可用声明（SUPPORTED / PARTIALLY_SUPPORTED） |
| `ABORT_WITH_LIMITATIONS` | 无可执行任务，且没有任何可用声明 |

裁决依据只有两件事：**还有没有可执行的任务**、**有没有通过校验的可用声明**。
达到轮次上限但仍有任务没跑完时，按 COMPILE 出报告，并在报告末尾列出未完成任务。

---

### Compiler（`nodes.py`）

1. 建立 `SourceCatalog`：把 `source_id / evidence_id` 编成 `[S1] / [E3]` 短编号，模型只用编号引用，不需要抄写 URL
2. 模型按编号写报告
3. `CitationVerifier` 校验：编号是否存在、引用的证据是否通过校验、可用声明是否被引用
4. 校验出 `unknown_ref` / `unverified_evidence` 时，把问题清单反馈给模型重跑一次（最多 1 次）
5. 程序追加"参考来源"列表与"本报告局限"小节

局限小节由程序生成，内容包括：未完成任务数量与 id、未通过校验的证据条数、可用声明不足提示。

---

## 状态与并发合并（`state.py`）

Researcher 通过 `Send` 并发执行，被并发写入的字段必须声明 reducer。
旧实现统一用 `operator.add`（列表追加），多个任务引用同一来源时会产生重复登记；
现改为按 id 合并的语义化 reducer：

| 字段 | reducer | 合并规则 |
|------|---------|---------|
| `sources` | `merge_sources` | 按 `source_id` 合并，只补齐空字段 |
| `evidences` | `merge_evidences` | 按 `evidence_id` 合并 |
| `claims` | `merge_claims` | 按 `claim_id` 合并，`evidence_ids` 取并集 |
| `relations` | `merge_relations` | 按 `(source_id, target_id, relation)` 去重 |
| `task_results` | `merge_task_results` | 按 `task_id` 覆盖 |
| `tasks` | `merge_tasks` | 按 `task_id` 覆盖（DAG 是可变结构） |
| `issues` | `merge_issues` | 按 `issue_id` 合并 |
| `searched_queries` | `merge_str_list` | 去重 |
| `source_texts` | `merge_dict` | 按 key 覆盖 |
| `ledger_events` | `append_list` | 只追加；节点只回写本节点新增的事件 |

配合内容哈希生成 id（`evidence.stable_id`、`claim.make_claim_id`），
同一来源、同一句话在不同并发分支里得到同一个 id，从而收敛为一条。

---

## 可追溯性：从结论回到检索式

- `ledger.py: ResearchLedger`：只追加的事件流水（任务调度、来源登记、证据抽取与校验、问题、裁决）
- `ledger.py: build_trace`：按"声明 → 证据 → 来源 → 任务 → 检索式"回溯，输出 `trace`

报告里的每个 `[E3]` 都能对应到一条 Evidence，Evidence 上有 `quote` 与 `verification_status`，
Evidence 指向 Source，Source 上有 url / doi / 年份 / 来源类型。

**边界**：引用校验只覆盖"报告里的编号是否指向真实且通过校验的证据"，
不校验"这条证据是否真的支持这句话"——后者是语义判断，当前由模型在抽取阶段完成，程序不复核。

---

## 相关文件

```
clawgent/core/research/
├── graph.py      # LangGraph 子图定义，节点连线与重试策略
├── nodes.py      # 8 个节点的具体实现
├── state.py      # ResearchStateDict 与语义化 reducer
├── dag.py        # ResearchTask / ResearchPacket / TaskDAG（确定性调度）
├── evidence.py   # Source / SourceRegistry / Evidence / EvidenceVerifier
├── claim.py      # Claim / ClaimRelation / ClaimGraph（支撑状态推导）
├── citation.py   # SourceCatalog / CitationVerifier（引用目录与校验）
├── ledger.py     # ResearchLedger / build_trace（过程记录与回溯）
├── search.py     # hybrid_search：学术MCP + Tavily + RAG 三路并发
└── academic.py   # 学术 MCP 客户端（arXiv / Semantic Scholar / PubMed）
```

测试：`tests/test_research_dag.py`（确定性逻辑）、`tests/test_research_flow.py`（离线整链路，不联网）。
触发入口：`clawgent/core/tools/research_tool.py` → `deep_research(query)` 工具。
