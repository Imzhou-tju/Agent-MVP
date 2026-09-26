# 多智能体调研系统

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
| `UNVERIFIED_EVIDENCE` | 证据未通过原文校验 |
| `UNSUPPORTED_CLAIM` | 声明无可用证据（UNSUPPORTED / CONTRADICTED） |
| `WEAK_SOURCE` | 来源质量不足 |
| `CONTRADICTORY_EVIDENCE` | 证据之间存在矛盾（含确定性冲突检测） |
| `SCOPE_MISMATCH` | 结论适用范围不一致（确定性范围检测） |
| `INCOMPLETE_TASK` | 任务未完成 |
| `DUPLICATE_RESEARCH` | 重复研究 |
| `COVERAGE_GAP` | 研究问题有未被覆盖的方面 |
| `LOGIC_GAP` | 从证据到结论的推理跳跃 |

每个问题带 `severity`、`target_type/target_id`、`target_claim_id` / `target_task_id`、
`recommended_action`（`add_task` / `reopen_task` / `none`）、`required_evidence`、`priority`。
`issue_id` 由 `(类型, 目标, 描述)` 哈希得到，下一轮重复提出同一问题时 id 相同。

确定性发现的冲突（§25-28）与范围不一致由 `ConflictDetector` 计算，作为 `CONTRADICTORY_EVIDENCE` /
`SCOPE_MISMATCH` 类问题写入，其 `recommended_action=none`（不触发补任务，仅作提示）。

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
3. `CitationVerifier` 校验（§45 在原文基础上新增三道闭环校验）：编号是否存在、引用的证据是否通过校验、
   证据引用的来源是否登记在 Source Catalog（`source_missing`）、证据是否归属于某个已登记 Claim
   （`evidence_orphan`）、证据归属的 Claim 是否存在（`claim_missing`）、可用声明是否被引用
4. 校验出 `unknown_ref` / `unverified_evidence` 时，把问题清单反馈给模型重跑一次（最多 1 次）
5. 程序追加"参考来源"列表与"本报告局限"小节
6. 成文后由 `CitationBinder`（§44）建立"草稿句子 → Claim → Evidence → Source"闭环，
   并统计未被引用的悬挂证据与未闭合 Claim（随 `citation_binding` 进入产出，供审计）
7. `UnsupportedClaimDetector`（§48）扫描正文，把 UNSUPPORTED / CONTRADICTED 声明从正文移除，
   并单列到"未支撑声明警示"小节，确保未支撑结论不被当作事实输出（§49）

局限小节由程序生成，内容包括：未完成任务数量与 id、未通过校验的证据条数、可用声明不足提示、
以及被检测器移出正文的未支撑声明清单。

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
├── conflict.py   # ConflictDetector（§25-28 确定性冲突/范围检测）
├── review.py     # ResearchReview / build_review（§29 确定性评审快照）
├── citation.py   # SourceCatalog / CitationVerifier / CitationBinder / UnsupportedClaimDetector
├── ledger.py     # ResearchLedger / build_trace（过程记录与回溯）
├── search.py     # hybrid_search：学术MCP + Tavily + RAG 三路并发
└── academic.py   # 学术 MCP 客户端（arXiv / Semantic Scholar / PubMed）
```

测试：`tests/test_research_dag.py`（确定性逻辑）、`tests/test_research_flow.py`（离线整链路，不联网）、
`tests/test_research_scenarios.py`（14 个核心闭环场景 + 检测器，离线、无需 langgraph）。
触发入口：`clawgent/core/tools/research_tool.py` → `deep_research(query)` 工具。

---

## 核心闭环补全实现（补实现，2026-09）

在 `research-dag` 分支既有 DAG 调度 + Evidence→Claim→Source 溯源骨架之上，把"调研核心闭环"
从多个散点能力补成端到端闭合：问题→计划/DAG→调度→任务→来源/检索→证据→校验→声明→
声明-证据关系→评审→修订→裁决→成文→引用闭合→未支撑拦截→可溯源报告。约束：不下载模型、
代码保持精简、测试以离线确定性为主（API 未必可用）。

### 修改/新增文件清单与用途

| 文件 | 动作 | 用途 |
|------|------|------|
| `dag.py` | 改 | `DONE` 增加 `COMPLETED`/`SKIPPED` 别名；`ResearchTask` 增加 `parent_task_id`/`revision_round`；新增 `ResearchPlan`（§4.1） |
| `claim.py` | 改 | `Claim` 增加 `scope_fields`/`polarity`/`value`（供 `ConflictDetector` 做确定性比较，§24-28） |
| `state.py` | 改 | 新增裁决 `COMPILE_WITH_LIMITATIONS`；状态增加 `plan`/`review`/`last_evidence_count`/`stagnant_rounds` |
| `conflict.py` | 新增 | `ConflictDetector` + `detect_pairwise`：四类判定 `NO_CONFLICT`/`DIRECT_CONFLICT`/`SCOPE_MISMATCH`/`POTENTIAL_CONFLICT`（§25-28），纯确定性，无 LLM |
| `review.py` | 新增 | `ResearchReview` + `build_review`：基于支撑状态与 `ConflictDetector` 计算覆盖率/未支撑/冲突/范围不一致（§29） |
| `citation.py` | 改 | 新增 `CitationBinder`（§44 闭环）、`CitationRenderer`（§47）、`UnsupportedClaimDetector`（§48）；`CitationVerifier` 增加 §45 三道闭环校验；新增 `source_missing`/`evidence_orphan`/`claim_missing` |
| `ledger.py` | 改 | 新增事件 `UNSUPPORTED_CLAIM` |
| `nodes.py` | 改 | `planner_node` 构建 `ResearchPlan`；`review_node` 用 `build_review` + 确定性冲突问题；`repair_node` 修订任务命名/去重 + `REVISION_CREATED`；`judge_node` 进展门控（ stagnant_rounds ）+ `COMPILE_WITH_LIMITATIONS`；`compiler_node` 接入 `CitationBinder` 与 `UnsupportedClaimDetector` |
| `research_tool.py` | 改 | 初始状态补 `plan`/`review`/`last_evidence_count`/`stagnant_rounds` |
| `tests/test_research_scenarios.py` | 新增 | 14 个核心场景 + 检测器/Plan 往返，共 19 个用例，离线确定性 |

### 架构变化（Before → After）

| 维度 | Before | After |
|------|--------|-------|
| 冲突/范围判断 | 无确定性判定，靠模型在抽取/评审时用自然语言描述 | `ConflictDetector`（§25-28）：基于 `scope_fields`+`polarity`+`value` 给出四类结果，同方法不同 dataset → `SCOPE_MISMATCH`（非直接冲突） |
| 评审结构 | Review 输出自由 JSON，含 7 类旧问题类型 | 新增 `ResearchReview` 快照（§29）+ 10 类问题（§30），每个问题带 `recommended_action`/`required_evidence`/`priority`/`target_claim_id`/`target_task_id`（§31） |
| 修订任务 | 仅 `add_task`/`reopen_task`，无修订溯源字段 | `ResearchTask` 带 `parent_task_id`/`revision_round`（§33），命名 `{parent}-R{round}` 或 `R{round}-{n}`，按 `issue_id` 去重（§34） |
| 终止裁决 | 仅 REVISE/COMPILE/ABORT | 增加进展门控（§38）：连续两轮 `new_evidence==0` 强制终止为 `COMPILE_WITH_LIMITATIONS`/`ABORT_WITH_LIMITATIONS` |
| 引用闭合 | `CitationVerifier` 只查编号存在 + 证据可引用 | 增加 §44 `CitationBinder`（句子→Claim→Evidence→Source）与 §45 三道闭环校验（来源登记 / 证据归属 Claim / Claim 存在） |
| 未支撑拦截 | 无 | §48 `UnsupportedClaimDetector`：UNSUPPORTED/CONTRADICTED 声明从正文移除并单列警示（§49） |
| 研究计划 | Planner 直接产出任务列表 | 新增 §4.1 `ResearchPlan`（plan_id/objective/constraints/tasks），结构化存储并进入 state |

### 测试结果

- 用例：`tests/test_research_scenarios.py`，共 **19** 个，全部通过（离线、`unittest`，无需 langgraph）。
- 覆盖的 14 个场景：DAG 正常调度、环检测、依赖失败阻塞、证据伪造（INVALID）、空源（UNVERIFIED）、
  无证据→UNSUPPORTED、两声明直接冲突、范围不一致、修订任务创建、重复修订拒绝、
  引用闭环、断引（UNKNOWN_REF）、假源/未登记来源拦截、无证据失败。
- 额外覆盖：`UnsupportedClaimDetector` 正文净化、ResearchPlan 往返。
- 加载方式：优先从正式包导入；环境缺 langgraph 时退回合成包加载纯逻辑子模块（`_load_pure`），
  绕开 `research/__init__` 对 graph 的导入，保证离线可跑。

### 明确未完项（如实标注）

1. **整链路集成测试未跑**：14 个场景测试只覆盖纯逻辑模块（dag/claim/evidence/conflict/review/citation/ledger），
   不调用 LangGraph 整图。新组件在 `compiler_node` 内的实际串联（CitationBinder / UnsupportedClaimDetector 在运行时触发）
   已落地但仅在单元测试层面验证，未在缺 API 环境下跑 `test_research_flow.py` 整链路。
2. **修订去重的单元覆盖偏窄**：`repair_node` 的"按 issue_id 去重、单轮上限 3、命名冲突后缀规避"已实现，
   但离线测试只在 DAG 层验证了"相同 task_id 被拒"，未对 repair 编排逻辑做端到端断言（需 LLM/网络）。
3. **`CitationRenderer`（§47）已实现但未接线**：`CitationRenderer.render_closure` 可用于把闭环渲染成附录，
   当前 `compiler_node` 只返回 `citation_binding` 结构、未调用该渲染器生成可视化附录。属可选项。
4. **`ResearchPlan` 已入 state 但未驱动额外校验**：目前仅作为结构化产出存储，尚未用作"计划覆盖率"校验输入。
5. **§46 禁止模型自由生成 URL**：已通过结构性保证（Evidence 必须引用 `SourceRegistry` 中已登记 `source_id`，
   未登记直接判 INVALID），未单独在 compiler 再加一道运行时白名单检查。
6. **`POTENTIAL_CONFLICT` 的呈现**：`build_review` 把 `DIRECT_CONFLICT` 与 `POTENTIAL_CONFLICT` 一并归入 `contradictions`，
   与 §28 "同 dataset/metric 不同数值→POTENTIAL_CONFLICT" 一致；若希望单独区分提示，可再细分，当前未做。

> 以上未完项均为"已实现了代码逻辑、但测试/接线层面未做满"，不影响已通过单元测试 correctness；
> 真正依赖外部 API 的整链路行为需在有 key 的环境中用 `tests/test_research_flow.py` 复验。
