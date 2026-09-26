# research-dag 科研智能体 · 完整逻辑闭环实现说明

> 分支：`research-dag`
> 范围：在既有 research-dag 架构之上补齐确定性缺口，使系统「逻辑闭环、状态一致、边界明确、可解释、可追溯」。
> 原则：不重新设计系统、复用既有对象与函数；本轮不以测试为交付目标（见 §5）。

---

## 1. 本次修改清单（按规范 §31 的 22 项映射）

| # | 规范要点 | 落点文件 | 确定性 / LLM |
|---|----------|----------|--------------|
| 1 | Planner 依赖回退（依据任务类型+标题文本推导依赖） | `research/dag.py`：`normalize_task_dependencies` + `_DEPENDENCY_RULES` | 确定性 |
| 2 | 统一 DAG 状态机（合法转移表，非法转移一律拒绝） | `research/dag.py`：`_TRANSITIONS` + `TaskDAG.transition_task` | 确定性 |
| 3 | Claim–Evidence 语义关系（SUPPORTS / PARTIALLY_SUPPORTS / CONTRADICTS / IRRELEVANT） | `research/claim.py`：`SEM_*`、`Claim.semantic_relation`、`ClaimRelation.semantic_relation` | 由 LLM 产出原始关系，程序规整 |
| 4 | Claim 状态确定性重算（校验状态 + 语义关系 + CONTRADICTS 组合） | `research/claim.py`：`ClaimGraph.recompute_statuses` / `recompute_claim_statuses` | 确定性 |
| 5 | 冲突/范围闭环（CONTRADICTORY / SCOPE_MISMATCH 确定性识别并写入 issues） | `research/conflict.py` + `nodes.py:review_node` | 确定性（前序已实现，本轮接入评审） |
| 6 | Review→Repair 闭环（issue 带 recommended_action / required_evidence，repair 只接受 add_task / reopen_task） | `nodes.py:review_node` / `repair_node` | 确定性 |
| 7 | Repair 去重与预算（同一 issue 只处理一次；`MAX_REPAIR_TASKS` 上限） | `nodes.py:repair_node`（`repaired_issue_ids` + 计数） | 确定性 |
| 8 | Judge 增强（每轮进展日志 + 原始/修订任务区分） | `nodes.py:judge_node` + `state.py`（`progress_log` / `last_claim_count` / `last_source_count`） | 确定性 |
| 9 | 原始任务 vs 修订任务区分（`CONTINUE_UNFINISHED` / `REVISE` / `COMPILE_WITH_LIMITATIONS`） | `nodes.py:judge_node` + `state.py:CONTINUE_UNFINISHED` | 确定性 |
| 10 | 停滞检测（连续两轮无新增证据即终止，防空转） | `nodes.py:judge_node`（`stagnant_rounds >= 2`） | 确定性 |
| 11 | CitationBinder 精细化（草稿句子→Claim→Evidence 绑定） | `research/citation.py` + `nodes.py:compiler_node` | 确定性（前序已实现，本轮保留） |
| 12 | CitationVerifier 精细化（UNKNOWN_REF / UNVERIFIED_EVIDENCE 阻断重跑） | `research/citation.py` + `nodes.py:compiler_node` | 确定性（前序已实现，本轮保留） |
| 13 | UnsupportedClaimDetector 接入 Compiler（UNSUPPORTED/CONTRADICTED 禁止作为事实写入正文） | `nodes.py:compiler_node`（`detect_and_sanitize`） | 确定性 |
| 14 | Web 摘要/原文边界（仅摘要来源的证据强制 `UNVERIFIED`） | `research/evidence.py`：`SEARCH_SNIPPET` + `EvidenceVerifier.check` | 确定性 |
| 15 | 来源质量元数据（content_type / quality / retrieval_method / retrieved_at + `source_quality_of` 规则） | `research/evidence.py` + `nodes.py`（Source 构造处） | 确定性 |
| 16 | RAG 检索去重 + rerank 状态透传（按 chunk_id/url 去重；`rerank_status` SUCCESS/FALLBACK） | `research/search.py`（`hybrid_search` / `rag_search`）+ `rag/service.py`：`rerank` | 确定性 |
| 17 | RAG 回退状态记录（rerank 失败时记 FALLBACK 而非静默） | `rag/service.py:rerank` + `nodes.py`（Evidence/Source 元数据存 `rerank_status`） | 确定性 |
| 18 | Deep Research LLM 可靠性包装（retry → 升级模型 → 熔断 → 死信 → 降级） | `research/reliability.py`：`reliable_llm_call` / `ReliableLLM` + `nodes.py` 各 LLM 调用点 | 复用 `rag/reliability` 的 CB/DLQ |
| 19 | Research Ledger 落盘（逐事件写 `logs/<thread_id>.jsonl`） | `core/logger.py`：`log_research_event` / `flush` + `tools/research_tool.py:deep_research` | 确定性（程序落盘） |
| 20 | Trace 闭环补全（报告末尾程序生成「可追溯性说明」） | `nodes.py:_render_trace_summary` + `_split_sections` + `compiler_node` | 确定性 |
| 21 | Skill 工具命名空间（`skill_<归一化名>` + `reserved_names` 去重） | `core/skill_loader.py` + `core/agent.py` | 确定性 |
| 22 | Compiler 终态闭环（固定七段式 + 程序追加参考来源与可追溯说明） | `nodes.py:compiler_node` + `REPORT_SECTIONS` | 程序规定结构，LLM 填内容 |

### 新增文件
- `clawgent/core/research/semantic.py`：`ClaimEvidenceSemanticVerifier`（LLM 判 quote→claim 语义关系，失败/缺失兜底 SUPPORTS）。
- `clawgent/core/research/reliability.py`：`reliable_llm_call` + `ReliableLLM`，复用 `rag/reliability` 的 `CircuitBreaker` / `DeadLetterQueue`。

---

## 2. 完整运行链路（run-chain）

```
deep_research(query, context, max_revisions, thread_id)       # tools/research_tool.py
  └─ graph.ainvoke(initial_state, recursion_limit=100)
       ├─ planner_node      LLM 出 ResearchPlan(DAG)；normalize_task_dependencies 补依赖
       ├─ scheduler_node    DAG 按依赖挑 READY 任务 → Command(goto=[Send(researcher,...)*])
       ├─ researcher_node*  (并发) 检索→登记 Source→抽 Evidence→原文校验→抽 Claim→语义关系
       │     ├─ hybrid_search（学术/联网/RAG 三路，chunk_id/url 去重，rerank_status 透传）
       │     ├─ EvidenceVerifier（摘要来源→UNVERIFIED；否则按原文命中判定 VERIFIED/PARTIAL）
       │     └─ ClaimEvidenceSemanticVerifier（LLM 判语义关系；失败兜底 SUPPORTS）
       ├─ aggregator_node   合并去重 → ClaimGraph.recompute_statuses(校验 + 语义 + CONTRADICTS)
       ├─ review_node       build_review(确定性冲突/范围) + LLM 结构化 issues
       ├─ repair_node       仅 add_task / reopen_task；repaired_issue_ids 去重 + 预算上限
       ├─ judge_node        进展日志 + 停滞检测 + 原始/修订区分 → Command(goto scheduler|compiler)
       └─ compiler_node     七段式成文 → CitationVerifier → CitationBinder → UnsupportedClaimDetector
                              → 程序追加「参考来源」「可追溯性说明」「本报告局限」
  └─ 收尾：把 ledger_events 逐条 log_research_event(thread_id, ...) 落盘 + audit_logger.flush()
```

关键状态传递：`researcher_node` 返回 `ResearchPacket`（来源/证据/声明），**不直接改全局 DAG**；`review_node` **不改 DAG**；`judge_node` **不依赖模型打分**，只看 DAG 状态与统计；`compiler_node` **只用已有 Claim**，且 UNSUPPORTED/CONTRADICTED 声明被程序剔除或预警。

---

## 3. 数据对象关系（data-object relationships）

- **TaskDAG ⇄ ResearchTask**：`TaskDAG.tasks: dict[task_id, ResearchTask]`；`transition_task` 统一驱动 `PENDING→READY→RUNNING→COMPLETED/FAILED/BLOCKED→REOPENED`。
- **Source ⇄ Evidence**：`SourceRegistry.by_id[source_id]`；`Evidence.source_id` 外键；`EvidenceVerifier.check` 通过 `registry.has(source_id)` 校验来源已登记，并按 `Source.content_type==SEARCH_SNIPPET` 强制 `UNVERIFIED`。
- **Evidence ⇄ Claim**：`Claim.evidence_ids` 一对多；`ClaimRelation(source_id=evidence_id, target_id=claim_id, semantic_relation)` 记录语义关系。
- **ClaimGraph.recompute_statuses**：输入 `{evidence_id: {verification_status}}` 与 `semantic_index={(evidence_id, claim_id): SEM_*}`，输出 `Claim.status`（SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED / CONTRADICTED）。**结论完全由程序推导，LLM 不参与 verdict。**
- **ResearchLedger ⇄ state**：`ledger_events` 经 reducer `append_list` 累积；`deep_research` 收尾时逐条写盘。
- **Trace ⇄ Claim/Evidence/Source/Task**：`build_trace(claims, evidences, sources, tasks)` 回溯「声明→证据→来源→任务→检索式」链；`_render_trace_summary` 渲染进报告。
- **SkillLoader ⇄ Agent**：`load_dynamic_skills(reserved_names=[BUILTIN_TOOLS 名])` → 工具名统一 `skill_<归一化>`，避开内置工具名冲突。

---

## 4. LLM / 确定性边界

**由 LLM 产出的（原始材料，非结论）：**
- planner：ResearchPlan 文本（任务拆分、依赖初稿）。
- researcher：声明文本、原文引用抽取、语义关系原始标签（`ClaimEvidenceSemanticVerifier`）。
- review：问题清单原始描述（类型/动作/所需证据）。
- compiler：七段式报告正文（结构与引用标记由程序约束）。

**由程序确定性推导/校验的（结论不可由 LLM 决定）：**
- 证据是否被原文「校验通过」（`EvidenceVerifier`，字符串相似度 + 来源登记）。
- Claim 是否被「支撑」（状态机 `recompute_statuses`）。
- 引用是否「有效」（`CitationVerifier`，UNKNOWN_REF / UNVERIFIED_EVIDENCE 阻断重跑）。
- 声明是否「可作为事实」（UnsupportedClaimDetector 剔除 UNSUPPORTED/CONTRADICTED）。
- 是否「继续调研」（judge_node：READY 任务 + 停滞计数 + 原始/修订区分）。
- 是否「有冲突/范围不一致」（ConflictDetector）。
- rerank 是否用远程重排器（`rerank_status` 落库，不靠模型自述）。
- 来源质量档位（`source_quality_of` 规则：学术全文=high，web 全文=medium，web 摘要=low）。

**可靠性边界（research/reliability.py）：**
- 所有研究链路 LLM 调用经 `reliable_llm_call`：重试 → 升级模型兜底 → 熔断跳过重试 → 死信入队（`DeadLetterQueue`）→ 返回 `fallback`。
- `ReliableLLM` 把 `reliable_llm_call` 适配为 langchain `invoke` 接口，语义校验器复用同一套熔断/死信。
- 熔断阈值沿用 `rag/reliability`：连续失败 3 次 OPEN，60s 后半开探测。

---

## 5. 已知局限

- **本轮未写测试**：按规范要求，本轮聚焦逻辑闭环补齐，未新增自动化测试；各模块已用独立加载（stub 掉 langchain/langgraph）做关键路径验证，但非回归套件。
- **语义关系依赖 LLM**：`ClaimEvidenceSemanticVerifier` 在无 LLM / 调用异常时整组兜底为 `SUPPORTS`，可能把本应 CONTRADICTS 的关系误判为支持；最终由 `recompute_statuses` 的 CONTRADICTS 关系与 ConflictDetector 双重兜底，但语义关系本身精度受 LLM 限制。
- **摘要来源证据上限**：仅返回摘要的 web 来源，其证据恒为 `UNVERIFIED`，无法升级为 VERIFIED（设计使然，防幻觉引用）。
- **LangGraph 运行态未在本机验证**：managed 运行时未安装 langgraph，端到端 `graph.ainvoke` 未实跑；仅对各节点函数做去依赖单测。真实运行需目标环境装齐 langgraph/langchain 及 `TAVILY_API_KEY` / `RAG_LLM_*` 配置。
- **rerank 远程依赖**：`rerank_status=FALLBACK` 时退回向量分，排序质量下降，已记录但报告未单独标注。
- **Skill 命名空间改造为前缀式**：所有动态技能工具名统一加 `skill_` 前缀，调用侧（如提示词里的工具名引用）需同步；既有按裸名引用的地方需复核。
- **Ledger 落盘依赖守护线程**：`log_research_event` 经内存队列异步写盘，`audit_logger.flush()` 仅 `queue.join()` 等待，不保证进程硬崩溃前的最后几条必达。

---

## 6. Plan Verification & Repair（追加，`research/plan_validation.py`）

在 Planner 生成 DAG 之后、Scheduler 之前补一层「计划验证」，并在研究执行后补「证据覆盖验证」。不改 Scheduler / Researcher / 检索 / Evidence 结构，全部复用现有 ResearchPlan / ResearchTask / TaskDAG。

### 流程

```
执行前：Planner → PlanSchemaValidator → PlanCoverageValidator → SemanticPlanCritic
              → LocalRepair → ValidatedPlan → Scheduler
执行后：Evidence/Claim → EvidenceCoverageValidator → PASS/GAP → LocalRepair Task
```

### 新增组件（`research/plan_validation.py`）

| 组件 | 类型 | 职责 |
|---|---|---|
| `PlanSchemaValidator` | 确定性 | 8 项结构检查：task_id 唯一、依赖存在、无自依赖、无环、task_type 合法、必要字段非空、root 可达、依赖「能提供输入」 |
| `PlanCoverageValidator` | 确定性 | `required_capabilities - covered_capabilities` 集合覆盖，缺失输出 `MISSING_CAPABILITY`（带 add_task 建议） |
| `TaskSuccessCriteriaEvaluator` | 确定性 | 按结构化 criteria 判 `PASS/PARTIAL/FAIL`（字段出现 / 来源类型 / 证据条数） |
| `EvidenceCoverageValidator` | 确定性 | 执行后校验 Evidence/Claim 是否覆盖 plan 关键维度，缺失输出 `COVERAGE_GAP` |
| `SemanticPlanCritic` | LLM | 对 (Question, Plan, DAG) 语义检查，只给局部修改建议，不重建 DAG；异常静默降级 |
| `validate_plan` / `_apply_local_repair` | 编排 | Schema→Coverage→Critic→LocalRepair 串联；local repair 只 add_task/remove_task，受 `max_repairs` 上限 |

### 字段扩展（向后兼容，全部带默认值）

- `ResearchTask.capabilities: list[str]`、`ResearchTask.criteria: dict`（结构化完成条件）。
- `ResearchPlan.required_dimensions / required_capabilities: list[str]`（计划必须覆盖的维度/能力）。
- `TaskDAG.reachable_from_roots()` / `roots()`（root 可达性检查用）。
- `state.plan_validation` / `state.coverage_validation`（验证结果快照，可追溯）。

### 兼容性

- 所有新字段带默认值，旧 dict 走 `from_dict` 白名单过滤，不丢旧字段、不报错。
- `repair_node` 动作集合扩展 `remove_task`（计划验证删除冗余/不可达任务），并保留原 add_task/reopen_task 逻辑。
- `planner_node` 内联调用验证链（不新增 LangGraph 节点，图拓扑不变）。
- `review_node` 内联 `EvidenceCoverageValidator`，把 COVERAGE_GAP 并入 issues 交给 repair。

### 已知局限

- `required_capabilities` 目前需 Planner 声明或由调用方注入；Planner 尚未从 query 自动推导维度（可后续加规则）。
- `SemanticPlanCritic` 语义判断依赖 LLM，异常时静默降级为空（不影响流程）。
- 未新增端到端测试（按边界要求）；各组件用 importlib+stub 做了确定性单测。
