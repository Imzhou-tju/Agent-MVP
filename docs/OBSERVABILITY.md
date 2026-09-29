# 监控、审计与可视化（Observability）

为一次 Research Run 提供从 `Run → Node → Task → Retrieval → Evidence → Claim → Citation → Report` 的追踪能力，可观察耗时、失败、Retry、Fallback、Repair、Judge 决策等关键行为。

本能力只增可观测性，不改 Planner/Scheduler/Researcher/Review/Repair/Judge 的决策规则。

---

## 1. 审计事件结构

所有审计事件统一结构（`clawgent/core/audit.py` 的 `AuditEvent`）：

```text
event_id        run_id        thread_id      timestamp
event_type      node          task_id        parent_event_id
status          duration_ms   metadata       error
```

约束：

- `metadata` 只放结构化摘要（数量、状态、id、短原因），不放完整 Prompt / LLM 返回 / 正文；超长字符串按 500 字符截断。
- `error` 只放错误类型与短 message（200 字符），不落堆栈。
- `run_id` 一次 Run 唯一；`task_id` 与任务有关时必须记录；`parent_event_id` 用于父子关系（`span` 自动填）。

## 2. 事件类型

| 分类 | 事件 |
|------|------|
| 运行生命周期 | `run_start` / `run_end` |
| 节点生命周期 | `node_start` / `node_end` |
| 任务生命周期 | `task_created` / `task_ready` / `task_started` / `task_completed` / `task_failed` / `task_reopened` / `task_skipped` |
| 调用与可靠性 | `llm_call` / `tool_call` / `retry` / `fallback` / `circuit_breaker` |
| 检索 | `rag_iteration` / `rag_retrieval` / `rerank` / `multi_hop_iteration` |
| 调研决策 | `plan_gate` / `aggregation` / `review_issue` / `repair` / `judge_decision` |
| 溯源 | `evidence_registered` / `citation_verified` |

## 3. 埋点位置

| 组件 | 埋点内容 | 位置 |
|------|---------|------|
| Planner | `run_start`（run_id 在此确定）、`plan_gate`（SOP/决策/repair 次数）、每个 `task_created` | `research/nodes.py` planner_node |
| Scheduler | 调度轮次 + 每个 READY 任务单独 `task_ready`/`task_started` | scheduler_node |
| Researcher | 每个任务 `task_started`→`task_completed`/`task_failed`、`evidence_registered` | researcher_node |
| Aggregator | `aggregation`（source/evidence 数、claim 支撑状态分布） | aggregator_node |
| Review | `review_issue`（issue 数与分类） | review_node |
| Repair | `repair`（新增/删除/重开任务数、原因）、`task_reopened` | repair_node |
| Judge | `judge_decision`（decision/unfinished/stagnant/reason） | judge_node |
| Compiler | `citation_verified`（引用校验结果）、`run_end` | compiler_node |
| RAG 单轮 | `rag_retrieval`（各路召回数量）、`rag_iteration`（sufficiency/query_rewrite）、`rerank`（fallback 标记） | `rag/service.py` |
| 多跳 | `multi_hop_iteration`（iteration/新增证据数/gap/action/stop_reason） | `rag/retrieval_decision.py` 的 `on_iteration` 回调 |
| 可靠性 | `retry` / `fallback` / `circuit_breaker` | `rag/reliability.py` |
| 三通道检索 | `rag_retrieval`（学术/联网/本地 RAG 数量） | `research/search.py` hybrid_search |

节点级 `node_start`/`node_end` 由 `@audit_node("节点名")` 装饰器统一注入，节点抛异常记 FAILED 后原样抛出，由 LangGraph `RetryPolicy` 决定重试。

## 4. Run Summary

Run 完成后由 `audit.summarize()` 从事件流聚合（纯函数，不维护第二套业务状态）：

```text
total_duration  final_status  verdict  termination_reason
total_tasks  completed/failed/reopened/skipped_tasks
total_llm_calls  total_tool_calls  total_retries  total_fallbacks
source_count  evidence_count
supported/partial/unsupported/contradicted_claims
review_issue_count  repair_rounds  rag_iterations  multi_hop_iterations
```

异常终止（无 `run_end`）时也会给出尽可能完整的统计，`final_status` 按「是否有 node FAILED」推导。

## 5. Dashboard

本地只读 Dashboard，不改 Research State，不改 JSONL。启动：

```bash
python -m entry.dashboard                # 默认 http://127.0.0.1:8765
python -m entry.dashboard --port 9000    # 指定端口
```

页面（`clawgent/dashboard/static/index.html`，单页无构建）：

信息层级自上而下依次收起：Run 头 → 核心指标 → 阶段流水线 → 下钻页签。

- **Run 列表（左栏）**：Run ID / 调研主题 / 时间 / 耗时 / 任务数 / 证据数 / 最终状态，支持状态筛选与关键词搜索。
- **Run 头**：调研主题、Run ID、线程、起止时间、Judge 裁决、终止原因、声明条数。
- **核心指标**：总耗时、任务闭环、证据/来源、声明支撑率（无声明数据时退回证据核验率）、多跳轮次、可靠性事件。
- **阶段流水线**：planner / scheduler / researcher / aggregator / review / repair / judge / compiler 八段，
  状态与耗时全部由事件推导（无事件的阶段显示「未执行」），不写死。
- **页签 1 · 任务拓扑**：从 `task_created` 还原依赖，按最长路径分层用 SVG 绘制；节点标
  PENDING/COMPLETED/FAILED/REOPENED/SKIPPED，点击展开该任务的工单详情与事件序列。
- **页签 2 · 瀑布时间线**：上方为各节点耗时排名，下方为事件瀑布（按时刻定位，有 duration 画条、
  无 duration 画刻度）。当事件时间戳集中在 50ms 内（例如同一批写入）时，横轴改为按事件顺序等距展开，
  并在标题标注实际跨度。
- **页签 3 · 检索迭代**：多跳每轮「对应任务 | 盲区探测 | 改写查询 | 新增证据 | 收敛判定 | 执行动作 | 停止原因」，
  以及单轮三通道召回数量。
- **页签 4 · 证据溯源**：`evidence_registered` 事件的 evidence_id / source_id / source_type /
  verification_status / claim_id / task_id。
- **页签 5 · 事件流**：全量事件表，支持按节点、事件类型、关键词（含 metadata 与 error）筛选。

视觉与可读性约定：颜色只承载状态语义（绿=完成、蓝=进行中、琥珀=需关注、红=失败、灰=未执行/未知），
状态标签同时带符号，不依赖颜色单独表意；无 `duration_ms` 数据时显示「—」并说明原因，不填造数值。

HTTP API：`/api/runs`（列表）、`/api/runs/<run_id>`（详情）、`/api/runs/<run_id>/dag`（拓扑图）、
`/api/refresh`（增量刷新）。

**离线预览**：直接用浏览器打开该 HTML 时，`fetch` 因 `file://` 协议失败，页面会回退到内置示例数据
并在顶部显示提示条；启动服务后刷新即自动切换为真实审计数据。

## 6. 数据存储与边界

- **JSONL 是唯一审计数据源**，落在 `logs/<thread_id>.jsonl`，与现有行为审计日志同目录、同格式（`event` 字段兼容旧读取器）。
- **SQLite 只做查询索引**（`logs/audit_index.sqlite`，路径 `AUDIT_INDEX_PATH`），增量扫描、删除可重建，不是第二套业务状态源。
- 写入异步：复用 `JSONLEventLogger` 的内存队列 + 后台写盘线程，主链路只做一次 `queue.put()`。
- 单次事件写失败只打 warning，不抛异常、不阻断调研。
- Prompt / Response 不落盘；Evidence 原文不重复保存；Dashboard 查询异常返回 500，不影响调研。
- 索引器按文件偏移增量读，不整包加载全部历史日志到内存。

## 7. 已知限制

- 本环境未安装项目依赖（`python-dotenv`/`langgraph` 等），`python -m entry.dashboard` 与演示脚本需先 `pip install -r requirements.txt`。
- `source_texts`（来源正文缓存）不落审计事件，Dashboard 只显示 evidence_id/source_id 等摘要字段，不显示原文。
- 证据溯源目前从 `evidence_registered` 事件组装，展示 evidence/source/claim 的 id 与状态；完整的 Claim→Evidence→Source 反向链以 `store.load_claim_chain`（业务库）为准，Dashboard 不读业务库。
- `llm_call` / `tool_call` 事件类型已定义但未在所有调用点埋点（RAG 侧判定调用走 `retry/fallback` 事件，调研侧走节点级事件），后续按需补齐。

## 8. 演示

生成一次完整事件链（不调模型、不联网）：

```bash
python scripts/demo_audit_run.py
python -m entry.dashboard
```

## 9. 测试

```bash
python -m unittest tests.test_research_audit tests.test_dashboard -v
```

覆盖：事件字段收敛、Summary 聚合、异常终止、写入失败隔离、索引构建、run 列表过滤、task DAG 还原、证据溯源组装、增量刷新。
