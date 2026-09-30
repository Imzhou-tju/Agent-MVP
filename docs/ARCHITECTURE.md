# ClawAgent 核心系统架构说明

ClawAgent 是一个基于 LangGraph 构建的**面向科研场景的可溯源智能体系统**。系统通过结构化状态机、数据范式解耦与异步审计机制，旨在缓解大语言模型在复杂多跳推演与长线调研过程中易出现的幻觉发散、多跳遗忘、并发写冲突与循环死锁等问题。

---

## 1. 核心系统拓扑 (System Topology)

系统整体采用**“分层流转、中枢决策、旁路监听、闭环回路”**的拓扑架构，对应可视化拓扑见 [docs/architecture_diagram.html](architecture_diagram.html)。核心逻辑拓扑如下：

```mermaid
flowchart TD
    %% 输入层
    subgraph LayerInput[一、输入层]
        HB[Heartbeat / 定时调度] --> GW[Gateway 上下文网关]
        UI[科研提问 / 任务输入] --> GW
    end

    %% 核心运行时中枢
    subgraph LayerCore[二、运行时核心中枢]
        direction TB
        subgraph Mem[记忆层 (Memory)]
            LRU[LRU 上下文修剪<br>40轮修剪至10轮]
            LEDGER[Entity Ledger<br>科研硬约束/实体账本]
        end

        subgraph Decision[智能决策层 (Decision)]
            REACT[Agent Loop 核心主循环<br>ReAct 推理与意图分类]
            ROUTER[工具调度与条件路由<br>tools_condition]
        end

        Mem <-->|双向存取与约束注入| Decision
    end

    %% 安全控制层
    subgraph LayerSec[三、安全与控制层]
        SANDBOX[沙盒路径越权拦截<br>OFFICE_DIR 隔离]
        PLANGATE[PlanGate 校验<br>I/O契约与推演]
        BREAKER[LLM 熔断机制<br>重试降级]
    end

    %% 工具与子图执行层
    subgraph LayerTools[四、工具与子图执行层]
        DAG[多智能体调研图 Research DAG<br>Planner ➔ Scheduler ➔ 并发 Researcher<br>Review ↔ Repair ↔ Judge]
        RAG[RAG 混合检索引擎<br>双路召回 + Rerank ➔ 残差多跳推理]
        SKILLS[基础 Skills 与 MCP 学术源]
    end

    %% 透明监控层
    subgraph LayerMonitor[五、透明监控与可观测层]
        AUDIT[@audit_node 异步事件埋点]
        EE[EventEmitter 状态捕获]
        LOG[JSONL 全链路持久化日志]
        DASH[Web Dashboard 实时监控]
    end

    %% 输出层
    subgraph LayerOutput[六、输出层]
        REPORT[证据溯源报告<br>Compiler 外键校验与角标渲染]
        TERMINAL[交互终端与监控看板]
    end

    %% 数据底座支撑关系 (虚线)
    subgraph LayerData[七、底层数据支撑 (1:N 证据溯源底座)]
        DB[(Source 1:N Evidence 1:N Claim)]
    end

    %% 关键拓扑连接
    GW -->|初始分发| Mem
    Decision -->|调用触发| LayerSec
    LayerSec -->|校验放行| LayerTools
    LayerTools -->|执行结果回路| Decision
    LayerTools -.->|长任务调度| HB
    LayerTools -.->|读写与外键绑定| DB

    Mem -->|状态变更| LayerMonitor
    Decision -->|决策流转| LayerMonitor
    
    Decision -->|成文输出| REPORT
    LayerMonitor -->|可观测同步| TERMINAL
```

---

## 2. 核心架构设计与工程决策

### 2.1 运行时基座与透明可观测性 (Runtime & Observability)

系统的基座负责保证长周期交互的稳定性、状态连续性以及执行过程的完全透明：

1. **上下文分级管理与修剪 (LRU Context Trimming)**：
   - **机制**：以前台交互轮次为单位进行监控，当会话超过设定阈值（如 40 轮）时，强制启动修剪逻辑，仅保留最近的 10 轮原始交互。
   - **设计考量**：避免无限追加原始消息导致上下文超出模型有效注意力窗口（Lost in the Middle）与推理成本线性膨胀。
2. **实体账本 (Entity Ledger)**：
   - **机制**：在执行上下文修剪时，由抽取模型将前序对话中的科研硬约束、核心实体和已决/未决问题结构化提炼为 Pydantic 数据模型。
   - **设计考量**：摒弃传统仅依赖向量检索（容易因语义漂移遗漏硬约束）的做法，将提炼出的高密度 JSON 账本直接锚定于 System Prompt，以确定性注入维持长效记忆。
3. **旁路异步审计与持久化 (Observability Pipeline)**：
   - **机制**：采用 `@audit_node` 装饰器对图节点进行无侵入切面埋点，由内部 `EventEmitter` 将节点转态、流转耗时、Token 开销等事件压入内存队列，后台由独立线程按 `thread_id` 写入 `logs/<thread_id>.jsonl`。
   - **设计考量**：将审计与监控逻辑完全从主推理流剥离，杜绝日志写入 IO 阻塞核心推理；同时为 Web Dashboard 提供确定性的事件重放与调用复现能力。
4. **交互式多智能体 DAG 拓扑可视化 (DAG Topology Visualization)**：
   - **机制**：基于 `AuditIndexer.task_dag_graph` 与 `/api/runs/<run_id>/dag` 专用接口，在 Web 看板中实现基于有向无环图算法的分层（Layered Ranking）拓扑网络。前端使用平滑 SVG 贝塞尔曲线连接依赖节点，实时映射任务状态机跃迁。
   - **设计考量**：彻底替代平铺卡片列表，直观暴露多智能体执行分支、菱形依赖汇聚，并对 Review/Repair 触发的 `REOPENED` 级联重置节点进行高亮警示，支持点击节点下钻查看依赖、耗时与类型元数据。

---

### 2.2 多智能体调研流水线 (Multi-Agent Research DAG)

针对需要大规模信息收集与交叉论证的复杂科研任务，系统采用重型状态机编排代替脆弱的单体 Agent 循环：

1. **结构化任务分解与调度 (Planner & Scheduler)**：
   - **机制**：`Planner` 依据预设研究 SOP（如 FACT、COMPARISON、SURVEY）将复杂主题拆解为带有依赖关系的 `ResearchTask` 有向无环图（DAG）；`Scheduler` 纯基于逻辑依赖推进，使用 LangGraph `Send` API 向满足依赖就绪条件的 `Researcher` 发起并发派发。
2. **并发安全与状态合并 (ID-based Reducer)**：
   - **机制**：在并发 `fan-in` 汇聚至 `Aggregator` 时，全局状态弃用追加式 `List`，采用自定义 `Dict` 配合 `operator.add` Reducer。`Source` 按 URL Hash 覆盖去重，`Evidence` 严格按全局唯一 UUID 追加。
   - **设计考量**：规避多子智能体并发写入全局状态时造成的脏写、覆盖与乱序现象。
3. **闭环质检与级联回退 (Review ↔ Repair ↔ Judge)**：
   - **审查 (Review)**：作为质检网关，输出包含事实冲突、证据缺失、范围不匹配的结构化 `issues` 清单。
   - **修补 (Repair)**：依据问题清单局部增删任务（派发 `{parent}-R{round_no}` 局部修复任务），并通过向下遍历 DAG，将其影响的下游节点状态级联重置为 `REOPENED`。
   - **终审与强制熔断 (Judge)**：裁决继续调度、二次修补还是直接交付。引入 `stagnant_rounds >= 2` 的停滞阈值判定，一旦陷入无效修补死循环，强制行使一票否决权（`COMPILE_WITH_LIMITATIONS`），合成附带局限性声明的研报，确保系统不挂起。

---

### 2.3 混合检索与残差多跳推理 (Hybrid RAG & Iterative Reasoning)

针对单一关键词检索无法触达的深层与交叉文档问题，提供弹性的检索机制：

1. **混合多路召回与二次精排 (Hybrid Search & Rerank)**：
   - **机制**：底层建立 Dense（向量语义检索）与 Sparse（BM25 词法检索）双路召回，通过 RRF（Reciprocal Rank Fusion, $k=60$）算法计算综合得分，并在候选截断后调用远端 `bge-reranker-v2-m3` 进行交叉编码精排。
   - **设计考量**：兼顾专有名词（如模型代号、特定公式）的精确匹配与宽泛语义的模糊关联。
2. **残差状态累加机制 (Cumulative Context / Stateful Memory)**：
   - **机制**：在微观的多跳循环（Retrieve ➔ Reason）中，系统将第 1 跳至第 $N-1$ 跳已抽取验证的高纯度事实片段（Factoids）保留在全局证据池中。进入第 $N$ 跳推演时，将全部历史残差事实联合输入模型。
   - **设计考量**：借鉴残差网络（Skip Connection）思想，避免多跳推理退化为仅依赖上一跳的马尔可夫链，防止跨步骤信息遗忘（Context Fragmentation）。
3. **缺口驱动与步进控制 (Gap-driven Decision)**：
   - **机制**：每一跳检索后，模型比对原始目标与当前累加事实池。若信息充分（`is_sufficient=True`）则退出；若存在知识缺口，则提取具体 `Gap` 并生成针对性 `next_query`；若上一跳查无增益（`information_gain=False`），则原地切换检索策略（补检索），受全局 `RAG_MAX_ITERS` 轮次硬上限约束。

---

### 2.4 证据溯源链 1:N 架构 (底层数据底座)

作为多智能体调研图与 RAG 引擎的统一数据底座，从数据结构层面约束结论生成，提升文献引用的可靠性：

1. **分层解耦的 Pydantic 数据范式**：
   - **Source 层 (`ResearchSource`)**：唯一标识原始网页、文档或论文，记录 `source_id`、URL、标题与元数据。
   - **Evidence 层 (`ResearchEvidence`)**：记录由模型抽取的关键客观切片原文（Quote）及去噪事实，通过外键 `source_id` 强绑定归属来源，并附带验证状态（`VERIFIED` / `UNVERIFIED`）。
   - **Claim 层 (`ResearchClaim`)**：记录综合推导得出的学术结论声明，通过外键列表 `support_evidences: List[UUID]` 显式绑定一条或多条支撑证据。
2. **架构收益与抗幻觉机制**：
   - **避免上下文膨胀**：同一网页来源被多次引用时，无需重复传递整页长文本，仅流转轻量级引用外键。
   - **程序化确定性校验 (Compiler Rendering)**：在报告合成阶段，由 `Compiler` 节点（确定性 Python 脚本逻辑，非模型直接生成）校验 Claim 所绑定的 Evidence 与 Source 是否有效存在。校验通过后，自动在 Markdown 文本中渲染对应的引文角标（如 `[1]`, `[2]`）并在文末生成标准参考文献列表，在架构层面阻断虚假引用的产生。
