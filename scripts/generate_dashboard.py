# -*- coding: utf-8 -*-
"""
生成高质量、极富流程感与科技感的多智能体科研可观测 Dashboard 预览页面。
支持宏观流水线阶梯 (Macro Pipeline Stepper) 与微观 DAG 动态流动拓扑图 (Micro Flowing DAG)。
"""

import json
import os

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

active_events = load_jsonl("logs/trace_active.jsonl")
sim_events = load_jsonl("logs/trace_simulation.jsonl")

runs_data = {
    "run-active-9002": {
        "run_id": "run-active-9002",
        "title": "语音大模型前沿架构对比与消融实验调研",
        "query": "语音大模型前沿架构对比与消融实验综合调研",
        "status": "SUCCESS",
        "duration": "18.2s",
        "tasks_summary": "4 / 4 任务闭环 (菱形并发全胜)",
        "evidence_count": "7 篇切片",
        "citation_status": "100% 支撑 (3条声明已程序化核验)",
        "pipeline_stages": [
            {
                "id": "stage-1",
                "name": "1. SOP 规划",
                "role": "Planner",
                "desc": "拆解 4 节点菱形并发 DAG 拓扑",
                "status": "COMPLETED",
                "time": "1.2s",
                "icon": "📋"
            },
            {
                "id": "stage-2",
                "name": "2. 并发调度",
                "role": "Scheduler",
                "desc": "Send API 异步派发 Worker-1 & Worker-2",
                "status": "COMPLETED",
                "time": "0.4s",
                "icon": "⚡"
            },
            {
                "id": "stage-3",
                "name": "3. 证据多跳检索",
                "role": "Researcher",
                "desc": "双路 RRF + Rerank，双跳填补盲区",
                "status": "COMPLETED",
                "time": "8.5s",
                "icon": "🔍"
            },
            {
                "id": "stage-4",
                "name": "4. 状态归约推导",
                "role": "Aggregator",
                "desc": "ID Reducer 去重，推导 3 条 Claim",
                "status": "COMPLETED",
                "time": "0.7s",
                "icon": "🧩"
            },
            {
                "id": "stage-5",
                "name": "5. 红队质检",
                "role": "Reviewer",
                "desc": "三方交叉核验，无漏洞无幻觉",
                "status": "COMPLETED",
                "time": "1.2s",
                "icon": "🛡️"
            },
            {
                "id": "stage-6",
                "name": "6. 引文成文交付",
                "role": "Compiler",
                "desc": "外键比对对账，输出带引文研报",
                "status": "COMPLETED",
                "time": "1.7s",
                "icon": "📄"
            }
        ],
        "dag_tasks": [
            {
                "task_id": "t1-FACT",
                "name": "基础事实与架构共识",
                "stage_tag": "Step 1: 基础基石",
                "task_type": "FACT",
                "status": "COMPLETED",
                "duration_ms": 2200,
                "priority": 1,
                "worker": "researcher_worker_0",
                "produced": "产出 3 篇核心文献切片",
                "summary": "调研 Conformer 融合局部卷积与全局时序特征的基础原理",
                "dependencies": []
            },
            {
                "task_id": "t2-BENCH",
                "name": "VoxCeleb 基准表现评测",
                "stage_tag": "Step 2A: 并发分支",
                "task_type": "COMPARISON",
                "status": "COMPLETED",
                "duration_ms": 6500,
                "priority": 2,
                "worker": "researcher_worker_1",
                "produced": "产出 2 篇基准实验数据",
                "summary": "微调大底模取得 EER 1.12% 前沿性能指标，多跳补足训练超参",
                "dependencies": ["t1-FACT"]
            },
            {
                "task_id": "t3-ABLATION",
                "name": "多头注意力层消融实验",
                "stage_tag": "Step 2B: 并发分支",
                "task_type": "MECHANISM",
                "status": "COMPLETED",
                "duration_ms": 7400,
                "priority": 2,
                "worker": "researcher_worker_2",
                "produced": "产出 2 篇消融对比切片",
                "summary": "移除 MHSA 导致长程建模失效，EER 指标相对恶化 28%",
                "dependencies": ["t1-FACT"]
            },
            {
                "task_id": "t4-SYNTHESIS",
                "name": "综合推导与结论收敛",
                "stage_tag": "Step 3: 汇聚总结",
                "task_type": "SURVEY",
                "status": "COMPLETED",
                "duration_ms": 2200,
                "priority": 3,
                "worker": "synthesizer_agent",
                "produced": "推导 3 条无懈可击声明",
                "summary": "综合基准测试与消融实验，输出全面调研结论，准备交付",
                "dependencies": ["t2-BENCH", "t3-ABLATION"]
            }
        ],
        "rag_hops": [
            {
                "iteration": 1,
                "task_id": "t2-BENCH",
                "gap": "缺失微调协议 (Fine-tuning protocol) 详细设置与超参数",
                "query": "SpeechFoundation fine-tuning protocol learning rate batch size",
                "hits": "10 篇粗排 / Top1 相关度: 0.93",
                "status": "INSUFFICIENT",
                "action": "TRIGGER_HOP_2"
            },
            {
                "iteration": 2,
                "task_id": "t2-BENCH",
                "gap": "无 (已补足基准 EER 1.12% 数据与训练超参，形成闭环)",
                "query": "SpeechFoundation VoxCeleb benchmark evaluation protocol",
                "hits": "8 篇粗排 / Top1 相关度: 0.91",
                "status": "SUFFICIENT",
                "action": "STOP"
            }
        ],
        "events": active_events,
        "report_content": """<h3>语音大模型前沿架构对比与消融实验综合调研报告</h3>
<p>1. <strong>架构基础</strong>：现代语音大模型普遍采用 Conformer 或混合时延卷积结构以融合局部与全局时序特征 <b style='color:var(--accent); cursor:pointer;' onclick='focusEvidence(\"evi-fact-01\")'>[1]</b>。</p>
<p>2. <strong>基准性能</strong>：在 VoxCeleb 说话人验证排行榜中，采用预训练大底模微调的架构取得了 EER 1.12% 的前沿性能 <b style='color:var(--accent); cursor:pointer;' onclick='focusEvidence(\"evi-bench-01\")'>[2]</b>。</p>
<p>3. <strong>消融分析</strong>：多头自注意力机制（MHSA）对长程上下文建模起决定性作用，消融测试表明移除自注意力层会导致 EER 恶化约 28% <b style='color:var(--accent); cursor:pointer;' onclick='focusEvidence(\"evi-ablation-01\")'>[3]</b>。</p>
<hr style='margin:16px 0; border:none; border-top:1px solid var(--border);'>
<h4 style='font-size:13px; color:var(--muted); margin-bottom:8px;'>参考文献与溯源支撑 (Verified References):</h4>
<div style='font-size:12px; color:#475569; display:flex; flex-direction:column; gap:6px;'>
  <div><strong>[1]</strong> Gulati et al. (2020). <em>Conformer: Convolution-augmented Transformer for Speech Recognition</em>. (关联证据: <code>evi-fact-01</code> · 状态: <code>SUPPORTED</code>)</div>
  <div><strong>[2]</strong> VoxCeleb Leaderboard (2023). <em>Fine-tuned SpeechFoundation Benchmarks</em>. (关联证据: <code>evi-bench-01</code> · 状态: <code>SUPPORTED</code>)</div>
  <div><strong>[3]</strong> Speech Transformer Ablation Studies (2022). <em>Multi-Head Attention Impact in ASR</em>. (关联证据: <code>evi-ablation-01</code> · 状态: <code>SUPPORTED</code>)</div>
</div>"""
    },
    "run-sim-8891": {
        "run_id": "run-sim-8891",
        "title": "对比 x-vector 与 ECAPA-TDNN 架构及性能",
        "query": "对比 x-vector 与 ECAPA-TDNN 的模型架构及识别性能差异",
        "status": "SUCCESS",
        "duration": "15.6s",
        "tasks_summary": "3 任务完成 (经历 1 轮红队发难与局部重开自愈)",
        "evidence_count": "5 篇切片",
        "citation_status": "100% 支撑 (2条声明核验通过)",
        "pipeline_stages": [
            {
                "id": "stage-1",
                "name": "1. SOP 规划",
                "role": "Planner",
                "desc": "拆解 Mechanism 与 Comparison 任务链",
                "status": "COMPLETED",
                "time": "0.9s",
                "icon": "📋"
            },
            {
                "id": "stage-2",
                "name": "2. 调度执行",
                "role": "Scheduler",
                "desc": "派发执行 t1-ARCH",
                "status": "COMPLETED",
                "time": "0.2s",
                "icon": "⚡"
            },
            {
                "id": "stage-3",
                "name": "3. 证据检索",
                "role": "Researcher",
                "desc": "双跳 RAG 混合检索捕获 5 篇证据",
                "status": "COMPLETED",
                "time": "5.2s",
                "icon": "🔍"
            },
            {
                "id": "stage-4",
                "name": "4. 状态归约",
                "role": "Aggregator",
                "desc": "ID Reducer 归约，推导 2 条 Claim",
                "status": "COMPLETED",
                "time": "1.3s",
                "icon": "🧩"
            },
            {
                "id": "stage-5",
                "name": "5. 审查与自愈",
                "role": "Review & Repair",
                "desc": "发现弱证据漏洞 ➔ 局部重开 t2-PERF-R1 补足证据",
                "status": "REOPENED",
                "time": "4.5s",
                "icon": "🔄"
            },
            {
                "id": "stage-6",
                "name": "6. 终审放行",
                "role": "Judge & Compiler",
                "desc": "复审合格，校验引文并放行成文",
                "status": "COMPLETED",
                "time": "1.2s",
                "icon": "📄"
            }
        ],
        "dag_tasks": [
            {
                "task_id": "t1-ARCH",
                "name": "时延卷积与通道注意力调研",
                "stage_tag": "Step 1: 机制拆解",
                "task_type": "MECHANISM",
                "status": "COMPLETED",
                "duration_ms": 3800,
                "priority": 1,
                "worker": "researcher_worker_0",
                "produced": "产出 3 篇核心证据",
                "summary": "调研 ECAPA-TDNN 的 Squeeze-and-Excitation 机制",
                "dependencies": []
            },
            {
                "task_id": "t2-PERF",
                "name": "VoxCeleb EER 性能初测",
                "stage_tag": "Step 2: 性能比对 (初测)",
                "task_type": "COMPARISON",
                "status": "FAILED",
                "duration_ms": 2100,
                "priority": 2,
                "worker": "researcher_worker_1",
                "produced": "弱证据 (缺少测试集拆分说明)",
                "summary": "被 Review 拦截判定为弱证据漏洞，下发局部重开",
                "dependencies": ["t1-ARCH"]
            },
            {
                "task_id": "t2-PERF-R1",
                "name": "VoxCeleb 协议细分补全 (自愈任务)",
                "stage_tag": "Step 2R: 局部自愈",
                "task_type": "COMPARISON",
                "status": "COMPLETED",
                "duration_ms": 1200,
                "priority": 2,
                "worker": "researcher_worker_1",
                "produced": "补齐 VoxCeleb1-O 划分证据",
                "summary": "精确补全测试集划分协议，成功自愈，复审通过",
                "dependencies": ["t1-ARCH"]
            },
            {
                "task_id": "t3-SYNTH",
                "name": "架构对比总结成文",
                "stage_tag": "Step 3: 汇聚成文",
                "task_type": "SURVEY",
                "status": "COMPLETED",
                "duration_ms": 1800,
                "priority": 3,
                "worker": "synthesizer_agent",
                "produced": "2 条无懈可击声明",
                "summary": "最终裁决放行成文交付",
                "dependencies": ["t2-PERF-R1"]
            }
        ],
        "rag_hops": [
            {
                "iteration": 1,
                "task_id": "t1-ARCH",
                "gap": "缺失原始 x-vector 统计池化层参数实现",
                "query": "ECAPA-TDNN architecture Squeeze-and-Excitation",
                "hits": "8 篇粗排 / Top1: 0.912",
                "status": "INSUFFICIENT",
                "action": "TRIGGER_HOP_2"
            },
            {
                "iteration": 2,
                "task_id": "t1-ARCH",
                "gap": "无 (已查全统计池化层时序上下文参数)",
                "query": "x-vector statistical pooling layer temporal context",
                "hits": "5 篇粗排 / Top1: 0.887",
                "status": "SUFFICIENT",
                "action": "STOP"
            }
        ],
        "events": sim_events,
        "report_content": """<h3>x-vector 与 ECAPA-TDNN 架构及性能对比研报</h3>
<p>1. <strong>核心架构</strong>：x-vector 基于标准 TDNN 与统计池化层 <b style='color:var(--accent); cursor:pointer;'>[1]</b>；而 ECAPA-TDNN 引入了 Squeeze-and-Excitation (SE) 通道注意力机制与一维扩张卷积 <b style='color:var(--accent); cursor:pointer;'>[2]</b>。</p>
<p>2. <strong>识别性能</strong>：在 VoxCeleb1-O 评估集上，ECAPA-TDNN 等错误率相比 x-vector 相对降低约 30% <b style='color:var(--accent); cursor:pointer;'>[2]</b>。</p>
<hr style='margin:16px 0; border:none; border-top:1px solid var(--border);'>
<h4 style='font-size:13px; color:var(--muted); margin-bottom:8px;'>参考文献与溯源支撑 (Verified References):</h4>
<div style='font-size:12px; color:#475569; display:flex; flex-direction:column; gap:6px;'>
  <div><strong>[1]</strong> Snyder et al. (2018). <em>X-vectors: Robust DNN Embeddings for Speaker Recognition</em>. (关联证据: <code>evi-9901</code> · 状态: <code>SUPPORTED</code>)</div>
  <div><strong>[2]</strong> Desplanques et al. (2020). <em>ECAPA-TDNN: Emphasized Channel Attention</em>. (关联证据: <code>evi-9902</code> · 状态: <code>SUPPORTED</code>)</div>
</div>"""
    }
}

# 序列化为安全合法的 JS 字面量
json_literal = json.dumps(runs_data, ensure_ascii=False)

html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CyberClaw 多智能体科研可观测 Dashboard</title>
<style>
  :root {{
    --bg: #f8fafc;
    --panel: #ffffff;
    --border: #e2e8f0;
    --text: #0f172a;
    --muted: #64748b;
    --accent: #2563eb;
    --accent-light: #eff6ff;
    --red: #ef4444;
    --green: #10b981;
    --amber: #f59e0b;
    --purple: #8b5cf6;
    --card-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02);
    --hover-shadow: 0 6px 16px -2px rgba(37,99,235,0.1), 0 2px 6px -1px rgba(0,0,0,0.06);
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 13.5px;
    line-height: 1.5;
  }}

  header {{
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    position: sticky;
    top: 0;
    z-index: 20;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
  }}
  header h1 {{
    font-size: 16px;
    font-weight: 700;
    color: #1e293b;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .logo-icon {{
    width: 24px;
    height: 24px;
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    color: #fff;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 800;
  }}
  .tag {{
    background: #ecfdf5;
    color: #059669;
    font-size: 11px;
    padding: 2px 9px;
    border-radius: 6px;
    font-weight: 600;
    border: 1px solid #a7f3d0;
  }}
  header .controls {{
    margin-left: auto;
    display: flex;
    gap: 10px;
    align-items: center;
  }}
  button {{
    padding: 6px 12px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: #fff;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s;
    font-weight: 500;
  }}
  button:hover {{ border-color: var(--accent); color: var(--accent); }}
  button.primary {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
  button.primary:hover {{ background: #1d4ed8; color: #fff; }}

  main {{
    padding: 20px 24px;
    display: grid;
    grid-template-columns: 310px 1fr;
    gap: 20px;
    max-width: 1480px;
    margin: 0 auto;
    min-height: calc(100vh - 65px);
  }}

  .panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
    box-shadow: var(--card-shadow);
  }}
  .panel h2 {{
    font-size: 14px;
    font-weight: 700;
    color: #334155;
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}

  /* 会话列表 */
  .run-item {{
    padding: 12px;
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 10px;
    cursor: pointer;
    transition: all 0.15s;
    background: #fff;
  }}
  .run-item:hover {{
    border-color: var(--accent);
    transform: translateY(-1px);
    box-shadow: var(--hover-shadow);
  }}
  .run-item.active {{
    border-color: var(--accent);
    background: #eff6ff;
    border-left: 4px solid var(--accent);
  }}
  .run-item .rid {{
    font-weight: 700;
    font-size: 13px;
    color: #1e293b;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .run-item .meta {{
    color: var(--muted);
    font-size: 12px;
    margin-top: 5px;
  }}

  /* 状态徽章 */
  .badge {{
    display: inline-block;
    padding: 2px 7px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
  }}
  .badge.SUCCESS, .badge.COMPLETED {{ background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }}
  .badge.RUNNING {{ background: #fef3c7; color: #b45309; border: 1px solid #fde68a; animation: pulse 1.8s infinite; }}
  .badge.REOPENED {{ background: #ffedd5; color: #c2410c; border: 1px solid #fed7aa; }}
  .badge.FAILED {{ background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }}
  .badge.PENDING {{ background: #f1f5f9; color: #64748b; border: 1px solid #e2e8f0; }}
  @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}

  /* 核心 KPI 卡片 */
  .stat-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
  }}
  .stat {{
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
    background: #fafafa;
  }}
  .stat .k {{ color: var(--muted); font-size: 12px; font-weight: 500; }}
  .stat .v {{ font-size: 18px; font-weight: 800; margin-top: 3px; color: #0f172a; }}

  /* 宏观流水线阶段步进器 (Macro Pipeline Stepper) */
  .pipeline-stepper-box {{
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 20px;
  }}
  .pipeline-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }}
  .pipeline-header-title {{
    font-size: 12.5px;
    font-weight: 700;
    color: #334155;
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .pipeline-flow {{
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 10px;
    position: relative;
  }}
  .pipeline-step {{
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px;
    position: relative;
    transition: all 0.2s;
  }}
  .pipeline-step:hover {{
    border-color: var(--accent);
    box-shadow: var(--card-shadow);
  }}
  .pipeline-step.COMPLETED {{ border-top: 3px solid var(--green); }}
  .pipeline-step.REOPENED {{ border-top: 3px solid var(--amber); }}
  .pipeline-step.RUNNING {{ border-top: 3px solid var(--accent); animation: pulse 2s infinite; }}
  .pipeline-step-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }}
  .pipeline-step-name {{
    font-size: 12px;
    font-weight: 700;
    color: #1e293b;
  }}
  .pipeline-step-role {{
    font-size: 10.5px;
    color: var(--accent);
    font-weight: 600;
    background: var(--accent-light);
    padding: 1px 5px;
    border-radius: 4px;
  }}
  .pipeline-step-desc {{
    font-size: 11px;
    color: var(--muted);
    margin-top: 4px;
    line-height: 1.4;
  }}
  .pipeline-step-time {{
    font-size: 10.5px;
    color: #94a3b8;
    margin-top: 4px;
    text-align: right;
  }}

  /* 选项卡导航 */
  .tabbar {{
    display: flex;
    gap: 6px;
    margin-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }}
  .tabbar button {{
    background: none;
    border: none;
    color: var(--muted);
    padding: 9px 16px;
    border-radius: 0;
    border-bottom: 2px solid transparent;
    font-weight: 600;
    font-size: 13px;
  }}
  .tabbar button.on {{
    color: var(--accent);
    border-bottom-color: var(--accent);
  }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}

  /* DAG 拓扑图画板 */
  .dag-container {{
    position: relative;
    min-height: 420px;
    background: #fafafa;
    background-image: radial-gradient(#e2e8f0 1.2px, transparent 1.2px);
    background-size: 20px 20px;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: auto;
    padding: 24px;
  }}
  .dag-canvas {{
    position: relative;
    min-width: 800px;
    min-height: 360px;
  }}
  .dag-svg-layer {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 1;
  }}

  /* 流动光效线条 */
  @keyframes dagFlow {{
    from {{ stroke-dashoffset: 24; }}
    to {{ stroke-dashoffset: 0; }}
  }}
  .dag-edge-bg {{
    stroke: #e2e8f0;
    stroke-width: 5px;
    stroke-linecap: round;
    fill: none;
  }}
  .dag-edge-flow {{
    stroke: #3b82f6;
    stroke-width: 2.2px;
    stroke-linecap: round;
    fill: none;
    stroke-dasharray: 6 5;
    animation: dagFlow 1.2s linear infinite;
  }}

  /* DAG 节点卡片 */
  .dag-node-item {{
    position: absolute;
    width: 200px;
    background: #ffffff;
    border: 1.5px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 12px;
    box-shadow: var(--card-shadow);
    cursor: pointer;
    transition: all 0.2s;
    z-index: 2;
  }}
  .dag-node-item:hover {{
    transform: translateY(-3px);
    box-shadow: var(--hover-shadow);
  }}
  .dag-node-item.active {{
    outline: 2px solid var(--accent);
    box-shadow: 0 0 0 4px rgba(37,99,235,0.15);
  }}
  .dag-node-item.COMPLETED {{ border-color: var(--green); border-left: 5px solid var(--green); }}
  .dag-node-item.RUNNING {{ border-color: var(--accent); border-left: 5px solid var(--accent); animation: pulse 1.8s infinite; }}
  .dag-node-item.REOPENED {{ border-color: var(--amber); border-left: 5px solid var(--amber); }}
  .dag-node-item.FAILED {{ border-color: var(--red); border-left: 5px solid var(--red); }}
  
  .dag-node-tag {{
    font-size: 10px;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .dag-node-title {{
    font-weight: 700;
    font-size: 13px;
    color: #1e293b;
    line-height: 1.3;
    margin-bottom: 6px;
  }}
  .dag-node-worker {{
    font-size: 11px;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 4px;
  }}
  .dag-node-output {{
    margin-top: 6px;
    padding-top: 6px;
    border-top: 1px dashed var(--border);
    font-size: 11px;
    color: #059669;
    font-weight: 600;
  }}

  /* 阶段列头指示 */
  .dag-stage-col-header {{
    position: absolute;
    top: 6px;
    font-size: 11px;
    font-weight: 700;
    color: #64748b;
    background: #f1f5f9;
    padding: 3px 10px;
    border-radius: 12px;
    border: 1px solid #cbd5e1;
    text-align: center;
  }}

  /* 节点下钻详情抽屉 */
  .dag-drawer {{
    margin-top: 14px;
    padding: 14px 18px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: #ffffff;
    box-shadow: var(--card-shadow);
  }}

  /* 表格样式 */
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; background: #f8fafc; font-size: 12px; }}
  tr:hover td {{ background: #f8fafc; }}

  /* 事件流列表 */
  .timeline {{ display: flex; flex-direction: column; gap: 8px; max-height: 540px; overflow-y: auto; }}
  .tl-item {{
    display: grid;
    grid-template-columns: 85px 120px 110px 1fr 70px;
    gap: 12px;
    align-items: baseline;
    padding: 9px 12px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: #fff;
    font-size: 12px;
    transition: all 0.15s;
  }}
  .tl-item:hover {{ background: #f8fafc; border-color: #cbd5e1; }}
  .tl-time {{ color: var(--muted); font-family: monospace; font-size: 11px; }}
  .tl-node {{
    font-weight: 600;
    color: #4338ca;
    background: #e0e7ff;
    padding: 2px 7px;
    border-radius: 5px;
    font-size: 11px;
    width: fit-content;
  }}
  .tl-evt {{ font-weight: 600; color: #1e293b; }}
  .tl-meta {{ color: #475569; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .tl-dur {{ text-align: right; color: var(--muted); font-family: monospace; font-size: 11px; }}
</style>
</head>
<body>

<header>
  <h1>
    <div class="logo-icon">C</div>
    CyberClaw Research 多智能体全景可观测看板
  </h1>
  <span class="tag">端到端全链路闭环 · 100% 审计追溯</span>
  <div class="controls">
    <button onclick="window.location.reload()">重新载入</button>
  </div>
</header>

<main>
  <!-- 左侧：多会话列表 -->
  <div id="run-list" class="panel">
    <h2>
      运行会话 (Runs)
      <span style="font-size:12px; font-weight:normal; color:var(--muted)">2 个会话</span>
    </h2>
    
    <!-- 闭环菱形并发会话 -->
    <div class="run-item active" id="item-run-active-9002" onclick="selectRun('run-active-9002')">
      <div class="rid">
        <span>run-active-9002</span>
        <span class="badge SUCCESS">SUCCESS</span>
      </div>
      <div class="meta" style="margin-top:6px; color:#1e293b; font-weight:600;">
        语音大模型前沿架构与消融调研
      </div>
      <div class="meta">
        <span>14:30:00</span> · <span>18.2s</span> · <span>菱形并发 (4节点闭环)</span>
      </div>
    </div>

    <!-- 历史重开自愈会话 -->
    <div class="run-item" id="item-run-sim-8891" onclick="selectRun('run-sim-8891')">
      <div class="rid">
        <span>run-sim-8891</span>
        <span class="badge SUCCESS">SUCCESS</span>
      </div>
      <div class="meta" style="margin-top:6px; color:#1e293b; font-weight:600;">
        对比 x-vector 与 ECAPA-TDNN 架构
      </div>
      <div class="meta">
        <span>10:00:00</span> · <span>15.6s</span> · <span>红队发难 ➔ 局部重开自愈</span>
      </div>
    </div>
  </div>

  <!-- 右侧：详情面板 -->
  <div id="detail" class="panel">
    <h2>
      <span id="detail-title">会话运行全景详情 - run-active-9002</span>
      <span class="badge SUCCESS" id="detail-badge">SUCCESS</span>
    </h2>

    <!-- 核心指标卡 -->
    <div class="stat-grid">
      <div class="stat">
        <div class="k">总执行耗时</div>
        <div class="v" id="kpi-duration" style="color:var(--accent)">18.2s</div>
      </div>
      <div class="stat">
        <div class="k">DAG 任务完成度</div>
        <div class="v" id="kpi-tasks">4 / 4 闭环</div>
      </div>
      <div class="stat">
        <div class="k">精准证据切片</div>
        <div class="v" id="kpi-evidences" style="color:var(--green)">7 篇切片</div>
      </div>
      <div class="stat">
        <div class="k">引文对账状态</div>
        <div class="v" id="kpi-citation" style="color:var(--green)">100% 支撑</div>
      </div>
    </div>

    <!-- 顶层宏观流程步进器 (Macro Workflow Stepper) -->
    <div class="pipeline-stepper-box">
      <div class="pipeline-header">
        <div class="pipeline-header-title">
          <span>🔄</span>
          <span>多智能体全生命周期流水线阶段 (Multi-Agent Lifecycle Stages)</span>
        </div>
        <span style="font-size:11px; color:var(--muted)">按时序自动协同推进 · 阶段全景流转</span>
      </div>
      <div class="pipeline-flow" id="pipeline-stepper-container">
        <!-- 动态生成 6 个阶梯卡片 -->
      </div>
    </div>

    <!-- 选项卡导航 -->
    <div class="tabbar">
      <button class="on" onclick="switchTab('tab-dag', this)">1. 多智能体任务拓扑图 (Task DAG & Flow)</button>
      <button onclick="switchTab('tab-rag', this)">2. RAG 多跳推演与盲区对账 (Iterative RAG)</button>
      <button onclick="switchTab('tab-events', this)">3. 全链路审计事件流 (Event Stream)</button>
      <button onclick="switchTab('tab-report', this)">4. 交付研报与证据外键核验 (Report & Citations)</button>
    </div>

    <!-- Tab 1: DAG 拓扑图与流程 -->
    <div id="tab-dag" class="tab-content active">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <div style="display:flex; gap:14px; font-size:11px; align-items:center;">
          <span><span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--green); margin-right:4px;"></span>COMPLETED (完成)</span>
          <span><span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--accent); margin-right:4px;"></span>RUNNING (执行中)</span>
          <span><span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--amber); margin-right:4px;"></span>REOPENED (重开自愈)</span>
          <span><span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--red); margin-right:4px;"></span>FAILED (拦截)</span>
        </div>
        <div style="font-size:12px; color:var(--muted)">流动虚线标识数据流向 · 点击节点下钻查看产出切片与前驱输入</div>
      </div>
      
      <div class="dag-container">
        <div id="dag-canvas-box" class="dag-canvas">
          <!-- 动态渲染 SVG 流程线与节点卡片 -->
        </div>
      </div>

      <div id="dag-detail-drawer" class="dag-drawer" style="display:none;"></div>
    </div>

    <!-- Tab 2: RAG 多跳 -->
    <div id="tab-rag" class="tab-content">
      <table>
        <thead>
          <tr>
            <th>跳数 (Hop)</th>
            <th>关联任务</th>
            <th>目标知识盲区 (Information Gap)</th>
            <th>精准搜索词 (Next Query)</th>
            <th>候选数 / Rerank 分</th>
            <th>充分性判定</th>
          </tr>
        </thead>
        <tbody id="rag-table-body">
        </tbody>
      </table>
    </div>

    <!-- Tab 3: 全链路事件流 -->
    <div id="tab-events" class="tab-content">
      <div class="timeline" id="events-timeline">
      </div>
    </div>

    <!-- Tab 4: 报告 -->
    <div id="tab-report" class="tab-content">
      <div id="report-box" style="background:#f8fafc; border:1px solid var(--border); border-radius:8px; padding:18px; font-size:14px; line-height:1.7;">
      </div>
    </div>

  </div>
</main>

<script>
const allRuns = {json_literal};
let currentRunId = "run-active-9002";
let currentDagTasks = [];

function selectRun(rid) {{
  currentRunId = rid;
  document.querySelectorAll('.run-item').forEach(el => el.classList.remove('active'));
  const activeItem = document.getElementById('item-' + rid);
  if (activeItem) activeItem.classList.add('active');

  const run = allRuns[rid];
  if (!run) return;

  // 更新 KPI 与 Header
  document.getElementById('detail-title').innerText = '会话运行全景详情 - ' + run.run_id + ' (' + run.title + ')';
  const badge = document.getElementById('detail-badge');
  badge.innerText = run.status;
  badge.className = 'badge ' + run.status;
  
  document.getElementById('kpi-duration').innerText = run.duration;
  document.getElementById('kpi-tasks').innerText = run.tasks_summary;
  document.getElementById('kpi-evidences').innerText = run.evidence_count;
  document.getElementById('kpi-citation').innerText = run.citation_status;

  // 渲染顶层流水线阶梯
  renderPipelineStepper(run.pipeline_stages || []);

  // 渲染 DAG 拓扑流程图
  renderDagGraph(run.dag_tasks || []);

  // 渲染 RAG 多跳表格
  renderRagTable(run.rag_hops || []);

  // 渲染事件流
  renderTimeline(run.events || []);

  // 渲染报告
  document.getElementById('report-box').innerHTML = run.report_content || '';
}}

function renderPipelineStepper(stages) {{
  const container = document.getElementById('pipeline-stepper-container');
  if (!stages || !stages.length) {{
    container.innerHTML = '<div style="color:var(--muted); font-size:12px;">无阶段记录</div>';
    return;
  }}
  let html = '';
  stages.forEach(st => {{
    html += `
      <div class="pipeline-step ${{st.status}}">
        <div class="pipeline-step-top">
          <span class="pipeline-step-name">${{st.icon || '📌'}} ${{st.name}}</span>
          <span class="pipeline-step-role">${{st.role}}</span>
        </div>
        <div class="pipeline-step-desc">${{st.desc}}</div>
        <div class="pipeline-step-time">${{st.time}} · <span class="badge ${{st.status}}" style="font-size:9px; padding:0 4px;">${{st.status}}</span></div>
      </div>
    `;
  }});
  container.innerHTML = html;
}}

function renderDagGraph(tasks) {{
  currentDagTasks = tasks || [];
  const container = document.getElementById('dag-canvas-box');
  document.getElementById('dag-detail-drawer').style.display = 'none';

  if (!tasks || !tasks.length) {{
    container.innerHTML = '<div class="empty">无任务 DAG</div>';
    return;
  }}

  // 1. 拓扑分层 (Kahn / Layered Ranking)
  const taskMap = {{}};
  tasks.forEach(t => taskMap[t.task_id] = t);
  const ranks = {{}};

  function getRank(tid, visited = new Set()) {{
    if (ranks[tid] !== undefined) return ranks[tid];
    if (visited.has(tid)) return 0;
    visited.add(tid);
    const t = taskMap[tid];
    if (!t || !t.dependencies || !t.dependencies.length) {{
      ranks[tid] = 0;
      return 0;
    }}
    let maxDepRank = -1;
    for (const dep of t.dependencies) {{
      if (taskMap[dep]) {{
        maxDepRank = Math.max(maxDepRank, getRank(dep, new Set(visited)));
      }}
    }}
    ranks[tid] = maxDepRank + 1;
    return ranks[tid];
  }}

  tasks.forEach(t => getRank(t.task_id));

  // 2. 按 Rank 划分列
  const columns = {{}};
  tasks.forEach(t => {{
    const r = ranks[t.task_id] || 0;
    if (!columns[r]) columns[r] = [];
    columns[r].push(t);
  }});

  const nodePositions = {{}};
  const colWidth = 270;
  const rowHeight = 135;
  const maxRank = Math.max(...Object.keys(columns).map(Number));

  let maxColCount = 1;
  Object.keys(columns).forEach(r => {{
    maxColCount = Math.max(maxColCount, columns[r].length);
  }});

  const canvasWidth = Math.max(820, (maxRank + 1) * colWidth + 60);
  const canvasHeight = Math.max(380, maxColCount * rowHeight + 90);

  // 3. 计算节点坐标
  Object.keys(columns).forEach(r => {{
    const colList = columns[r];
    const x = 40 + r * colWidth;
    const startY = 60 + ((maxColCount - colList.length) * rowHeight) / 2;
    colList.forEach((t, idx) => {{
      const y = startY + idx * rowHeight;
      nodePositions[t.task_id] = {{ x, y, width: 210, height: 95, task: t }};
    }});
  }});

  // 4. 生成带流向光效的平滑贝塞尔曲线边
  let svgPaths = '';
  tasks.forEach(t => {{
    if (t.dependencies) {{
      t.dependencies.forEach(dep => {{
        const from = nodePositions[dep];
        const to = nodePositions[t.task_id];
        if (from && to) {{
          const x1 = from.x + from.width;
          const y1 = from.y + from.height / 2;
          const x2 = to.x;
          const y2 = to.y + to.height / 2;
          const cx1 = x1 + (x2 - x1) * 0.45;
          const cx2 = x2 - (x2 - x1) * 0.45;
          const d = `M ${{x1}} ${{y1}} C ${{cx1}} ${{y1}}, ${{cx2}} ${{y2}}, ${{x2}} ${{y2}}`;
          // 底层加宽轨道线 + 上层动画流动虚线
          svgPaths += `
            <path d="${{d}}" class="dag-edge-bg" />
            <path d="${{d}}" class="dag-edge-flow" marker-end="url(#dag-arrow)" />
          `;
        }}
      }});
    }}
  }});

  // 5. 渲染列头阶段标识
  let stageHeaders = '';
  const stageNames = ["阶段一：基础基石", "阶段二：并发调研", "阶段三：汇聚收敛", "阶段四：终审交付"];
  Object.keys(columns).forEach(r => {{
    const colX = 40 + r * colWidth + 30;
    const name = stageNames[r] || ('阶段 ' + (Number(r) + 1));
    stageHeaders += `<div class="dag-stage-col-header" style="left:${{colX}}px;">${{name}}</div>`;
  }});

  let html = `
    ${{stageHeaders}}
    <svg class="dag-svg-layer" width="${{canvasWidth}}" height="${{canvasHeight}}">
      <defs>
        <marker id="dag-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#3b82f6" />
        </marker>
      </defs>
      ${{svgPaths}}
    </svg>`;

  // 6. 渲染各节点卡片
  Object.values(nodePositions).forEach(item => {{
    const t = item.task;
    html += `
      <div class="dag-node-item ${{t.status}}" id="node-${{t.task_id}}"
           style="left:${{item.x}}px; top:${{item.y}}px;" onclick="clickDagNode('${{t.task_id}}')">
        <div class="dag-node-tag">
          <span>${{t.stage_tag || t.task_id}}</span>
          <span class="badge ${{t.status}}" style="font-size:9px; padding:0 5px;">${{t.status}}</span>
        </div>
        <div class="dag-node-title">${{t.name || t.task_id}}</div>
        <div class="dag-node-worker">
          <span>🤖 ${{t.worker || 'Worker'}}</span>
          <span>·</span>
          <span>${{t.duration_ms ? t.duration_ms + 'ms' : '-'}}</span>
        </div>
        <div class="dag-node-output">
          <span>${{t.produced || '已生成证据'}}</span>
        </div>
      </div>`;
  }});

  container.style.width = canvasWidth + 'px';
  container.style.height = canvasHeight + 'px';
  container.innerHTML = html;
}}

function clickDagNode(tid) {{
  document.querySelectorAll('.dag-node-item').forEach(el => el.classList.remove('active'));
  const activeEl = document.getElementById('node-' + tid);
  if (activeEl) activeEl.classList.add('active');

  const task = currentDagTasks.find(t => t.task_id === tid);
  const drawer = document.getElementById('dag-detail-drawer');
  if (!task || !drawer) return;

  drawer.style.display = 'block';
  drawer.innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
      <h3 style="font-size:13.5px; font-weight:700; color:#1e293b;">
        <span>📌 节点工单详情: ${{task.task_id}} - ${{task.name || ''}}</span>
      </h3>
      <span class="badge ${{task.status}}">${{task.status}}</span>
    </div>
    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; font-size:12px; background:#f8fafc; padding:10px 12px; border-radius:6px; margin-bottom:10px;">
      <div><span style="color:var(--muted)">任务类型:</span> <b>${{task.task_type || '-'}}</b></div>
      <div><span style="color:var(--muted)">执行主体 (Worker):</span> <b>${{task.worker || '-'}}</b></div>
      <div><span style="color:var(--muted)">执行耗时:</span> <b>${{task.duration_ms ? task.duration_ms + 'ms' : '-'}}</b></div>
      <div><span style="color:var(--muted)">前置依赖节点:</span> <b>${{(task.dependencies && task.dependencies.length) ? task.dependencies.join(', ') : '无 (根基石节点)'}}</b></div>
    </div>
    <div style="font-size:12.5px; color:#334155; line-height:1.6;">
      <div><strong>任务摘要：</strong>${{task.summary || '-'}}</div>
      <div style="margin-top:4px;"><strong>产出资产：</strong><span style="color:var(--green); font-weight:600;">${{task.produced || '-'}}</span></div>
    </div>
  `;
}}

function renderRagTable(hops) {{
  const tbody = document.getElementById('rag-table-body');
  if (!hops || !hops.length) {{
    tbody.innerHTML = '<tr><td colspan="6" style="color:var(--muted); text-align:center; padding:20px;">无 RAG 检索事件</td></tr>';
    return;
  }}
  let html = '';
  for (const h of hops) {{
    html += `<tr>
      <td><strong>第 ${{h.iteration}} 跳</strong></td>
      <td><code>${{h.task_id || '-'}}</code></td>
      <td style="color:#b45309; font-weight:500;">${{h.gap || '-'}}</td>
      <td><code>${{h.query || '-'}}</code></td>
      <td>${{h.hits}}</td>
      <td><span class="badge ${{h.status}}">${{h.status}}</span></td>
    </tr>`;
  }}
  tbody.innerHTML = html;
}}

function renderTimeline(events) {{
  const container = document.getElementById('events-timeline');
  if (!events || !events.length) {{
    container.innerHTML = '<div style="color:var(--muted); text-align:center; padding:20px;">无事件记录</div>';
    return;
  }}
  let html = '';
  for (const e of events) {{
    const dur = e.duration_ms ? e.duration_ms + 'ms' : '-';
    const timeStr = e.timestamp ? e.timestamp.split('T')[1].replace('Z','') : '';
    let metaSummary = '';
    if (e.metadata) {{
      if (e.metadata.query) metaSummary = 'query: ' + e.metadata.query;
      else if (e.metadata.gap) metaSummary = 'Gap: ' + e.metadata.gap;
      else if (e.metadata.issues) metaSummary = 'Issues: ' + JSON.stringify(e.metadata.issues);
      else if (e.metadata.decision) metaSummary = 'Decision: ' + e.metadata.decision;
      else if (e.metadata.objective) metaSummary = 'Objective: ' + e.metadata.objective;
      else if (e._annotation) metaSummary = e._annotation;
    }}
    
    html += `<div class="tl-item">
      <span class="tl-time">${{timeStr}}</span>
      <span class="tl-node">${{e.node || 'core'}}</span>
      <span class="tl-evt">${{e.event_type || e.event}}</span>
      <span class="tl-meta" title="${{metaSummary}}">${{metaSummary || '-'}}</span>
      <span class="tl-dur">${{dur}}</span>
    </div>`;
  }}
  container.innerHTML = html;
}}

function switchTab(tabId, btn) {{
  document.querySelectorAll('.tabbar button').forEach(b => b.classList.remove('on'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  btn.classList.add('on');
  document.getElementById(tabId).classList.add('active');
}}

function focusEvidence(eviId) {{
  alert('定位到证据外键: ' + eviId + '，可在全链路事件流与知识库中反查对应 PDF 篇章与切片。');
}}

// 默认激活闭环会话
selectRun("run-active-9002");
</script>
</body>
</html>
"""

with open("docs/dashboard_preview.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Successfully written to docs/dashboard_preview.html, length:", len(html_content))
