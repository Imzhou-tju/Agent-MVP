"""统一审计事件层：Run → Node → Task → Retrieval → Evidence → Claim → Citation → Report。

设计约束（对应监控/审计 spec）：

1. JSONL 是唯一审计数据源，落盘位置与现有行为审计日志一致：`logs/<thread_id>.jsonl`；
   旧事件（llm_input / tool_call / tool_result / ai_message）不受影响。
2. 写入是异步的：复用 `logger.JSONLEventLogger` 的内存队列 + 后台写盘线程，
   主链路只做一次 `queue.put()`，不阻塞在磁盘 IO。
3. 单次事件写入失败只打印 warning，不抛异常、不中断主业务流程。
4. `metadata` 只放结构化摘要（数量、状态、id、短原因），不放完整 Prompt、
   完整 LLM 返回、来源正文；超长字符串按 `MAX_METADATA_VALUE_CHARS` 截断。
5. `error` 只放错误类型与短 message，不放完整堆栈。
"""

from __future__ import annotations

import functools
import inspect
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from . import config

AUDIT_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------

STATUS_RUNNING = "RUNNING"      # 生命周期事件已开始、未结束
STATUS_SUCCESS = "SUCCESS"      # 正常完成
STATUS_DEGRADED = "DEGRADED"    # 完成但走了降级（fallback / 部分失败）
STATUS_FAILED = "FAILED"        # 失败


# ---------------------------------------------------------------------------
# 事件类型
# ---------------------------------------------------------------------------

# 运行生命周期
RUN_START = "run_start"
RUN_END = "run_end"

# 节点生命周期
NODE_START = "node_start"
NODE_END = "node_end"

# 任务生命周期
TASK_CREATED = "task_created"
TASK_READY = "task_ready"
TASK_STARTED = "task_started"
TASK_COMPLETED = "task_completed"
TASK_FAILED = "task_failed"
TASK_REOPENED = "task_reopened"
TASK_SKIPPED = "task_skipped"

# 调用与可靠性
LLM_CALL = "llm_call"
TOOL_CALL = "tool_call"
RETRY = "retry"
FALLBACK = "fallback"
CIRCUIT_BREAKER = "circuit_breaker"

# 检索
RAG_ITERATION = "rag_iteration"
RAG_RETRIEVAL = "rag_retrieval"
RERANK = "rerank"
MULTI_HOP_ITERATION = "multi_hop_iteration"

# 调研决策
PLAN_GATE = "plan_gate"
AGGREGATION = "aggregation"
REVIEW_ISSUE = "review_issue"
REPAIR = "repair"
JUDGE_DECISION = "judge_decision"

# 溯源
EVIDENCE_REGISTERED = "evidence_registered"
CITATION_VERIFIED = "citation_verified"

ALL_EVENT_TYPES = (
    RUN_START, RUN_END, NODE_START, NODE_END,
    TASK_CREATED, TASK_READY, TASK_STARTED, TASK_COMPLETED,
    TASK_FAILED, TASK_REOPENED, TASK_SKIPPED,
    LLM_CALL, TOOL_CALL, RETRY, FALLBACK, CIRCUIT_BREAKER,
    RAG_ITERATION, RAG_RETRIEVAL, RERANK, MULTI_HOP_ITERATION,
    PLAN_GATE, AGGREGATION, REVIEW_ISSUE, REPAIR, JUDGE_DECISION,
    EVIDENCE_REGISTERED, CITATION_VERIFIED,
)

# metadata 里单个字符串值的最大长度（防止把正文/响应塞进审计）
MAX_METADATA_VALUE_CHARS = 500
# error message 的最大长度
MAX_ERROR_CHARS = 200


# ---------------------------------------------------------------------------
# 事件结构
# ---------------------------------------------------------------------------

@dataclass
class AuditEvent:
    event_type: str
    event_id: str = ""
    run_id: str = ""
    thread_id: str = ""
    timestamp: str = ""
    node: str = ""
    task_id: str = ""
    parent_event_id: str = ""
    status: str = ""
    duration_ms: int = 0
    metadata: dict = field(default_factory=dict)
    error: dict | None = None

    def __post_init__(self) -> None:
        if not self.event_id:
            self.event_id = uuid.uuid4().hex[:16]
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "event_type": self.event_type,
            "event_id": self.event_id,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "timestamp": self.timestamp,
            "node": self.node,
            "task_id": self.task_id,
            "parent_event_id": self.parent_event_id,
            "status": self.status,
            "duration_ms": int(self.duration_ms or 0),
            "metadata": dict(self.metadata or {}),
            "error": dict(self.error) if self.error else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AuditEvent":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


# ---------------------------------------------------------------------------
# 上下文（run_id / thread_id / node / task_id / parent_event_id）
# ---------------------------------------------------------------------------

_CTX: ContextVar[dict] = ContextVar("clawgent_audit_ctx", default={})


def bind(**kwargs: Any) -> Any:
    """合并进当前上下文，返回用于 unbind 的 token。值为 None 表示不覆盖。"""
    cur = dict(_CTX.get())
    cur.update({k: v for k, v in kwargs.items() if v is not None})
    return _CTX.set(cur)


def unbind(token: Any) -> None:
    try:
        _CTX.reset(token)
    except ValueError:
        pass


def current_context() -> dict:
    return dict(_CTX.get())


def set_node_metadata(**kwargs: Any) -> None:
    """节点在结束前塞入的摘要字段，会合并进 node_end 的 metadata。"""
    cur = dict(_CTX.get())
    merged = dict(cur.get("node_end_metadata") or {})
    merged.update(kwargs)
    cur["node_end_metadata"] = merged
    _CTX.set(cur)


# ---------------------------------------------------------------------------
# Sink
# ---------------------------------------------------------------------------

_sinks: list[Callable[[dict], None]] = []
_sink_lock = threading.Lock()


def add_sink(fn: Callable[[dict], None]) -> None:
    with _sink_lock:
        if fn not in _sinks:
            _sinks.append(fn)


def remove_sink(fn: Callable[[dict], None]) -> None:
    with _sink_lock:
        if fn in _sinks:
            _sinks.remove(fn)


def clear_sinks() -> None:
    with _sink_lock:
        _sinks.clear()


def _jsonl_sink(event: dict) -> None:
    """默认 sink：写 logs/<thread_id>.jsonl（异步队列）。

    兼容旧读取器：日志行同时带 `event`（= event_type）与 `ts`，
    entry/monitor.py 那类旧逻辑不会因新事件而报错。
    """
    from .logger import audit_logger

    payload = {k: v for k, v in event.items() if k != "thread_id"}
    audit_logger.log_event(event.get("thread_id") or "research", event["event_type"], **payload)


def install_default_sink() -> None:
    """安装 JSONL sink（幂等）。AUDIT_ENABLED=false 时不安装。"""
    if not getattr(config, "AUDIT_ENABLED", True):
        return
    if _jsonl_sink not in _sinks:
        add_sink(_jsonl_sink)


def _warn(msg: str) -> None:
    print(f"[Audit Warning] {msg}")


# ---------------------------------------------------------------------------
# 字段收敛
# ---------------------------------------------------------------------------

def _clip(value: Any, limit: int) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…(truncated)"
    return value


def _safe_metadata(metadata: dict | None) -> dict:
    if not metadata:
        return {}
    out: dict = {}
    for k, v in metadata.items():
        if isinstance(v, dict):
            out[k] = {kk: _clip(vv, MAX_METADATA_VALUE_CHARS) for kk, vv in list(v.items())[:20]}
        elif isinstance(v, (list, tuple)):
            out[k] = [_clip(x, MAX_METADATA_VALUE_CHARS) for x in list(v)[:20]]
        else:
            out[k] = _clip(v, MAX_METADATA_VALUE_CHARS)
    return out


def _safe_error(error: Any) -> dict | None:
    if error is None:
        return None
    if isinstance(error, dict):
        return {
            "type": str(error.get("type", ""))[:100],
            "message": str(error.get("message", ""))[:MAX_ERROR_CHARS],
        }
    if isinstance(error, BaseException):
        return {"type": type(error).__name__, "message": str(error)[:MAX_ERROR_CHARS]}
    return {"type": "Error", "message": str(error)[:MAX_ERROR_CHARS]}


# ---------------------------------------------------------------------------
# emit
# ---------------------------------------------------------------------------

def emit(
    event_type: str,
    *,
    status: str = "",
    duration_ms: float | int | None = None,
    metadata: dict | None = None,
    error: Any = None,
    run_id: str | None = None,
    thread_id: str | None = None,
    node: str | None = None,
    task_id: str | None = None,
    parent_event_id: str | None = None,
) -> dict | None:
    """发一条审计事件。返回落盘的事件 dict；任何异常都被吞掉并返回 None。"""
    try:
        ctx = _CTX.get()
        ev = AuditEvent(
            event_type=event_type,
            run_id=run_id or ctx.get("run_id", ""),
            thread_id=thread_id or ctx.get("thread_id", ""),
            node=node or ctx.get("node", ""),
            task_id=task_id or ctx.get("task_id", ""),
            parent_event_id=parent_event_id or ctx.get("parent_event_id", ""),
            status=status,
            duration_ms=int(duration_ms or 0),
            metadata=_safe_metadata(metadata),
            error=_safe_error(error),
        )
        data = ev.to_dict()
        install_default_sink()
        for sink in list(_sinks):
            try:
                sink(data)
            except Exception as e:  # 单个 sink 故障不影响其他 sink 与主流程
                _warn(f"sink {getattr(sink, '__name__', sink)} 写入失败: {e}")
        return data
    except Exception as e:
        _warn(f"事件 {event_type} 生成失败（不阻断流程）: {e}")
        return None


@contextmanager
def span(start_type: str, end_type: str, *, metadata: dict | None = None,
         status: str = STATUS_SUCCESS):
    """开始/结束事件对，自动计算 duration_ms；异常时结束事件记 FAILED。"""
    t0 = time.perf_counter()
    start = emit(start_type, status=STATUS_RUNNING, metadata=metadata)
    parent = start["event_id"] if start else ""
    token = bind(parent_event_id=parent)
    err: BaseException | None = None
    try:
        yield parent
    except Exception as e:
        err = e
        raise
    finally:
        dur = (time.perf_counter() - t0) * 1000.0
        if err is not None:
            emit(end_type, status=STATUS_FAILED, duration_ms=dur, error=err)
        else:
            emit(end_type, status=status, duration_ms=dur)
        unbind(token)


# ---------------------------------------------------------------------------
# 节点埋点装饰器
# ---------------------------------------------------------------------------

def _node_begin(node_name: str, state: Any) -> tuple[Any, Any, float]:
    ctx_state = state if isinstance(state, dict) else {}
    task = ctx_state.get("task") or {}
    token = bind(
        run_id=str(ctx_state.get("run_id", "") or ""),
        thread_id=str(ctx_state.get("thread_id", "") or ""),
        node=node_name,
        task_id=str(task.get("task_id", "") or "") if isinstance(task, dict) else "",
    )
    ev = emit(NODE_START, status=STATUS_RUNNING, metadata={
        "round_no": ctx_state.get("round_no", 0),
    })
    parent = ev["event_id"] if ev else ""
    token2 = bind(parent_event_id=parent)
    return (token, token2), time.perf_counter(), parent


def _node_end(tokens: tuple[Any, Any], t0: float, err: BaseException | None = None) -> None:
    token, token2 = tokens
    dur = (time.perf_counter() - t0) * 1000.0
    extra = dict(_CTX.get().get("node_end_metadata") or {})
    if err is not None:
        emit(NODE_END, status=STATUS_FAILED, duration_ms=dur, error=err, metadata=extra)
    else:
        emit(NODE_END, status=STATUS_SUCCESS, duration_ms=dur, metadata=extra)
    unbind(token2)
    unbind(token)


def audit_node(node_name: str) -> Callable:
    """给 Research 子图节点加 node_start / node_end 埋点（不改返回值与业务行为）。

    同步节点与异步节点都支持；节点抛异常时记 FAILED 后原样抛出，
    由 LangGraph 的 RetryPolicy 决定是否重试。
    """
    def deco(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(state, *args, **kwargs):
                tokens, t0, _parent = _node_begin(node_name, state)
                try:
                    return await fn(state, *args, **kwargs)
                except Exception as e:
                    _node_end(tokens, t0, e)
                    raise
                else:
                    _node_end(tokens, t0)
            return async_wrapper

        @functools.wraps(fn)
        def wrapper(state, *args, **kwargs):
            tokens, t0, _parent = _node_begin(node_name, state)
            try:
                return fn(state, *args, **kwargs)
            except Exception as e:
                _node_end(tokens, t0, e)
                raise
            else:
                _node_end(tokens, t0)
        return wrapper
    return deco


# ---------------------------------------------------------------------------
# Run Summary（纯函数，从事件流聚合，不维护第二套业务状态）
# ---------------------------------------------------------------------------

def _parse_ts(ts: str) -> float:
    if not ts:
        return 0.0
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _unique_task_ids(events: list[dict]) -> set[str]:
    return {e.get("task_id", "") for e in events if e.get("task_id")}


def summarize(events: Iterable[dict]) -> dict:
    """按 spec §6 聚合 Run Summary。异常终止（无 run_end）也会给出尽可能完整的统计。"""
    evs = [e for e in (events or []) if isinstance(e, dict) and e.get("event_type")]
    evs.sort(key=lambda e: e.get("timestamp", ""))

    if not evs:
        return {"total_duration": 0, "final_status": "UNKNOWN", "total_tasks": 0}

    def count(event_type: str) -> int:
        return sum(1 for e in evs if e.get("event_type") == event_type)

    def last_meta(event_type: str) -> dict:
        for e in reversed(evs):
            if e.get("event_type") == event_type and isinstance(e.get("metadata"), dict):
                return e["metadata"]
        return {}

    run_id = next((e.get("run_id", "") for e in evs if e.get("run_id")), "")
    thread_id = next((e.get("thread_id", "") for e in evs if e.get("thread_id")), "")

    starts = [e for e in evs if e.get("event_type") == RUN_START]
    ends = [e for e in evs if e.get("event_type") == RUN_END]
    first_ts = (starts[0] if starts else evs[0]).get("timestamp", "")
    last_ts = (ends[-1] if ends else evs[-1]).get("timestamp", "")
    total_duration = int(max(0.0, (_parse_ts(last_ts) - _parse_ts(first_ts)) * 1000))

    if ends:
        final_status = ends[-1].get("status") or STATUS_SUCCESS
    elif any(e.get("status") == STATUS_FAILED for e in evs if e.get("event_type") == NODE_END):
        final_status = STATUS_FAILED
    else:
        final_status = STATUS_RUNNING  # 没有 run_end：运行中或异常终止

    agg = last_meta(AGGREGATION)
    claim_by_status = agg.get("claim_by_status", {}) or {}

    source_count = int(agg.get("source_count", 0) or 0)
    evidence_count = int(agg.get("evidence_count", 0) or 0)
    if not evidence_count:
        evidence_count = count(EVIDENCE_REGISTERED)

    judge = last_meta(JUDGE_DECISION)
    repair_rounds = count(REPAIR)

    return {
        "run_id": run_id,
        "thread_id": thread_id,
        "query": (starts[0].get("metadata", {}) or {}).get("query", "") if starts else "",
        "start_time": first_ts,
        "end_time": last_ts,
        "total_duration": total_duration,
        "final_status": final_status,
        "verdict": judge.get("decision", ""),

        "total_tasks": len(_unique_task_ids(evs)),
        "completed_tasks": count(TASK_COMPLETED),
        "failed_tasks": count(TASK_FAILED),
        "reopened_tasks": count(TASK_REOPENED),
        "skipped_tasks": count(TASK_SKIPPED),

        "total_llm_calls": count(LLM_CALL),
        "total_tool_calls": count(TOOL_CALL),
        "total_retries": count(RETRY),
        "total_fallbacks": count(FALLBACK),
        "circuit_breaker_events": count(CIRCUIT_BREAKER),

        "source_count": source_count,
        "evidence_count": evidence_count,

        "supported_claims": int(claim_by_status.get("SUPPORTED", 0)),
        "partial_claims": int(claim_by_status.get("PARTIALLY_SUPPORTED", 0)),
        "unsupported_claims": int(claim_by_status.get("UNSUPPORTED", 0)),
        "contradicted_claims": int(claim_by_status.get("CONTRADICTED", 0)),

        "review_issue_count": sum(
            int((e.get("metadata") or {}).get("issue_count", 0))
            for e in evs if e.get("event_type") == REVIEW_ISSUE
        ),
        "repair_rounds": repair_rounds,

        "rag_iterations": count(RAG_ITERATION),
        "multi_hop_iterations": count(MULTI_HOP_ITERATION),

        "termination_reason": judge.get("reason", "") or last_meta(RUN_END).get("reason", ""),
    }
