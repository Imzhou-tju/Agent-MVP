from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class CircuitState(Enum):
    CLOSED = "closed"      # 正常，允许调用
    OPEN = "open"          # 熔断，直接走降级
    HALF_OPEN = "half_open"  # 探测期，允许一次调用试探


@dataclass
class CircuitBreaker:
    """方法级熔断器。

    failure_threshold: 连续失败 N 次触发熔断
    recovery_timeout: OPEN 后等待 N 秒进入 HALF_OPEN 探测
    """
    name: str
    failure_threshold: int = 3
    recovery_timeout: float = 60.0

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def is_open(self) -> bool:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._opened_at >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    print(f"[CircuitBreaker:{self.name}] → HALF_OPEN，开始探测")
                    return False
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                print(f"[CircuitBreaker:{self.name}] → CLOSED，恢复正常")

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == CircuitState.HALF_OPEN or self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                print(f"[CircuitBreaker:{self.name}] → OPEN，连续失败 {self._failures} 次，熔断 {self.recovery_timeout}s")

    @property
    def state(self) -> CircuitState:
        return self._state


@dataclass
class DLQEntry:
    method: str
    query: str
    context: dict
    error: str
    created_at: float
    attempt: int = 0
    status: str = "pending"  # pending / resolved / failed


class DeadLetterQueue:
    """SQLite 持久化的死信队列 + 后台异步重试线程。

    后台线程每隔 retry_interval 秒消费一次 pending 条目，
    调用 retry_fn(entry) 重试；失败超过 max_attempts 次标记 failed。
    """

    def __init__(
        self,
        db_path: str,
        retry_fn: Callable[[DLQEntry], bool] | None = None,
        retry_interval: float = 30.0,
        max_attempts: int = 3,
    ) -> None:
        self.db_path = db_path
        self.retry_fn = retry_fn
        self.retry_interval = retry_interval
        self.max_attempts = max_attempts
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._init_db()
        if retry_fn:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dead_letters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    method TEXT NOT NULL,
                    query TEXT NOT NULL,
                    context TEXT NOT NULL,
                    error TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    attempt INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending'
                )
            """)
            conn.commit()

    def enqueue(self, method: str, query: str, context: dict, error: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO dead_letters (method, query, context, error, created_at) VALUES (?,?,?,?,?)",
                (method, query, json.dumps(context, ensure_ascii=False), error, time.time()),
            )
            conn.commit()
        print(f"[DLQ] 入队: method={method}, query={query[:50]}...")

    def _get_pending(self) -> list[tuple]:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT id, method, query, context, error, created_at, attempt "
                "FROM dead_letters WHERE status='pending' ORDER BY created_at LIMIT 10"
            ).fetchall()

    def _update(self, row_id: int, status: str, attempt: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE dead_letters SET status=?, attempt=? WHERE id=?",
                (status, attempt, row_id),
            )
            conn.commit()

    def _worker(self) -> None:
        while not self._stop.wait(self.retry_interval):
            rows = self._get_pending()
            for row in rows:
                row_id, method, query, context_str, error, created_at, attempt = row
                attempt += 1
                entry = DLQEntry(
                    method=method,
                    query=query,
                    context=json.loads(context_str),
                    error=error,
                    created_at=created_at,
                    attempt=attempt,
                )
                try:
                    success = self.retry_fn(entry)
                    if success:
                        self._update(row_id, "resolved", attempt)
                        print(f"[DLQ] 重试成功: id={row_id}, method={method}")
                    else:
                        status = "failed" if attempt >= self.max_attempts else "pending"
                        self._update(row_id, status, attempt)
                        if status == "failed":
                            print(f"[DLQ] 放弃重试: id={row_id}, method={method}, 已尝试 {attempt} 次")
                except Exception as e:
                    status = "failed" if attempt >= self.max_attempts else "pending"
                    self._update(row_id, status, attempt)
                    print(f"[DLQ] 重试异常: id={row_id}, error={e}")

    def stop(self) -> None:
        self._stop.set()

    def summary(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM dead_letters GROUP BY status"
            ).fetchall()
        return {r[0]: r[1] for r in rows}


def llm_call_with_reliability(
    method_name: str,
    circuit_breaker: CircuitBreaker,
    dlq: DeadLetterQueue,
    fn: Callable[[], Any],
    fallback: Any,
    query: str = "",
    context: dict | None = None,
    max_retries: int = 2,
) -> Any:
    """统一的可靠性包装器：retry → 熔断记录 → 死信入队 → 降级返回。

    fn 是无参数的 callable，内部 close over 了真正的参数。
    fallback 是降级返回值（如 None 或空 list）。
    """
    if circuit_breaker.is_open():
        print(f"[Reliability] {method_name} 熔断中，直接降级")
        return fallback

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            result = fn()
            circuit_breaker.record_success()
            return result
        except Exception as e:
            last_error = e
            print(f"[Reliability] {method_name} 第 {attempt}/{max_retries} 次失败: {e}")

    # 所有重试耗尽
    circuit_breaker.record_failure()
    dlq.enqueue(
        method=method_name,
        query=query,
        context=context or {},
        error=str(last_error),
    )
    return fallback
