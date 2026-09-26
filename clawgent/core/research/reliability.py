"""Research 子图的统一 LLM 可靠性包装（§18）。

复用 rag.reliability 的 CircuitBreaker / DeadLetterQueue / llm_call_with_reliability，
把 planner / researcher / review / compiler / semantic 的模型调用统一收口到
reliable_llm_call：

    retry（默认 2 次）→ 升级模型兜底（可选）→ 熔断记录 → 死信入队 → 降级返回

确定性边界：
- LLM 只产出文本，可靠性包装只负责「调用能不能成功返回」，不负责「内容对不对」。
- 主模型彻底失败时返回 fallback（如空串），由调用方已有的解析兜底逻辑处理，
  绝不因为模型抖动而让整条研究链路崩溃。
- 不修改任何结论推导逻辑，仅包住「invoke」这一动作。
"""

from __future__ import annotations

import os
from typing import Any, Callable

from langchain_core.messages import HumanMessage

from ..rag.reliability import (
    CircuitBreaker,
    DeadLetterQueue,
    llm_call_with_reliability,
)

# 死信队列落盘位置：与 Ledger 落盘（logs/）保持一致，便于运维排查。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_DLQ_PATH = os.path.join(_PROJECT_ROOT, "logs", "research_dlq.sqlite")

_breakers: dict[str, CircuitBreaker] = {}
_dlq_path: str = _DEFAULT_DLQ_PATH
_dlq: DeadLetterQueue | None = None


def set_dlq_path(path: str) -> None:
    """重定向死信队列路径（测试用，避免污染项目目录）。"""
    global _dlq_path, _dlq
    _dlq_path = path
    _dlq = None


def get_breaker(method_name: str) -> CircuitBreaker:
    """按方法名复用熔断器的单例。"""
    cb = _breakers.get(method_name)
    if cb is None:
        cb = CircuitBreaker(name=method_name)
        _breakers[method_name] = cb
    return cb


def get_dlq() -> DeadLetterQueue:
    """懒加载共享死信队列（不挂后台重试线程，仅作持久化记录）。"""
    global _dlq
    if _dlq is None:
        _dlq = DeadLetterQueue(_dlq_path, retry_fn=None)
    return _dlq


def reliable_llm_call(
    method_name: str,
    prompt: str,
    *,
    llm: Any,
    fallback: Any = "",
    stronger_llm: Any | None = None,
    max_retries: int = 2,
    query: str = "",
    context: dict | None = None,
) -> Any:
    """统一的 LLM 可靠性调用。

    返回模型输出的文本（str）；熔断 / 重试耗尽 / 升级模型也失败则返回 fallback。
    """

    def _fn() -> Any:
        return llm.invoke([HumanMessage(content=prompt)]).content

    stronger_fn: Callable[[], Any] | None = None
    if stronger_llm is not None:
        def stronger_fn() -> Any:  # type: ignore[misc]
            return stronger_llm.invoke([HumanMessage(content=prompt)]).content

    return llm_call_with_reliability(
        method_name=method_name,
        circuit_breaker=get_breaker(method_name),
        dlq=get_dlq(),
        fn=_fn,
        fallback=fallback,
        query=query or prompt[:200],
        context=context or {},
        max_retries=max_retries,
        stronger_fn=stronger_fn,
    )


class ReliableLLM:
    """把 ChatOpenAI 适配成「走 reliability 包装」的调用对象。

    用在需要把 llm 当对象传进去、内部自己调 .invoke 的地方
    （如 ClaimEvidenceSemanticVerifier），避免散落的 try/except。
    """

    def __init__(self, llm: Any, method_name: str, fallback: Any = ""):
        self._llm = llm
        self._method = method_name
        self._fallback = fallback

    def invoke(self, messages: list, **kwargs: Any) -> Any:
        prompt = ""
        for m in reversed(messages):
            content = getattr(m, "content", None)
            if content:
                prompt = content
                break
        content = reliable_llm_call(
            self._method, prompt, llm=self._llm, fallback=self._fallback
        )
        return _Message(content)

    @property
    def model(self) -> str:
        return getattr(self._llm, "model", "")


class _Message:
    """最小 AIMessage 替身，只暴露 .content。"""

    def __init__(self, content: Any):
        self.content = content
