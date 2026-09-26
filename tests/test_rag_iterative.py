"""多跳检索决策机制的离线测试（§多跳检索决策机制升级）。

全程不联网、不加载 langchain：循环编排在 retrieval_decision.run_iterative_loop，
依赖由回调注入，因此可以用假检索 / 假压缩 / 假决策跑完整 14 步顺序。

加载方式：rag 包的 __init__.py 会 import service（→ langchain_openai，本机未装），
故用 importlib 直接加载纯模块 retrieval_decision.py，与 tests/test_plan_gate.py 一致。
"""

import importlib.util
import os
import sys
import unittest
from types import ModuleType


def _load(name, path, parent=None):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if parent is not None:
        setattr(parent, name.rsplit(".", 1)[-1], mod)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_HERE = os.path.dirname(os.path.abspath(__file__))
_RAG = os.path.join(_HERE, "..", "clawgent", "core", "rag")

# 构造假父包，避免触发 rag/__init__.py → service → langchain
_pkg = ModuleType("clawgent.core.rag")
_pkg.__path__ = [_RAG]
sys.modules["clawgent.core.rag"] = _pkg
_core = ModuleType("clawgent.core")
_core.__path__ = [os.path.join(_HERE, "..", "clawgent", "core")]
sys.modules["clawgent.core"] = _core
_clawgent = ModuleType("clawgent")
_clawgent.__path__ = [os.path.join(_HERE, "..", "clawgent")]
sys.modules["clawgent"] = _clawgent
_clawgent.core = _core
_core.rag = _pkg

rd = _load("clawgent.core.rag.retrieval_decision",
           os.path.join(_RAG, "retrieval_decision.py"), _pkg)

ACTION_CONTINUE = rd.ACTION_CONTINUE
ACTION_STOP = rd.ACTION_STOP
MAX_EXPECTED_EVIDENCE = rd.MAX_EXPECTED_EVIDENCE
STOP_INVALID_DECISION = rd.STOP_INVALID_DECISION
STOP_MAX_ITERATIONS = rd.STOP_MAX_ITERATIONS
STOP_NO_EVIDENCE_GAIN = rd.STOP_NO_EVIDENCE_GAIN
STOP_NO_NEW_SUBQUESTION = rd.STOP_NO_NEW_SUBQUESTION
STOP_REPEATED_GAP = rd.STOP_REPEATED_GAP
STOP_SUFFICIENT = rd.STOP_SUFFICIENT
IterativeLoopState = rd.IterativeLoopState
RetrievalDecision = rd.RetrievalDecision
normalize_query = rd.normalize_query
run_iterative_loop = rd.run_iterative_loop


def _doc(chunk_id: str, name: str = "制度.txt") -> dict:
    return {"chunk_id": chunk_id, "document_name": name, "text": f"{chunk_id} 的正文"}


class _Script:
    """按轮次给出检索结果与决策。"""

    def __init__(self, docs_by_query: dict, decisions: list):
        self.docs_by_query = docs_by_query
        self.decisions = list(decisions)
        self.calls = 0

    def retrieve(self, query: str) -> list[dict]:
        return list(self.docs_by_query.get(query, []))

    @staticmethod
    def compress(query: str, docs: list[dict]) -> str:
        return f"结论:{query}"

    def decide(self, query: str, scratchpad: dict):
        idx = min(self.calls, len(self.decisions) - 1)
        self.calls += 1
        return self.decisions[idx]

    @staticmethod
    def synthesize(query: str, scratchpad: dict) -> str:
        return "最终答案"


def _run(docs_by_query: dict, decisions: list, max_iters: int = 4) -> dict:
    s = _Script(docs_by_query, decisions)
    return run_iterative_loop(
        "原始问题",
        max_iters=max_iters,
        retrieve_fn=s.retrieve,
        compress_fn=s.compress,
        decide_fn=s.decide,
        synthesize_fn=s.synthesize,
    )


class TestDecisionParsing(unittest.TestCase):
    def test_continue_keeps_fields(self):
        d = RetrievalDecision.from_dict({
            "action": "CONTINUE", "gap": "缺机制", "search_intent": "MECHANISM",
            "next_query": "X 的原理", "expected_evidence": ["结构图", "流程"],
        })
        self.assertEqual(d.action, ACTION_CONTINUE)
        self.assertEqual(d.search_intent, "MECHANISM")
        self.assertEqual(d.next_query, "X 的原理")
        self.assertEqual(len(d.expected_evidence), 2)

    def test_unknown_intent_falls_back_to_other(self):
        d = RetrievalDecision.from_dict(
            {"action": "CONTINUE", "search_intent": "NOT_A_TYPE", "next_query": "q"})
        self.assertEqual(d.search_intent, "OTHER")

    def test_expected_evidence_truncated(self):
        d = RetrievalDecision.from_dict(
            {"action": "CONTINUE", "next_query": "q",
             "expected_evidence": ["a", "b", "c", "d", "e"]})
        self.assertEqual(len(d.expected_evidence), MAX_EXPECTED_EVIDENCE)

    def test_continue_without_next_query_is_invalid(self):
        d = RetrievalDecision.from_dict({"action": "CONTINUE", "next_query": "  "})
        self.assertEqual(d.action, ACTION_STOP)
        self.assertEqual(d.stop_reason, STOP_INVALID_DECISION)
        self.assertEqual(d.next_query, "")

    def test_stop_ignores_next_query_and_expected(self):
        d = RetrievalDecision.from_dict({
            "action": "STOP", "next_query": "C", "expected_evidence": ["x"]})
        self.assertEqual(d.action, ACTION_STOP)
        self.assertEqual(d.next_query, "")
        self.assertEqual(d.expected_evidence, [])
        self.assertEqual(d.stop_reason, STOP_SUFFICIENT)

    def test_garbage_input_is_stop(self):
        self.assertEqual(RetrievalDecision.from_dict(None).action, ACTION_STOP)
        self.assertEqual(RetrievalDecision.from_dict("不是 dict").action, ACTION_STOP)


class TestLoopControl(unittest.TestCase):
    def test_normal_two_hops(self):
        # Round1 CONTINUE → Round2 STOP
        r = _run(
            {"原始问题": [_doc("c1")], "X 的原理": [_doc("c2")]},
            [{"action": "CONTINUE", "gap": "缺机制", "search_intent": "MECHANISM",
              "next_query": "X 的原理", "expected_evidence": ["原理图"]},
             {"action": "STOP", "stop_reason": "SUFFICIENT"}],
        )
        self.assertEqual(r["iterations"], 2)
        self.assertEqual(r["stop_reason"], STOP_SUFFICIENT)
        self.assertEqual(r["answer"], "最终答案")
        self.assertEqual(len(r["findings"]), 2)
        self.assertEqual([e["evidence_id"] for e in r["evidence_state"]], ["c1", "c2"])
        self.assertEqual(r["iteration_decisions"][0]["search_intent"], "MECHANISM")
        self.assertEqual(r["iteration_decisions"][0]["new_evidence_count"], 1)
        self.assertEqual(r["iteration_decisions"][1]["stop_reason"], STOP_SUFFICIENT)

    def test_repeated_query(self):
        # Round2 给出的 next_query 已经执行过
        r = _run(
            {"原始问题": [_doc("c1")], "B": [_doc("c2")], "原始问题2": []},
            [{"action": "CONTINUE", "gap": "缺 A", "next_query": "B"},
             {"action": "CONTINUE", "gap": "缺 B", "next_query": "原始问题"}],
        )
        self.assertEqual(r["iterations"], 2)
        self.assertEqual(r["stop_reason"], STOP_NO_NEW_SUBQUESTION)

    def test_repeated_gap(self):
        r = _run(
            {"原始问题": [_doc("c1")], "B": [_doc("c2")], "C": [_doc("c3")]},
            [{"action": "CONTINUE", "gap": "缺 X", "next_query": "B"},
             {"action": "CONTINUE", "gap": "缺 X", "next_query": "C"}],
        )
        self.assertEqual(r["iterations"], 2)
        self.assertEqual(r["stop_reason"], STOP_REPEATED_GAP)

    def test_no_evidence_gain(self):
        # Round2 检索到的还是同一个 chunk
        r = _run(
            {"原始问题": [_doc("c1")], "B": [_doc("c1")]},
            [{"action": "CONTINUE", "gap": "缺 B", "next_query": "B"}],
        )
        self.assertEqual(r["iterations"], 2)
        self.assertEqual(r["stop_reason"], STOP_NO_EVIDENCE_GAIN)
        self.assertEqual(r["iteration_decisions"][-1]["new_evidence_count"], 0)
        # evidence_state 不重复登记同一条证据
        self.assertEqual(len(r["evidence_state"]), 1)

    def test_max_iterations_is_hard_limit(self):
        docs = {f"q{i}": [_doc(f"c{i}")] for i in range(6)}
        decisions = [{"action": "CONTINUE", "gap": f"缺{i}", "next_query": f"q{i}"}
                     for i in range(1, 6)]
        r = _run({"原始问题": [_doc("c0")], **docs}, decisions, max_iters=4)
        self.assertEqual(r["iterations"], 4)
        self.assertEqual(r["stop_reason"], STOP_MAX_ITERATIONS)

    def test_invalid_decision(self):
        r = _run({"原始问题": [_doc("c1")]},
                 [{"action": "CONTINUE", "next_query": ""}])
        self.assertEqual(r["iterations"], 1)
        self.assertEqual(r["stop_reason"], STOP_INVALID_DECISION)

    def test_stop_with_next_query_does_not_continue(self):
        r = _run({"原始问题": [_doc("c1")], "C": [_doc("c9")]},
                 [{"action": "STOP", "next_query": "C", "expected_evidence": ["x"]}])
        self.assertEqual(r["iterations"], 1)
        self.assertEqual(r["stop_reason"], STOP_SUFFICIENT)
        self.assertNotIn("C", r["sources"])


class TestLoopState(unittest.TestCase):
    def test_query_normalization(self):
        self.assertEqual(normalize_query("  X-Vector "), "x-vector")
        loop = IterativeLoopState()
        loop.note_query("  X-Vector ")
        self.assertIn("x-vector", loop.seen_queries)

    def test_evidence_dedup_by_id(self):
        loop = IterativeLoopState()
        self.assertEqual(loop.register_evidence([_doc("c1"), _doc("c1"), _doc("c2")]), 2)
        self.assertEqual(loop.register_evidence([_doc("c1")]), 0)


if __name__ == "__main__":
    unittest.main()
