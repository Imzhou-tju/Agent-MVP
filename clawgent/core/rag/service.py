from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .. import config
from ..audit import (
    MULTI_HOP_ITERATION,
    RAG_ITERATION,
    RAG_RETRIEVAL,
    RERANK,
    STATUS_DEGRADED,
    STATUS_FAILED,
    STATUS_SUCCESS,
    emit,
)
from .vector_store import SimpleVectorStore
from .reliability import CircuitBreaker, DeadLetterQueue, llm_call_with_reliability
from .retrieval_decision import (
    RetrievalDecision,
    STOP_SUFFICIENT,
    run_iterative_loop,
)


class KnowledgeBaseService:
    def __init__(self) -> None:
        self.upload_dir = Path(config.KB_UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.store = SimpleVectorStore()
        self.llm = ChatOpenAI(
            model=config.RAG_LLM_MODEL,
            api_key=config.RAG_LLM_API_KEY,
            base_url=config.RAG_LLM_BASE_URL,
            temperature=config.RAG_LLM_TEMPERATURE,
        )
        # JSON 强约束模型：response_format=json_object 从源头杜绝非 JSON 输出，
        # 用于所有需要解析结构化结果的内部判定调用（assess/critique/crag/reason）。
        _json_kwargs = {"response_format": {"type": "json_object"}}
        self.llm_json = ChatOpenAI(
            model=config.RAG_LLM_MODEL,
            api_key=config.RAG_LLM_API_KEY,
            base_url=config.RAG_LLM_BASE_URL,
            temperature=config.RAG_LLM_TEMPERATURE,
            model_kwargs=_json_kwargs,
        ) if config.RAG_LLM_FORCE_JSON else self.llm
        # 升级模型：主模型重试耗尽/熔断时才用一次的更强兜底，未配置则为 None（不启用）
        self.llm_escalation = ChatOpenAI(
            model=config.RAG_LLM_ESCALATION_MODEL,
            api_key=config.RAG_LLM_ESCALATION_API_KEY,
            base_url=config.RAG_LLM_ESCALATION_BASE_URL,
            temperature=config.RAG_LLM_ESCALATION_TEMPERATURE,
            model_kwargs=_json_kwargs if config.RAG_LLM_FORCE_JSON else {},
        ) if config.RAG_LLM_ESCALATION_MODEL else None
        # 纯文本升级模型：给散文类调用（压缩结论/综合答案）兜底，不加 json 约束
        self.llm_escalation_text = ChatOpenAI(
            model=config.RAG_LLM_ESCALATION_MODEL,
            api_key=config.RAG_LLM_ESCALATION_API_KEY,
            base_url=config.RAG_LLM_ESCALATION_BASE_URL,
            temperature=config.RAG_LLM_ESCALATION_TEMPERATURE,
        ) if config.RAG_LLM_ESCALATION_MODEL else None
        # 同义词词典可选：放 workspace/knowledge_base/synonyms.json
        self.synonyms: dict = {}
        synonyms_path = self.upload_dir / "synonyms.json"
        if synonyms_path.exists():
            try:
                self.synonyms = json.loads(synonyms_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[RAG] 加载同义词词典失败: {e}")

        # 方法级熔断器：各自独立，互不影响
        self._cb_assess = CircuitBreaker("assess_query", failure_threshold=3, recovery_timeout=60.0)
        self._cb_critique = CircuitBreaker("critique_docs", failure_threshold=3, recovery_timeout=60.0)
        self._cb_crag = CircuitBreaker("crag_gate", failure_threshold=3, recovery_timeout=60.0)

        import os
        dlq_path = os.path.join(config.KB_INDEX_DIR, "dlq.sqlite")
        self._dlq = DeadLetterQueue(db_path=dlq_path, retry_fn=None)  # 只持久化，不自动重试（同步链路）

    def _apply_synonym_expansion(self, query: str) -> list[str]:
        expanded = []
        for key, values in self.synonyms.items():
            if key in query:
                for val in values:
                    expanded.append(query.replace(key, val))
        return expanded

    def _assess_query(self, query: str) -> dict:
        """一次 LLM 调用判定查询形态，决定检索策略。

        证据依据：无条件多查询扩写/HyDE 在查询已具体时反而降指标（-4pp），
        且自主迭代改写会引入 query drift。因此仅在“不完整/复杂多跳”时才扩写或分解。

        返回: {"expand": bool, "decompose": bool, "sub_queries": [str, ...]}
        """
        prompt = ChatPromptTemplate.from_template(
            "你是检索策略分析器。判断下面这个用于知识库检索的查询属于哪种情况，并只输出一个 JSON。\n\n"
            "查询: {query}\n\n"
            "判断规则:\n"
            "- 如果查询已经具体、单一、指向明确（如“差旅住宿报销上限”），则不需要改写也不需要分解。\n"
            "- 如果查询表述模糊/不完整/口语化（如“那个怎么弄”），标记 expand=true。\n"
            "- 如果查询包含多个并列子方面或需要多跳（如“报销和请假流程分别是什么”“A和B哪个标准更高”），"
            "标记 decompose=true，并给出 2-4 个可独立检索的子问题。\n\n"
            "严格只输出如下 JSON，不要任何解释:\n"
            '{{"expand": false, "decompose": false, "sub_queries": []}}'
        )
        def _call(llm):
            raw = (prompt | llm | StrOutputParser()).invoke({"query": query}).strip()
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                raise ValueError(f"JSON 解析失败，原始输出: {raw[:200]}")
            data = json.loads(m.group(0))
            subs = [s for s in data.get("sub_queries", []) if isinstance(s, str) and s.strip()]
            return {
                "expand": bool(data.get("expand", False)),
                "decompose": bool(data.get("decompose", False)) and len(subs) >= 2,
                "sub_queries": subs[:4],
            }

        fallback = {"expand": False, "decompose": False, "sub_queries": []}
        result = llm_call_with_reliability(
            method_name="assess_query",
            circuit_breaker=self._cb_assess,
            dlq=self._dlq,
            fn=lambda: _call(self.llm_json),
            stronger_fn=(lambda: _call(self.llm_escalation)) if self.llm_escalation else None,
            fallback=fallback,
            query=query,
            context={"query": query},
        )
        return result if result is not None else fallback

    def generate_multi_queries(self, query: str, n: int = 3) -> list[str]:
        prompt = ChatPromptTemplate.from_template(
            "作为一位资深的知识管理专家，你的任务是基于用户的原始提问，生成 {n} 个意思相近但表述不同、侧重点不同的搜索查询语句。\n"
            "这有助于我们在知识库中进行多角度的检索，克服单次检索的局限性。\n\n"
            "原始问题: {query}\n\n"
            "请直接输出 {n} 个变体查询，每行一个，不要包含任何序号、前缀或额外的解释说明。"
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            response = chain.invoke({"query": query, "n": n})
            clean = []
            for line in response.strip().split('\n'):
                v = re.sub(r'^(\d+\.|-|\*)\s*', '', line.strip()).strip()
                if v:
                    clean.append(v)
            return clean[:n]
        except Exception as e:
            print(f"[RAG] 多查询生成失败: {e}")
            return []

    def rebuild_index(self) -> int:
        return self.store.add_documents_from_folder(str(self.upload_dir))

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        # 证据驱动的条件策略：默认只用原始 query（避免 query drift / 具体查询被扩写降分）；
        # 仅在评估为“不完整”时才多查询扩写，评估为“复杂多跳”时才分解为子问题各自检索。
        assessment = self._assess_query(query)

        retrieval_queries = [query]
        if assessment["decompose"]:
            # 分解扩大召回：子问题各自检索，汇入同一候选池；精度由外层对“原始 query”的 rerank 恢复。
            retrieval_queries += assessment["sub_queries"]
        if assessment["expand"]:
            retrieval_queries += self.generate_multi_queries(query, n=3)
        retrieval_queries += self._apply_synonym_expansion(query)
        all_queries = list(dict.fromkeys(q for q in retrieval_queries if q and q.strip()))

        search_top_k = top_k or config.RAG_TOP_K
        all_ranked_lists = []
        vector_count = 0
        bm25_count = 0
        for q in all_queries:
            vec = self.store.search(q, top_k=search_top_k)
            bm25 = self.store.bm25_search(q, top_k=search_top_k)
            if vec:
                all_ranked_lists.append(vec)
                vector_count += len(vec)
            if bm25:
                all_ranked_lists.append(bm25)
                bm25_count += len(bm25)

        rrf_k = 60
        merged: dict = {}
        for ranked_list in all_ranked_lists:
            for rank, res in enumerate(ranked_list):
                doc_id = res.get('chunk_id')
                if doc_id not in merged:
                    merged[doc_id] = res.copy()
                    merged[doc_id]['rrf_score'] = 0.0
                merged[doc_id]['rrf_score'] += 1.0 / (rrf_k + rank + 1)

        final_list = sorted(merged.values(), key=lambda x: x.get('rrf_score', 0), reverse=True)
        emit(RAG_RETRIEVAL, status=STATUS_SUCCESS, metadata={
            "query": query,
            "query_count": len(all_queries),
            "vector_result_count": vector_count,
            "bm25_result_count": bm25_count,
            "rrf_result_count": len(final_list),
        })
        return final_list[:search_top_k * 2]

    def _critique_docs(self, query: str, documents: list[dict]) -> list[dict]:
        """Self-RAG：LLM 批量判断每个 doc 与 query 的相关性，过滤无关片段。

        用 LLM 而非 rerank 分数做门控的理由：rerank 给相对排序，LLM critique
        给绝对判断（"这段话能否帮助回答这个问题"），两者互补。
        """
        if not documents:
            return []

        docs_text = "\n\n".join(
            f"[{i}] {doc['text'][:300]}" for i, doc in enumerate(documents)
        )
        prompt = ChatPromptTemplate.from_template(
            "你是一个检索质量评估器。判断以下每个文档片段对于回答用户问题是否有实质帮助。\n\n"
            "用户问题: {query}\n\n"
            "文档片段（格式: [索引] 内容）:\n{docs}\n\n"
            "对每个片段输出 JSON 数组，每项包含 index 和 relevant(bool)。\n"
            "严格只输出 JSON 数组，不要解释:\n"
            '[{{"index": 0, "relevant": true}}, ...]'
        )
        def _call(llm):
            raw = (prompt | llm | StrOutputParser()).invoke({"query": query, "docs": docs_text}).strip()
            m = re.search(r'\[.*\]', raw, re.DOTALL)
            if not m:
                raise ValueError(f"JSON 数组解析失败，原始输出: {raw[:200]}")
            results = json.loads(m.group(0))
            relevant_indices = {r["index"] for r in results if r.get("relevant", True)}
            filtered = [doc for i, doc in enumerate(documents) if i in relevant_indices]
            return filtered if filtered else documents[:2]

        result = llm_call_with_reliability(
            method_name="critique_docs",
            circuit_breaker=self._cb_critique,
            dlq=self._dlq,
            fn=lambda: _call(self.llm_json),
            stronger_fn=(lambda: _call(self.llm_escalation)) if self.llm_escalation else None,
            fallback=None,
            query=query,
            context={"query": query, "doc_count": len(documents)},
        )
        # fallback=None 表示 critique 失败，跳过过滤返回原始列表
        return result if result is not None else documents

    def _crag_gate(self, query: str, documents: list[dict]) -> dict:
        """CRAG 门控：LLM 评估当前检索结果是否足以回答问题。

        返回: {"sufficient": bool, "rewrite_query": str | None}
        sufficient=False 时提供改写后的 query 用于补充检索。
        """
        if not documents:
            return {"sufficient": False, "rewrite_query": query}

        docs_text = "\n\n".join(
            f"[{i+1}] {doc['text'][:400]}" for i, doc in enumerate(documents[:5])
        )
        prompt = ChatPromptTemplate.from_template(
            "你是一个检索充分性评估器。判断以下检索结果是否足以回答用户问题。\n\n"
            "用户问题: {query}\n\n"
            "已检索到的文档片段:\n{docs}\n\n"
            "判断规则:\n"
            "- 如果文档包含直接或间接回答问题所需的核心信息，标记 sufficient=true。\n"
            "- 如果文档与问题基本无关、或信息明显不完整，标记 sufficient=false，"
            "并给出一个更精准的改写查询 rewrite_query。\n\n"
            "严格只输出 JSON，不要解释:\n"
            '{{"sufficient": true, "rewrite_query": null}}'
        )
        def _call(llm):
            raw = (prompt | llm | StrOutputParser()).invoke({"query": query, "docs": docs_text}).strip()
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                raise ValueError(f"JSON 解析失败，原始输出: {raw[:200]}")
            data = json.loads(m.group(0))
            return {
                "sufficient": bool(data.get("sufficient", True)),
                "rewrite_query": data.get("rewrite_query") or None,
            }

        fallback = {"sufficient": True, "rewrite_query": None}
        result = llm_call_with_reliability(
            method_name="crag_gate",
            circuit_breaker=self._cb_crag,
            dlq=self._dlq,
            fn=lambda: _call(self.llm_json),
            stronger_fn=(lambda: _call(self.llm_escalation)) if self.llm_escalation else None,
            fallback=fallback,
            query=query,
            context={"query": query, "doc_count": len(documents)},
        )
        return result if result is not None else fallback

    def search_agentic(self, query: str, top_k: int | None = None) -> list[dict]:
        """Adaptive + Self-RAG + CRAG 三层检索管线。

        流程: Adaptive 策略决策 → 多路检索+RRF → rerank
              → Self-RAG critique 过滤 → CRAG 充分性评估
              → 不足时 query 改写补检索一轮（max 1 次，防 drift）
        """
        search_top_k = top_k or config.RAG_TOP_K

        # 第一轮：Adaptive 策略 + 多路检索（已有 search() 含 _assess_query）
        docs = self.search(query, top_k=search_top_k)
        if not docs:
            return []

        reranked = self.rerank(query, docs)
        reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        top_docs = reranked[:search_top_k * 2]

        # Self-RAG：批量 critique，过滤无关片段
        critiqued = self._critique_docs(query, top_docs)

        # CRAG 门控：评估是否足够，不足则改写补检索（最多 1 次）
        gate = self._crag_gate(query, critiqued)
        rewrite_query = ""
        if not gate["sufficient"] and gate["rewrite_query"]:
            rewrite_q = gate["rewrite_query"]
            extra_docs = self.search(rewrite_q, top_k=search_top_k)
            if extra_docs:
                extra_reranked = self.rerank(query, extra_docs)  # 注意：仍用原始 query rerank
                seen = {d["chunk_id"] for d in critiqued}
                new_docs = [d for d in extra_reranked if d["chunk_id"] not in seen]
                critiqued = critiqued + new_docs
                critiqued.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
                rewrite_query = rewrite_q

        emit(RAG_ITERATION, status=STATUS_SUCCESS, metadata={
            "query": query,
            "vector_result_count": sum(1 for d in top_docs if d.get("rerank_status") == "SUCCESS"),
            "rrf_result_count": len(top_docs),
            "rerank_result_count": len(top_docs),
            "rerank_fallback": any(d.get("rerank_status") == "FALLBACK" for d in top_docs),
            "llm_filter_result_count": len(critiqued),
            "sufficiency": bool(gate["sufficient"]),
            "query_rewrite": rewrite_query,
        })

        return critiqued

    # -----------------------------------------------------------------------
    # 多轮推理 RAG（IRCoT 式 retrieve → reason → retrieve 循环）
    # 复用 search_agentic() 作单轮检索原语；scratchpad 只存压缩结论防 context 爆炸。
    # -----------------------------------------------------------------------

    def _compress_to_finding(self, sub_query: str, documents: list[dict]) -> str:
        """把一轮检索结果压缩成 1-2 句带来源的结论，存入 scratchpad。

        关键：scratchpad 存压缩结论而非原始 chunk，避免多轮累积 context 爆炸。
        """
        if not documents:
            return f"（针对「{sub_query}」未检索到相关内容）"

        docs_text = "\n\n".join(
            f"[来源:{d.get('document_name','?')}] {d.get('text','')[:300]}"
            for d in documents[:4]
        )
        prompt = ChatPromptTemplate.from_template(
            "根据以下检索到的文档片段，针对子问题给出 1-2 句简洁结论，必须标注来源文档名。\n"
            "若片段无法回答该子问题，如实说明「未找到」。\n\n"
            "子问题: {sub_query}\n\n文档片段:\n{docs}\n\n"
            "只输出结论（含来源），不要解释过程:"
        )
        def _call(llm):
            return (prompt | llm | StrOutputParser()).invoke(
                {"sub_query": sub_query, "docs": docs_text}
            ).strip()

        result = llm_call_with_reliability(
            method_name="compress_finding",
            circuit_breaker=self._cb_critique,  # 复用 critique 熔断器（同属检索后处理）
            dlq=self._dlq,
            fn=lambda: _call(self.llm),
            stronger_fn=(lambda: _call(self.llm_escalation_text)) if self.llm_escalation_text else None,
            fallback=f"针对「{sub_query}」检索到 {len(documents)} 条片段（压缩失败，保留原始）",
            query=sub_query,
            context={"sub_query": sub_query, "doc_count": len(documents)},
        )
        return result

    def _reason_next(self, original_query: str, scratchpad: dict) -> RetrievalDecision:
        """IRCoT 推理：看已有中间结论，输出结构化的下一步检索决策。

        返回 RetrievalDecision（action / gap / search_intent / next_query /
        expected_evidence / stop_reason）。字段校验与终止判定交给
        retrieval_decision 模块，这里只负责拿到模型的原始 JSON。
        """
        findings_text = "\n".join(f"- {f}" for f in scratchpad["intermediate_findings"]) or "（暂无）"
        asked_text = "\n".join(f"- {q}" for q in scratchpad["sub_questions_asked"]) or "（暂无）"

        prompt = ChatPromptTemplate.from_template(
            "你正在进行多跳检索。\n\n"
            "根据：\n"
            "1) 原始问题\n"
            "2) 已获得的证据\n"
            "3) 当前 scratchpad 的中间结论\n"
            "4) 已经发现的信息缺口\n\n"
            "判断是否需要继续检索：\n"
            "- 已有证据足以回答原始问题 → action=STOP，并给出 stop_reason。\n"
            "- 仍存在明确且可检索的信息缺口 → action=CONTINUE，必须给出 gap、"
            "search_intent、一个具体的 next_query、最多 3 条 expected_evidence。\n"
            "- 禁止为了增加检索轮次而人为制造新的 gap；\n"
            "  没有明确可通过检索获得的新信息时应当 STOP。\n\n"
            "search_intent 只能取: FACT / MECHANISM / COMPARISON / RELATION / "
            "PERFORMANCE / CAUSE / LIMITATION / OTHER\n"
            "stop_reason 只能取: SUFFICIENT / NO_NEW_SUBQUESTION / NO_EVIDENCE_GAIN / "
            "REPEATED_GAP / MAX_ITERATIONS / INVALID_DECISION\n\n"
            "原始问题: {query}\n\n"
            "已问过的子问题:\n{asked}\n\n"
            "已收集的中间结论:\n{findings}\n\n"
            "严格只输出 JSON:\n"
            '{{"action": "STOP", "gap": "", "search_intent": "", "next_query": "",'
            ' "expected_evidence": [], "stop_reason": "SUFFICIENT"}}'
        )
        def _call(llm):
            raw = (prompt | llm | StrOutputParser()).invoke({
                "query": original_query,
                "asked": asked_text,
                "findings": findings_text,
            }).strip()
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                raise ValueError(f"JSON 解析失败: {raw[:200]}")
            return RetrievalDecision.from_dict(json.loads(m.group(0)))

        # 降级：推理失败时默认停止，终止循环（避免带着坏状态空转）
        fallback = RetrievalDecision(action="STOP", stop_reason=STOP_SUFFICIENT)
        result = llm_call_with_reliability(
            method_name="reason_next",
            circuit_breaker=self._cb_crag,  # 复用 crag 熔断器（同属充分性推理）
            dlq=self._dlq,
            fn=lambda: _call(self.llm_json),
            stronger_fn=(lambda: _call(self.llm_escalation)) if self.llm_escalation else None,
            fallback=fallback,
            query=original_query,
            context={"query": original_query, "iteration": len(scratchpad["sub_questions_asked"])},
        )
        if isinstance(result, RetrievalDecision):
            return result
        return RetrievalDecision.from_dict(result)

    def _synthesize(self, original_query: str, scratchpad: dict) -> str:
        """基于 scratchpad 全部中间结论，综合成最终答案（带推理链和来源）。"""
        findings_text = "\n".join(
            f"{i+1}. {f}" for i, f in enumerate(scratchpad["intermediate_findings"])
        ) or "（未收集到任何结论）"

        prompt = ChatPromptTemplate.from_template(
            "你是严谨的知识库问答助手。基于以下多轮检索得到的中间结论，综合回答原始问题。\n"
            "要求：结论必须有据可依、标注来源；若证据不足以回答，如实说明缺口，不要编造。\n\n"
            "原始问题: {query}\n\n"
            "多轮检索的中间结论:\n{findings}\n\n"
            "请给出结构化的最终回答（含关键依据与来源）:"
        )
        def _call(llm):
            return (prompt | llm | StrOutputParser()).invoke(
                {"query": original_query, "findings": findings_text}
            ).strip()

        result = llm_call_with_reliability(
            method_name="synthesize",
            circuit_breaker=self._cb_crag,
            dlq=self._dlq,
            fn=lambda: _call(self.llm),
            stronger_fn=(lambda: _call(self.llm_escalation_text)) if self.llm_escalation_text else None,
            fallback="",  # 综合失败时返回空，由上层拼接原始结论兜底
            query=original_query,
            context={"query": original_query, "finding_count": len(scratchpad["intermediate_findings"])},
        )
        return result or findings_text  # 综合失败则直接返回中间结论列表

    def search_iterative(self, query: str, top_k: int | None = None) -> dict:
        """多轮推理检索：retrieve → reason → retrieve 循环，处理多跳复杂问题。

        每轮顺序（见 retrieval_decision.run_iterative_loop）：
          检索 → 结论压缩 → 更新 evidence_state → 计算新证据数
          → 达 RAG_MAX_ITERS 停 → 无新证据停
          → 取 RetrievalDecision → 决策校验与终止判定 → 继续则换 next_query

        返回: {"answer": str, "findings": [...], "iterations": int, "sources": [...],
               "evidence_state": [...], "iteration_decisions": [...], "stop_reason": str}
        """
        return run_iterative_loop(
            query,
            max_iters=config.RAG_MAX_ITERS,
            retrieve_fn=lambda q: self.search_agentic(q, top_k=top_k),
            compress_fn=self._compress_to_finding,
            decide_fn=self._reason_next,
            synthesize_fn=self._synthesize,
            on_iteration=lambda it: emit(MULTI_HOP_ITERATION, status=STATUS_SUCCESS,
                                         metadata=dict(it)),
        )

    def rerank(self, query: str, documents: list[dict]) -> list[dict]:
        if not documents:
            return []

        import requests

        headers = {
            "Authorization": f"Bearer {config.RAG_RERANKER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.RAG_RERANKER_MODEL,
            "query": query,
            "documents": [doc["text"] for doc in documents],
            "return_documents": False,
        }
        try:
            resp = requests.post(config.RAG_RERANKER_BASE_URL, json=payload, headers=headers, timeout=15)
            resp.raise_for_status()
            for res in resp.json().get("results", []):
                idx = res["index"]
                if 0 <= idx < len(documents):
                    documents[idx]["rerank_score"] = float(res["relevance_score"])
                    documents[idx]["rerank_status"] = "SUCCESS"
            for doc in documents:
                doc.setdefault("rerank_score", 0.0)
                doc.setdefault("rerank_status", "SUCCESS")
            emit(RERANK, status=STATUS_SUCCESS, metadata={
                "query": query, "rerank_result_count": len(documents), "fallback": False})
            return documents
        except Exception as e:
            print(f"[RAG] Reranking 失败，回退向量分: {e}")
            for doc in documents:
                doc["rerank_score"] = doc.get("score", 0.0)
                doc["rerank_status"] = "FALLBACK"
            emit(RERANK, status=STATUS_DEGRADED, metadata={
                "query": query, "rerank_result_count": len(documents),
                "fallback": True, "error": str(e)[:200]})
            return documents

    def stats(self) -> dict:
        return self.store.stats()
