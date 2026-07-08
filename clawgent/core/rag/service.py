from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .. import config
from .vector_store import SimpleVectorStore
from .reliability import CircuitBreaker, DeadLetterQueue, llm_call_with_reliability


class KnowledgeBaseService:
    def __init__(self) -> None:
        self.upload_dir = Path(config.KB_UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.store = SimpleVectorStore()
        self.llm = ChatOpenAI(
            model=config.RAG_LLM_MODEL,
            api_key=config.RAG_LLM_API_KEY,
            base_url=config.RAG_LLM_BASE_URL,
            temperature=0.7,
        )
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
        chain = prompt | self.llm | StrOutputParser()

        def _call():
            raw = chain.invoke({"query": query}).strip()
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
            fn=_call,
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
        for q in all_queries:
            vec = self.store.search(q, top_k=search_top_k)
            bm25 = self.store.bm25_search(q, top_k=search_top_k)
            if vec:
                all_ranked_lists.append(vec)
            if bm25:
                all_ranked_lists.append(bm25)

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
        chain = prompt | self.llm | StrOutputParser()

        def _call():
            raw = chain.invoke({"query": query, "docs": docs_text}).strip()
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
            fn=_call,
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
        chain = prompt | self.llm | StrOutputParser()

        def _call():
            raw = chain.invoke({"query": query, "docs": docs_text}).strip()
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
            fn=_call,
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
        if not gate["sufficient"] and gate["rewrite_query"]:
            rewrite_q = gate["rewrite_query"]
            extra_docs = self.search(rewrite_q, top_k=search_top_k)
            if extra_docs:
                extra_reranked = self.rerank(query, extra_docs)  # 注意：仍用原始 query rerank
                seen = {d["chunk_id"] for d in critiqued}
                new_docs = [d for d in extra_reranked if d["chunk_id"] not in seen]
                critiqued = critiqued + new_docs
                critiqued.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)

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
        chain = prompt | self.llm | StrOutputParser()

        def _call():
            return chain.invoke({"sub_query": sub_query, "docs": docs_text}).strip()

        result = llm_call_with_reliability(
            method_name="compress_finding",
            circuit_breaker=self._cb_critique,  # 复用 critique 熔断器（同属检索后处理）
            dlq=self._dlq,
            fn=_call,
            fallback=f"针对「{sub_query}」检索到 {len(documents)} 条片段（压缩失败，保留原始）",
            query=sub_query,
            context={"sub_query": sub_query, "doc_count": len(documents)},
        )
        return result

    def _reason_next(self, original_query: str, scratchpad: dict) -> dict:
        """IRCoT 推理：看已有中间结论，判断信息是否足够，不够则生成下一个子问题。

        返回: {"sufficient": bool, "next_sub_question": str | None, "gap": str}
        """
        findings_text = "\n".join(f"- {f}" for f in scratchpad["intermediate_findings"]) or "（暂无）"
        asked_text = "\n".join(f"- {q}" for q in scratchpad["sub_questions_asked"]) or "（暂无）"

        prompt = ChatPromptTemplate.from_template(
            "你在对本地知识库做多跳推理问答。请判断当前已收集的中间结论是否足以回答原始问题。\n\n"
            "原始问题: {query}\n\n"
            "已问过的子问题:\n{asked}\n\n"
            "已收集的中间结论:\n{findings}\n\n"
            "判断规则:\n"
            "- 如果中间结论已足以完整回答原始问题，sufficient=true。\n"
            "- 如果还缺关键信息，sufficient=false，并给出下一个应检索的子问题"
            "（必须与已问过的不同，聚焦当前缺口）。\n"
            "- gap 字段用一句话说明还缺什么。\n\n"
            "严格只输出 JSON:\n"
            '{{"sufficient": false, "next_sub_question": "...", "gap": "..."}}'
        )
        chain = prompt | self.llm | StrOutputParser()

        def _call():
            raw = chain.invoke({
                "query": original_query,
                "asked": asked_text,
                "findings": findings_text,
            }).strip()
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                raise ValueError(f"JSON 解析失败: {raw[:200]}")
            data = json.loads(m.group(0))
            nxt = data.get("next_sub_question") or None
            return {
                "sufficient": bool(data.get("sufficient", False)),
                "next_sub_question": nxt,
                "gap": data.get("gap", ""),
            }

        # 降级：推理失败时默认「已足够」，终止循环（避免带着坏状态空转）
        fallback = {"sufficient": True, "next_sub_question": None, "gap": ""}
        result = llm_call_with_reliability(
            method_name="reason_next",
            circuit_breaker=self._cb_crag,  # 复用 crag 熔断器（同属充分性推理）
            dlq=self._dlq,
            fn=_call,
            fallback=fallback,
            query=original_query,
            context={"query": original_query, "iteration": len(scratchpad["sub_questions_asked"])},
        )
        return result if result is not None else fallback

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
        chain = prompt | self.llm | StrOutputParser()

        def _call():
            return chain.invoke({"query": original_query, "findings": findings_text}).strip()

        result = llm_call_with_reliability(
            method_name="synthesize",
            circuit_breaker=self._cb_crag,
            dlq=self._dlq,
            fn=_call,
            fallback="",  # 综合失败时返回空，由上层拼接原始结论兜底
            query=original_query,
            context={"query": original_query, "finding_count": len(scratchpad["intermediate_findings"])},
        )
        return result or findings_text  # 综合失败则直接返回中间结论列表

    def search_iterative(self, query: str, top_k: int | None = None) -> dict:
        """多轮推理检索：retrieve → reason → retrieve 循环，处理多跳复杂问题。

        流程:
          init scratchpad
          loop (最多 RAG_MAX_ITERS 轮):
            search_agentic(当前子问题)  → 复用单轮完整管线
            _compress_to_finding()      → 压成结论存 scratchpad
            _reason_next()              → 判断是否足够 / 生成下一子问题
            三重终止判断
          _synthesize()                 → 综合最终答案

        返回: {"answer": str, "findings": [...], "iterations": int, "sources": [...]}
        """
        max_iters = config.RAG_MAX_ITERS
        scratchpad = {
            "sub_questions_asked": [],
            "intermediate_findings": [],
            "open_gaps": [],
            "all_sources": [],
        }

        current_query = query
        prev_gap = None
        gap_unchanged_count = 0

        for iteration in range(max_iters):
            # ⑴ 单轮检索（复用现有完整管线）
            docs = self.search_agentic(current_query, top_k=top_k)
            scratchpad["sub_questions_asked"].append(current_query)

            # 收集来源
            for d in docs:
                name = d.get("document_name", "")
                if name and name not in scratchpad["all_sources"]:
                    scratchpad["all_sources"].append(name)

            # ⑵ 压缩成结论
            finding = self._compress_to_finding(current_query, docs)
            scratchpad["intermediate_findings"].append(finding)

            # ⑶ 推理下一步
            decision = self._reason_next(query, scratchpad)

            # ⑷ 三重终止判断
            if decision["sufficient"]:
                break
            if not decision["next_sub_question"]:
                break  # 提不出新子问题
            gap = decision.get("gap", "")
            if gap and gap == prev_gap:
                gap_unchanged_count += 1
                if gap_unchanged_count >= 1:  # 连续 2 轮缺口没变 → 原地打转，停
                    break
            else:
                gap_unchanged_count = 0
            prev_gap = gap
            scratchpad["open_gaps"].append(gap)
            current_query = decision["next_sub_question"]

        # ⑸ 综合最终答案
        answer = self._synthesize(query, scratchpad)

        return {
            "answer": answer,
            "findings": scratchpad["intermediate_findings"],
            "iterations": len(scratchpad["sub_questions_asked"]),
            "sources": scratchpad["all_sources"],
        }

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
            for doc in documents:
                doc.setdefault("rerank_score", 0.0)
            return documents
        except Exception as e:
            print(f"[RAG] Reranking 失败，回退向量分: {e}")
            for doc in documents:
                doc["rerank_score"] = doc.get("score", 0.0)
            return documents

    def stats(self) -> dict:
        return self.store.stats()
