from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .. import config
from .vector_store import SimpleVectorStore


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
        try:
            raw = chain.invoke({"query": query}).strip()
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}
            subs = [s for s in data.get("sub_queries", []) if isinstance(s, str) and s.strip()]
            return {
                "expand": bool(data.get("expand", False)),
                "decompose": bool(data.get("decompose", False)) and len(subs) >= 2,
                "sub_queries": subs[:4],
            }
        except Exception as e:
            print(f"[RAG] 查询评估失败，回退直查: {e}")
            return {"expand": False, "decompose": False, "sub_queries": []}

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
