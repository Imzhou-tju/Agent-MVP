from __future__ import annotations

import asyncio
import os
from typing import Any


_web_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """调研否定了 max_concurrency 参数的存在，用 Semaphore 自行限流。"""
    global _web_semaphore
    if _web_semaphore is None:
        max_concurrent = int(os.getenv("RESEARCH_MAX_CONCURRENT", "5"))
        _web_semaphore = asyncio.Semaphore(max_concurrent)
    return _web_semaphore


async def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Tavily 联网搜索。多 Agent 路径只支持 Tavily（调研已验证）。"""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return []

    async with _get_semaphore():
        try:
            from tavily import AsyncTavilyClient
            client = AsyncTavilyClient(api_key=api_key)
            resp = await client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_raw_content=False,
            )
            results = []
            for r in resp.get("results", []):
                results.append({
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "snippet": r.get("content", ""),
                    "search_query": query,
                    "source": "web",
                })
            return results
        except ImportError:
            print("[Research] tavily 未安装，跳过联网搜索。pip install tavily-python")
            return []
        except Exception as e:
            print(f"[Research] Tavily 搜索失败: {e}")
            return []


def rag_search(query: str, top_k: int = 4) -> list[dict]:
    """复用已有 RAG 检索（本地知识库）。"""
    try:
        from ..rag.service import KnowledgeBaseService
        kb = KnowledgeBaseService()
        docs = kb.search(query, top_k=top_k)
        if not docs:
            return []
        reranked = kb.rerank(query, docs)
        reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        results = []
        for d in reranked[:top_k]:
            results.append({
                "url": f"local://{d.get('document_name', 'unknown')}",
                "title": d.get("document_name", ""),
                "snippet": d.get("text", ""),
                "search_query": query,
                "source": "rag",
                "relevance": d.get("rerank_score", 0.0),
            })
        return results
    except Exception as e:
        print(f"[Research] RAG 检索失败: {e}")
        return []


async def hybrid_search(
    query: str,
    web_results: int = 5,
    rag_results: int = 3,
    academic_results: int = 5,
) -> list[dict]:
    """学术源(MCP) + 联网(Tavily) + 本地 RAG 混合检索，去重后合并。

    学术源经 research/academic.py 的 MCP 通道接入（arXiv 为主），
    未启用（ACADEMIC_MCP_ENABLED=false）时自动返回空，不影响其余通道。
    """
    from .academic import academic_search

    web_task = asyncio.create_task(tavily_search(query, max_results=web_results))
    academic_task = asyncio.create_task(academic_search(query, max_results=academic_results))
    # RAG 是同步的，在 executor 里跑
    loop = asyncio.get_event_loop()
    rag_task = loop.run_in_executor(None, rag_search, query, rag_results)

    academic, web, rag = await asyncio.gather(
        academic_task, web_task, rag_task, return_exceptions=True
    )
    academic = academic if isinstance(academic, list) else []
    web = web if isinstance(web, list) else []
    rag = rag if isinstance(rag, list) else []

    seen_urls: set[str] = set()
    merged = []
    # 学术源优先（科研定位：论文证据权重高于普通网页）
    for r in academic + web + rag:
        url = r.get("url", "") or (r.get("title", "") + r.get("snippet", "")[:40])
        if url and url not in seen_urls:
            seen_urls.add(url)
            merged.append(r)
    return merged
