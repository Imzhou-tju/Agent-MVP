"""学术检索：原生 MCP 接入。

通过 langchain-mcp-adapters 连接多个学术数据源的 MCP server，
把它们的检索工具统一封装成 academic_search(query)->list[dict]，
输出结构与 research/search.py 的 web/rag 结果一致（url/title/snippet/source），
从而无缝汇入 deep_research 的证据链，被 Critic/Judge 溯源。

数据源（按 config 动态启用）：
- arxiv         : 前期主力，公开 MCP，无需 key（预印本，CS/物理/数学）
- semantic-scholar : 跨学科 + 引用图谱 + TLDR 摘要（可选 key 提额度）
- pubmed        : 生物医学（NCBI，可选 key）
- google-scholar: 覆盖最广，走爬虫，稳定性差（可选）

任何 server 连接失败都跳过，不阻断整体检索。敏感信息（key）仅从 .env 读取，不硬编码。
"""

from __future__ import annotations

import asyncio
import json

from .. import config

_mcp_client = None
_mcp_tools_cache: list | None = None
_client_lock = asyncio.Lock()


def _build_server_config() -> dict:
    """按 .env 组装 MCP server 连接配置。仅启用已具备运行条件的源。

    这里用 stdio 方式拉起社区 MCP server（需对应包已 pip 安装）；
    Google Scholar 若走远程 HTTP MCP，可在此追加 {"transport": "streamable_http", "url": ...}。
    """
    servers: dict = {}

    # arXiv —— 前期主力，无需 key
    servers["arxiv"] = {
        "command": "uvx",
        "args": ["arxiv-mcp-server"],
        "transport": "stdio",
    }

    # Semantic Scholar —— key 可选（有 key 提高速率上限）
    ss_env = {}
    if config.SEMANTIC_SCHOLAR_API_KEY:
        ss_env["SEMANTIC_SCHOLAR_API_KEY"] = config.SEMANTIC_SCHOLAR_API_KEY
    servers["semantic-scholar"] = {
        "command": "uvx",
        "args": ["semantic-scholar-mcp"],
        "transport": "stdio",
        "env": ss_env,
    }

    # PubMed —— 生物医学，key 可选
    if config.PUBMED_API_KEY:
        servers["pubmed"] = {
            "command": "uvx",
            "args": ["pubmed-mcp-server"],
            "transport": "stdio",
            "env": {"PUBMED_API_KEY": config.PUBMED_API_KEY},
        }

    return servers


async def _get_mcp_tools() -> list:
    """懒初始化 MCP 客户端并拉取全部工具，进程内缓存。"""
    global _mcp_client, _mcp_tools_cache
    if _mcp_tools_cache is not None:
        return _mcp_tools_cache

    async with _client_lock:
        if _mcp_tools_cache is not None:
            return _mcp_tools_cache
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError:
            print("[Academic] 未安装 langchain-mcp-adapters，学术检索不可用。"
                  "pip install langchain-mcp-adapters")
            _mcp_tools_cache = []
            return _mcp_tools_cache

        try:
            _mcp_client = MultiServerMCPClient(_build_server_config())
            _mcp_tools_cache = await _mcp_client.get_tools()
            names = ", ".join(t.name for t in _mcp_tools_cache)
            print(f"[Academic] MCP 学术工具已加载: {names or '无'}")
        except Exception as e:
            print(f"[Academic] MCP 客户端初始化失败: {e}")
            _mcp_tools_cache = []

    return _mcp_tools_cache


def _pick_search_tools(tools: list) -> list:
    """从 MCP 工具集中挑出'检索类'工具（按名字启发式匹配）。"""
    keywords = ("search", "query", "find", "paper", "article")
    picked = [t for t in tools if any(k in t.name.lower() for k in keywords)]
    return picked or tools  # 匹配不到就全用


def _normalize_result(raw, source: str, query: str) -> list[dict]:
    """把 MCP 工具的返回（字符串/JSON/list）归一化成统一 dict 结构。"""
    items: list = []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            items = parsed if isinstance(parsed, list) else parsed.get("results", parsed.get("papers", [parsed]))
        except (json.JSONDecodeError, AttributeError):
            # 非 JSON 文本，整体作为一条 snippet
            return [{"url": "", "title": f"{source} result", "snippet": raw[:1500],
                     "search_query": query, "source": source}]
    elif isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = raw.get("results", raw.get("papers", [raw]))

    out: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            out.append({"url": "", "title": "", "snippet": str(it)[:1500],
                        "search_query": query, "source": source})
            continue
        url = it.get("url") or it.get("pdf_url") or it.get("link") or it.get("doi", "")
        title = it.get("title", "")
        snippet = (it.get("abstract") or it.get("summary") or it.get("tldr")
                   or it.get("snippet") or it.get("content", ""))
        out.append({
            "url": str(url),
            "title": str(title),
            "snippet": str(snippet)[:1500],
            "search_query": query,
            "source": source,
            "authors": it.get("authors", ""),
            "year": it.get("year", ""),
        })
    return out


async def academic_search(query: str, max_results: int = 5) -> list[dict]:
    """并发调用所有已连的学术 MCP 检索工具，合并归一化后返回。"""
    if not config.ACADEMIC_MCP_ENABLED:
        return []

    tools = await _get_mcp_tools()
    if not tools:
        return []

    search_tools = _pick_search_tools(tools)

    async def _run(tool):
        try:
            # 多数学术 MCP 工具接受 {"query": ..., "max_results": ...}
            raw = await tool.ainvoke({"query": query, "max_results": max_results})
            source = tool.name.split("_")[0] if "_" in tool.name else tool.name
            return _normalize_result(raw, source=f"academic:{source}", query=query)
        except Exception as e:
            print(f"[Academic] 工具 {tool.name} 调用失败: {e}")
            return []

    gathered = await asyncio.gather(*[_run(t) for t in search_tools], return_exceptions=True)

    merged: list[dict] = []
    seen: set[str] = set()
    for res in gathered:
        if not isinstance(res, list):
            continue
        for r in res:
            key = r.get("url") or (r.get("title", "") + r.get("snippet", "")[:40])
            if key and key not in seen:
                seen.add(key)
                merged.append(r)
    return merged[:max_results * len(search_tools)]
