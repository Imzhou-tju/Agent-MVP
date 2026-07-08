import asyncio

from .base import clawgent_tool


@clawgent_tool
def search_academic(query: str, max_results: int = 5) -> str:
    """检索学术论文（arXiv / Semantic Scholar / PubMed 等，经 MCP 接入）。
    当用户要查找论文、文献、研究综述、某方向的最新进展、或需要可引用的学术来源时调用。
    与 search_knowledge_base（本地私有知识库）、deep_research（多智能体深度调研报告）的区别：
    本工具是【面向公开学术数据库的直接检索】，快速返回带标题/作者/年份/摘要/链接的论文列表。

    需在 .env 设置 ACADEMIC_MCP_ENABLED=true 并安装对应 MCP server（默认走 arXiv）。

    参数:
    query (str): 检索关键词或研究主题（支持自然语言）。
    max_results (int): 每个数据源返回的最大条数，默认 5。

    返回:
    论文列表（标题、作者、年份、摘要片段、来源与链接）；未启用或无命中时返回提示。
    """
    from ..research.academic import academic_search
    from .. import config

    if not config.ACADEMIC_MCP_ENABLED:
        return ("学术检索未启用。请在 .env 设置 ACADEMIC_MCP_ENABLED=true，"
                "并安装 langchain-mcp-adapters 及对应学术 MCP server（默认 arXiv）。")

    try:
        results = asyncio.run(academic_search(query, max_results=max_results))
    except RuntimeError:
        # 已在事件循环中（如被异步上下文调用），退回新线程跑
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as ex:
            results = ex.submit(
                lambda: asyncio.run(academic_search(query, max_results=max_results))
            ).result()
    except Exception as e:
        return f"学术检索失败：{e}（请检查 MCP server 是否安装、网络是否可用）。"

    if not results:
        return f"未检索到与「{query}」相关的论文（数据源可能未连接或无结果）。"

    lines = [f"【学术检索结果 · 共 {len(results)} 条】\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "无标题")
        authors = r.get("authors", "")
        year = r.get("year", "")
        source = r.get("source", "")
        url = r.get("url", "")
        snippet = r.get("snippet", "")[:400]
        meta = " · ".join(x for x in [str(authors)[:80], str(year), source] if x)
        lines.append(f"[{i}] {title}")
        if meta:
            lines.append(f"    {meta}")
        if url:
            lines.append(f"    链接: {url}")
        if snippet:
            lines.append(f"    摘要: {snippet}")
        lines.append("")
    return "\n".join(lines)
