from .base import clawgent_tool
from ..config import RAG_INITIAL_TOP_K, RAG_TOP_K, KB_UPLOAD_DIR

# 懒初始化：KnowledgeBaseService 的构造会建立 Chroma / embedding 客户端，
# 放到首次调用时再做，避免启动阶段就依赖远程 API 与索引可用。
_kb_service = None


def _get_kb():
    global _kb_service
    if _kb_service is None:
        from ..rag.service import KnowledgeBaseService
        _kb_service = KnowledgeBaseService()
    return _kb_service


@clawgent_tool
def search_knowledge_base(query: str) -> str:
    """在企业知识库中做【单轮快速检索】，适合单点事实查询。
    当用户问及公司规定、内部文档、报销流程、政策制度等，且问题指向单一明确、一次检索即可回答时，调用此工具。
    例如「差旅住宿报销上限是多少」「年假有几天」这类单点问题。
    工具会做多查询扩写 + 向量/BM25 混合检索 + RRF 融合 + rerank 精排，返回最相关的文档片段与出处。

    注意：若问题需要串联多份文档、多步条件推理才能回答，请改用 deep_query_knowledge_base（多轮深度检索）。

    参数:
    query (str): 用于检索的关键词或自然语言问句。

    返回:
    多个相关文档片段的合并字符串，带出处与相关性得分；若知识库为空或未命中则返回提示。
    """
    kb = _get_kb()
    try:
        top_docs = kb.search_agentic(query, top_k=RAG_INITIAL_TOP_K)
    except Exception as e:
        return f"知识库检索失败：{e}（常见原因：embedding API key 无效或余额不足，请检查 RAG_EMBEDDING_API_KEY）。"
    if not top_docs:
        return "没有在知识库中找到相关文档（知识库可能为空，请先调用 rebuild_knowledge_index 建立索引）。"
    top_docs = top_docs[:RAG_TOP_K]
    if not top_docs:
        return "没有在知识库中找到相关文档。"

    return '\n\n'.join(
        f"[来源: {r.get('document_name', '未知')} | 相关性: {r.get('rerank_score', 0):.4f}]\n{r.get('text', '')}"
        for r in top_docs
    )


@clawgent_tool
def deep_query_knowledge_base(query: str) -> str:
    """对企业知识库做【多轮推理式深度检索】，用于需要多跳推理、综合多份文档才能回答的复杂问题。
    与 search_knowledge_base 的区别：
    - search_knowledge_base：单轮快查，适合「XX 报销上限是多少」这类单点事实查询。
    - deep_query_knowledge_base：多轮 retrieve→reason→retrieve 循环，适合需要串联多个信息点的问题，
      例如「根据请假制度和考勤制度，连续病假超过多少天会影响全勤奖」——需要先查病假规则、再查全勤判定、最后综合。

    何时选择本工具：问题包含多个子方面、需要条件推理、需要跨多份文档综合、或单轮查询容易漏信息时。
    工具会自动拆解推理步骤、逐步补充检索、累积中间结论，最终给出带推理依据和来源的综合答案。

    参数:
    query (str): 用户的复杂问题（自然语言）。

    返回:
    综合回答 + 推理经过的子问题 + 来源文档列表。
    """
    kb = _get_kb()
    try:
        result = kb.search_iterative(query, top_k=RAG_INITIAL_TOP_K)
    except Exception as e:
        return f"深度检索失败：{e}（常见原因：embedding/LLM API key 无效或余额不足）。"

    answer = result.get("answer", "")
    if not answer:
        return "没有在知识库中找到足以回答该问题的内容（知识库可能为空，请先 rebuild_knowledge_index）。"

    iterations = result.get("iterations", 0)
    sources = result.get("sources", [])
    findings = result.get("findings", [])

    parts = [f"【多轮深度检索完成 · 共 {iterations} 轮推理】\n\n{answer}"]
    if findings:
        parts.append("\n---\n推理经过：")
        for i, f in enumerate(findings, 1):
            parts.append(f"  第{i}轮：{f}")
    if sources:
        parts.append(f"\n涉及来源：{', '.join(sources)}")
    return "\n".join(parts)


@clawgent_tool
def rebuild_knowledge_index() -> str:
    """重建企业知识库的向量索引。
    当用户向知识库目录新增、修改或删除了 txt/md/pdf 文档后，
    调用此工具重新扫描该目录并全量重建索引。首次使用知识库前也需先调用一次。

    返回:
    重建结果说明，包含索引到的文档片段数量。
    """
    kb = _get_kb()
    try:
        count = kb.rebuild_index()
    except Exception as e:
        return f"索引重建失败：{e}（常见原因：embedding API key 无效或余额不足，请检查 RAG_EMBEDDING_API_KEY）。"
    if count == 0:
        return f"知识库目录 {KB_UPLOAD_DIR} 中没有可索引的文档（支持 txt/md/pdf）。请先放入文档再重建。"
    stats = kb.stats()
    return f"索引重建完成：共 {stats.get('total_documents', 0)} 个文档、{count} 个片段已入库。"
