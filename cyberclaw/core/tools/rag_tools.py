from .base import cyberclaw_tool
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


@cyberclaw_tool
def search_knowledge_base(query: str) -> str:
    """在企业知识库中检索与用户问题相关的官方文档。
    当用户问及公司规定、内部文档、操作指南、报销流程、政策制度等需要依据资料回答的问题时，必须调用此工具。
    工具会做多查询扩写 + 向量/BM25 混合检索 + RRF 融合 + rerank 精排，返回最相关的文档片段与出处。

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


@cyberclaw_tool
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
