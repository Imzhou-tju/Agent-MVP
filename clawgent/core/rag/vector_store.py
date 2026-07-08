from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from .. import config
from .text_utils import chunk_text, chunk_text_hierarchical


class SimpleVectorStore:
    """两级分层向量库：子切片集合用于检索，父块集合用于上下文回溯。

    child_chunks  (clawgent_knowledge)   — 细粒度，向量检索命中
    parent_chunks (clawgent_parents)     — 粗粒度，按 parent_chunk_id 回溯取完整上下文
    BM25 语料绑定子切片，与向量检索 RRF 融合后统一走父块回溯。
    """

    def __init__(self) -> None:
        self.index_dir = Path(config.KB_INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings = OpenAIEmbeddings(
            model=config.RAG_EMBEDDING_MODEL,
            api_key=config.RAG_EMBEDDING_API_KEY,
            base_url=config.RAG_EMBEDDING_BASE_URL,
            chunk_size=1,
            check_embedding_ctx_length=False,
        )

        # 子切片：细粒度，用于向量检索
        self.vector_store = Chroma(
            collection_name="clawgent_knowledge",
            embedding_function=self.embeddings,
            persist_directory=str(self.index_dir),
        )
        # 父块：粗粒度，仅按 ID 取，无需 embedding（存 raw 即可）
        self._parent_store = Chroma(
            collection_name="clawgent_parents",
            embedding_function=self.embeddings,
            persist_directory=str(self.index_dir),
        )

        self.bm25 = None
        self.bm25_docs: List[Document] = []
        self._reload_bm25_from_store()

    def _reload_bm25_from_store(self) -> None:
        """从子切片集合恢复 BM25 语料。"""
        try:
            data = self.vector_store._collection.get(include=["documents", "metadatas"])
            docs_text = data.get("documents", []) or []
            metas = data.get("metadatas", []) or []
            self.bm25_docs = [
                Document(page_content=t, metadata=m or {})
                for t, m in zip(docs_text, metas)
                if t and t.strip()
            ]
            if self.bm25_docs:
                tokenized = [list(jieba.cut(d.page_content)) for d in self.bm25_docs]
                self.bm25 = BM25Okapi(tokenized)
        except Exception as e:
            print(f"[RAG] BM25 语料恢复失败（首次使用可忽略）: {e}")

    def _fetch_parent(self, parent_chunk_id: str, doc_name: str, fallback_text: str) -> dict:
        """按 parent_chunk_id 从父块集合回溯，取不到时降级返回子切片文本。"""
        try:
            res = self._parent_store._collection.get(
                ids=[parent_chunk_id],
                include=["documents", "metadatas"],
            )
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []
            if docs and docs[0]:
                meta = metas[0] if metas else {}
                return {
                    "chunk_id": parent_chunk_id,
                    "document_name": meta.get("document_name", doc_name),
                    "text": meta.get("raw_text", docs[0]),
                    "score": 0.0,
                }
        except Exception:
            pass
        # 降级：父块取不到，返回子切片文本
        return {
            "chunk_id": parent_chunk_id,
            "document_name": doc_name,
            "text": fallback_text,
            "score": 0.0,
        }

    def rebuild_from_documents(self, documents: list[tuple[str, str]]) -> int:
        """重建两级索引：滚动处理每个文档，父块→子切片流式写入，控制内存峰值。"""
        # 清空两个集合
        self.vector_store.delete_collection()
        self._parent_store.delete_collection()
        self.vector_store = Chroma(
            collection_name="clawgent_knowledge",
            embedding_function=self.embeddings,
            persist_directory=str(self.index_dir),
        )
        self._parent_store = Chroma(
            collection_name="clawgent_parents",
            embedding_function=self.embeddings,
            persist_directory=str(self.index_dir),
        )

        all_child_docs: List[Document] = []
        total_children = 0

        for doc_name, content in documents:
            # 逐文档分层切分（滚动处理，不一次性展开全部文档）
            hierarchy = chunk_text_hierarchical(
                content,
                parent_size=config.RAG_PARENT_CHUNK_SIZE,
                parent_overlap=config.RAG_PARENT_CHUNK_OVERLAP,
                child_size=config.RAG_CHILD_CHUNK_SIZE,
                child_overlap=config.RAG_CHILD_CHUNK_OVERLAP,
                fallback_sizes=config.RAG_CHUNK_FALLBACK_SIZES,
            )

            parent_docs: List[Document] = []
            child_docs: List[Document] = []

            for h in hierarchy:
                p_id = h["parent_id"]
                parent_chunk_id = f"{doc_name}::parent_{p_id}"
                parent_text = h["parent_text"]

                # 父块：存入 parent_store（无需向量化，仅按 ID 取）
                parent_docs.append(Document(
                    page_content=f"[文档: {doc_name}]\n{parent_text}",
                    metadata={
                        "document_name": doc_name,
                        "chunk_id": parent_chunk_id,
                        "raw_text": parent_text,
                    },
                ))

                # 子切片：每个子切片记录其父块 ID，用于检索命中后回溯
                for c_idx, child_text in enumerate(h["children"]):
                    child_chunk_id = f"{doc_name}::parent_{p_id}::child_{c_idx}"
                    expanded = f"[文档: {doc_name}]\n{child_text}"
                    child_docs.append(Document(
                        page_content=expanded,
                        metadata={
                            "document_name": doc_name,
                            "parent_chunk_id": parent_chunk_id,
                            "chunk_index": c_idx,
                            "chunk_id": child_chunk_id,
                            "raw_text": child_text,
                        },
                    ))

            # 批量写入父块（不做向量化，用 add_texts 跳过 embedding）
            if parent_docs:
                self._parent_store._collection.add(
                    ids=[d.metadata["chunk_id"] for d in parent_docs],
                    documents=[d.page_content for d in parent_docs],
                    metadatas=[d.metadata for d in parent_docs],
                )

            all_child_docs.extend(child_docs)
            total_children += len(child_docs)

        valid_children = [d for d in all_child_docs if d.page_content.strip()]
        if valid_children:
            self.vector_store.add_documents(valid_children)
            self.bm25_docs = valid_children
            tokenized_corpus = [list(jieba.cut(d.page_content)) for d in valid_children]
            self.bm25 = BM25Okapi(tokenized_corpus)

        return total_children

    def add_documents_from_folder(self, folder: str) -> int:
        from .loader import DocumentLoader, SUPPORTED_EXTENSIONS

        loader = DocumentLoader()
        docs: list[tuple[str, str]] = []
        for path in sorted(Path(folder).glob('*')):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file():
                docs.append((path.name, loader.load(str(path))))
        return self.rebuild_from_documents(docs)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """子切片向量检索 → 回溯父块，返回父块文本。"""
        top_k = top_k or config.RAG_TOP_K
        try:
            results = self.vector_store.similarity_search_with_relevance_scores(query, k=top_k)
        except Exception:
            raw = self.vector_store.similarity_search_with_score(query, k=top_k)
            results = [(doc, max(0.0, 1.0 - score)) for doc, score in raw]

        seen_parents: set[str] = set()
        output = []
        for doc, score in results:
            meta = doc.metadata
            parent_chunk_id = meta.get("parent_chunk_id") or meta.get("chunk_id", "unknown")
            if parent_chunk_id in seen_parents:
                continue
            seen_parents.add(parent_chunk_id)
            doc_name = meta.get("document_name", "unknown")
            fallback = meta.get("raw_text", doc.page_content)
            parent = self._fetch_parent(parent_chunk_id, doc_name, fallback)
            parent["score"] = float(score)
            output.append(parent)
        return output

    def bm25_search(self, query: str, top_k: int | None = None) -> list[dict]:
        """BM25 子切片检索 → 回溯父块，去重后返回。"""
        top_k = top_k or config.RAG_TOP_K
        if not self.bm25 or not self.bm25_docs:
            return []

        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        seen_parents: set[str] = set()
        output = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0:
                continue
            doc = self.bm25_docs[idx]
            meta = doc.metadata
            parent_chunk_id = meta.get("parent_chunk_id") or meta.get("chunk_id", "unknown")
            if parent_chunk_id in seen_parents:
                continue
            seen_parents.add(parent_chunk_id)
            doc_name = meta.get("document_name", "unknown")
            fallback = meta.get("raw_text", doc.page_content)
            parent = self._fetch_parent(parent_chunk_id, doc_name, fallback)
            parent["score"] = float(score)
            output.append(parent)
        return output

    def stats(self) -> dict:
        try:
            collection = self.vector_store._collection
            count = collection.count()
            parent_count = self._parent_store._collection.count()
            results = collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", []) or []
            docs = sorted({m.get("document_name") for m in metadatas if m and "document_name" in m})
            return {
                'total_documents': len(docs),
                'total_child_chunks': count,
                'total_parent_chunks': parent_count,
                'documents': docs,
            }
        except Exception:
            return {'total_documents': 0, 'total_child_chunks': 0, 'total_parent_chunks': 0, 'documents': []}
