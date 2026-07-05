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
from .text_utils import chunk_text


class SimpleVectorStore:
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

        self.vector_store = Chroma(
            collection_name="cyberclaw_knowledge",
            embedding_function=self.embeddings,
            persist_directory=str(self.index_dir),
        )

        self.bm25 = None
        self.bm25_docs: List[Document] = []
        # 进程内重建 BM25：Chroma 持久化了向量，但 BM25 语料需运行时重载
        self._reload_bm25_from_store()

    def _reload_bm25_from_store(self) -> None:
        """从已持久化的 Chroma 集合恢复 BM25 语料，避免每次启动都要重建索引。"""
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

    def rebuild_from_documents(self, documents: list[tuple[str, str]]) -> int:
        self.vector_store.delete_collection()
        self.vector_store = Chroma(
            collection_name="cyberclaw_knowledge",
            embedding_function=self.embeddings,
            persist_directory=str(self.index_dir),
        )

        docs_to_add: List[Document] = []
        for doc_name, content in documents:
            chunks = chunk_text(
                content,
                chunk_size=config.RAG_CHUNK_SIZE,
                chunk_overlap=config.RAG_CHUNK_OVERLAP,
            )
            for idx, chunk in enumerate(chunks):
                # document expansion：把文档名前缀拼进被索引的正文，提升召回（经验证 ~+2pp）。
                # 原始 chunk 存 metadata.raw_text，检索返回时用它，避免前缀污染上下文。
                expanded = f"[文档: {doc_name}]\n{chunk}"
                metadata = {
                    "document_name": doc_name,
                    "chunk_index": idx,
                    "chunk_id": f"{doc_name}::chunk_{idx}",
                    "raw_text": chunk,
                }
                docs_to_add.append(Document(page_content=expanded, metadata=metadata))

        valid_docs = [d for d in docs_to_add if d.page_content.strip()]
        if not valid_docs:
            return 0

        self.vector_store.add_documents(valid_docs)

        self.bm25_docs = valid_docs
        tokenized_corpus = [list(jieba.cut(d.page_content)) for d in valid_docs]
        self.bm25 = BM25Okapi(tokenized_corpus)
        return len(valid_docs)

    def add_documents_from_folder(self, folder: str) -> int:
        from .loader import DocumentLoader, SUPPORTED_EXTENSIONS

        loader = DocumentLoader()
        docs: list[tuple[str, str]] = []
        for path in sorted(Path(folder).glob('*')):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file():
                docs.append((path.name, loader.load(str(path))))
        return self.rebuild_from_documents(docs)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or config.RAG_TOP_K
        try:
            results = self.vector_store.similarity_search_with_relevance_scores(query, k=top_k)
        except Exception:
            raw = self.vector_store.similarity_search_with_score(query, k=top_k)
            results = [(doc, max(0.0, 1.0 - score)) for doc, score in raw]

        return [
            {
                'chunk_id': doc.metadata.get('chunk_id', 'unknown'),
                'document_name': doc.metadata.get('document_name', 'unknown'),
                'text': doc.metadata.get('raw_text', doc.page_content),
                'score': float(score),
            }
            for doc, score in results
        ]

    def bm25_search(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or config.RAG_TOP_K
        if not self.bm25 or not self.bm25_docs:
            return []

        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        output = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0:
                continue
            doc = self.bm25_docs[idx]
            output.append({
                'chunk_id': doc.metadata.get('chunk_id', 'unknown'),
                'document_name': doc.metadata.get('document_name', 'unknown'),
                'text': doc.metadata.get('raw_text', doc.page_content),
                'score': float(score),
            })
        return output

    def stats(self) -> dict:
        try:
            collection = self.vector_store._collection
            count = collection.count()
            results = collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", []) or []
            docs = sorted({m.get("document_name") for m in metadatas if m and "document_name" in m})
            return {'total_documents': len(docs), 'total_chunks': count, 'documents': docs}
        except Exception:
            return {'total_documents': 0, 'total_chunks': 0, 'documents': []}
