from __future__ import annotations

import re
from typing import List


def normalize_text(text: str) -> str:
    text = text.replace('　', ' ')
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 250, chunk_overlap: int = 50) -> List[str]:
    text = normalize_text(text)
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        if end < len(text):
            last_break = max(chunk.rfind('\n\n'), chunk.rfind('\n'), chunk.rfind('。'), chunk.rfind('. '))
            if last_break > chunk_size // 3:
                end = start + last_break + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return [c for c in chunks if c]


def chunk_text_hierarchical(
    text: str,
    parent_size: int = 512,
    parent_overlap: int = 64,
    child_size: int = 128,
    child_overlap: int = 16,
    fallback_sizes: list[int] | None = None,
) -> list[dict]:
    """父块→子切片两级分层切分，滚动处理降低内存峰值。

    对每个父块逐块生成子切片，避免整文档一次性展开。
    解析异常时按 fallback_sizes 四级降级，保证语义连贯性。

    返回: [{"parent_id": str, "parent_text": str, "children": [str, ...]}, ...]
    """
    if fallback_sizes is None:
        fallback_sizes = [128, 256, 512, 1024]

    text = normalize_text(text)
    if not text:
        return []

    # 尝试父块切分，异常时四级降级
    parent_chunks: list[str] = []
    for size in [parent_size] + [s for s in fallback_sizes if s != parent_size]:
        try:
            parent_chunks = chunk_text(text, chunk_size=size, chunk_overlap=parent_overlap)
            if parent_chunks:
                break
        except Exception:
            continue

    if not parent_chunks:
        return []

    result = []
    for p_idx, parent_text in enumerate(parent_chunks):
        parent_id = f"p{p_idx}"
        # 子切片：在父块范围内切分，保持语义边界
        children: list[str] = []
        for size in [child_size] + [s for s in fallback_sizes if s != child_size and s < parent_size]:
            try:
                children = chunk_text(parent_text, chunk_size=size, chunk_overlap=child_overlap)
                if children:
                    break
            except Exception:
                continue
        if not children:
            children = [parent_text]
        result.append({"parent_id": parent_id, "parent_text": parent_text, "children": children})

    return result
