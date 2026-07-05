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
