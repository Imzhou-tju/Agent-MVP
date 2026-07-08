from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup
import markdown

from .pdf_backends import PyPDF2Backend, get_pdf_backend
from .text_utils import normalize_text


SUPPORTED_EXTENSIONS = {'.txt', '.md', '.markdown', '.pdf'}


class DocumentLoader:
    def __init__(self) -> None:
        # 可插拔 PDF 后端（pypdf2 / mineru），由 config.RAG_PDF_BACKEND 决定
        self._pdf_backend = get_pdf_backend()

    def load(self, file_path: str) -> str:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f'Unsupported file type: {suffix}')
        if suffix == '.pdf':
            return self._load_pdf(path)
        if suffix in {'.md', '.markdown'}:
            return self._load_markdown(path)
        return normalize_text(path.read_text(encoding='utf-8', errors='ignore'))

    def _load_pdf(self, path: Path) -> str:
        try:
            return self._pdf_backend.parse(path)
        except Exception as e:
            # 任何后端失败（MinerU 超时/网络等）都降级本地 PyPDF2，绝不阻断索引
            print(f"[PDF] {type(self._pdf_backend).__name__} 解析失败，降级 PyPDF2: {e}")
            return PyPDF2Backend().parse(path)

    def _load_markdown(self, path: Path) -> str:
        raw = path.read_text(encoding='utf-8', errors='ignore')
        html = markdown.markdown(raw)
        text = BeautifulSoup(html, 'html.parser').get_text(separator='\n')
        return normalize_text(text)
