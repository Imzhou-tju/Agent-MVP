from __future__ import annotations

from pathlib import Path

from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import markdown

from .text_utils import normalize_text


SUPPORTED_EXTENSIONS = {'.txt', '.md', '.markdown', '.pdf'}


class DocumentLoader:
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
        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages:
            texts.append(page.extract_text() or '')
        return normalize_text('\n'.join(texts))

    def _load_markdown(self, path: Path) -> str:
        raw = path.read_text(encoding='utf-8', errors='ignore')
        html = markdown.markdown(raw)
        text = BeautifulSoup(html, 'html.parser').get_text(separator='\n')
        return normalize_text(text)
