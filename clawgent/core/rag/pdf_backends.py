"""可插拔 PDF 解析后端。

设计：抽象 parse(path)->str 接口，按 config.RAG_PDF_BACKEND 选择实现。
- PyPDF2Backend：纯本地、零外部依赖，作为默认与最终降级。
- MinerUBackend：MinerU 官方 API，异步任务轮询，解析质量高（公式/表格/双栏），
  任何异常（缺 key / 超时 / 网络）都抛出，由上层 loader 降级回 PyPDF2，绝不阻断索引。
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import requests

from .. import config
from .text_utils import normalize_text


class PDFBackend:
    def parse(self, path: Path) -> str:
        raise NotImplementedError


class PyPDF2Backend(PDFBackend):
    def parse(self, path: Path) -> str:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(path))
        texts = [(page.extract_text() or "") for page in reader.pages]
        return normalize_text("\n".join(texts))


class MinerUBackend(PDFBackend):
    """MinerU 官方 API（v4）解析。

    流程：申请上传链接 → PUT 上传 PDF → 轮询任务状态 → 下载解析结果 zip → 抽取 markdown。
    MinerU 输出结构化 Markdown（保留标题/公式/表格），比 PyPDF2 的裸文本更适合学术 PDF。
    """

    def __init__(self) -> None:
        self.api_key = config.MINERU_API_KEY
        self.api_base = config.MINERU_API_BASE.rstrip("/")
        self.poll_timeout = config.MINERU_POLL_TIMEOUT
        if not self.api_key:
            raise RuntimeError("MINERU_API_KEY 未配置，无法使用 MinerU 后端")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def parse(self, path: Path) -> str:
        file_name = path.name

        # ① 申请上传链接（同时创建解析任务）
        resp = requests.post(
            f"{self.api_base}/file-urls/batch",
            headers=self._headers(),
            json={
                "enable_formula": True,
                "enable_table": True,
                "language": "auto",
                "files": [{"name": file_name, "is_ocr": True}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
        batch_id = data.get("batch_id")
        upload_urls = data.get("file_urls", [])
        if not batch_id or not upload_urls:
            raise RuntimeError(f"MinerU 申请上传链接失败: {resp.text[:200]}")

        # ② 上传 PDF 到预签名 URL（PUT，不带鉴权头）
        with open(path, "rb") as f:
            put_resp = requests.put(upload_urls[0], data=f, timeout=120)
        put_resp.raise_for_status()

        # ③ 轮询任务状态
        deadline = time.monotonic() + self.poll_timeout
        result_url = None
        while time.monotonic() < deadline:
            status_resp = requests.get(
                f"{self.api_base}/extract-results/batch/{batch_id}",
                headers=self._headers(),
                timeout=30,
            )
            status_resp.raise_for_status()
            results = status_resp.json().get("data", {}).get("extract_result", [])
            if results:
                state = results[0].get("state")
                if state == "done":
                    result_url = results[0].get("full_zip_url")
                    break
                if state == "failed":
                    raise RuntimeError(f"MinerU 解析失败: {results[0].get('err_msg', '未知错误')}")
            time.sleep(3)

        if not result_url:
            raise TimeoutError(f"MinerU 解析超时（>{self.poll_timeout}s）")

        # ④ 下载结果 zip，抽取 markdown
        zip_resp = requests.get(result_url, timeout=60)
        zip_resp.raise_for_status()
        return normalize_text(self._extract_markdown(zip_resp.content))

    @staticmethod
    def _extract_markdown(zip_bytes: bytes) -> str:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            md_names = [n for n in zf.namelist() if n.lower().endswith(".md")]
            if not md_names:
                raise RuntimeError("MinerU 结果 zip 中未找到 markdown 文件")
            # 优先取 full.md，否则第一个
            target = next((n for n in md_names if "full" in n.lower()), md_names[0])
            return zf.read(target).decode("utf-8", errors="ignore")


def get_pdf_backend() -> PDFBackend:
    """按配置返回后端；MinerU 初始化失败时降级回 PyPDF2。"""
    if config.RAG_PDF_BACKEND == "mineru":
        try:
            return MinerUBackend()
        except Exception as e:
            print(f"[PDF] MinerU 后端初始化失败，降级 PyPDF2: {e}")
    return PyPDF2Backend()
