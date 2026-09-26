"""Evidence Layer：Source Registry、Evidence 与 Evidence Verifier。

职责边界：
- RAG 负责找到候选材料
- 本模块负责把候选材料登记为 Source，并把 LLM 抽出的 quote 绑定、验证到真实来源

LLM 不生成 url：所有 Evidence 必须引用 Source Registry 里已登记的 source_id。
"""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable

VERIFIED = "VERIFIED"
PARTIAL = "PARTIAL"
INVALID = "INVALID"
UNVERIFIED = "UNVERIFIED"

# 来源内容形态（§14）：full_text = 可获取完整/可定位原文；
# snippet = 仅检索摘要，无法对原文做引文定位校验。
FULL_TEXT = "full_text"
SEARCH_SNIPPET = "snippet"

# quote 与原文本完全命中即 VERIFIED；否则按最长公共片段占比判定 PARTIAL / INVALID
PARTIAL_THRESHOLD = 0.6
MIN_QUOTE_CHARS = 8
# 参与比对时原文的最大长度，避免超长文本拖慢 SequenceMatcher
MAX_COMPARE_CHARS = 4000


def normalize_text(text: str) -> str:
    """归一化：小写、去掉标点与空白，便于做原文定位比对。"""
    if not text:
        return ""
    lowered = text.lower()
    return re.sub(r"[\s\W_]+", "", lowered)


def stable_id(prefix: str, *parts: str) -> str:
    """由内容生成稳定 id。

    Researcher 通过 Send 并发执行，多个分支可能对同一来源/同一引文重复登记。
    用内容哈希做 id 可让重复登记收敛到同一个 id，从而在 reducer 里按 id 去重。
    """
    joined = "|".join(normalize_text(p) for p in parts if p)
    if not joined:
        return ""
    return prefix + hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]


@dataclass
class Source:
    source_id: str
    source_type: str = ""        # academic | web | local_kb
    title: str = ""
    authors: str = ""
    url: str = ""
    doi: str = ""
    publication_year: str = ""
    venue: str = ""
    provider: str = ""           # arxiv / semantic-scholar / pubmed / tavily / rag
    content_type: str = FULL_TEXT   # full_text | snippet（仅检索摘要，§14 边界）
    quality: str = ""               # 质量标签 high/medium/low，由 source_quality_of 推导
    retrieval_method: str = ""      # academic / web / local_kb / rag
    retrieved_at: str = ""          # 检索时间戳（ISO 8601），仅作元数据
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "title": self.title,
            "authors": self.authors,
            "url": self.url,
            "doi": self.doi,
            "publication_year": self.publication_year,
            "venue": self.venue,
            "provider": self.provider,
            "content_type": self.content_type,
            "quality": self.quality,
            "retrieval_method": self.retrieval_method,
            "retrieved_at": self.retrieved_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Source":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def dedupe_key(self) -> str:
        url = (self.url or "").strip().lower()
        if url:
            return f"url:{url}"
        doi = (self.doi or "").strip().lower()
        if doi:
            return f"doi:{doi}"
        return f"title:{self.title.strip().lower()}|{self.source_type}"


def source_quality_of(source: "Source") -> str:
    """确定性来源质量标签（§15）。

    - 仅检索摘要（snippet）：low（无法定位原文，证据最多 UNVERIFIED）
    - academic：high（同行评议，可定位原文）
    - local_kb / web：medium
    - 其它：low
    """
    if getattr(source, "content_type", "") == SEARCH_SNIPPET:
        return "low"
    st = source.source_type
    if st == "academic":
        return "high"
    if st in ("local_kb", "web"):
        return "medium"
    return "low"


class SourceRegistry:
    """所有正式来源的登记表。去重键：url > doi > title+source_type。"""

    def __init__(self, sources: Iterable[dict | Source] | None = None):
        self.by_id: dict[str, Source] = {}
        self.key_to_id: dict[str, str] = {}
        for s in sources or []:
            src = s if isinstance(s, Source) else Source.from_dict(s)
            self.upsert(src)

    def upsert(self, source: Source) -> Source:
        key = source.dedupe_key()
        existed = self.key_to_id.get(key)
        if existed:
            # 已登记过：保留原 source_id，只补齐空字段
            old = self.by_id[existed]
            for f in ("source_type", "title", "authors", "url", "doi",
                      "publication_year", "venue", "provider",
                      "content_type", "retrieval_method", "retrieved_at"):
                if not getattr(old, f) and getattr(source, f):
                    setattr(old, f, getattr(source, f))
            old.metadata.update(source.metadata or {})
            old.quality = source_quality_of(old)
            return old

        if not source.source_id:
            source.source_id = stable_id("S", key)
        source.quality = source_quality_of(source)
        self.by_id[source.source_id] = source
        self.key_to_id[key] = source.source_id
        return source

    def get(self, source_id: str) -> Source | None:
        return self.by_id.get(source_id)

    def has(self, source_id: str) -> bool:
        return source_id in self.by_id

    def to_dicts(self) -> list[dict]:
        return [s.to_dict() for s in self.by_id.values()]


@dataclass
class Evidence:
    evidence_id: str
    source_id: str
    document_id: str = ""
    chunk_id: str = ""
    quote: str = ""               # 必须来自真实来源，程序校验
    interpretation: str = ""      # 可由 LLM 生成
    locator: str = ""             # 形如 chunk:xxx / doc:yyy
    retrieval_query: str = ""
    retrieval_rank: int = 0
    source_type: str = ""
    task_id: str = ""
    verification_status: str = UNVERIFIED
    verification_reason: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "quote": self.quote,
            "interpretation": self.interpretation,
            "locator": self.locator,
            "retrieval_query": self.retrieval_query,
            "retrieval_rank": self.retrieval_rank,
            "source_type": self.source_type,
            "task_id": self.task_id,
            "verification_status": self.verification_status,
            "verification_reason": self.verification_reason,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Evidence":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


def make_evidence_id(source_id: str, quote: str, locator: str = "") -> str:
    """按来源 + 引文内容生成稳定 evidence_id。"""
    return stable_id("E", source_id, locator, quote)


class EvidenceVerifier:
    """quote 定位校验：确认 Evidence 的原文确实出现在来源文本中。"""

    def __init__(
        self,
        registry: SourceRegistry,
        partial_threshold: float = PARTIAL_THRESHOLD,
        min_quote_chars: int = MIN_QUOTE_CHARS,
    ):
        self.registry = registry
        self.partial_threshold = partial_threshold
        self.min_quote_chars = min_quote_chars

    def verify(self, ev: Evidence, source_text: str, known_chunk_ids: Iterable[str] | None = None) -> Evidence:
        status, reason = self.check(ev, source_text, known_chunk_ids)
        ev.verification_status = status
        ev.verification_reason = reason
        return ev

    def check(
        self,
        ev: Evidence,
        source_text: str,
        known_chunk_ids: Iterable[str] | None = None,
    ) -> tuple[str, str]:
        if not ev.source_id or not self.registry.has(ev.source_id):
            return INVALID, "source_id 未登记在 Source Registry"

        # §14 来源边界：仅检索摘要（snippet）的来源无法对原文做引文定位校验，
        # 即使摘要里恰好出现了 quote，也只能记为 UNVERIFIED，不能作为支撑依据。
        src = self.registry.get(ev.source_id)
        if src is not None and getattr(src, "content_type", "") == SEARCH_SNIPPET:
            return UNVERIFIED, "来源仅提供检索摘要，无法对原文做定位校验"

        if len((ev.quote or "").strip()) < self.min_quote_chars:
            return INVALID, "quote 过短或为空"
        if not source_text:
            return UNVERIFIED, "缺少来源原文，无法定位"

        locator_ok, locator_reason = self._check_locator(ev, known_chunk_ids)
        if not locator_ok:
            return INVALID, locator_reason

        norm_quote = normalize_text(ev.quote)
        norm_text = normalize_text(source_text)[:MAX_COMPARE_CHARS]
        if not norm_quote:
            return INVALID, "quote 归一化后为空"
        if norm_quote in norm_text:
            return VERIFIED, "quote 完整命中来源原文"

        ratio = self._match_ratio(norm_quote, norm_text)
        if ratio >= self.partial_threshold:
            return PARTIAL, f"quote 与来源原文部分匹配（占比 {ratio:.2f}）"
        return INVALID, f"quote 无法在来源原文中定位（占比 {ratio:.2f}）"

    @staticmethod
    def _check_locator(ev: Evidence, known_chunk_ids: Iterable[str] | None) -> tuple[bool, str]:
        locator = (ev.locator or "").strip()
        if not locator:
            return True, ""
        if locator.startswith("chunk:"):
            cid = locator.split(":", 1)[1]
            if cid != ev.chunk_id:
                return False, "locator 与 chunk_id 不一致"
            if known_chunk_ids is not None and ev.chunk_id not in set(known_chunk_ids):
                return False, "chunk 不在本次检索结果中"
            return True, ""
        if locator.startswith("doc:"):
            did = locator.split(":", 1)[1]
            if did != ev.document_id:
                return False, "locator 与 document_id 不一致"
            return True, ""
        return False, "locator 格式不支持（应为 chunk:xxx 或 doc:yyy）"

    @staticmethod
    def _match_ratio(norm_quote: str, norm_text: str) -> float:
        if not norm_text:
            return 0.0
        matcher = difflib.SequenceMatcher(None, norm_quote, norm_text, autojunk=False)
        block = matcher.find_longest_match(0, len(norm_quote), 0, len(norm_text))
        return block.size / len(norm_quote)

    def verify_all(
        self,
        evidences: Iterable[Evidence],
        texts_by_source: dict[str, str],
        known_chunk_ids: Iterable[str] | None = None,
    ) -> list[Evidence]:
        """批量校验。texts_by_source 以 source_id 为键，值为该来源可用于定位的原文。

        已经判定为 VERIFIED / INVALID 的证据不重复校验。
        注意 PARTIAL 会重跑：同一证据在拿到更完整的原文后可能升级为 VERIFIED。
        """
        out = []
        for ev in evidences:
            if ev.verification_status in (VERIFIED, INVALID):
                out.append(ev)
                continue
            out.append(self.verify(ev, texts_by_source.get(ev.source_id, ""), known_chunk_ids))
        return out
