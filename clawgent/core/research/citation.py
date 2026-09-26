"""Citation Layer：来源编号目录、引用校验与引用渲染。

职责边界：
- LLM 负责在报告中插入引用标记
- 本模块负责：把标记解析回真实的 Source / Evidence，校验它们是否可用，
  并渲染参考来源列表

目录（编号 → source_id / evidence_id）由程序建立，不交给模型记忆，
因此模型只需要使用 [S1] / [E3] 这类短标记，不需要抄写 URL。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .claim import normalize_claim_text

# 报告里的引用标记：[S1] 指向来源，[E3] 指向具体证据
CITATION_PATTERN = re.compile(r"\[(?P<kind>[SE])(?P<num>\d+)\]")

UNKNOWN_REF = "unknown_ref"
UNVERIFIED_EVIDENCE = "unverified_evidence"
UNCITED_CLAIM = "uncited_claim"
SOURCE_MISSING = "source_missing"
EVIDENCE_ORPHAN = "evidence_orphan"
CLAIM_MISSING = "claim_missing"

# 允许进入报告的证据校验状态
CITABLE_STATUSES = ("VERIFIED", "PARTIAL")


@dataclass
class CitationIssue:
    issue_type: str
    ref: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        return {"issue_type": self.issue_type, "ref": self.ref, "detail": self.detail}


@dataclass
class CitationRef:
    raw: str
    kind: str            # "S" | "E"
    number: int
    target_id: str = ""  # 解析后的 source_id / evidence_id，解析失败为空
    resolved: bool = False

    def to_dict(self) -> dict:
        return {
            "raw": self.raw, "kind": self.kind, "number": self.number,
            "target_id": self.target_id, "resolved": self.resolved,
        }


class SourceCatalog:
    """来源编号目录：把 source_id / evidence_id 映射为报告里的短编号。"""

    def __init__(self, sources: Iterable[dict] | None = None,
                 evidences: Iterable[dict] | None = None):
        self.source_number: dict[str, int] = {}
        self.number_source: dict[int, str] = {}
        self.evidence_number: dict[str, int] = {}
        self.number_evidence: dict[int, str] = {}
        self.sources: dict[str, dict] = {}
        self.evidences: dict[str, dict] = {}

        for s in sources or []:
            sid = s.get("source_id", "")
            if not sid or sid in self.source_number:
                continue
            n = len(self.source_number) + 1
            self.source_number[sid] = n
            self.number_source[n] = sid
            self.sources[sid] = s

        for e in evidences or []:
            eid = e.get("evidence_id", "")
            if not eid or eid in self.evidence_number:
                continue
            n = len(self.evidence_number) + 1
            self.evidence_number[eid] = n
            self.number_evidence[n] = eid
            self.evidences[eid] = e

    # ------------------------------------------------------------------
    # 编号查询
    # ------------------------------------------------------------------

    def source_marker(self, source_id: str) -> str:
        n = self.source_number.get(source_id)
        return f"[S{n}]" if n else ""

    def evidence_marker(self, evidence_id: str) -> str:
        n = self.evidence_number.get(evidence_id)
        return f"[E{n}]" if n else ""

    def resolve(self, kind: str, number: int) -> str:
        if kind == "S":
            return self.number_source.get(number, "")
        return self.number_evidence.get(number, "")

    def parse(self, text: str) -> list[CitationRef]:
        refs: list[CitationRef] = []
        for m in CITATION_PATTERN.finditer(text or ""):
            kind = m.group("kind")
            num = int(m.group("num"))
            target = self.resolve(kind, num)
            refs.append(CitationRef(
                raw=m.group(0), kind=kind, number=num,
                target_id=target, resolved=bool(target),
            ))
        return refs

    # ------------------------------------------------------------------
    # 给模型的目录文本
    # ------------------------------------------------------------------

    def render_for_prompt(self, claims: Iterable[dict] | None = None,
                          max_quote_chars: int = 220) -> str:
        """生成"可用引用目录"，模型只能引用目录里出现的编号。"""
        lines: list[str] = ["## 可用来源"]
        for sid, n in sorted(self.source_number.items(), key=lambda kv: kv[1]):
            s = self.sources.get(sid, {})
            meta = []
            if s.get("publication_year"):
                meta.append(str(s["publication_year"]))
            if s.get("venue"):
                meta.append(s["venue"])
            if s.get("source_type"):
                meta.append(s["source_type"])
            suffix = f"（{'，'.join(meta)}）" if meta else ""
            lines.append(f"[S{n}] {s.get('title', '')}{suffix} — {s.get('url', '')}")

        lines.append("")
        lines.append("## 可用证据")
        for eid, n in sorted(self.evidence_number.items(), key=lambda kv: kv[1]):
            e = self.evidences.get(eid, {})
            quote = (e.get("quote", "") or "")[:max_quote_chars]
            sm = self.source_marker(e.get("source_id", ""))
            lines.append(
                f"[E{n}] {sm} 原文：{quote}"
                + (f"｜解读：{e['interpretation'][:80]}" if e.get("interpretation") else "")
            )

        if claims:
            lines.append("")
            lines.append("## 可用声明（每条必须带引用）")
            for c in claims:
                marks = " ".join(
                    self.evidence_marker(eid) for eid in c.get("evidence_ids", [])
                    if self.evidence_marker(eid)
                )
                scope = f"（适用范围：{c['scope']}）" if c.get("scope") else ""
                lines.append(f"- {c.get('text','')}{scope} {marks}".rstrip())

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 参考来源渲染
    # ------------------------------------------------------------------

    def render_references(self, used_only: bool = True,
                          used_source_ids: Iterable[str] | None = None) -> str:
        used = set(used_source_ids or [])
        lines: list[str] = []
        for sid, n in sorted(self.source_number.items(), key=lambda kv: kv[1]):
            if used_only and used and sid not in used:
                continue
            s = self.sources.get(sid, {})
            parts = []
            if s.get("authors"):
                parts.append(str(s["authors"]))
            if s.get("title"):
                parts.append(str(s["title"]))
            if s.get("venue"):
                parts.append(str(s["venue"]))
            if s.get("publication_year"):
                parts.append(str(s["publication_year"]))
            if s.get("doi"):
                parts.append(f"DOI:{s['doi']}")
            if s.get("url"):
                parts.append(str(s["url"]))
            lines.append(f"[S{n}] " + "，".join(p for p in parts if p))
        return "\n".join(lines)


class CitationVerifier:
    """校验报告中的引用标记是否指向真实、可引用、且闭环的 Claim/Evidence/Source。

    在原始"引用标记能否解析 + 证据是否通过校验"之上，按规范 §45 增加三道闭环校验：
    - 证据引用的来源必须登记在 Source Catalog（SOURCE_MISSING）
    - 证据必须归属于某个已登记的 Claim（EVIDENCE_ORPHAN）
    - 证据归属的 Claim 本身必须存在（CLAIM_MISSING）
    """

    def __init__(self, catalog: SourceCatalog):
        self.catalog = catalog
        self.claim_set: set[str] = set()
        self.evidence_claim: dict[str, str] = {}

    def _index_claims(self, claims: Iterable[dict] | None) -> None:
        self.claim_set = set()
        self.evidence_claim = {}
        for c in claims or []:
            cid = c.get("claim_id", "")
            if cid:
                self.claim_set.add(cid)
            for eid in (c.get("evidence_ids") or []):
                if eid:
                    self.evidence_claim[str(eid)] = cid

    def verify(self, report: str, claims: Iterable[dict] | None = None) -> dict:
        """返回校验结果：{issues, refs, used_source_ids, used_evidence_ids}。"""
        self._index_claims(claims)
        refs = self.catalog.parse(report)
        issues: list[CitationIssue] = []
        used_sources: set[str] = set()
        used_evidences: set[str] = set()

        for r in refs:
            if not r.resolved:
                issues.append(CitationIssue(UNKNOWN_REF, r.raw, "编号不在可用目录中"))
                continue
            if r.kind == "S":
                used_sources.add(r.target_id)
                if r.target_id not in self.catalog.sources:
                    issues.append(CitationIssue(
                        SOURCE_MISSING, r.raw, "来源未登记在 Source Catalog"))
                continue

            # r.kind == "E"
            used_evidences.add(r.target_id)
            ev = self.catalog.evidences.get(r.target_id, {})
            status = ev.get("verification_status", "UNVERIFIED")
            if status not in CITABLE_STATUSES:
                issues.append(CitationIssue(
                    UNVERIFIED_EVIDENCE, r.raw,
                    f"证据校验状态为 {status}，不能作为引用依据",
                ))
            sid = ev.get("source_id", "")
            if sid:
                used_sources.add(sid)
                if sid not in self.catalog.sources:
                    issues.append(CitationIssue(
                        SOURCE_MISSING, r.raw, "证据对应的来源未登记在 Source Catalog"))
            # §45：证据必须归属于某个已登记的 Claim
            cid = self.evidence_claim.get(r.target_id)
            if not cid:
                issues.append(CitationIssue(
                    EVIDENCE_ORPHAN, r.raw, "证据未绑定到任何 Claim"))
            elif cid not in self.claim_set:
                issues.append(CitationIssue(
                    CLAIM_MISSING, r.raw, "证据绑定的 Claim 不存在"))

        # 通过证据标记间接引用到的来源
        for eid in list(used_evidences):
            sid = self.catalog.evidences.get(eid, {}).get("source_id", "")
            if sid:
                used_sources.add(sid)

        for c in claims or []:
            if c.get("status") not in ("SUPPORTED", "PARTIALLY_SUPPORTED"):
                continue
            marks = [self.catalog.evidence_marker(eid) for eid in c.get("evidence_ids", [])]
            marks = [m for m in marks if m]
            if marks and not any(m in (report or "") for m in marks):
                issues.append(CitationIssue(
                    UNCITED_CLAIM, marks[0],
                    f"可用声明未在报告中被引用：{c.get('text','')[:60]}",
                ))

        return {
            "issues": [i.to_dict() for i in issues],
            "refs": [r.to_dict() for r in refs],
            "used_source_ids": sorted(used_sources),
            "used_evidence_ids": sorted(used_evidences),
        }


# ---------------------------------------------------------------------------
# 句子切分与引用闭环
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """按换行与句末标点切分句子，保留非空片段。"""
    parts = re.split(r'(?<=[。！？!?])|\n+', text or "")
    return [p for p in parts if p.strip()]


class CitationBinder:
    """§44 引用绑定：建立 草稿句子 → Claim ID → Evidence IDs 的闭环。

    输入报告正文，按句子切分；对含引用标记的每句，解析标记并回溯到
    Evidence → Claim → Source，给出每句的引用闭环结构。同时统计未被引用的
    悬挂证据与未闭合 Claim，供审计与测试使用。
    """

    def __init__(self, catalog: SourceCatalog, claims: Iterable[dict] | None = None):
        self.catalog = catalog
        self.claim_evidence: dict[str, list[str]] = {}
        self.evidence_claim: dict[str, str] = {}
        for c in claims or []:
            cid = c.get("claim_id", "")
            if not cid:
                continue
            self.claim_evidence.setdefault(cid, [])
            for eid in (c.get("evidence_ids") or []):
                if not eid:
                    continue
                eid = str(eid)
                if eid not in self.claim_evidence[cid]:
                    self.claim_evidence[cid].append(eid)
                self.evidence_claim[eid] = cid

    def bind_sentence(self, sentence: str) -> dict:
        refs = self.catalog.parse(sentence)
        evidence_ids: set[str] = set()
        source_ids: set[str] = set()
        claim_ids: set[str] = set()
        for r in refs:
            if not r.resolved:
                continue
            if r.kind == "S":
                source_ids.add(r.target_id)
            else:
                evidence_ids.add(r.target_id)
                sid = self.catalog.evidences.get(r.target_id, {}).get("source_id", "")
                if sid:
                    source_ids.add(sid)
                cid = self.evidence_claim.get(r.target_id)
                if cid:
                    claim_ids.add(cid)
        return {
            "sentence": sentence,
            "refs": [r.to_dict() for r in refs],
            "evidence_ids": sorted(evidence_ids),
            "source_ids": sorted(source_ids),
            "claim_ids": sorted(claim_ids),
        }

    def bind(self, report: str) -> dict:
        bindings = [self.bind_sentence(s) for s in _split_sentences(report)]
        cited_evidence = {e for b in bindings for e in b["evidence_ids"]}
        cited_claims = {c for b in bindings for c in b["claim_ids"]}
        dangling_evidence = [e for e in self.evidence_claim if e not in cited_evidence]
        unbound_claims = [c for c in self.claim_evidence if c not in cited_claims]
        return {
            "bindings": bindings,
            "cited_evidence_ids": sorted(cited_evidence),
            "cited_claim_ids": sorted(cited_claims),
            "dangling_evidence_ids": sorted(dangling_evidence),
            "unbound_claims": sorted(unbound_claims),
        }


class CitationRenderer:
    """§47 引用渲染：把 CitationBinder 的闭环结构渲染为可读文本。"""

    @staticmethod
    def render_closure(binding: dict) -> str:
        lines: list[str] = []
        for b in binding.get("bindings", []):
            if not (b.get("claim_ids") or b.get("evidence_ids")):
                continue
            lines.append(
                f"句子: {b['sentence']}\n"
                f"  → Claim: {', '.join(b['claim_ids']) or '—'}\n"
                f"  → Evidence: {', '.join(b['evidence_ids']) or '—'}\n"
                f"  → Source: {', '.join(b['source_ids']) or '—'}"
            )
        if binding.get("unbound_claims"):
            lines.append(f"未闭合 Claim: {', '.join(binding['unbound_claims'])}")
        if binding.get("dangling_evidence_ids"):
            lines.append(f"悬挂 Evidence: {', '.join(binding['dangling_evidence_ids'])}")
        return "\n".join(lines)


class UnsupportedClaimDetector:
    """§48 未支撑声明检测器。

    报告成文后，禁止把 UNSUPPORTED / CONTRADICTED 的声明当作事实写入正文。
    确定性实现（不依赖模型）：逐行扫描正文，跳过"局限 / 矛盾"等豁免小节；
    若某条风险声明的归一化文本出现在正文中，则将其所在行从正文移除，
    并归集为违规，供 compiler 单独列出警示，确保未支撑结论不被当作事实输出（§49）。
    """

    _EXEMPT_HEADERS = (
        "本报告局限", "本报告限制", "局限", "矛盾与局限", "限制", "未支撑声明警示",
    )

    def __init__(self, claims: Iterable[dict] | None = None):
        self.risky: list[dict] = []
        for c in claims or []:
            if c.get("status") in ("UNSUPPORTED", "CONTRADICTED"):
                text = (c.get("text") or "").strip()
                if text:
                    self.risky.append({
                        "claim_id": c.get("claim_id", ""),
                        "text": text,
                        "status": c.get("status", ""),
                    })

    def detect_and_sanitize(self, report: str) -> tuple[list[dict], str]:
        """返回 (违规列表, 净化后的报告)。无风险声明时原样返回。"""
        if not self.risky:
            return [], report
        lines = (report or "").split("\n")
        violations: list[dict] = []
        kept: list[str] = []
        in_exempt = False
        for line in lines:
            header = line.strip()
            if header.startswith("#"):
                in_exempt = any(k in header for k in self._EXEMPT_HEADERS)
                kept.append(line)
                continue
            if in_exempt:
                kept.append(line)
                continue
            norm_line = normalize_claim_text(line)
            hit = None
            for c in self.risky:
                norm_claim = normalize_claim_text(c["text"])
                if norm_claim and norm_line and norm_claim in norm_line:
                    hit = c
                    break
            if hit:
                violations.append({
                    "claim_id": hit["claim_id"],
                    "text": hit["text"],
                    "status": hit["status"],
                    "segment": line.strip(),
                })
            else:
                kept.append(line)
        return violations, "\n".join(kept)
