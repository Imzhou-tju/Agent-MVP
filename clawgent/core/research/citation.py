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

# 报告里的引用标记：[S1] 指向来源，[E3] 指向具体证据
CITATION_PATTERN = re.compile(r"\[(?P<kind>[SE])(?P<num>\d+)\]")

UNKNOWN_REF = "unknown_ref"
UNVERIFIED_EVIDENCE = "unverified_evidence"
UNCITED_CLAIM = "uncited_claim"

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
    """校验报告中的引用标记是否指向真实、可引用的来源与证据。"""

    def __init__(self, catalog: SourceCatalog):
        self.catalog = catalog

    def verify(self, report: str, claims: Iterable[dict] | None = None) -> dict:
        """返回校验结果：{issues, refs, used_source_ids, used_evidence_ids}。"""
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
            else:
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
