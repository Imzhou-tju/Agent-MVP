"""Claim-Evidence 语义关系判定（§8）。

Evidence 的 verification_status 只说明「quote 是否能在原文里定位」，
而「quote 是否真的支持 claim」是语义问题，由本模块用 LLM 判定。

语义关系不影响「能否引用」，只影响 Claim 的支撑状态推导：
- SUPPORTS / PARTIALLY_SUPPORTS → 计入支撑
- CONTRADICTS → 计入反证
- IRRELEVANT → 不计入支撑

确定性边界（与 spec 一致）：
- LLM 只产出「语义关系」这一判断，不产出「claim 是否被支撑」的结论；
  claim 的支撑状态由 ClaimGraph.recompute_statuses 按证据校验 + 语义关系确定性推导。
- LLM 不可用时整体回退到 SUPPORTS（抽取阶段已让模型逐字配对 quote 与 claim，
  兜底为支持的最坏后果是「过度支持」，而非「错误否定」已有证据）。
- 本模块任何异常都不外抛，统一返回回退结果，调用方无需 try/except 包裹。
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from .claim import (
    SEM_SUPPORTS,
    SEM_PARTIALLY_SUPPORTS,
    SEM_CONTRADICTS,
    SEM_IRRELEVANT,
    SEMANTIC_RELATIONS,
)


def _parse_json(raw: str) -> Any:
    m = re.search(r'(\{.*\}|\[.*\])', raw, re.DOTALL)
    if not m:
        raise ValueError("无法从输出中提取 JSON")
    return json.loads(m.group(0))


class ClaimEvidenceSemanticVerifier:
    """判定若干 (evidence, claim) 对的语义关系。

    做法是把本任务的 claim/evidence 配对列表一次性发给模型，让其对每对
    给出语义关系，落回确定性推导。模型失败时整体回退 SUPPORTS。
    """

    def __init__(self, llm: ChatOpenAI | None = None):
        self._llm = llm

    def verify(
        self,
        evidences: list[dict],
        claims: list[dict],
    ) -> dict[tuple[str, str], str]:
        """返回 {(evidence_id, claim_id): semantic_relation}。

        所有配对都会给出值（默认 SUPPORTS），未命中模型输出的配对保持 SUPPORTS。
        本方法不抛异常：任何失败都回退到 SUPPORTS。
        """
        pairs = self._build_pairs(evidences, claims)
        index: dict[tuple[str, str], str] = {}
        for ev_id, c_id, _rid in pairs:
            index[(ev_id, c_id)] = SEM_SUPPORTS  # 默认回退值

        if not pairs or self._llm is None:
            return index

        prompt = self._build_prompt(pairs, evidences, claims)
        try:
            raw = self._llm.invoke([HumanMessage(content=prompt)]).content.strip()
            parsed = _parse_json(raw)
            rows = parsed.get("relations", []) if isinstance(parsed, dict) else (parsed or [])
            id_to_pair = {rid: (ev_id, c_id) for ev_id, c_id, rid in pairs}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                rid = str(row.get("id", ""))
                rel = str(row.get("semantic_relation", "")).upper()
                pair = id_to_pair.get(rid)
                if pair is None:
                    continue
                if rel not in SEMANTIC_RELATIONS:
                    rel = SEM_SUPPORTS
                index[pair] = rel
        except Exception as e:  # noqa: BLE001 - 兜底语义判定失败不阻断主流程
            print(f"[Semantic] 语义关系判定失败，回退 SUPPORTS: {e}")
            for ev_id, c_id, _rid in pairs:
                index[(ev_id, c_id)] = SEM_SUPPORTS

        return index

    def _build_pairs(
        self, evidences: list[dict], claims: list[dict]
    ) -> list[tuple[str, str, str]]:
        """配对：每个 evidence 归属的 claim（claim.evidence_ids 反查）。"""
        pairs: list[tuple[str, str, str]] = []
        counter = 0
        for ev in evidences:
            ev_id = ev.get("evidence_id", "")
            for c in claims:
                if ev_id and ev_id in (c.get("evidence_ids") or []):
                    counter += 1
                    pairs.append((ev_id, c.get("claim_id", ""), f"R{counter}"))
        return pairs

    def _build_prompt(
        self,
        pairs: list[tuple[str, str, str]],
        evidences: list[dict],
        claims: list[dict],
    ) -> str:
        ev_by_id = {e.get("evidence_id"): e for e in evidences}
        c_by_id = {c.get("claim_id"): c for c in claims}
        items = []
        for ev_id, c_id, rid in pairs:
            ev = ev_by_id.get(ev_id, {})
            c = c_by_id.get(c_id, {})
            items.append(
                f"{rid}: 声明「{c.get('text', '')}」\n   原文引文「{ev.get('quote', '')}」"
            )
        body = "\n".join(items)
        return (
            "判断每条「原文引文」是否真正支持对应的「声明」。\n"
            "可选语义关系：\n"
            "- SUPPORTS：引文直接支持声明\n"
            "- PARTIALLY_SUPPORTS：引文部分支持或只支持声明的某一方面\n"
            "- CONTRADICTS：引文反驳声明\n"
            "- IRRELEVANT：引文与该声明无关\n\n"
            f"{body}\n\n"
            "对每条给出 JSON（只输出 JSON，不要解释）:\n"
            '{"relations":[{"id":"R1","semantic_relation":"SUPPORTS"}]}'
        )
