"""Research SOP（研究标准作业程序）：定义「应该覆盖什么」，不规定固定拓扑。

与固定 DAG 模板的区别：
- 模板规定「先做 A、再做 B、最后做 C」的固定拓扑；
- SOP 只声明某类研究「至少要覆盖哪些维度 / 能力」，拓扑仍由 Planner 决定。

SOPSelector 用关键词规则选型，不调用 LLM（§3）。

本模块纯确定性，无外部依赖，可离线测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# SOP 覆盖维度的三类定性：required（必须覆盖）/ optional（可选，不强求）
REQUIRED = "required"
OPTIONAL = "optional"

# 四类 SOP（§3）。每类声明「至少应覆盖的能力维度」与「可选维度」。
# Ablation / Complexity / Implementation 等只作 optional，不得强制（§3 明确）。
_SOP_DEFINITIONS: dict[str, dict] = {
    "FACT": {
        "name": "事实型调研",
        "required": ["object_definition", "evidence"],
        "optional": ["background", "method", "limitation"],
    },
    "COMPARISON": {
        "name": "对比型调研",
        # COMPARISON 至少要求（§3 示例）：对象定义 / 方法或机制 / 评价或对比 / 证据 / 综合
        "required": [
            "object_definition",
            "method_or_mechanism",
            "evaluation_or_comparison",
            "evidence",
            "synthesis",
        ],
        "optional": ["ablation", "complexity", "implementation", "benchmark", "limitation"],
    },
    "MECHANISM": {
        "name": "机制型调研",
        "required": ["object_definition", "method_or_mechanism", "evidence"],
        "optional": ["evaluation", "limitation", "background"],
    },
    "SURVEY": {
        "name": "综述型调研",
        "required": [
            "object_definition",
            "background",
            "method_or_mechanism",
            "evaluation_or_comparison",
            "synthesis",
        ],
        "optional": ["limitation", "trend", "benchmark", "ablation"],
    },
}

# SOP 能力维度 → 建议的 task_type（供 repair 补任务 / coverage 兜底使用）
_CAPABILITY_TO_TASK_TYPE = {
    "object_definition": "BACKGROUND",
    "background": "BACKGROUND",
    "method_or_mechanism": "MECHANISM",
    "method": "METHOD",
    "mechanism": "MECHANISM",
    "evaluation_or_comparison": "COMPARISON",
    "evaluation": "EVALUATION",
    "comparison": "COMPARISON",
    "evidence": "FACT",
    "fact": "FACT",
    "synthesis": "SYNTHESIS",
    "trend": "TREND",
    "limitation": "LIMITATION",
    "benchmark": "EVALUATION",
    "ablation": "EVALUATION",
    "complexity": "EVALUATION",
    "implementation": "METHOD",
}


@dataclass
class ResearchSOP:
    """一类研究的标准作业程序：声明应覆盖的维度，不规定固定拓扑。"""

    sop_type: str = "FACT"                    # FACT / COMPARISON / MECHANISM / SURVEY
    required_dimensions: list[str] = field(default_factory=list)
    optional_dimensions: list[str] = field(default_factory=list)

    @property
    def dimensions(self) -> list[str]:
        """required 在前、optional 在后的完整维度列表。"""
        return list(self.required_dimensions) + list(self.optional_dimensions)

    def to_dict(self) -> dict:
        return {
            "sop_type": self.sop_type,
            "required_dimensions": list(self.required_dimensions),
            "optional_dimensions": list(self.optional_dimensions),
        }


def build_sop(sop_type: str) -> ResearchSOP:
    """按类型构建一个 ResearchSOP（不存在时回退到 FACT）。"""
    key = str(sop_type or "FACT").upper()
    if key not in _SOP_DEFINITIONS:
        key = "FACT"
    d = _SOP_DEFINITIONS[key]
    return ResearchSOP(
        sop_type=key,
        required_dimensions=list(d["required"]),
        optional_dimensions=list(d["optional"]),
    )


# ---------------------------------------------------------------------------
# SOPSelector：关键词规则选型，不调 LLM（§3）
# ---------------------------------------------------------------------------

# (关键词, sop_type)。按顺序匹配，命中即返回；因此「对比/比较」要在「综述」之前，
# 避免「对比综述」这类词被误判成 SURVEY。
_SELECTOR_RULES: list[tuple[tuple[str, ...], str]] = [
    (("对比", "比较", "区别", "差异", "优劣", "vs", "versus", "compare", "comparison",
      "哪个更好", "选型", "横评"), "COMPARISON"),
    (("机制", "原理", "如何工作", "为什么", "机理", "作用机制", "mechanism",
      "how does", "why"), "MECHANISM"),
    (("综述", "survey", "review", "发展历程", "研究现状", "梳理", "总结现状",
      "概述", "全景"), "SURVEY"),
    # 其余一律 FACT
]


def select_sop(query: str) -> ResearchSOP:
    """关键词规则选 SOP 类型，返回 ResearchSOP。不调用 LLM。

    匹配顺序：COMPARISON → MECHANISM → SURVEY → 默认 FACT。
    """
    q = (query or "").lower()
    for keywords, sop_type in _SELECTOR_RULES:
        if any(kw.lower() in q for kw in keywords):
            return build_sop(sop_type)
    return build_sop("FACT")


def capability_to_task_type(capability: str) -> str:
    """能力维度 → 建议 task_type（repair 补任务时用）。"""
    return _CAPABILITY_TO_TASK_TYPE.get(str(capability).lower(), "FACT")


# task_type → 它「能提供」的 SOP 维度集合（供 PlanValidator 做覆盖判断）。
# 方向与 _CAPABILITY_TO_TASK_TYPE 相反：这里回答「某类任务产出哪些维度」。
_TASK_TYPE_TO_DIMENSIONS: dict[str, set[str]] = {
    "BACKGROUND": {"object_definition", "background"},
    "FACT": {"evidence", "fact", "object_definition"},
    "METHOD": {"method_or_mechanism", "method", "mechanism"},
    "MECHANISM": {"method_or_mechanism", "method", "mechanism"},
    "COMPARISON": {"evaluation_or_comparison", "comparison", "evaluation"},
    "EVALUATION": {"evaluation_or_comparison", "comparison", "evaluation"},
    "SYNTHESIS": {"synthesis"},
    "TREND": {"trend"},
    "LIMITATION": {"limitation"},
    "RESULT": {"evidence", "evaluation"},
}


def task_type_dimensions(task_type: str) -> set[str]:
    """返回某 task_type 能提供的 SOP 维度集合（未知名 → 空集）。"""
    return set(_TASK_TYPE_TO_DIMENSIONS.get(str(task_type or "").upper(), set()))
