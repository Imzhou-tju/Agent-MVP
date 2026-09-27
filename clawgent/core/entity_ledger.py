"""Entity Ledger（实体白板）运行时状态管理模块。

职责边界：
- 仅负责跨回合长期有效的研究状态（科研硬约束、已确认实体、未决研究问题）。
- 不负责搜索、证据存储、citation 管理、DAG 生成、最终答案生成，亦不替代 evidence_state。
- 采用结构化 Schema，在历史回合裁剪时通过 LLM 增量抽取与冲突解决，替代旧有的 150 字有损自然语言摘要。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal
from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)

# 约束状态定义
CONSTRAINT_ACTIVE = "active"
CONSTRAINT_INACTIVE = "inactive"
CONSTRAINT_REMOVED = "removed"

# 实体类型定义
ENTITY_TYPE_PAPER = "paper"
ENTITY_TYPE_METHOD = "method"
ENTITY_TYPE_DATASET = "dataset"
ENTITY_TYPE_MODEL = "model"
ENTITY_TYPE_PERSON = "person"
ENTITY_TYPE_ORGANIZATION = "organization"
ENTITY_TYPE_OTHER = "other"

# 问题状态定义
QUESTION_PENDING = "pending"
QUESTION_RESOLVED = "resolved"


class GlobalConstraint(BaseModel):
    """科研全局硬性约束模型。"""
    id: str = Field(description="约束唯一标识，如 constraint_1")
    content: str = Field(description="约束具体内容，如：只关注 2024 年以后的无监督方法")
    source_turn: int = Field(default=0, description="引入该约束的回合数")
    status: Literal["active", "inactive", "removed"] = Field(
        default=CONSTRAINT_ACTIVE,
        description="约束当前状态：active(生效中)、inactive(已停用)、removed(已撤销)"
    )


class ResolvedEntity(BaseModel):
    """已确认或消歧的事实实体模型。"""
    id: str = Field(description="实体唯一标识，如 entity_1")
    type: Literal["paper", "method", "dataset", "model", "person", "organization", "other"] = Field(
        default=ENTITY_TYPE_OTHER,
        description="实体类型"
    )
    name: str = Field(description="实体标准名称或消歧后的名称")
    description: str = Field(default="", description="实体关键描述或确认事实")
    source_turn: int = Field(default=0, description="确立该实体的回合数")


class PendingQuestion(BaseModel):
    """未决或进行中的科研问题模型。"""
    id: str = Field(description="问题唯一标识，如 question_1")
    question: str = Field(description="未决问题的具体内容")
    source_turn: int = Field(default=0, description="提出该问题的回合数")
    status: Literal["pending", "resolved"] = Field(
        default=QUESTION_PENDING,
        description="问题状态：pending(待解决)、resolved(已解决)"
    )


class EntityLedger(BaseModel):
    """实体白板完整状态模型。"""
    global_constraints: list[GlobalConstraint] = Field(default_factory=list, description="科研全局硬约束列表")
    resolved_entities: list[ResolvedEntity] = Field(default_factory=list, description="已确认实体列表")
    pending_questions: list[PendingQuestion] = Field(default_factory=list, description="未决研究问题列表")

    def to_dict(self) -> dict[str, Any]:
        """导出为纯 JSON 可序列化字典。"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Any) -> "EntityLedger":
        """从字典或任意对象安全反序列化，容错缺失字段与非标数据。"""
        if not data or not isinstance(data, dict):
            return cls()
        try:
            return cls.model_validate(data)
        except ValidationError as e:
            logger.warning("EntityLedger 校验失败，尝试安全清洗: %s", e)
            cleaned: dict[str, Any] = {
                "global_constraints": [],
                "resolved_entities": [],
                "pending_questions": []
            }
            # 容错清洗 constraints
            for item in data.get("global_constraints", []):
                if isinstance(item, dict) and "content" in item:
                    try:
                        cleaned["global_constraints"].append(GlobalConstraint.model_validate(item))
                    except Exception:
                        pass
            # 容错清洗 entities
            for item in data.get("resolved_entities", []):
                if isinstance(item, dict) and "name" in item:
                    try:
                        cleaned["resolved_entities"].append(ResolvedEntity.model_validate(item))
                    except Exception:
                        pass
            # 容错清洗 questions
            for item in data.get("pending_questions", []):
                if isinstance(item, dict) and "question" in item:
                    try:
                        cleaned["pending_questions"].append(PendingQuestion.model_validate(item))
                    except Exception:
                        pass
            return cls(**cleaned)

    @classmethod
    def empty(cls) -> "EntityLedger":
        """生成空白板。"""
        return cls()

    def is_empty(self) -> bool:
        """检查白板是否完全为空。"""
        return (
            len(self.global_constraints) == 0
            and len(self.resolved_entities) == 0
            and len(self.pending_questions) == 0
        )

    def get_active_constraints(self) -> list[GlobalConstraint]:
        """获取所有当前生效中的约束。"""
        return [c for c in self.global_constraints if c.status == CONSTRAINT_ACTIVE]

    def get_pending_questions(self) -> list[PendingQuestion]:
        """获取所有待解决的研究问题。"""
        return [q for q in self.pending_questions if q.status == QUESTION_PENDING]

    def to_prompt_context(self) -> str:
        """生成注入下游 Agent/Planner 的结构化上下文文本。"""
        active_constraints = self.get_active_constraints()
        unresolved_questions = self.get_pending_questions()

        lines = [
            "===========================================================",
            "【Research State / Entity Ledger (科研实体白板 - 长期约束与状态)】",
            "说明：此白板记录跨回合长期有效的科研硬约束与实体，供后续 Planner 及 Research Agent 强制遵循。",
            "1. active constraints 为必须严格遵守的全局约束；",
            "2. resolved entities 为已建立的消歧实体与事实；",
            "3. pending questions 为尚未解决的核心问题；",
            "4. 不得无依据篡改已记录信息；若用户在最新回合明确提出变更，以用户最新指令为准。",
            ""
        ]

        if active_constraints:
            lines.append("● 必须遵守的全局硬性约束 (Active Constraints):")
            for c in active_constraints:
                lines.append(f"  - [{c.id}] {c.content} (来源回合: {c.source_turn})")
        else:
            lines.append("● 必须遵守的全局硬性约束 (Active Constraints): 暂无生效约束")

        if self.resolved_entities:
            lines.append("\n● 已确认科研实体 (Resolved Entities):")
            for e in self.resolved_entities:
                desc = f" - {e.description}" if e.description else ""
                lines.append(f"  - [{e.id}] ({e.type}) {e.name}{desc} (来源回合: {e.source_turn})")
        else:
            lines.append("\n● 已确认科研实体 (Resolved Entities): 暂无")

        if unresolved_questions:
            lines.append("\n● 待解决研究问题 (Pending Questions):")
            for q in unresolved_questions:
                lines.append(f"  - [{q.id}] {q.question} (来源回合: {q.source_turn})")
        else:
            lines.append("\n● 待解决研究问题 (Pending Questions): 暂无")

        lines.append("\n[JSON Schema 状态]:")
        lines.append(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))
        lines.append("===========================================================")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 更新 Prompt 与核心逻辑
# ---------------------------------------------------------------------------

LEDGER_UPDATE_PROMPT_TEMPLATE = """你是一个负责科研智能体「Entity Ledger（实体白板）」状态管理的后台模块。
当前运行时正准备对历史会话进行裁剪。你的职责是：仔细审查即将被裁剪的历史对话，并结合当前的 Entity Ledger，提取需要长期保留到白板中的关键科研状态。

【核心原则】
1. **绝非全文摘要**：严禁概括“双方说了什么”或输出过程性叙述。白板只记录对未来科研决策有持续约束力的实体和状态。
2. **提取对象仅限三类**：
   - `global_constraints`：必须持续遵守的科研硬约束（时间窗口、指定数据集、排除条件、特定模型版本、硬件限制等）。
   - `resolved_entities`：已明确确认、消歧或建立事实的实体（论文、方法、模型、数据集、作者等）。
   - `pending_questions`：当前明确提出但尚未解决的研究问题。
3. **冲突与更新消除（核心规则）**：
   - **用户修改约束**（例如：此前要求“只看2024年”，后修改为“放宽到2022年”）：必须更新对应 constraint 的 content，并保持 status="active"，绝对禁止同时保留两个互相冲突的 active 约束！
   - **用户撤销约束**（例如：“不限制数据集了”）：必须将对应 constraint 的 status 改为 "inactive" 或 "removed"，不得继续保留生效状态。
   - **实体消歧与明确**（例如：此前记录了“BERT”，后续明确为“BERT-base-uncased”）：必须更新原有 entity 的 name/description，不得并列生成重复实体。
   - **问题已解决**：若历史对话中 pending question 已被解答或已有定论，将其 status 标记为 "resolved"。
4. **确定性与抗噪性**：
   - 严禁将模型自身的猜测、模糊推测升级为 confirmed 约束或实体。
   - 严禁将普通的寒暄、闲聊、临时工具调试、过程性讨论写入白板。
   - 若即将被裁剪的对话中没有产生任何新的状态变化，保持原有 Ledger 不变即可。

【当前 Entity Ledger】
{current_ledger_json}

【即将被裁剪的历史对话】
{discarded_text}

【输出格式要求】
请严格输出合法的 JSON 对象，不要添加任何前缀、解释、自然语言或 Markdown 包装外的文字。
JSON 格式定义：
{{
  "global_constraints": [
    {{"id": "constraint_1", "content": "内容", "source_turn": 1, "status": "active|inactive|removed"}}
  ],
  "resolved_entities": [
    {{"id": "entity_1", "type": "paper|method|dataset|model|person|organization|other", "name": "名称", "description": "说明", "source_turn": 1}}
  ],
  "pending_questions": [
    {{"id": "question_1", "question": "问题内容", "source_turn": 1, "status": "pending|resolved"}}
  ]
}}
"""


def _extract_json_from_llm_response(text: str) -> dict[str, Any]:
    """从 LLM 响应中稳健提取 JSON 字典。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 正则提取最外层的大括号 JSON
    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    raise ValueError(f"无法从 LLM 响应中解析出有效的 JSON 对象: {text[:200]}")


def update_entity_ledger(
    current_ledger: EntityLedger | dict[str, Any] | None,
    discarded_msgs: list[BaseMessage],
    llm: Any,
    current_turn: int = 0
) -> EntityLedger:
    """根据即将裁剪的消息和当前白板，调用 LLM 增量更新 Entity Ledger。

    容错策略：
    - 若 discarded_msgs 为空，直接返回当前 ledger；
    - 若 LLM 调用失败、输出非法 JSON 或 Schema 校验未通过，记录 warning 并返回旧 ledger；
    - 绝不因 Ledger 更新失败而阻断整个对话或上下文裁剪流程。
    """
    if isinstance(current_ledger, EntityLedger):
        base_ledger = current_ledger
    elif isinstance(current_ledger, dict):
        base_ledger = EntityLedger.from_dict(current_ledger)
    else:
        base_ledger = EntityLedger.empty()

    if not discarded_msgs or not llm:
        return base_ledger

    # 过滤出有意义的文本内容
    discarded_lines = []
    for m in discarded_msgs:
        content = m.content if hasattr(m, "content") else str(m)
        if content:
            sender = getattr(m, "type", "message")
            discarded_lines.append(f"{sender}: {content}")

    if not discarded_lines:
        return base_ledger

    discarded_text = "\n".join(discarded_lines)
    prompt = LEDGER_UPDATE_PROMPT_TEMPLATE.format(
        current_ledger_json=json.dumps(base_ledger.to_dict(), ensure_ascii=False, indent=2),
        discarded_text=discarded_text
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)], config={"callbacks": []})
        raw_text = response.content if hasattr(response, "content") else str(response)
        parsed_dict = _extract_json_from_llm_response(raw_text)
        new_ledger = EntityLedger.model_validate(parsed_dict)
        return new_ledger
    except (ValidationError, ValueError, json.JSONDecodeError) as e:
        logger.warning("[EntityLedger] 更新校验失败，保留旧状态: %s", e)
        return base_ledger
    except Exception as e:
        logger.warning("[EntityLedger] 调用 LLM 发生异常，保留旧状态: %s", e)
        return base_ledger
