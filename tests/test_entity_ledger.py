import unittest
import os
import sys
import json
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from clawgent.core.entity_ledger import (
    EntityLedger,
    GlobalConstraint,
    ResolvedEntity,
    PendingQuestion,
    update_entity_ledger,
    CONSTRAINT_ACTIVE,
    CONSTRAINT_INACTIVE,
    CONSTRAINT_REMOVED,
    QUESTION_PENDING,
    QUESTION_RESOLVED
)
from clawgent.core.context import AgentState, get_entity_ledger_from_state, trim_context_messages


class TestEntityLedger(unittest.TestCase):

    # 1. 空 Ledger 初始化
    def test_1_empty_ledger_initialization(self):
        ledger = EntityLedger.empty()
        self.assertTrue(ledger.is_empty())
        self.assertEqual(ledger.global_constraints, [])
        self.assertEqual(ledger.resolved_entities, [])
        self.assertEqual(ledger.pending_questions, [])
        self.assertEqual(ledger.get_active_constraints(), [])
        self.assertEqual(ledger.get_pending_questions(), [])

        # 测试从空字典反序列化
        from_dict_ledger = EntityLedger.from_dict({})
        self.assertTrue(from_dict_ledger.is_empty())

    # 2. 新增 global constraint
    def test_2_add_global_constraint(self):
        llm = Mock()
        llm_response = {
            "global_constraints": [
                {
                    "id": "constraint_1",
                    "content": "只关注 2024 年以后的无监督方法",
                    "source_turn": 5,
                    "status": "active"
                }
            ],
            "resolved_entities": [],
            "pending_questions": []
        }
        llm.invoke.return_value = AIMessage(content=json.dumps(llm_response))

        discarded = [HumanMessage(content="调研时只关注 2024 年以后的无监督方法")]
        new_ledger = update_entity_ledger(EntityLedger.empty(), discarded, llm, current_turn=5)

        self.assertFalse(new_ledger.is_empty())
        self.assertEqual(len(new_ledger.global_constraints), 1)
        c = new_ledger.global_constraints[0]
        self.assertEqual(c.id, "constraint_1")
        self.assertEqual(c.content, "只关注 2024 年以后的无监督方法")
        self.assertEqual(c.status, CONSTRAINT_ACTIVE)
        self.assertEqual(len(new_ledger.get_active_constraints()), 1)

    # 3. 新增 resolved entity
    def test_3_add_resolved_entity(self):
        llm = Mock()
        llm_response = {
            "global_constraints": [],
            "resolved_entities": [
                {
                    "id": "entity_ecapa",
                    "type": "model",
                    "name": "ECAPA-TDNN",
                    "description": "基于通道注意力的声纹识别时延神经网络模型",
                    "source_turn": 8
                }
            ],
            "pending_questions": []
        }
        llm.invoke.return_value = AIMessage(content=json.dumps(llm_response))

        discarded = [AIMessage(content="我们确认采用 ECAPA-TDNN 作为主干特征提取模型。")]
        new_ledger = update_entity_ledger(EntityLedger.empty(), discarded, llm, current_turn=8)

        self.assertEqual(len(new_ledger.resolved_entities), 1)
        e = new_ledger.resolved_entities[0]
        self.assertEqual(e.name, "ECAPA-TDNN")
        self.assertEqual(e.type, "model")
        self.assertEqual(e.source_turn, 8)

    # 4. 新增 pending question
    def test_4_add_pending_question(self):
        llm = Mock()
        llm_response = {
            "global_constraints": [],
            "resolved_entities": [],
            "pending_questions": [
                {
                    "id": "question_loss",
                    "question": "在短语音场景下，哪种角边际损失（AAM-Softmax vs Sub-center）表现更优？",
                    "source_turn": 12,
                    "status": "pending"
                }
            ]
        }
        llm.invoke.return_value = AIMessage(content=json.dumps(llm_response))

        discarded = [HumanMessage(content="短语音场景下损失函数选型还没定，需要后续测试。")]
        new_ledger = update_entity_ledger(EntityLedger.empty(), discarded, llm, current_turn=12)

        self.assertEqual(len(new_ledger.pending_questions), 1)
        q = new_ledger.pending_questions[0]
        self.assertEqual(q.id, "question_loss")
        self.assertEqual(q.status, QUESTION_PENDING)
        self.assertEqual(len(new_ledger.get_pending_questions()), 1)

    # 5. 同一 constraint 被用户修改
    def test_5_modify_same_constraint(self):
        initial = EntityLedger(
            global_constraints=[
                GlobalConstraint(
                    id="constraint_time",
                    content="只关注 2024 年之后的方法",
                    source_turn=5,
                    status=CONSTRAINT_ACTIVE
                )
            ]
        )

        llm = Mock()
        # 用户在后续回合放宽了限制，LLM 更新原有 constraint，而不是追加互相冲突的新 constraint
        llm_response = {
            "global_constraints": [
                {
                    "id": "constraint_time",
                    "content": "放宽到 2022 年之后的方法",
                    "source_turn": 18,
                    "status": "active"
                }
            ],
            "resolved_entities": [],
            "pending_questions": []
        }
        llm.invoke.return_value = AIMessage(content=json.dumps(llm_response))

        discarded = [HumanMessage(content="2024年的文章太少，把时间范围放宽到2022年之后吧")]
        updated = update_entity_ledger(initial, discarded, llm, current_turn=18)

        # 验证仅有一个生效约束，且内容已更新
        self.assertEqual(len(updated.global_constraints), 1)
        self.assertEqual(updated.global_constraints[0].id, "constraint_time")
        self.assertEqual(updated.global_constraints[0].content, "放宽到 2022 年之后的方法")
        self.assertEqual(updated.global_constraints[0].status, CONSTRAINT_ACTIVE)

    # 6. constraint 被用户撤销
    def test_6_revoke_constraint(self):
        initial = EntityLedger(
            global_constraints=[
                GlobalConstraint(
                    id="constraint_dataset",
                    content="仅在 VoxCeleb1 数据集上评估",
                    source_turn=3,
                    status=CONSTRAINT_ACTIVE
                )
            ]
        )

        llm = Mock()
        llm_response = {
            "global_constraints": [
                {
                    "id": "constraint_dataset",
                    "content": "仅在 VoxCeleb1 数据集上评估",
                    "source_turn": 3,
                    "status": "removed"
                }
            ],
            "resolved_entities": [],
            "pending_questions": []
        }
        llm.invoke.return_value = AIMessage(content=json.dumps(llm_response))

        discarded = [HumanMessage(content="取消数据集限定，不限制只跑 VoxCeleb1 了")]
        updated = update_entity_ledger(initial, discarded, llm, current_turn=20)

        self.assertEqual(len(updated.global_constraints), 1)
        self.assertEqual(updated.global_constraints[0].status, CONSTRAINT_REMOVED)
        # get_active_constraints 应该过滤掉已撤销的
        self.assertEqual(len(updated.get_active_constraints()), 0)

    # 7. entity 消歧/更新
    def test_7_entity_disambiguation(self):
        initial = EntityLedger(
            resolved_entities=[
                ResolvedEntity(
                    id="entity_bert",
                    type="model",
                    name="BERT",
                    description="预训练语言模型",
                    source_turn=2
                )
            ]
        )

        llm = Mock()
        llm_response = {
            "global_constraints": [],
            "resolved_entities": [
                {
                    "id": "entity_bert",
                    "type": "model",
                    "name": "BERT-base-uncased",
                    "description": "12层、768隐藏维度的无大小写英文预训练BERT模型",
                    "source_turn": 14
                }
            ],
            "pending_questions": []
        }
        llm.invoke.return_value = AIMessage(content=json.dumps(llm_response))

        discarded = [HumanMessage(content="这里明确一下，我们使用的基线模型是 BERT-base-uncased")]
        updated = update_entity_ledger(initial, discarded, llm, current_turn=14)

        self.assertEqual(len(updated.resolved_entities), 1)
        self.assertEqual(updated.resolved_entities[0].name, "BERT-base-uncased")
        self.assertIn("12层", updated.resolved_entities[0].description)

    # 8. pending question 被解决
    def test_8_pending_question_resolved(self):
        initial = EntityLedger(
            pending_questions=[
                PendingQuestion(
                    id="q_cluster",
                    question="谱聚类与 AHC 哪个在未知说话人数量场景下更鲁棒？",
                    source_turn=6,
                    status=QUESTION_PENDING
                )
            ]
        )

        llm = Mock()
        llm_response = {
            "global_constraints": [],
            "resolved_entities": [],
            "pending_questions": [
                {
                    "id": "q_cluster",
                    "question": "谱聚类与 AHC 哪个在未知说话人数量场景下更鲁棒？",
                    "source_turn": 6,
                    "status": "resolved"
                }
            ]
        }
        llm.invoke.return_value = AIMessage(content=json.dumps(llm_response))

        discarded = [AIMessage(content="根据文献调研结论，自适应阈值的谱聚类在未知说话人数场景下显著优于 AHC。")]
        updated = update_entity_ledger(initial, discarded, llm, current_turn=19)

        self.assertEqual(len(updated.pending_questions), 1)
        self.assertEqual(updated.pending_questions[0].status, QUESTION_RESOLVED)
        self.assertEqual(len(updated.get_pending_questions()), 0)

    # 9. 无有效状态信息时 Ledger 不应发生无意义变化
    def test_9_no_new_info_ledger_unchanged(self):
        initial = EntityLedger(
            global_constraints=[
                GlobalConstraint(id="c1", content="限定单麦克风", source_turn=1, status=CONSTRAINT_ACTIVE)
            ]
        )

        llm = Mock()
        # 模型识别到没有新信息，返回原样结构
        llm.invoke.return_value = AIMessage(content=json.dumps(initial.to_dict()))

        discarded = [HumanMessage(content="帮我打印一下刚才的目录结构"), AIMessage(content="这是文件目录列表...")]
        updated = update_entity_ledger(initial, discarded, llm, current_turn=25)

        self.assertEqual(updated.to_dict(), initial.to_dict())

    # 10. LLM 返回非法 JSON
    def test_10_llm_returns_invalid_json(self):
        initial = EntityLedger(
            global_constraints=[
                GlobalConstraint(id="c1", content="限定单麦克风", source_turn=1, status=CONSTRAINT_ACTIVE)
            ]
        )

        llm = Mock()
        llm.invoke.return_value = AIMessage(content="对不起，我无法生成合法的 JSON 内容，这里有语法错误 {broken json")

        discarded = [HumanMessage(content="一些讨论")]
        # 不抛异常，优雅保留旧状态
        updated = update_entity_ledger(initial, discarded, llm, current_turn=30)
        self.assertEqual(updated.to_dict(), initial.to_dict())

    # 11. schema 校验失败
    def test_11_schema_validation_failure(self):
        initial = EntityLedger(
            global_constraints=[
                GlobalConstraint(id="c1", content="限定单麦克风", source_turn=1, status=CONSTRAINT_ACTIVE)
            ]
        )

        llm = Mock()
        # 返回缺失必要字段（如 constraint 缺少 content）的非法 schema
        bad_response = {
            "global_constraints": [{"id": "bad_one"}],  # 缺失 content
            "resolved_entities": "this should be a list not string"
        }
        llm.invoke.return_value = AIMessage(content=json.dumps(bad_response))

        discarded = [HumanMessage(content="一些讨论")]
        updated = update_entity_ledger(initial, discarded, llm, current_turn=30)
        # 容错降级保证旧状态留存
        self.assertEqual(updated.global_constraints[0].id, "c1")

    # 12. Ledger 更新失败时旧状态仍然保留
    def test_12_update_failure_preserves_old_state(self):
        initial = EntityLedger(
            global_constraints=[
                GlobalConstraint(id="c_keep", content="必须保留的硬约束", source_turn=1, status=CONSTRAINT_ACTIVE)
            ]
        )

        llm = Mock()
        llm.invoke.side_effect = TimeoutError("LLM API 连接超时")

        discarded = [HumanMessage(content="更新指令")]
        updated = update_entity_ledger(initial, discarded, llm, current_turn=30)
        self.assertEqual(len(updated.global_constraints), 1)
        self.assertEqual(updated.global_constraints[0].content, "必须保留的硬约束")

    # 13. 历史裁剪后 Ledger 仍可被下游 Agent 正确读取
    def test_13_context_trimming_downstream_read(self):
        # 构造超过 40 回合的长会话
        raw_msgs = []
        for i in range(45):
            raw_msgs.append(HumanMessage(content=f"用户提问 第{i+1}回合"))
            raw_msgs.append(AIMessage(content=f"AI回答 第{i+1}回合"))

        kept_msgs, discarded_msgs = trim_context_messages(raw_msgs, trigger_turns=40, keep_turns=10)
        self.assertEqual(len(discarded_msgs), 70)  # 前 35 回合 (35*2=70条消息)
        self.assertEqual(len(kept_msgs), 20)       # 后 10 回合 (10*2=20条消息)

        # 模拟白板从裁剪消息中提取并固化了第 3 回合提出的硬约束
        persisted_ledger = EntityLedger(
            global_constraints=[
                GlobalConstraint(
                    id="c_unsupervised",
                    content="只关注 2024 年以后的无监督方法",
                    source_turn=3,
                    status=CONSTRAINT_ACTIVE
                )
            ]
        )

        prompt_context = persisted_ledger.to_prompt_context()
        self.assertIn("【Research State / Entity Ledger (科研实体白板 - 长期约束与状态)】", prompt_context)
        self.assertIn("只关注 2024 年以后的无监督方法", prompt_context)
        self.assertIn("来源回合: 3", prompt_context)

    # 14. 老 checkpoint/state 不包含 Ledger 时能够正常恢复
    def test_14_legacy_checkpoint_compatibility(self):
        legacy_state: AgentState = {
            "messages": [HumanMessage(content="旧消息")],
            "summary": "旧版本生成的普通自然语言摘要"
        }

        # 从不含 entity_ledger 键的老 state 中提取，应安全返回空白板
        ledger = get_entity_ledger_from_state(legacy_state)
        self.assertIsInstance(ledger, EntityLedger)
        self.assertTrue(ledger.is_empty())

    # 15. Ledger 不应保存普通闲聊内容
    def test_15_chitchat_ignored(self):
        llm = Mock()
        # 面对普通问候与闲聊，LLM 返回空白板或无新增状态
        llm_response = {
            "global_constraints": [],
            "resolved_entities": [],
            "pending_questions": []
        }
        llm.invoke.return_value = AIMessage(content=json.dumps(llm_response))

        chitchat_msgs = [
            HumanMessage(content="你好啊！"),
            AIMessage(content="你好！请问有什么我可以帮你的？"),
            HumanMessage(content="今天天气真好"),
            AIMessage(content="是的，祝您今天心情愉快！")
        ]
        result = update_entity_ledger(EntityLedger.empty(), chitchat_msgs, llm, current_turn=1)
        self.assertTrue(result.is_empty())

    # 16. Ledger 与 evidence_state 可以同时存在且职责不冲突
    def test_16_coexistence_with_evidence_state(self):
        # 1. 运行时长期语义状态（EntityLedger）
        ledger = EntityLedger(
            global_constraints=[
                GlobalConstraint(
                    id="c_time",
                    content="只关注 2024 年以后",
                    source_turn=2,
                    status=CONSTRAINT_ACTIVE
                )
            ],
            resolved_entities=[
                ResolvedEntity(
                    id="e_mamba",
                    type="model",
                    name="Mamba-2",
                    description="状态空间序列模型",
                    source_turn=5
                )
            ]
        )

        # 2. 多跳检索与调研证据链状态（evidence_state）
        evidence_state = [
            {
                "evidence_id": "Ef45ff715",
                "summary": "Mamba-2 在长序列推理上吞吐量相较 Transformer 提升 3 倍",
                "source_id": "S10740835"
            }
        ]

        # 验证两者数据结构完全正交，互不污染
        ledger_dict = ledger.to_dict()
        self.assertIn("global_constraints", ledger_dict)
        self.assertIn("resolved_entities", ledger_dict)
        self.assertNotIn("evidence_id", ledger_dict)

        self.assertEqual(evidence_state[0]["evidence_id"], "Ef45ff715")
        self.assertEqual(evidence_state[0]["source_id"], "S10740835")


if __name__ == '__main__':
    unittest.main()
