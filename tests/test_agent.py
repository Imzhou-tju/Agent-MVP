import unittest
import os
import sys
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clawgent.core.context import AgentState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage


class TestAgent(unittest.TestCase):

    def test_agent_state_initialization(self):
        """测试 AgentState 的初始化"""
        from clawgent.core.context import AgentState

        initial_state = AgentState(
            messages=[],
            summary=""
        )

        self.assertEqual(initial_state["messages"], [])
        self.assertEqual(initial_state["summary"], "")

    @patch('clawgent.core.agent.get_provider')
    @patch('clawgent.core.agent.load_dynamic_skills')
    @patch('clawgent.core.agent.BUILTIN_TOOLS', [])
    def test_create_agent_app_basic(self, mock_load_skills, mock_get_provider):
        """测试创建基础代理应用（带 Mock）"""
        from clawgent.core.agent import create_agent_app

        # Mock provider 返回值
        mock_provider = Mock()
        mock_provider.bind_tools.return_value = Mock()
        mock_get_provider.return_value = mock_provider

        # Mock 动态技能加载
        mock_load_skills.return_value = []

        try:
            app = create_agent_app(provider_name="openai", model_name="gpt-4o-mini")
            self.assertIsNotNone(app)
        except Exception as e:
            # 即使出现其他错误也记录
            print(f"Unexpected error: {e}")
            raise

    @patch('clawgent.core.agent.get_provider')
    @patch('clawgent.core.agent.load_dynamic_skills')
    @patch('clawgent.core.agent.BUILTIN_TOOLS', [])
    def test_create_agent_app_with_custom_tools(self, mock_load_skills, mock_get_provider):
        """测试创建带有自定义工具的代理应用（带 Mock）"""
        from clawgent.core.agent import create_agent_app
        from langchain_core.tools import tool

        # Mock provider 返回值
        mock_provider = Mock()
        mock_provider.bind_tools.return_value = Mock()
        mock_get_provider.return_value = mock_provider

        # Mock 动态技能加载
        mock_load_skills.return_value = []

        # 创建一个真正的 mock 工具（使用@tool 装饰器）
        @tool
        def mock_tool(test_param: str) -> str:
            """A mock tool for testing"""
            return f"mock result: {test_param}"

        try:
            app = create_agent_app(
                provider_name="openai",
                model_name="gpt-4o-mini",
                tools=[mock_tool]
            )
            self.assertIsNotNone(app)
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    @patch('clawgent.core.agent.get_provider')
    @patch('clawgent.core.agent.load_dynamic_skills')
    @patch('clawgent.core.agent.BUILTIN_TOOLS', [])
    def test_create_agent_app_with_checkpointer(self, mock_load_skills, mock_get_provider):
        """测试创建带有检查点的代理应用（带 Mock）"""
        from clawgent.core.agent import create_agent_app
        from langgraph.checkpoint.memory import MemorySaver

        # Mock provider 返回值
        mock_provider = Mock()
        mock_provider.bind_tools.return_value = Mock()
        mock_get_provider.return_value = mock_provider

        # Mock 动态技能加载
        mock_load_skills.return_value = []

        memory_saver = MemorySaver()
        try:
            app = create_agent_app(
                provider_name="openai",
                model_name="gpt-4o-mini",
                checkpointer=memory_saver
            )
            self.assertIsNotNone(app)
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    @patch('clawgent.core.agent.get_provider')
    @patch('clawgent.core.agent.load_dynamic_skills')
    @patch('clawgent.core.agent.BUILTIN_TOOLS', [])
    @patch('clawgent.core.agent.update_entity_ledger')
    def test_agent_node_with_entity_ledger_trimming(self, mock_update_ledger, mock_load_skills, mock_get_provider):
        """测试长对话达到裁剪阈值时，agent_node 触发 Entity Ledger 更新与 Prompt 注入"""
        from clawgent.core.agent import create_agent_app
        from clawgent.core.entity_ledger import EntityLedger, GlobalConstraint

        mock_load_skills.return_value = []
        mock_llm = Mock()
        mock_llm_with_tools = Mock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_get_provider.return_value = mock_llm

        # 模拟返回的更新白板
        sample_ledger = EntityLedger(
            global_constraints=[
                GlobalConstraint(
                    id="constraint_time",
                    content="只关注 2024 年以后的无监督方法",
                    source_turn=5,
                    status="active"
                )
            ]
        )
        mock_update_ledger.return_value = sample_ledger

        # 捕获最终传递给 mock_llm_with_tools.invoke 的消息列表
        captured_messages = []
        def fake_invoke(msgs, **kwargs):
            captured_messages.extend(msgs)
            return AIMessage(content="已按照 2024 年后的无监督要求开始分析。")

        mock_llm_with_tools.invoke.side_effect = fake_invoke
        mock_llm.invoke.return_value = AIMessage(content="旧对话摘要融合结果")

        app = create_agent_app(provider_name="openai", model_name="gpt-4o-mini")

        # 构造 42 个回合的消息，触发 trigger_turns=40 裁剪
        long_messages = []
        for i in range(42):
            long_messages.append(HumanMessage(content=f"提问第 {i+1} 回合"))
            long_messages.append(AIMessage(content=f"回答第 {i+1} 回合"))

        initial_state = {
            "messages": long_messages,
            "summary": "",
            "entity_ledger": {}
        }

        node = app.nodes["agent"]
        if hasattr(node, "invoke"):
            result_state = node.invoke(initial_state)
        elif hasattr(node, "runnable") and hasattr(node.runnable, "invoke"):
            result_state = node.runnable.invoke(initial_state)
        else:
            result_state = node(initial_state)

        # 验证 update_entity_ledger 被调用
        self.assertTrue(mock_update_ledger.called)
        # 验证 state_updates 中包含了 entity_ledger
        self.assertIn("entity_ledger", result_state)
        self.assertEqual(result_state["entity_ledger"]["global_constraints"][0]["content"], "只关注 2024 年以后的无监督方法")

        # 验证传递给模型的 SystemPrompt 中包含了 Entity Ledger 的硬约束
        sys_msgs = [m for m in captured_messages if isinstance(m, SystemMessage)]
        self.assertTrue(len(sys_msgs) > 0)
        self.assertIn("只关注 2024 年以后的无监督方法", sys_msgs[0].content)
        self.assertIn("Research State / Entity Ledger", sys_msgs[0].content)


if __name__ == '__main__':
    unittest.main()
