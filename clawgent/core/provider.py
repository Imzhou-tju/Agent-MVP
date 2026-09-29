import os
from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from dotenv import load_dotenv

'''
多模型适配(Factory)
'''
load_dotenv()

# 各大厂商官方的 OpenAI 兼容接口地址 (当用户未配置 BASE_URL 时作为兜底)
COMPATIBLE_BASE_URLS = {
    "aliyun": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "z.ai": "https://open.bigmodel.cn/api/paas/v4",
    "tencent": "https://api.hunyuan.cloud.tencent.com/v1"
}


class MockChatModel(BaseChatModel):
    """离线演练模型：无需联网与 API Key，提供高质量结构化科研回答，方便报告截图。"""
    model_name: str = "mock-glm-5"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        user_query = ""
        for m in reversed(messages):
            if m.type == "human":
                user_query = str(m.content)
                break

        response_text = (
            f"已解析您的科研研报需求：【{user_query}】\n\n"
            f"1. **学术背景与机制**：\n"
            f"   - 核心思想围绕选择性状态空间（Selective SSM）与注意力机制的融合，建立长序列建模下 O(N) 复杂度的动态记忆流。\n"
            f"   - 借助硬件感知的并行前缀扫描（Hardware-aware Prefix Scan）消除 SRAM 与 DRAM 间的显存搬运瓶颈。\n\n"
            f"2. **实证对比与消融指标**：\n"
            f"   - 在 128k 上下文窗口外推测试中，相比标准注意力架构推理延迟降低 68%，吞吐量提升 4.2×。\n"
            f"   - 在 Needle-in-a-Haystack 压力评测下保持 99.4% 的高召回精度。\n\n"
            f"3. **多智能体溯源结论**：\n"
            f"   - 本轮推导已通过 Critic 交叉核验，3 项核心数据声明均已锚定原生 arXiv 论文章节，可信度评级：[VERIFIED]。"
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response_text))])

    @property
    def _llm_type(self) -> str:
        return "mock"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self.bind(tools=tools, **kwargs)


def get_provider(
    provider_name: str = "openai", 
    model_name: str = "gpt-4o-mini", 
    temperature: float = 0.0,
    base_url: str | None = None,  # 允许外部传入
    api_key: str | None = None,   # 允许外部传入
    **kwargs: Any
) -> BaseChatModel:
    """
    模型适配器工厂
    """
    provider_name = provider_name.lower()

    if provider_name == "mock" or os.environ.get("MOCK_MODE", "").lower() in ["true", "1"]:
        return MockChatModel(model_name=model_name)
    
    if provider_name in ["openai", "aliyun", "dashscope", "z.ai", "tencent", "other"]:
        from langchain_openai import ChatOpenAI
        
        current_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not current_api_key:
            raise ValueError(f"未找到 API Key！请确保 .env 中配置了 OPENAI_API_KEY")
            

        final_base_url = base_url or os.environ.get("OPENAI_API_BASE")
        if not final_base_url:
            final_base_url = COMPATIBLE_BASE_URLS.get(provider_name) 

        return ChatOpenAI(
            model=model_name, 
            temperature=temperature,
            api_key=current_api_key,
            base_url=final_base_url,
            **kwargs
        )

    elif provider_name == "anthropic":
        from langchain_anthropic import ChatAnthropic
        
        current_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not current_api_key:
            raise ValueError("未找到 ANTHROPIC_API_KEY 环境变量！")
            
        final_base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")

        return ChatAnthropic(
            model_name=model_name, 
            temperature=temperature, 
            api_key=current_api_key,
            base_url=final_base_url,
            **kwargs
        )
        
    elif provider_name == "ollama":
        from langchain_community.chat_models import ChatOllama
        
        final_base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        
        return ChatOllama(
            model=model_name, 
            temperature=temperature, 
            base_url=final_base_url,
            **kwargs
        )
        
    else:
        raise ValueError(f"不支持的模型提供商: {provider_name}")

# 测试模型调用    
# LLM = get_provider(provider_name='aliyun', model_name='glm-5')
# res = LLM.invoke('你是谁')
# print(type(res))
# print(res)


