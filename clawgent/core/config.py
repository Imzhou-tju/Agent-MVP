import os
from dotenv import load_dotenv

load_dotenv()

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(CORE_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)

WORKSPACE_DIR = os.getenv("CLAWGENT_WORKSPACE", os.path.join(PROJECT_ROOT, "workspace"))


DB_PATH = os.path.join(WORKSPACE_DIR, "state.sqlite3")     # 状态机：潜意识与短期记忆
MEMORY_DIR = os.path.join(WORKSPACE_DIR, "memory")         # 显性记忆：Markdown 画像
PERSONAS_DIR = os.path.join(WORKSPACE_DIR, "personas")     # 人设区：系统 Prompt
SCRIPTS_DIR = os.path.join(WORKSPACE_DIR, "scripts")       # 脚本区：自动化武器库
OFFICE_DIR = os.path.join(WORKSPACE_DIR, "office")         # 沙盒工位 唯一被允许执行文件与shell操作的空间
SKILLS_DIR = os.path.join(OFFICE_DIR, "skills")            # 技能卡槽
TASKS_FILE = os.path.join(WORKSPACE_DIR, "tasks.json")

# ==================== RAG 知识库 ====================
KB_UPLOAD_DIR = os.path.join(WORKSPACE_DIR, "knowledge_base")   # 语料底库：放入待检索的 txt/md/pdf
KB_INDEX_DIR = os.path.join(WORKSPACE_DIR, "kb_index")          # Chroma 向量索引持久化目录

# 检索用远程 API（OpenAI 兼容）。缺省值指向 SiliconFlow，key 由 .env 注入。
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
RAG_EMBEDDING_API_KEY = os.getenv("RAG_EMBEDDING_API_KEY", "")
RAG_EMBEDDING_BASE_URL = os.getenv("RAG_EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")

RAG_RERANKER_MODEL = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RAG_RERANKER_API_KEY = os.getenv("RAG_RERANKER_API_KEY", "")
RAG_RERANKER_BASE_URL = os.getenv("RAG_RERANKER_BASE_URL", "https://api.siliconflow.cn/v1/rerank")

# 多查询扩写用的 chat 模型（OpenAI 兼容）。留空则复用 ANTHROPIC/OPENAI 主配置由 service 侧回退。
RAG_LLM_MODEL = os.getenv("RAG_LLM_MODEL", "DeepSeek-V4-Flash")
RAG_LLM_API_KEY = os.getenv("RAG_LLM_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))
RAG_LLM_BASE_URL = os.getenv("RAG_LLM_BASE_URL", "https://www.sophnet.com/api/open-apis/v1")

# 父块→子切片两级分层切分
RAG_PARENT_CHUNK_SIZE = int(os.getenv("RAG_PARENT_CHUNK_SIZE", "512"))
RAG_PARENT_CHUNK_OVERLAP = int(os.getenv("RAG_PARENT_CHUNK_OVERLAP", "64"))
RAG_CHILD_CHUNK_SIZE = int(os.getenv("RAG_CHILD_CHUNK_SIZE", "128"))
RAG_CHILD_CHUNK_OVERLAP = int(os.getenv("RAG_CHILD_CHUNK_OVERLAP", "16"))
# 四级降级文档切片尺寸（解析异常时逐级 fallback）
RAG_CHUNK_FALLBACK_SIZES = [128, 256, 512, 1024]
# 向后兼容保留（rerank 等处引用）
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "250"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
RAG_INITIAL_TOP_K = int(os.getenv("RAG_INITIAL_TOP_K", "15"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
# 多轮推理 RAG（IRCoT 式 retrieve-reason 循环）最大迭代轮数
RAG_MAX_ITERS = int(os.getenv("RAG_MAX_ITERS", "4"))

# PDF 解析后端：pypdf2（默认/离线）| mineru（MinerU 官方 API，解析质量高）
RAG_PDF_BACKEND = os.getenv("RAG_PDF_BACKEND", "pypdf2").lower()
MINERU_API_KEY = os.getenv("MINERU_API_KEY", "")
MINERU_API_BASE = os.getenv("MINERU_API_BASE", "https://mineru.net/api/v4")
# MinerU 异步任务轮询上限（秒）
MINERU_POLL_TIMEOUT = int(os.getenv("MINERU_POLL_TIMEOUT", "180"))

# Research (Multi-Agent 调研系统)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
RESEARCH_MAX_CONCURRENT = int(os.getenv("RESEARCH_MAX_CONCURRENT", "5"))

# 学术检索 MCP servers（原生 MCP 接入）。为空则退化为无学术源。
# 配置示例见 .env.example；键为 server 名，值为 langchain-mcp-adapters 的 connection dict。
ACADEMIC_MCP_ENABLED = os.getenv("ACADEMIC_MCP_ENABLED", "false").lower() == "true"
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")

for d in [WORKSPACE_DIR, MEMORY_DIR, PERSONAS_DIR, SCRIPTS_DIR, OFFICE_DIR, SKILLS_DIR, KB_UPLOAD_DIR, KB_INDEX_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"[Config] Workspace ready: {WORKSPACE_DIR}")