# 1. 导入依赖包
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import langchain
from dotenv import load_dotenv
import os
from secret_utils import decrypt_env_value
import httpx
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import InMemorySaver

# 2. 设置 API 密钥
load_dotenv()
encrypted_api_key = os.getenv("DEEPSEEK_API_KEY_ENCRYPTED")
api_key_password = os.environ["DEEPSEEK_API_KEY_PASSWORD"]
api_key = decrypt_env_value(encrypted_api_key, api_key_password)

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=api_key,
    base_url=os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
    reasoning_effort="medium",
    extra_body={"thinking": {"type": "enabled"}},
    http_client=httpx.Client(trust_env=False),
    http_socket_options=(),
)

# 建议放到 .env:
# TAVILY_API_KEY=你的 Tavily Key
# OPENAI_API_KEY=你的 OpenAI Key
os.environ["TAVILY_API_KEY"] = "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"

# 3. 定义搜索工具
search_tool = TavilySearchResults(max_results=2)

# 5. 定义记忆组件
# 新版 create_agent 使用 checkpointer 保存同一个 thread_id 下的对话状态
checkpointer = InMemorySaver()

# 6. 创建 Agent
agent = create_agent(
    model=llm,
    tools=[search_tool],
    system_prompt=(
        "你是一个可以联网搜索的中文助手。"
        "当用户询问天气、新闻、实时信息时，必须使用 tavily_search_results_json 工具。"
        "请结合上下文理解用户的追问，例如“上海呢”表示继续询问上海的同类信息。"
    ),
    checkpointer=checkpointer,
    debug=True,
)

# 7. 同一个 thread_id 表示同一轮连续对话
config = {
    "configurable": {
        "thread_id": "weather_chat_001"
    }
}

# 8. 第一个查询
query1 = "请查询北京 2026-06-03 的天气预报？"

result1 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": query1,
            }
        ]
    },
    config=config,
)

print("查询结果:", result1["messages"][-1].content)

print("\n=== 继续对话 ===")

# 9. 第二个查询，会继承前面的上下文
query2 = "上海呢"

result2 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": query2,
            }
        ]
    },
    config=config,
)

print("分析结果:", result2["messages"][-1].content)