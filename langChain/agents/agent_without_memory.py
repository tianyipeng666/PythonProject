from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import langchain
from dotenv import load_dotenv
import os
from secret_utils import decrypt_env_value
import httpx

langchain.debug = True


@tool
def math_calculator(expression: str) -> str:
    """用于数学计算，输入必须是纯数学表达式，如 '3+5' 或 '3**2'。"""
    print(f"\n[工具调用] 计算表达式: {expression}")
    try:
        print("只因为在人群中多看了你一眼，确认下你调用了我^_^")
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {str(e)}"


# 初始化大模型
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

agent = create_agent(
    model=llm,
    tools=[math_calculator],
    system_prompt="你是一个数学工具调用助手。"
        "只要用户提出任何数学计算问题，你必须调用 math_calculator 工具，"
        "不能自己心算，不能直接给出计算结果。"
        "拿到工具返回结果后，再用中文回答用户。",
    debug=True,
)

print("=== 测试：正常工具调用 ===")

try:
    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "计算3的平方",
            }
        ]
    })

    print("最终答案:", response["messages"][-1].content)

except Exception as e:
    print(f"请求失败: {str(e)}")
    print("建议：1. 检查网络连接 2. 降低 OpenAI 请求频率")