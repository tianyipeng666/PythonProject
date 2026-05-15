import os

import httpx
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from secret_utils import decrypt_env_value


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

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好、耐心的 AI 助手。"),
    # 历史消息占位符，后面RunnableWithMessageHistory会把历史对话塞到这里
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])


def print_prompt_input(data: dict) -> dict:
    print("\n===== 即将传入 prompt 的 history 变量 =====")
    for index, message in enumerate(data.get("history", []), start=1):
        print(f"{index}. {message.type}: {message.content}")
    print("===== 即将传入 prompt 的 input 变量 =====")
    print(data["input"])
    return data


chain = RunnableLambda(print_prompt_input) | prompt | llm | StrOutputParser()

# 保存历史
store = {}
# 根据session_id找到对应的历史记录
def get_session_history(session_id: str):
    if session_id not in store:
        # 不存在则创建一个
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def print_memory(session_id: str, label: str):
    history = get_session_history(session_id)
    print(f"\n===== {label}: InMemoryChatMessageHistory.messages =====")
    if not history.messages:
        print("(empty)")
    for index, message in enumerate(history.messages, start=1):
        print(f"{index}. {message.type}: {message.content}")


# 将chain包装成带记忆的chain
conversation_chain = RunnableWithMessageHistory(
    chain,
    # RunnableWithMessageHistory每次执行时，会通过它拿到当前会话的历史消息，当前为拿到历史信息的函数
    get_session_history,
    # 告知RunnableWithMessageHistory用户当前输入在传入dict的哪个字段中
    input_messages_key="input",
    # 告知历史信息塞入prompt的哪个字段中
    history_messages_key="history",
)

config = {
    "configurable": {
        # 会话id，历史是会话级别的，具体见get_session_history
        "session_id": "user_001"
    }
}

session_id = config["configurable"]["session_id"]

print_memory(session_id, "第一次调用前")

# 根据session_id=user_001获取历史，此时历史为空，把空history塞进prompt，把input塞进human消息，调用llm，得到response1，把Human: 你好，我叫田毅鹏。和AI: ... 保存进user_001的历史
response1 = conversation_chain.invoke(
    {"input": "你好，我叫田毅鹏。"},
    config=config,
)
print("\n===== 第一次模型返回 =====")
print(response1)
print_memory(session_id, "第一次调用后")

# 取出user_001的历史，把历史塞进MessagesPlaceholder("history")，把当前input="我叫什么名字？" 塞进human消息，调用llm，模型看到之前你说“我叫田毅鹏”，回答你的名字，再把这一轮对话保存进历史
response2 = conversation_chain.invoke(
    {"input": "我叫什么名字？"},
    config=config,
)
print("\n===== 第二次模型返回 =====")
print(response2)
print_memory(session_id, "第二次调用后")
