from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from secret_utils import decrypt_env_value
import httpx
import os

load_dotenv()

encrypted_api_key = os.getenv("DEEPSEEK_API_KEY_ENCRYPTED")
api_key_password = os.environ["DEEPSEEK_API_KEY_PASSWORD"]
api_key = decrypt_env_value(encrypted_api_key, api_key_password)

# 对话模型
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=api_key,
    base_url=os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
    reasoning_effort="medium",
    extra_body={"thinking": {"type": "enabled"}},
    http_client=httpx.Client(trust_env=False),
    http_socket_options=(),
)

messages = [
    SystemMessage(content="你是一个诗人"),
    HumanMessage(content="写一首关于夏天的诗")
]
response = llm.invoke(messages)
# AIMessage
print(type(response))
print(response.content)
