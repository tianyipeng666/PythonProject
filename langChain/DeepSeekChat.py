from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from secret_utils import decrypt_env_value
import httpx
import os

load_dotenv()

encrypted_api_key = os.getenv("DEEPSEEK_API_KEY_ENCRYPTED")
api_key_password = ""
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

response = llm.invoke("你当前使用的是什么模型，确定我的用量信息，账户中已经花费了多少钱")
print(response.content)
