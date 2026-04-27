from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import httpx
import os
from secret_utils import decrypt_env_value

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

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个数学家，你可以计算任何算式"),
    ("human", "{text}"),
])

messages = chat_prompt.format_messages(text="我今年18岁，我的舅舅今年38岁，我的爷爷今年72岁，我和舅舅一共多少岁了？")

response = llm.invoke(messages)

# AIMessage
print(type(response))
print(response.content)
