from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.output_parsers import DatetimeOutputParser
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
    ("system","{format_instructions}"),
    ("human", "{request}")
])

output_parser = DatetimeOutputParser()

chain = chat_prompt | llm | output_parser
resp = chain.invoke({"request":"中华人民共和国是什么时候成立的",
                     "format_instructions":output_parser.get_format_instructions()})
print(resp)
print(type(resp))
