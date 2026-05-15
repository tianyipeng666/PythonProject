import os

import httpx
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
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

router_parser = JsonOutputParser()
text_parser = StrOutputParser()

router_prompt = ChatPromptTemplate.from_template(
    """你是一个路由分类器，请根据用户问题选择最合适的目的地。

可选目的地：
- math: 数学、计算、方程、数字推理
- history: 历史事件、历史人物、朝代、战争
- programming: 编程、代码、软件开发、调试
- default: 其他无法明确归类的问题

只返回 JSON，不要输出其他内容。
{format_instructions}

用户问题：{input}
"""
).partial(format_instructions=router_parser.get_format_instructions())

math_prompt = ChatPromptTemplate.from_template(
    "你是一个严谨的数学老师，请一步一步解答这个问题：\n{input}"
)
history_prompt = ChatPromptTemplate.from_template(
    "你是一个历史老师，请用清晰的时间线回答这个问题：\n{input}"
)
programming_prompt = ChatPromptTemplate.from_template(
    "你是一个资深程序员，请用简洁准确的方式回答这个问题：\n{input}"
)
default_prompt = ChatPromptTemplate.from_template(
    "请用清晰、准确、易懂的方式回答这个问题：\n{input}"
)

destination_chains = {
    "math": math_prompt | llm | text_parser,
    "history": history_prompt | llm | text_parser,
    "programming": programming_prompt | llm | text_parser,
    "default": default_prompt | llm | text_parser,
}


def route(info: dict) -> str:
    destination = info["route"].get("destination", "default")
    selected_chain = destination_chains.get(destination, destination_chains["default"])
    return selected_chain.invoke({"input": info["input"]})


router_chain = router_prompt | llm | router_parser

chain = (
    RunnablePassthrough.assign(route=router_chain)
    | RunnableLambda(route)
)

response = chain.invoke({
    "input": "秦始皇统一六国是在什么时候？"
})

print(response)
