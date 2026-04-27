import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


load_dotenv()


def _get_required_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError(f"请先设置环境变量: {' 或 '.join(names)}")


api_key = _get_required_env("EMBEDDINGS_API_KEY", "OPENAI_API_KEY")
base_url = os.getenv("EMBEDDINGS_BASE_URL") or os.getenv("OPENAI_BASE_URL")
model = os.getenv("EMBEDDINGS_MODEL") or "text-embedding-ada-002"

if base_url and "api.deepseek.com" in base_url:
    raise RuntimeError(
        "DeepSeek 官方 API 当前没有提供 embeddings 模型，不能用 "
        "OpenAIEmbeddings 直接调用 DeepSeek 生成向量。请把 "
        "EMBEDDINGS_BASE_URL/OPENAI_BASE_URL 改成支持 embeddings 的服务地址。"
    )

# 嵌入模型实例。DeepSeek 官方目前不能用于 embeddings，这里用于 OpenAI 或兼容 embeddings 的服务。
embeddings_model = OpenAIEmbeddings(
    model=model,
    api_key=api_key,
    base_url=base_url,
)

res1 = embeddings_model.embed_query("这是第一个测试文档")
print(res1)

res2 = embeddings_model.embed_documents(["这是第一个测试文档", "这是第二个测试文档"])
print(res2)
