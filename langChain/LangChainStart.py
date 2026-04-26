from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os


load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=1.0,
)

response = llm.invoke("什么是大模型？")
print(response.content)
