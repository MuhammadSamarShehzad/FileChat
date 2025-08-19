from langchain_openai import ChatOpenAI
from config import api_key, OPENAI_BASE, OPENAI_MODEL

llm = ChatOpenAI(
    openai_api_base=OPENAI_BASE,
    openai_api_key=api_key,
    model=OPENAI_MODEL,
)