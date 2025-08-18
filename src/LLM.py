from langchain_openai import ChatOpenAI
from .config import API_KEY

llm = ChatOpenAI(
    model="mistral-medium",
    openai_api_key=API_KEY,
    openai_api_base="https://api.mistral.ai/v1"
)