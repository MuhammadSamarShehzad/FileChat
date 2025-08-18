from langchain_mistralai import ChatMistralAI
from .config import API_KEY

llm = ChatMistralAI(
    model="mistral-medium",
    api_key=API_KEY,
    temperature=0,
)