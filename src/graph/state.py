from typing import TypedDict, List
from langchain_core.messages import BaseMessage


class QAState(TypedDict, total=False):
    question: str
    retrieved: List[str]
    answer: str
    messages: List[BaseMessage]  # Chat history for conversation context