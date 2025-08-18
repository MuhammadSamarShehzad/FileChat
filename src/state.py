from typing import Dict, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated

class QAState(TypedDict):
    doc_text: str
    query: str
    queries: List[str]
    bm25_terms: List[str]
    retrieved: List[str]
    answer: Annotated[list[BaseMessage], add_messages]