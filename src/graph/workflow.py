# src/workflow/qa_workflow.py

from __future__ import annotations

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage

from .state import QAState
from .nodes import make_load_docs, llm_answer, generate_alternative_queries
from src.db.database import db
from config import MAX_MESSAGES

# ---------------------------
# Reducers
# ---------------------------
def message_accumulator(state: dict) -> dict:
    """
    Accumulate messages for conversation context with memory management.

    Keeps the last N messages to avoid unbounded growth of conversation history.
    """
    current_messages = state.get("messages", [])
    question = state.get("question", "")
    answer = state.get("answer", "")

    new_messages = []
    if question:
        new_messages.append(HumanMessage(content=question))
    if answer:
        new_messages.append(AIMessage(content=answer))

    # Merge old and new
    all_messages = current_messages + new_messages

    # Smart memory management
    if len(all_messages) > MAX_MESSAGES:
        all_messages = all_messages[-MAX_MESSAGES:]

    return {"messages": all_messages}


# ---------------------------
# Workflow Builder
# ---------------------------
def create_workflow(vectorstore, k: int = 4):
    """
    Create and compile the QA workflow graph.

    Args:
        vectorstore: Vector store instance used for document retrieval.
        k (int, optional): Number of documents to retrieve. Defaults to 4.

    Returns:
        Compiled workflow graph with persistent checkpointing.
    """
    workflow = StateGraph(QAState)

    # Register nodes
    workflow.add_node("alternate_queries", generate_alternative_queries)
    workflow.add_node("LOAD_DOCS", make_load_docs(vectorstore, k=k))
    workflow.add_node("LLM_ANSWER", llm_answer)
    workflow.add_node("MESSAGE_ACCUMULATOR", message_accumulator)

    # Define edges
    workflow.add_edge(START, "alternate_queries")
    workflow.add_edge("alternate_queries", "LOAD_DOCS")
    workflow.add_edge("LOAD_DOCS", "LLM_ANSWER")
    workflow.add_edge("LLM_ANSWER", "MESSAGE_ACCUMULATOR")
    workflow.add_edge("MESSAGE_ACCUMULATOR", END)

    # Add persistence layer
    checkpointer = SqliteSaver(conn=db.get_langgraph_connection())

    return workflow.compile(checkpointer=checkpointer)
