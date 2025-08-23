from .state import QAState
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite import SqliteSaver
from .nodes import make_load_docs, llm_answer
from src.db.database import db


def create_workflow(vectorstore, k: int = 4):
    workflow = StateGraph(QAState)

    workflow.add_node("LOAD_DOCS", make_load_docs(vectorstore, k=k))
    workflow.add_node("LLM_ANSWER", llm_answer)

    workflow.add_edge(START, "LOAD_DOCS")
    workflow.add_edge("LOAD_DOCS", "LLM_ANSWER")
    workflow.add_edge("LLM_ANSWER", END)

    # Message accumulator reducer - builds conversation history over time
    def message_accumulator(state):
        """Accumulate messages for conversation context with smart memory management."""
        current_messages = state.get("messages", [])
        question = state.get("question", "")
        answer = state.get("answer", "")
        
        # Add current Q&A to conversation history
        new_messages = []
        if question:
            from langchain_core.messages import HumanMessage, AIMessage
            new_messages.append(HumanMessage(content=question))
        if answer:
            new_messages.append(AIMessage(content=answer))
        
        # Combine with existing messages
        all_messages = current_messages + new_messages
        
        # SMART MEMORY MANAGEMENT: Keep only recent relevant messages
        # This prevents the conversation from growing indefinitely
        max_messages = 10  # Keep last 10 messages (5 Q&A pairs)
        if len(all_messages) > max_messages:
            all_messages = all_messages[-max_messages:]
        
        return {"messages": all_messages}

    # Add the reducer to the workflow
    workflow.add_node("MESSAGE_ACCUMULATOR", message_accumulator)
    workflow.add_edge("LLM_ANSWER", "MESSAGE_ACCUMULATOR")
    workflow.add_edge("MESSAGE_ACCUMULATOR", END)

    # Use SQLite checkpointer for persistent state
    checkpointer = SqliteSaver(conn=db.get_langgraph_connection())

    graph = workflow.compile(checkpointer=checkpointer)

    return graph
