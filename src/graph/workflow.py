from .state import QAState
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import InMemorySaver
from .nodes import make_load_docs, llm_answer


def create_workflow(vectorstore, k: int = 4):
    workflow = StateGraph(QAState)

    workflow.add_node("LOAD_DOCS", make_load_docs(vectorstore, k=k))
    workflow.add_node("LLM_ANSWER", llm_answer)

    workflow.add_edge(START, "LOAD_DOCS")
    workflow.add_edge("LOAD_DOCS", "LLM_ANSWER")
    workflow.add_edge("LLM_ANSWER", END)

    checkpointer = InMemorySaver()

    graph = workflow.compile(checkpointer=checkpointer)

    return graph
