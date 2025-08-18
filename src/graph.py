from .state import QAState
from langgraph.graph import StateGraph, END
from .nodes import gen_answer, gen_queries, gen_bm25_terms, retrieve
from langgraph.checkpoint.memory import InMemorySaver

def build_graph(vs, bm25, raw_docs):
    g = StateGraph(QAState)
    g.add_node("gen_queries", gen_queries)
    g.add_node("gen_bm25_terms", gen_bm25_terms)
    g.add_node("retrieve", lambda s: retrieve(s, vs, bm25, raw_docs))
    g.add_node("gen_answer", gen_answer)

    g.add_edge("gen_queries", "gen_bm25_terms")
    g.add_edge("gen_bm25_terms", "retrieve")
    g.add_edge("retrieve", "gen_answer")
    g.add_edge("gen_answer", END)
    g.set_entry_point("gen_queries")

    checkpointer = InMemorySaver()
    return g.compile(checkpointer=checkpointer), checkpointer