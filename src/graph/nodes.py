import logging
from .state import QAState
from src.llm.llm import llm
from src.utils.text_cleaner import clean_text
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


def make_load_docs(vectorstore, k: int = 4):
    """Factory to create a load_docs node that uses a captured vectorstore.

    The returned function matches LangGraph's expected node signature: (state) -> dict
    """
    def load_docs(state: QAState):
        query = state.get("question") or ""
        if not query:
            logger.warning("No question found in state; skipping retrieval.")
            return {"retrieved": []}
        try:
            results = vectorstore.similarity_search(query, k=k)
        except Exception as exc:
            logger.exception("Vector store retrieval failed: %s", exc)
            return {"retrieved": []}
        return {
            "retrieved": [r.page_content for r in results],
            "question": query,
        }

    return load_docs


def llm_answer(state: QAState):
    """Answer the question using the LLM."""
    retrieved = state.get("retrieved") or []
    question = state.get("question") or ""

    if not question:
        return {"answer": "No question provided."}

    context_str = "\n\n".join(retrieved[:10])

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. Use the provided context if relevant. "
                "If the context does not contain the answer, say 'I don't know'. "
                "Be concise."
            )
        ),
        HumanMessage(
            content=(
                f"Context:\n{context_str}\n\n"
                f"Question: {question}\n\n"
                "Answer concisely."
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        content = getattr(response, "content", str(response))
    except Exception as exc:
        logger.exception("LLM invocation failed: %s", exc)
        content = "I'm sorry, I couldn't generate an answer due to an internal error."

    return {"answer": clean_text(content)}