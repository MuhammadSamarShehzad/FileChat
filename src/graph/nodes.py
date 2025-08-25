import logging
from .state import QAState
from src.llm.llm import llm
from src.utils.text_cleaner import clean_text
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)


def make_load_docs(vectorstore, k: int = 4):
    """Factory to create a load_docs node that uses a captured vectorstore,
    retrieving docs for both the main query and its alternatives.
    """
    def load_docs(state: QAState):
        logger.info("Loading documents for query retrieval")
        query = state.get("question") or ""
        alternatives = state.get("alternative_queries", [])
        logger.debug(f"Main query: {query}, Alternative queries: {alternatives}")

        if not query and not alternatives:
            logger.warning("No question or alternatives found in state; skipping retrieval.")
            return {"retrieved": [], "question": query, "alternative_queries": alternatives}

        retrieved_docs = []

        def _search(q: str):
            try:
                return vectorstore.similarity_search(q, k=k)
            except Exception as exc:
                logger.exception("Vector store retrieval failed for query '%s': %s", q, exc)
                return []

        # Search for main query
        if query:
            retrieved_docs.extend(_search(query))

        # Search for alternative queries
        for alt in alternatives:
            retrieved_docs.extend(_search(alt))

        # Deduplicate while preserving order
        seen = set()
        unique_docs = []
        for r in retrieved_docs:
            if r.page_content not in seen:
                seen.add(r.page_content)
                unique_docs.append(r.page_content)

        return {
            "retrieved": unique_docs,
            "question": query
        }

    return load_docs



def select_relevant_context(question: str, messages: list, max_messages: int = 5):
    """Smartly select only relevant conversation context instead of entire history."""
    if not messages or len(messages) <= max_messages:
        return messages
    
    # STRATEGY 1: Recent context (current implementation)
    # Return last N messages for immediate conversation continuity
    recent_messages = messages[-max_messages:]
    
    # TODO: STRATEGY 2: Semantic relevance (future enhancement)
    # Could use embeddings to find most semantically relevant messages
    # This would be more intelligent than just recent messages
    
    # TODO: STRATEGY 3: Topic-based clustering
    # Group messages by topic and select relevant topic clusters
    
    return recent_messages


def llm_answer(state: QAState):
    """Answer the question using the LLM with smart conversation context selection."""
    logger.info("Generating answer using LLM")
    retrieved = state.get("retrieved") or []
    question = state.get("question") or ""
    messages = state.get("messages", [])  # Get existing conversation history
    logger.debug(f"Retrieved {len(retrieved)} documents, {len(messages)} conversation messages")

    if not question:
        logger.warning("No question provided to LLM")
        return {"answer": "No question provided."}

    # SMART CONTEXT SELECTION: Only send relevant conversation history
    relevant_messages = select_relevant_context(question, messages, max_messages=20)
    
    # Build conversation context with limited history
    conversation_messages = []
    
    # Add system message
    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. Use the provided context if relevant. "
            "If the context does not contain the answer, say 'I don't know'. "
            "Be concise. You can reference recent conversation context when relevant."
        )
    )
    conversation_messages.append(system_message)
    
    # Add only relevant conversation history (not entire chat)
    if relevant_messages:
        conversation_messages.extend(relevant_messages)
    
    # Add current question with retrieved context
    context_str = "\n\n".join(retrieved[:10]) if retrieved else "No relevant context found."
    
    human_message = HumanMessage(
        content=(
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Answer concisely."
        )
    )
    conversation_messages.append(human_message)

    try:
        response = llm.invoke(conversation_messages)
        content = getattr(response, "content", str(response))
    except Exception as exc:
        logger.exception("LLM invocation failed: %s", exc)
        content = "I'm sorry, I couldn't generate an answer due to an internal error."

    return {"answer": clean_text(content)}


def generate_alternative_queries(state: QAState):
    """Generate alternative queries to improve document retrieval."""
    logger.info("Generating alternative queries")
    question = state.get("question") or ""
    logger.debug(f"Original question: {question}")
    if not question:
        logger.warning("No question provided for alternative query generation")
        return {"alternative_queries": []}

    messages = [
        SystemMessage(
            content=(
                "You are an expert query reformulator. Given a user's question, "
                "generate 3 concise and relevant alternative queries that could help "
                "in retrieving better documents. Avoid redundancy."
            )
        ),
        HumanMessage(
            content=(
                f"Original Question: {question}\n\n"
                "Provide 3 alternative queries."
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        content = getattr(response, "content", str(response))
        # Split response into separate queries
        alternatives = [clean_text(q) for q in content.split('\n') if q.strip()]
        # Limit to 3 alternatives
        alternatives = alternatives[:3]
    except Exception as exc:
        logger.exception("LLM invocation for alternative queries failed: %s", exc)
        alternatives = []

    logger.info(f"Following alternative queries generated: \n{alternatives}\n")
    return {"alternative_queries": alternatives}