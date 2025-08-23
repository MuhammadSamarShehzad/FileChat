from typing import Callable, List, Any

from langchain_core.documents import Document

from src.loader.pdf_loader import load_pdf_from_bytes as default_pdf_bytes_loader
from src.splitter.semantic_chunker import split_pdf_into_chunks as default_splitter
from src.vector_store.faiss_store import create_vector_store as default_vector_store_builder
from src.graph.workflow import create_workflow as default_graph_builder


def load_docs_from_pdf_bytes(name: str, data: bytes, *, pdf_bytes_loader: Callable[[bytes], List[Document]] = default_pdf_bytes_loader) -> List[Document]:
	"""Load a PDF provided as bytes into Documents using the project's loader."""
	return pdf_bytes_loader(data)


def build_graph_from_documents(
	documents: List[Document],
	*,
	splitter: Callable[[List[Document]], List[Document]] = default_splitter,
	vector_store_builder: Callable[[List[Document]], Any] = default_vector_store_builder,
	graph_builder: Callable[[Any, int], Any] = default_graph_builder,
	k: int = 4,
):
	"""Create a RAG graph from in-memory documents using provided components."""
	chunks = splitter(documents)
	vectorstore = vector_store_builder(chunks)
	graph = graph_builder(vectorstore, k=k)
	return graph


def ask_question(graph: Any, question: str, thread_id: str) -> str:
	"""Invoke the graph with a question and thread id, returning the answer string."""
	# Get existing chat history for conversation context
	from src.db.database import db
	existing_messages = db.get_chat_history(thread_id)
	
	# Convert database messages to LangChain message format
	from langchain_core.messages import HumanMessage, AIMessage
	langchain_messages = []
	for msg in existing_messages:
		if msg["role"] == "user":
			langchain_messages.append(HumanMessage(content=msg["content"]))
		elif msg["role"] == "assistant":
			langchain_messages.append(AIMessage(content=msg["content"]))
	
	# Initialize state with conversation history
	initial_state = {
		"question": question,
		"retrieved": [],
		"messages": langchain_messages
	}
	
	# Pass configurable thread_id for checkpointer state management
	result = graph.invoke(
		initial_state,
		config={"configurable": {"thread_id": thread_id}}
	)
	return result.get("answer", "") 
	