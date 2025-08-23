import logging
from langchain_community.vectorstores import FAISS
from config import emb

logger = logging.getLogger(__name__)


def create_vector_store(chunks):
    """Create a FAISS vector store from document chunks."""
    return FAISS.from_documents(chunks, emb)