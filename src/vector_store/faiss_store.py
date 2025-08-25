import logging
from langchain_community.vectorstores import FAISS
from config import emb

logger = logging.getLogger(__name__)


def create_vector_store(chunks):
    """Create a FAISS vector store from document chunks."""
    logger.info(f"Creating vector store with {len(chunks)} chunks")
    try:
        vectorstore = FAISS.from_documents(chunks, emb)
        logger.info("Vector store created successfully")
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise