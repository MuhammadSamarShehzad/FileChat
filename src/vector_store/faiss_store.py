import os
import logging
from langchain_community.vectorstores import FAISS
from config import emb

logger = logging.getLogger(__name__)

DEFAULT_FAISS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "faiss_index")
)


def create_vector_store(chunks, emb=emb):
    vectorstore = FAISS.from_documents(chunks, emb)
    return vectorstore


def save_vector_store(vectorstore: FAISS, path: str = DEFAULT_FAISS_DIR) -> None:
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)


def load_vector_store(path: str = DEFAULT_FAISS_DIR, emb=emb):
    index_file = os.path.join(path, "index.faiss")
    if os.path.isdir(path) and os.path.exists(index_file):
        try:
            return FAISS.load_local(path, emb, allow_dangerous_deserialization=True)
        except Exception as exc:
            logger.warning("Failed to load FAISS index at %s: %s", path, exc)
    return None


def load_or_create_vector_store(chunks, path: str = DEFAULT_FAISS_DIR, emb=emb):
    vs = load_vector_store(path, emb)
    if vs is not None:
        logger.info("Loaded existing FAISS index from %s", path)
        return vs
    logger.info("Creating new FAISS index at %s", path)
    vs = FAISS.from_documents(chunks, emb)
    save_vector_store(vs, path)
    return vs