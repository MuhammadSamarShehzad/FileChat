import logging
from langchain_community.document_loaders import PDFPlumberLoader
from typing import List
from langchain_core.documents import Document
import tempfile

logger = logging.getLogger(__name__)


def load_pdf_from_bytes(data: bytes) -> List[Document]:
    """Load a PDF provided as bytes into Documents, without persisting it."""
    logger.info("Loading PDF from bytes")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
        logger.debug(f"Temporary PDF file created at: {tmp_path}")
    
    try:
        logger.debug("Loading PDF with PDFPlumberLoader")
        loader = PDFPlumberLoader(tmp_path)
        docs = loader.load()
        logger.info(f"Successfully loaded {len(docs)} documents from PDF")
        return docs
    except Exception as e:
        logger.error(f"Failed to load PDF: {e}")
        raise
    finally:
        import os
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.debug("Temporary PDF file removed")