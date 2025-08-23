from langchain_community.document_loaders import PDFPlumberLoader
from typing import List
from langchain_core.documents import Document
import tempfile


def load_pdf_from_bytes(data: bytes) -> List[Document]:
    """Load a PDF provided as bytes into Documents, without persisting it."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    
    try:
        loader = PDFPlumberLoader(tmp_path)
        docs = loader.load()
    finally:
        import os
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    return docs