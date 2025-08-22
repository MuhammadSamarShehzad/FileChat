from langchain_community.document_loaders import PDFPlumberLoader
from typing import List
from langchain_core.documents import Document
import os
import tempfile

def load_pdf():
    loader = PDFPlumberLoader(r"/home/muhammad-samar/Projects/FileChat/data/Muhammad Samar Shehzad.pdf")
    docs = loader.load()
    return docs


def load_pdf_from_bytes(data: bytes) -> List[Document]:
    """Load a PDF provided as bytes into Documents, without persisting it.

    Uses a temporary file only for the duration of loading, then removes it.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        loader = PDFPlumberLoader(tmp_path)
        docs = loader.load()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return docs