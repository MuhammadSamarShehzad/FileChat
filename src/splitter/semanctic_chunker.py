from langchain_experimental.text_splitter import SemanticChunker
from config import emb


def split_pdf_into_chunks(docs):
    splitter = SemanticChunker(emb)
    chunks = splitter.split_documents(docs)
    return chunks