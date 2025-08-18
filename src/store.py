from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from .config import EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from .utils import clean_for_indexing

@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def build_store(pdf_path: str):
    raw_docs = PyPDFLoader(pdf_path).load()
    cleaned_docs = []
    for d in raw_docs:
        dc = d.copy()
        dc.page_content = clean_for_indexing(d.page_content)
        cleaned_docs.append(dc)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(cleaned_docs)

    vs = FAISS.from_documents(chunks, get_embeddings())
    bm25 = BM25Retriever.from_documents(chunks)
    intro = cleaned_docs[0].page_content[:1000] if cleaned_docs else ""
    return vs, bm25, raw_docs, intro