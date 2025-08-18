from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from .utils import clean_for_indexing

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_store(pdf_path):
    raw_docs = PyPDFLoader(pdf_path).load()
    # Cleaned copies for indexing
    cleaned_docs = []
    for d in raw_docs:
        dc = d.copy()
        dc.page_content = clean_for_indexing(d.page_content)
        cleaned_docs.append(dc)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(cleaned_docs)

    vs = FAISS.from_documents(chunks, emb)
    bm25 = BM25Retriever.from_documents(chunks)
    intro = cleaned_docs[0].page_content[:1000] if cleaned_docs else ""
    return vs, bm25, raw_docs, intro

