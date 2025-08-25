import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

 
def split_pdf_into_chunks(docs):
    """Split documents into chunks using stable text splitter."""
    logger.info(f"Splitting {len(docs)} documents into chunks")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents")
    return chunks