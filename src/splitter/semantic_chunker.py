from langchain_text_splitters import RecursiveCharacterTextSplitter

 
def split_pdf_into_chunks(docs):
    """Split documents into chunks using stable text splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
	chunks = splitter.split_documents(docs)
	return chunks 