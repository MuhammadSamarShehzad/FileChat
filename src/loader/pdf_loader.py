from langchain_community.document_loaders import PDFPlumberLoader

def load_pdf():
    loader = PDFPlumberLoader(r"/home/muhammad-samar/Projects/FileChat/data/Muhammad Samar Shehzad.pdf")
    docs = loader.load()
    return docs