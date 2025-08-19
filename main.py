import logging
import os
from src.graph.workflow import create_workflow
from src.vector_store.faiss_store import load_or_create_vector_store
from src.splitter.semantic_chunker import split_pdf_into_chunks
from src.loader.pdf_loader import load_pdf


# Configure logging to console and to a file in a dynamic project-relative path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

logging.basicConfig(
	level=logging.INFO,
	handlers=[
		logging.FileHandler(LOG_FILE),
	],
)
logger = logging.getLogger(__name__)


def run():
	docs = load_pdf()
	chunks = split_pdf_into_chunks(docs)
	vectorstore = load_or_create_vector_store(chunks)
	graph = create_workflow(vectorstore, k=4)

	config1 = {"configurable": {"thread_id": "1"}}

	input = {"question": "what is name of the person?"}
	result = graph.invoke(input, config=config1)
	answer = result.get("answer", "")
	logger.info("Answer: %s", answer)
	print(answer)


if __name__ == "__main__":
	run()