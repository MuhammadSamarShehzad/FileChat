import logging
from langchain_openai import ChatOpenAI
from config import api_key, OPENAI_BASE, OPENAI_MODEL

logger = logging.getLogger(__name__)

logger.info(f"Initializing LLM with model: {OPENAI_MODEL} and base: {OPENAI_BASE}")
llm = ChatOpenAI(
    openai_api_base=OPENAI_BASE,
    openai_api_key=api_key,
    model=OPENAI_MODEL,
)
logger.info("LLM initialized successfully")