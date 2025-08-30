import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

MAX_MESSAGES = 10  # Keep last 10 messages (≈5 Q&A pairs)

# API Configuration
api_key = os.getenv("HG_API_KEY")
if not api_key:
    raise RuntimeError("Missing HG_API_KEY in environment. Please set it in your .env file.")

OPENAI_BASE = os.getenv("OPENAI_BASE", "https://router.huggingface.co/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-ai/DeepSeek-V3.1")

# Embeddings Configuration
emb = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
)
