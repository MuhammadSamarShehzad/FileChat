import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

api_key = os.getenv("HG_API_KEY")
if not api_key:
    raise RuntimeError("Missing HG_API_KEY in environment. Please set it in your .env file.")

OPENAI_BASE = os.getenv("OPENAI_BASE", "https://router.huggingface.co/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b:cerebras")

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
