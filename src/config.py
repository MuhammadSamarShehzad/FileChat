from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# 🔑 API Keys
API_KEY = os.getenv("MISTRAL_API_KEY")


# ⚙️ Model Configs
LLM_MODEL_NAME = "openai/gpt-oss-120b:cerebras"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking Configs
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50