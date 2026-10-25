import os
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "postgres").lower()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_MIN_POOL = int(os.getenv("POSTGRES_MIN_POOL", "1"))
POSTGRES_MAX_POOL = int(os.getenv("POSTGRES_MAX_POOL", "10"))

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
