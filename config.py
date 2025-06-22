# backend/config.py (chỉ cập nhật phần import)
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent

DATA_DIR = os.path.join(ROOT_DIR, "data")
BENCHMARK_DIR = os.path.join(ROOT_DIR, "benchmark")
BENCHMARK_RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results")

class DatabaseConfig:
    """Cấu hình database - MongoDB và ChromaDB"""
    MONGODB_USERNAME = os.getenv("MONGODB_USERNAME")
    MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")
    MONGODB_HOST = os.getenv("MONGODB_HOST")
    MONGO_DB_NAME = os.getenv("MONGODB_DATABASE", "chatbot_db")
    MONGO_URI = f"mongodb+srv://{MONGODB_USERNAME}:{MONGODB_PASSWORD}{MONGODB_HOST}/{MONGO_DB_NAME}?retryWrites=true&w=majority"
    
    # ChromaDB Local Configuration
    CHROMA_PERSIST_PATH = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")
    CHROMA_COLLECTION = os.getenv("COLLECTION_NAME", "law_data")
    # Tạo đường dẫn tuyệt đối cho ChromaDB
    CHROMA_PERSIST_DIRECTORY = os.path.join(ROOT_DIR, CHROMA_PERSIST_PATH.lstrip('./'))

class AIConfig:
    """Cấu hình AI models và parameters"""
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL")
    USE_GPU = os.getenv("USE_GPU", "True").lower() == "true"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL")
    TOP_K = int(os.getenv("TOP_K"))
    MAX_TOKENS_PER_DOC = int(os.getenv("MAX_CONTEXT_LENGTH"))

class PerformanceConfig:
    """Cấu hình performance và caching"""
    CACHE_TTL_DAYS = int(os.getenv("CACHE_TTL_DAYS"))
    MAX_CONVERSATION_TOKENS = int(os.getenv("MAX_CONVERSATION_TOKENS"))

class APIConfig:
    """Cấu hình API server"""
    API_HOST = os.getenv("API_HOST")
    API_PORT = int(os.getenv("API_PORT"))

# Khởi tạo instances
DB_CONFIG = DatabaseConfig()
AI_CONFIG = AIConfig()
PERF_CONFIG = PerformanceConfig()
API_CONFIG = APIConfig()

# Legacy exports để tương thích ngược
MONGODB_USERNAME = DB_CONFIG.MONGODB_USERNAME
MONGODB_PASSWORD = DB_CONFIG.MONGODB_PASSWORD
MONGODB_HOST = DB_CONFIG.MONGODB_HOST
MONGO_DB_NAME = DB_CONFIG.MONGO_DB_NAME
MONGO_URI = DB_CONFIG.MONGO_URI

CHROMA_PERSIST_PATH = DB_CONFIG.CHROMA_PERSIST_PATH
CHROMA_COLLECTION = DB_CONFIG.CHROMA_COLLECTION
CHROMA_PERSIST_DIRECTORY = DB_CONFIG.CHROMA_PERSIST_DIRECTORY

EMBEDDING_MODEL_NAME = AI_CONFIG.EMBEDDING_MODEL_NAME
USE_GPU = AI_CONFIG.USE_GPU
GEMINI_API_KEY = AI_CONFIG.GEMINI_API_KEY
GEMINI_MODEL = AI_CONFIG.GEMINI_MODEL
TOP_K = AI_CONFIG.TOP_K
MAX_TOKENS_PER_DOC = AI_CONFIG.MAX_TOKENS_PER_DOC

CACHE_TTL_DAYS = PERF_CONFIG.CACHE_TTL_DAYS
MAX_CONVERSATION_TOKENS = PERF_CONFIG.MAX_CONVERSATION_TOKENS

API_HOST = API_CONFIG.API_HOST
API_PORT = API_CONFIG.API_PORT