# Production configuration overrides
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Force CPU usage in production
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["USE_GPU"] = "False"

# Optimize for Railway environment
ROOT_DIR = Path(__file__).parent

# Database settings
class ProductionDatabaseConfig:
    """Production database configuration"""
    MONGODB_USERNAME = os.getenv("MONGODB_USERNAME")
    MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")
    MONGODB_HOST = os.getenv("MONGODB_HOST")
    MONGO_DB_NAME = os.getenv("MONGODB_DATABASE", "chatbot_db")
    MONGO_URI = f"mongodb+srv://{MONGODB_USERNAME}:{MONGODB_PASSWORD}{MONGODB_HOST}/{MONGO_DB_NAME}?retryWrites=true&w=majority"
    
    # ChromaDB - use memory for Railway (lightweight)
    CHROMA_PERSIST_PATH = os.getenv("CHROMA_PERSIST_PATH", "/tmp/chroma_db")
    CHROMA_COLLECTION = os.getenv("COLLECTION_NAME", "law_data")
    CHROMA_PERSIST_DIRECTORY = CHROMA_PERSIST_PATH

class ProductionAIConfig:
    """Production AI configuration - optimized for CPU"""
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    USE_GPU = False  # Force CPU
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    TOP_K = int(os.getenv("TOP_K", "3"))  # Reduce for faster processing
    MAX_TOKENS_PER_DOC = int(os.getenv("MAX_CONTEXT_LENGTH", 10000)) 

class ProductionPerformanceConfig:
    """Production performance settings"""
    CACHE_TTL_DAYS = int(os.getenv("CACHE_TTL_DAYS", "7"))
    MAX_CONVERSATION_TOKENS = int(os.getenv("MAX_CONVERSATION_TOKENS", 50000))

class ProductionAPIConfig:
    """Production API settings"""
    API_HOST = "0.0.0.0"
    API_PORT = int(os.getenv("PORT", 8001))

# Export for compatibility
DB_CONFIG = ProductionDatabaseConfig()
AI_CONFIG = ProductionAIConfig()
PERF_CONFIG = ProductionPerformanceConfig()
API_CONFIG = ProductionAPIConfig()

# Legacy exports
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

# Production data directories
DATA_DIR = "/tmp/data"
BENCHMARK_DIR = "/tmp/benchmark"
BENCHMARK_RESULTS_DIR = "/tmp/benchmark/results"