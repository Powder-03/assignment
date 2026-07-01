import os

def load_dotenv():
    # Helper to load .env variables into os.environ
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

load_dotenv()

class Settings:
    # Groq API Configurations (For Chat Completions)
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "dummy")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_LLM_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_FALLBACK_MODEL: str = "llama-3.1-8b-instant"
    
    # Gemini AI Studio Configurations (Primary for completions)
    GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "dummy")
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    GEMINI_LLM_MODEL: str = os.environ.get("GEMINI_LLM_MODEL", "gemini-2.5-flash")
    
    # OpenRouter API Configurations (For Embeddings)
    OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "dummy")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    EMBEDDING_MODEL: str = "google/gemini-embedding-2"
    EMBEDDING_DIMENSIONS: int = 768
    
    # File Paths
    CATALOG_FILE: str = "clean_catalog.json"
    INDEX_FILE: str = "index.faiss"
    
    # Latency/Timeout Limits (in seconds)
    STATE_EXTRACTOR_TIMEOUT: float = 8.0
    CLARIFY_GENERATOR_TIMEOUT: float = 8.0
    RESPONSE_GENERATOR_TIMEOUT: float = 8.0
    EMBEDDING_TIMEOUT: float = 6.0

settings = Settings()
