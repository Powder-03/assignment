import os

class Settings:
    # Groq API Configurations (For Chat Completions)
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "dummy")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_LLM_MODEL: str = "llama-3.1-8b-instant"
    
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
