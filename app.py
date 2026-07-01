import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from shl_recommender.core.config import settings
from shl_recommender.services import catalog as catalog_svc
from shl_recommender.services import search as search_svc
from shl_recommender.api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown lifecycle:
    - Loads the cleaned catalog JSON O(1) map.
    - Loads the FAISS vector index.
    - Instantiates the Groq AsyncOpenAI client for chat completions.
    - Instantiates the OpenRouter AsyncOpenAI client for query embeddings.
    - Mounts them on app.state for dependency injection.
    """
    print("Application booting up...")
    
    # 1. Load catalog
    catalog_data = catalog_svc.load_catalog(settings.CATALOG_FILE)
    app.state.catalog = catalog_data
    print(f"Loaded catalog containing {len(catalog_data)} items.")

    # 2. Load FAISS index
    index_data = search_svc.load_index(settings.INDEX_FILE)
    app.state.index = index_data
    if index_data is not None:
        print(f"Loaded FAISS index containing {index_data.ntotal} items.")
    else:
        print("FAISS index loading skipped or failed.")

    # 3. Initialize Groq AsyncOpenAI client (for completions)
    groq_api_key = os.environ.get("GROQ_API_KEY", settings.GROQ_API_KEY)
    app.state.groq_client = AsyncOpenAI(
        base_url=settings.GROQ_BASE_URL,
        api_key=groq_api_key
    )
    print("Groq AsyncOpenAI client initialized.")

    # 4. Initialize Gemini AsyncOpenAI client (for completions)
    gemini_api_key = os.environ.get("GEMINI_API_KEY", settings.GEMINI_API_KEY)
    app.state.gemini_client = AsyncOpenAI(
        base_url=settings.GEMINI_BASE_URL,
        api_key=gemini_api_key
    )
    print("Gemini AsyncOpenAI client initialized.")

    # 5. Initialize OpenRouter AsyncOpenAI client (for embeddings)
    or_api_key = os.environ.get("OPENROUTER_API_KEY", settings.OPENROUTER_API_KEY)
    app.state.embed_client = AsyncOpenAI(
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=or_api_key
    )
    print("OpenRouter AsyncOpenAI client initialized.")

    yield
    
    print("Application shutting down...")

# Initialize FastAPI application with clean lifecycle management
app = FastAPI(
    title="SHL Assessment Recommender API",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the modular API endpoints router
app.include_router(router)
