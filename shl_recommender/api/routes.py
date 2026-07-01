from fastapi import APIRouter, Request, HTTPException
from shl_recommender.domain.schemas import ChatRequest, ChatResponse
from shl_recommender.services.pipeline import run_recommender_pipeline

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, http_request: Request):
    """
    POST /chat endpoint.
    Retrieves global catalog, FAISS index, Groq client (aclient), and OpenRouter client (embed_client)
    from app state, and runs the recommender pipeline coordinator.
    """
    catalog = getattr(http_request.app.state, "catalog", None)
    index = getattr(http_request.app.state, "index", None)
    gemini_client = getattr(http_request.app.state, "gemini_client", None)
    groq_clients = getattr(http_request.app.state, "groq_clients", None)
    embed_client = getattr(http_request.app.state, "embed_client", None)
    
    if not catalog or index is None or not gemini_client or not groq_clients or not embed_client:
        raise HTTPException(
            status_code=503,
            detail="The service is starting up or missing catalogue/client components."
        )
        
    response = await run_recommender_pipeline(request, catalog, index, gemini_client, groq_clients, embed_client)
    return response
