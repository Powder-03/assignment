import faiss
import asyncio
import numpy as np
from typing import List, Optional
from openai import AsyncOpenAI
from shl_recommender.core.config import settings

def load_index(index_path: str) -> Optional[faiss.IndexIDMap]:
    """
    Loads FAISS index from disk.
    """
    try:
        return faiss.read_index(index_path)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

async def get_query_embedding(query_text: str, embed_client: AsyncOpenAI) -> Optional[List[float]]:
    """
    Calls OpenRouter embeddings API using embed_client to embed query_text. Retries once on failure.
    """
    if not query_text or not query_text.strip():
        return None
        
    try:
        response = await asyncio.wait_for(
            embed_client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=[query_text],
                dimensions=settings.EMBEDDING_DIMENSIONS,
                encoding_format="float"
            ),
            timeout=settings.EMBEDDING_TIMEOUT
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}. Retrying once...")
        try:
            response = await asyncio.wait_for(
                embed_client.embeddings.create(
                    model=settings.EMBEDDING_MODEL,
                    input=[query_text],
                    dimensions=settings.EMBEDDING_DIMENSIONS,
                    encoding_format="float"
                ),
                timeout=settings.EMBEDDING_TIMEOUT
            )
            return response.data[0].embedding
        except Exception as e2:
            print(f"Embedding retry error: {e2}. Returning None.")
            return None

def search_index(
    query_embedding: Optional[List[float]],
    allowed_ids: List[int],
    faiss_index: faiss.IndexIDMap
) -> List[int]:
    """
    Queries FAISS index using SearchParameters with ID selector.
    If query_embedding is None, degrades to returning the allowed_ids in catalog order (first 10).
    """
    if not allowed_ids:
        return []

    if query_embedding is None:
        print("Query embedding is None. Degrading to catalog order.")
        return allowed_ids[:7]

    try:
        allowed_ids_np = np.array(allowed_ids, dtype=np.int64)
        selector = faiss.IDSelectorBatch(allowed_ids_np)
        
        query_vector = np.array([query_embedding], dtype=np.float32)
        params = faiss.SearchParameters(sel=selector)
        
        distances, indices = faiss_index.search(query_vector, 7, params=params)
        return [int(idx) for idx in indices[0] if idx != -1]
    except Exception as e:
        print(f"FAISS search failed: {e}. Falling back to catalog order.")
        return allowed_ids[:7]
