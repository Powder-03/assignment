from pydantic import BaseModel, Field
from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class RecommendationItem(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[RecommendationItem]
    end_of_conversation: bool

class ExtractorState(BaseModel):
    is_vague: bool
    is_out_of_scope: bool
    recommendations_ready: bool
    end_of_conversation: bool
    allowed_test_types: List[str]
    semantic_search_term: str
    items_to_compare: List[str]

class GeneratorResponse(BaseModel):
    assistant_reply: str
    selected_ids: List[int]
