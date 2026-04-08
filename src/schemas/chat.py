from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None


class ContextItem(BaseModel):
    source: str
    text: str


class ChatResponse(BaseModel):
    answer: str
    relevant_context: List[ContextItem]
    metadata: Dict[str, Any] = {}