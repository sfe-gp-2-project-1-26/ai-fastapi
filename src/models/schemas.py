from pydantic import BaseModel, Field
from typing import Optional, List

class SearchQueryRequest(BaseModel):
    query: str = Field(..., description="User search query in natural language (Arabic or English)")
    top_k: int = Field(default=10, description="Number of product IDs to return")
    min_price: Optional[float] = Field(default=None, description="Minimum price filter")
    max_price: Optional[float] = Field(default=None, description="Maximum price filter")

class SearchQueryResponse(BaseModel):
    query: str
    total_found: int
    product_ids: List[int] = Field(..., description="Sorted list of matched product_ids (1 to N)")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User chat message in Arabic or English")
    session_id: Optional[str] = Field(default=None, description="Session UUID for multi-turn conversation. Omit for new session.")
    top_k: int = Field(default=5, description="Number of catalog items to retrieve for context")

class ChatResponse(BaseModel):
    session_id: str = Field(..., description="Unique Session UUID")
    reply: str = Field(..., description="Gemini generated response")
    recommended_product_ids: List[int] = Field(..., description="List of product IDs retrieved and passed to Gemini")
