"""Pydantic models for standalone ai-search-2 endpoint."""
from typing import Optional, List
from pydantic import BaseModel, Field

from app.models.ai_search_models import CandidateResult


class AISearch2Request(BaseModel):
    """Query-only request for ai-search-2."""

    query: str = Field(..., description="Natural language search query")
    user_id: Optional[int] = Field(None, description="Optional user ID for tracking")
    top_k: Optional[int] = Field(100, description="Number of results (default: 100)")


class AISearch2Response(BaseModel):
    """Response for ai-search-2."""

    query: str
    total_results: int
    results: List[CandidateResult]
    search_type: Optional[str] = Field(None, description="'semantic' | 'name' | 'hybrid'")
    results_from: Optional[str] = Field(None, description="'primary' | 'name'")
