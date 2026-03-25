"""Pydantic models for standalone ai-search-2 endpoint."""
from typing import Optional, List
from pydantic import BaseModel, Field, model_validator

from app.models.ai_search_models import CandidateResult


class AISearch2Request(BaseModel):
    """Query-only request for ai-search-2."""

    query: str = Field(..., description="Natural language search query")
    user_id: Optional[int] = Field(None, description="Optional user ID for tracking")
    top_k: Optional[int] = Field(100, description="Number of results (default: 100)")

    @model_validator(mode="before")
    @classmethod
    def _accept_top_k_with_whitespace_key(cls, data):
        """
        Some clients accidentally send `"top_k "` (trailing space) instead of `top_k`.
        Accept it and map to the correct field.
        """
        if isinstance(data, dict):
            if "top_k" not in data:
                for k, v in data.items():
                    if isinstance(k, str) and k.strip() == "top_k":
                        data["top_k"] = v
                        break
        return data


class AISearch2Response(BaseModel):
    """Response for ai-search-2."""

    query: str
    total_results: int
    results: List[CandidateResult]
    search_type: Optional[str] = Field(None, description="'semantic' | 'name' | 'hybrid'")
    results_from: Optional[str] = Field(None, description="'primary' | 'name'")
