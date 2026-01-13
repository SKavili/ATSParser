"""Pydantic models for AI search operations."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class AISearchRequest(BaseModel):
    """Request model for AI search."""
    query: str = Field(..., description="Natural language search query")
    user_id: Optional[int] = Field(None, description="Optional user ID for tracking")
    top_k: Optional[int] = Field(20, description="Number of results to return (default: 20)")


class CandidateResult(BaseModel):
    """Individual candidate result model."""
    candidate_id: str
    resume_id: Optional[int]
    name: str
    category: Optional[str] = None
    mastercategory: Optional[str] = None
    experience_years: Optional[int] = None
    skills: List[str] = []
    location: Optional[str] = None
    score: float = Field(..., ge=0.0, le=1.0, description="Semantic similarity score")
    fit_tier: str = Field(..., description="Fit tier: Perfect Match, Good Match, Partial Match, Low Match")


class AISearchResponse(BaseModel):
    """Response model for AI search."""
    query: str
    total_results: int
    results: List[CandidateResult]
