"""Pydantic models for AI search operations."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class AISearchRequest(BaseModel):
    """Request model for AI search."""
    query: str = Field(..., description="Natural language search query")
    mastercategory: Optional[str] = Field(None, description="Mastercategory (IT/NON_IT) - optional, if not provided will search all categories")
    category: Optional[str] = Field(None, description="Category namespace - optional, if not provided will search all categories")
    user_id: Optional[int] = Field(None, description="Optional user ID for tracking")
    top_k: Optional[int] = Field(20, description="Number of results to return (default: 20)")


class CandidateResult(BaseModel):
    """Individual candidate result model."""
    candidate_id: str
    resume_id: Optional[int]
    name: str
    category: Optional[str] = None
    mastercategory: Optional[str] = None
    designation: Optional[str] = Field(None, description="Candidate's current or most recent job title/designation")
    jobrole: Optional[str] = Field(None, description="Candidate's job role")
    experience_years: Optional[int] = None
    skills: List[str] = []
    location: Optional[str] = None
    score: float = Field(..., ge=0.0, le=100.0, description="Semantic similarity score as percentage (0-100)")
    fit_tier: str = Field(..., description="Fit tier: Perfect Match, Good Match, Partial Match, Low Match")


class AISearchResponse(BaseModel):
    """Response model for AI search."""
    query: str
    mastercategory: Optional[str] = Field(None, description="Mastercategory (IT/NON_IT) used for search (None if broad search)")
    category: Optional[str] = Field(None, description="Category namespace used for search (None if broad search)")
    total_results: int
    results: List[CandidateResult]
