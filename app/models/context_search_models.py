"""Pydantic models for context-based ATS search API."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ContextSearchRequest(BaseModel):
    """Request model for ats-context semantic candidate search."""

    role: str = Field(..., description="Target role/designation to search")
    skills: List[str] = Field(default_factory=list, description="Required/preferred skills")
    experience: Optional[str] = Field(
        "",
        description="Optional experience requirement, e.g. '3+ years'",
    )
    top_k: int = Field(20, ge=1, le=200, description="Max results to return")
    metadata_filter: Optional[Dict[str, Any]] = Field(
        None, description="Optional Pinecone metadata filter"
    )


class ContextCandidateResult(BaseModel):
    """ATS-compatible candidate result from context index."""

    candidate_id: str
    resume_id: Optional[int] = None
    name: Optional[str] = None
    category: Optional[str] = None
    mastercategory: Optional[str] = None
    designation: Optional[str] = None
    jobrole: Optional[str] = None
    experience_years: int = 0
    skills: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    email: Optional[str] = None
    mobile: Optional[str] = None
    filename: Optional[str] = None
    score: float = Field(..., ge=0.0, le=100.0)
    fit_tier: str


class ContextSearchResponse(BaseModel):
    """Final response model for context-search endpoint."""

    query: str
    total_results: int
    results: List[ContextCandidateResult]
    search_type: str = "semantic"
    results_from: str = "context_index"

