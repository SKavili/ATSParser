"""Pydantic models for AI search operations."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


def _normalize_mastercategory(value: Optional[str]) -> Optional[str]:
    """Normalize mastercategory to canonical IT or NON_IT (e.g. 'Non - IT' -> 'NON_IT')."""
    if value is None or not str(value).strip():
        return None
    s = str(value).strip()
    # Collapse spaces/hyphens to single token and lowercase for comparison
    normalized = s.replace(" ", "").replace("-", "").replace("_", "").lower()
    if normalized == "it":
        return "IT"
    if normalized == "nonit":
        return "NON_IT"
    # Already canonical
    if s.upper() in ("IT", "NON_IT"):
        return s.upper()
    return s


class AISearchRequest(BaseModel):
    """Request model for AI search."""
    query: str = Field(..., description="Natural language search query")
    mastercategory: Optional[str] = Field(None, description="Mastercategory (IT/NON_IT) - optional, if not provided will search all categories")
    category: Optional[str] = Field(None, description="Category namespace - optional, if not provided will search all categories")
    user_id: Optional[int] = Field(None, description="Optional user ID for tracking")
    top_k: Optional[int] = Field(20, description="Number of results to return (default: 20)")

    @field_validator("mastercategory", mode="before")
    @classmethod
    def normalize_mastercategory(cls, v: Optional[str]) -> Optional[str]:
        """Normalize 'Non - IT', 'Non-IT', 'NON IT' etc. to 'NON_IT'; 'it'/'IT' to 'IT'."""
        return _normalize_mastercategory(v)


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
