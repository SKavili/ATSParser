"""Pydantic models for job-related operations."""
from typing import Optional, List
from pydantic import BaseModel, Field, model_validator


class JobCreate(BaseModel):
    """Request model for job creation."""
    title: str
    description: str
    required_skills: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    job_id: Optional[str] = None


class JobCreateResponse(BaseModel):
    """Response model for job creation."""
    job_id: str
    title: str
    embedding_id: Optional[str] = None
    message: str


class MatchRequest(BaseModel):
    """Request model for job matching."""
    job_id: Optional[str] = None
    job_description: Optional[str] = None
    top_k: Optional[int] = None
    
    @model_validator(mode='after')
    def validate_required_fields(self):
        """Validate that either job_id or job_description is provided."""
        if not self.job_id and not self.job_description:
            raise ValueError("Either job_id or job_description must be provided")
        return self


class MatchResult(BaseModel):
    """Single match result."""
    resume_id: int
    candidate_name: str
    similarity_score: float
    candidate_summary: str
    filename: str


class MatchResponse(BaseModel):
    """Response model for job matching."""
    matches: List[MatchResult]
    total_results: int
    job_id: Optional[str] = None

