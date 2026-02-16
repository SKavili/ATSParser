"""Pydantic models for job-related operations."""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, model_validator

from app.jd_parser.jd_models import ParsedJD



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


class JDParseRequest(BaseModel):
    """Request model for parsing a raw job description (optionally with candidate matching)."""
    text: str = Field(..., description="Raw job description text to parse")
    mastercategory: Literal["IT", "Non-IT"] = Field(..., description="Master category: IT or Non-IT (required)")
    category: Optional[str] = Field("", description="Specific category name (e.g., 'Full Stack Development (Python)'). Empty string queries all namespaces.")
    top_k: int = Field(10, ge=1, le=100, description="Number of candidate matches to return when mode is parse-and-match")
    mode: Literal["parse-only", "parse-and-match"] = Field(
        "parse-and-match",
        description="parse-only: return only parsed JD; parse-and-match: also return matching candidates from vector DB"
    )


class JDMatchCandidate(BaseModel):
    """Single candidate match returned for a parsed JD."""
    candidate_id: str = Field(..., description="Resume/candidate ID")
    similarity_score: float = Field(..., ge=0, le=100, description="Match score 0-100 (cosine similarity * 100)")
    name: Optional[str] = None
    email: Optional[str] = Field(None, description="Candidate email")
    designation: Optional[str] = None
    experience_years: Optional[float] = None
    skills: Optional[List[str]] = None
    location: Optional[str] = None
    # Extended candidate output for UI (Power Apps, etc.)
    domain_match: Optional[str] = Field(None, description="Domain match description (e.g. Match, Partial, No match)")
    domain_score: Optional[float] = Field(None, ge=0, le=100, description="Domain match score 0-100")
    education_score: Optional[float] = Field(None, ge=0, le=100, description="Education match score 0-100")
    experience_match: Optional[str] = Field(None, description="Experience match description (e.g. Meets, Above, Below)")
    experience_score: Optional[float] = Field(None, ge=0, le=100, description="Experience match score 0-100")
    match_percent: Optional[float] = Field(None, ge=0, le=100, description="Overall match percentage 0-100")
    total_score: Optional[float] = Field(None, ge=0, le=100, description="Total combined score 0-100")
    rank: Optional[int] = Field(None, ge=1, description="Candidate rank in result list (1-based)")
    matched_skills: Optional[List[str]] = Field(None, description="JD required skills that candidate has")
    missing_skills: Optional[List[str]] = Field(None, description="JD required skills that candidate does not have")


class ParseJDMatchInfo(BaseModel):
    """Summary of the candidate match run (when mode=parse-and-match)."""
    namespace_used: str = Field(..., description="Category/namespace used for search (e.g. inferred_category or 'default')")
    top_k_requested: int = Field(..., description="Requested top_k")
    candidates_found: int = Field(..., description="Number of candidates returned")


class ParseJDResponse(BaseModel):
    """Response for POST /parse-jd: parsed JD and optional candidate matches."""
    parsed_jd: ParsedJD = Field(..., description="Structured parsed job description")
    candidates: List[JDMatchCandidate] = Field(default_factory=list, description="Matching candidates (empty when mode=parse-only or on match failure)")
    match_info: ParseJDMatchInfo = Field(..., description="Match run summary (namespace, top_k, count)")


