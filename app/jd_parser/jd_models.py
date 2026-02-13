"""Pydantic models for Job Description parsing."""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ParsedJD(BaseModel):
    """Parsed Job Description schema with validation."""
    
    designation: Optional[str] = Field(None, description="Primary normalized job title")
    must_have_skills: List[str] = Field(default_factory=list, description="Explicitly required skills")
    nice_to_have_skills: List[str] = Field(default_factory=list, description="Preferred/optional skills")
    min_experience_years: Optional[int] = Field(None, ge=0, description="Minimum years of experience")
    max_experience_years: Optional[int] = Field(None, ge=0, description="Maximum years of experience")
    education_requirements: List[str] = Field(default_factory=list, description="Required degrees/certifications")
    domain_focus: Optional[str] = Field(None, description="Short functional domain")
    inferred_mastercategory: Literal["IT", "Non-IT"] = Field(..., description="IT or Non-IT classification (required)")
    inferred_category: Optional[str] = Field(None, description="Granular snake_case category namespace")
    inferred_category_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in category inference (0.0-1.0)")
    location: Optional[str] = Field(None, description="Explicit location if mentioned")
    location_type: Optional[Literal["strict", "preferred", "remote"]] = Field(None, description="Location requirement type")
    other_requirements: List[str] = Field(default_factory=list, description="Soft skills / non-tech requirements")
    text_for_embedding: str = Field(..., min_length=1, description="Concise keyword summary for embedding")
