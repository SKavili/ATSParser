"""Pydantic models for resume-related operations."""
from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr


class ResumeUpload(BaseModel):
    """Request model for resume upload (metadata only)."""
    candidate_name: Optional[str] = Field(None, alias="candidateName")
    job_role: Optional[str] = Field(None, alias="jobRole")
    source: Optional[str] = None
    
    class Config:
        populate_by_name = True


class ResumeUploadResponse(BaseModel):
    """Response model for resume upload."""
    id: int
    candidateName: str
    jobrole: str
    designation: Optional[str] = None  # Current or most recent job title
    experience: str
    domain: str
    mobile: str
    email: str
    education: str
    filename: str
    skillset: str
    status: Optional[str] = None  # Processing status
    created_at: str
    
    class Config:
        from_attributes = True

