"""Designation extraction module for extracting job titles/designations from resumes."""
from app.designation.designation_extractor import DesignationExtractor, DESIGNATION_PROMPT
from app.designation.designation_service import DesignationService

__all__ = [
    "DesignationExtractor",
    "DesignationService",
    "DESIGNATION_PROMPT",
]

