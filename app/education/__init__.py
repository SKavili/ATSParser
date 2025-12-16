"""Education extraction module for extracting education details from resumes."""
from app.education.education_extractor import EducationExtractor, EDUCATION_PROMPT
from app.education.education_service import EducationService

__all__ = [
    "EducationExtractor",
    "EducationService",
    "EDUCATION_PROMPT",
]

