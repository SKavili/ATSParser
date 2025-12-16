"""Domain extraction module for extracting industry domain from resumes."""
from app.domain.domain_extractor import DomainExtractor, DOMAIN_PROMPT
from app.domain.domain_service import DomainService

__all__ = [
    "DomainExtractor",
    "DomainService",
    "DOMAIN_PROMPT",
]

