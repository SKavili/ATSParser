"""Email extraction module for extracting email addresses from resumes."""
from app.email.email_extractor import EmailExtractor, EMAIL_PROMPT
from app.email.email_service import EmailService

__all__ = [
    "EmailExtractor",
    "EmailService",
    "EMAIL_PROMPT",
]

