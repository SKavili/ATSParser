"""Name extraction module for extracting candidate names from resumes."""
from app.name.name_extractor import NameExtractor, NAME_PROMPT
from app.name.name_service import NameService

__all__ = [
    "NameExtractor",
    "NameService",
    "NAME_PROMPT",
]

