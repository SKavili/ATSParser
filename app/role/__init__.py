"""Role extraction module for extracting job roles from resumes."""
from app.role.role_extractor import RoleExtractor, ROLE_PROMPT
from app.role.role_service import RoleService

__all__ = [
    "RoleExtractor",
    "RoleService",
    "ROLE_PROMPT",
]

