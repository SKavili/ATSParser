"""Skills extraction module for extracting technical skills from resumes."""
from app.skills.skills_extractor import SkillsExtractor, SKILLS_PROMPT
from app.skills.skills_service import SkillsService

__all__ = [
    "SkillsExtractor",
    "SkillsService",
    "SKILLS_PROMPT",
]

