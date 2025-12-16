"""Service for extracting and saving experience to database."""
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.experience.experience_extractor import ExperienceExtractor
from app.repositories.resume_repo import ResumeRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ExperienceService:
    """Service for extracting experience from resume and saving to database."""
    
    def __init__(self, session: AsyncSession):
        self.experience_extractor = ExperienceExtractor()
        self.resume_repo = ResumeRepository(session)
    
    async def extract_and_save_experience(
        self,
        resume_text: str,
        resume_id: int,
        filename: str = "resume"
    ) -> Optional[str]:
        """
        Extract experience from resume text and update the database record.
        
        Args:
            resume_text: The text content of the resume
            resume_id: The ID of the resume record in the database
            filename: Name of the resume file (for logging)
        
        Returns:
            The extracted experience string or None if not found
        """
        try:
            logger.info(
                f"🔍 STARTING EXPERIENCE EXTRACTION for resume ID {resume_id}",
                extra={
                    "resume_id": resume_id, 
                    "file_name": filename,
                }
            )
            
            experience = await self.experience_extractor.extract_experience(resume_text, filename)
            
            logger.info(
                f"📊 EXPERIENCE EXTRACTION RESULT for resume ID {resume_id}: {experience}",
                extra={"resume_id": resume_id, "experience": experience, "file_name": filename}
            )
            
            if experience:
                logger.info(
                    f"💾 UPDATING DATABASE: Resume ID {resume_id} with experience: '{experience}'",
                    extra={"resume_id": resume_id, "experience": experience, "file_name": filename}
                )
                
                updated_resume = await self.resume_repo.update(resume_id, {"experience": experience})
                if updated_resume:
                    logger.info(
                        f"✅ DATABASE UPDATED: Successfully saved experience for resume ID {resume_id}",
                        extra={"resume_id": resume_id, "experience": experience}
                    )
                else:
                    logger.error(f"❌ DATABASE UPDATE FAILED: Resume ID {resume_id} - record not found")
            else:
                logger.warning(
                    f"💾 SAVING NULL: No experience found for resume ID {resume_id}, saving as NULL",
                    extra={"resume_id": resume_id, "file_name": filename}
                )
                await self.resume_repo.update(resume_id, {"experience": None})
            
            return experience
            
        except Exception as e:
            logger.error(
                f"ERROR: Failed to extract and save experience for resume ID {resume_id}: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "resume_id": resume_id,
                    "file_name": filename,
                },
                exc_info=True
            )
            try:
                await self.resume_repo.update(resume_id, {"experience": None})
                logger.info(f"Saved NULL experience for resume ID {resume_id} after extraction failure")
            except Exception as db_error:
                logger.error(f"Failed to update database with NULL experience: {db_error}")
            
            return None

