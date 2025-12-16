"""Service for extracting and saving education to database."""
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.education.education_extractor import EducationExtractor
from app.repositories.resume_repo import ResumeRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)


class EducationService:
    """Service for extracting education from resume and saving to database."""
    
    def __init__(self, session: AsyncSession):
        self.education_extractor = EducationExtractor()
        self.resume_repo = ResumeRepository(session)
    
    async def extract_and_save_education(
        self,
        resume_text: str,
        resume_id: int,
        filename: str = "resume"
    ) -> Optional[str]:
        """
        Extract education from resume text and update the database record.
        
        Args:
            resume_text: The text content of the resume
            resume_id: The ID of the resume record in the database
            filename: Name of the resume file (for logging)
        
        Returns:
            The extracted education string or None if not found
        """
        try:
            logger.info(
                f"🔍 STARTING EDUCATION EXTRACTION for resume ID {resume_id}",
                extra={
                    "resume_id": resume_id, 
                    "file_name": filename,
                }
            )
            
            education = await self.education_extractor.extract_education(resume_text, filename)
            
            logger.info(
                f"📊 EDUCATION EXTRACTION RESULT for resume ID {resume_id}",
                extra={"resume_id": resume_id, "education_length": len(education) if education else 0, "file_name": filename}
            )
            
            if education:
                logger.info(
                    f"💾 UPDATING DATABASE: Resume ID {resume_id} with education",
                    extra={"resume_id": resume_id, "file_name": filename}
                )
                
                updated_resume = await self.resume_repo.update(resume_id, {"education": education})
                if updated_resume:
                    logger.info(
                        f"✅ DATABASE UPDATED: Successfully saved education for resume ID {resume_id}",
                        extra={"resume_id": resume_id}
                    )
                else:
                    logger.error(f"❌ DATABASE UPDATE FAILED: Resume ID {resume_id} - record not found")
            else:
                logger.warning(
                    f"💾 SAVING NULL: No education found for resume ID {resume_id}, saving as NULL",
                    extra={"resume_id": resume_id, "file_name": filename}
                )
                await self.resume_repo.update(resume_id, {"education": None})
            
            return education
            
        except Exception as e:
            logger.error(
                f"ERROR: Failed to extract and save education for resume ID {resume_id}: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "resume_id": resume_id,
                    "file_name": filename,
                },
                exc_info=True
            )
            try:
                await self.resume_repo.update(resume_id, {"education": None})
                logger.info(f"Saved NULL education for resume ID {resume_id} after extraction failure")
            except Exception as db_error:
                logger.error(f"Failed to update database with NULL education: {db_error}")
            
            return None

