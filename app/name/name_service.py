"""Service for extracting and saving candidate name to database."""
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.name.name_extractor import NameExtractor
from app.repositories.resume_repo import ResumeRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)


class NameService:
    """Service for extracting candidate name from resume and saving to database."""
    
    def __init__(self, session: AsyncSession):
        self.name_extractor = NameExtractor()
        self.resume_repo = ResumeRepository(session)
    
    async def extract_and_save_name(
        self,
        resume_text: str,
        resume_id: int,
        filename: str = "resume"
    ) -> Optional[str]:
        """
        Extract candidate name from resume text and update the database record.
        
        Args:
            resume_text: The text content of the resume
            resume_id: The ID of the resume record in the database
            filename: Name of the resume file (for logging)
        
        Returns:
            The extracted candidate name or None if not found
        """
        try:
            logger.info(
                f"🔍 STARTING NAME EXTRACTION for resume ID {resume_id}",
                extra={
                    "resume_id": resume_id, 
                    "file_name": filename,
                }
            )
            
            name = await self.name_extractor.extract_name(resume_text, filename)
            
            logger.info(
                f"📊 NAME EXTRACTION RESULT for resume ID {resume_id}: {name}",
                extra={"resume_id": resume_id, "extracted_name": name, "file_name": filename}
            )
            
            if name:
                logger.info(
                    f"💾 UPDATING DATABASE: Resume ID {resume_id} with name: '{name}'",
                    extra={"resume_id": resume_id, "extracted_name": name, "file_name": filename}
                )
                
                updated_resume = await self.resume_repo.update(resume_id, {"candidatename": name})
                if updated_resume:
                    logger.info(
                        "Name extracted and saved",
                        extra={
                            "extracted_name": name,
                            "resume_id": resume_id,
                            "file_name": filename
                        }
                    )
                else:
                    logger.error(f"❌ DATABASE UPDATE FAILED: Resume ID {resume_id} - record not found")
            else:
                logger.warning(
                    f"💾 SAVING NULL: No name found for resume ID {resume_id}, saving as NULL",
                    extra={"resume_id": resume_id, "file_name": filename}
                )
                await self.resume_repo.update(resume_id, {"candidatename": None})
            
            return name
            
        except Exception as e:
            logger.error(
                f"ERROR: Failed to extract and save name for resume ID {resume_id}: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "resume_id": resume_id,
                    "file_name": filename,
                },
                exc_info=True
            )
            try:
                await self.resume_repo.update(resume_id, {"candidatename": None})
                logger.info(f"Saved NULL name for resume ID {resume_id} after extraction failure")
            except Exception as db_error:
                logger.error(f"Failed to update database with NULL name: {db_error}")
            
            return None

