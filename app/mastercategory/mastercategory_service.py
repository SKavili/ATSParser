"""Service for extracting and saving master category to database."""
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.mastercategory.mastercategory_extractor import MasterCategoryExtractor
from app.repositories.resume_repo import ResumeRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)


class MasterCategoryService:
    """Service for extracting master category from resume and saving to database."""
    
    def __init__(self, session: AsyncSession):
        self.mastercategory_extractor = MasterCategoryExtractor()
        self.resume_repo = ResumeRepository(session)
    
    async def extract_and_save_mastercategory(
        self,
        resume_text: str,
        resume_id: int,
        filename: str = "resume"
    ) -> str:
        """
        Extract master category (IT/NON_IT) from resume text and update the database record.
        
        Args:
            resume_text: The text content of the resume
            resume_id: The ID of the resume record in the database
            filename: Name of the resume file (for logging)
        
        Returns:
            The extracted master category ("IT" or "NON_IT")
        """
        try:
            logger.info(
                f"[MASTERCATEGORY] STARTING MASTER CATEGORY EXTRACTION for resume ID {resume_id}",
                extra={
                    "resume_id": resume_id, 
                    "file_name": filename,
                }
            )
            
            mastercategory = await self.mastercategory_extractor.extract_mastercategory(resume_text, filename)
            
            logger.info(
                f"[MASTERCATEGORY] MASTER CATEGORY EXTRACTION RESULT for resume ID {resume_id}: {mastercategory}",
                extra={
                    "resume_id": resume_id, 
                    "mastercategory": mastercategory, 
                    "file_name": filename,
                }
            )
            
            # Update database with mastercategory
            logger.info(
                f"[MASTERCATEGORY] UPDATING DATABASE: Resume ID {resume_id} with mastercategory: '{mastercategory}'",
                extra={"resume_id": resume_id, "mastercategory": mastercategory, "file_name": filename}
            )
            
            updated_resume = await self.resume_repo.update(resume_id, {"mastercategory": mastercategory})
            if updated_resume:
                logger.info(
                    f"[MASTERCATEGORY] DATABASE UPDATED: Successfully saved mastercategory for resume ID {resume_id}",
                    extra={"resume_id": resume_id, "mastercategory": mastercategory}
                )
            else:
                logger.error(f"[MASTERCATEGORY] DATABASE UPDATE FAILED: Resume ID {resume_id} - record not found")
            
            return mastercategory
            
        except Exception as e:
            logger.error(
                f"[MASTERCATEGORY] ERROR: Failed to extract and save mastercategory for resume ID {resume_id}: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "resume_id": resume_id,
                    "file_name": filename,
                },
                exc_info=True
            )
            # Default to NON_IT on error
            try:
                await self.resume_repo.update(resume_id, {"mastercategory": "NON_IT"})
                logger.info(f"[MASTERCATEGORY] Saved default NON_IT mastercategory for resume ID {resume_id} after extraction failure")
            except Exception as db_error:
                logger.error(f"[MASTERCATEGORY] Failed to update database with default mastercategory: {db_error}")
            
            return "NON_IT"

