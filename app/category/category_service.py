"""Service for extracting and saving category to database."""
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.category.category_extractor import CategoryExtractor
from app.repositories.resume_repo import ResumeRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)


class CategoryService:
    """Service for extracting category from resume and saving to database."""
    
    def __init__(self, session: AsyncSession):
        self.category_extractor = CategoryExtractor()
        self.resume_repo = ResumeRepository(session)
    
    async def extract_and_save_category(
        self,
        resume_text: str,
        resume_id: int,
        mastercategory: str,
        filename: str = "resume"
    ) -> Optional[str]:
        """
        Extract category from resume text based on mastercategory and update the database record.
        
        Args:
            resume_text: The text content of the resume
            resume_id: The ID of the resume record in the database
            mastercategory: "IT" or "NON_IT" (from database)
            filename: Name of the resume file (for logging)
        
        Returns:
            The extracted category string or None if not found
        """
        try:
            logger.info(
                f"[CATEGORY] STARTING CATEGORY EXTRACTION for resume ID {resume_id}",
                extra={
                    "resume_id": resume_id, 
                    "mastercategory": mastercategory,
                    "file_name": filename,
                }
            )
            
            category = await self.category_extractor.extract_category(
                resume_text=resume_text,
                mastercategory=mastercategory,
                filename=filename
            )
            
            logger.info(
                f"[CATEGORY] CATEGORY EXTRACTION RESULT for resume ID {resume_id}: {category}",
                extra={
                    "resume_id": resume_id, 
                    "mastercategory": mastercategory,
                    "category": category, 
                    "file_name": filename,
                }
            )
            
            # Update database with category
            if category:
                logger.info(
                    f"[CATEGORY] UPDATING DATABASE: Resume ID {resume_id} with category: '{category}'",
                    extra={"resume_id": resume_id, "category": category, "file_name": filename}
                )
                
                updated_resume = await self.resume_repo.update(resume_id, {"category": category})
                if updated_resume:
                    logger.info(
                        f"[CATEGORY] DATABASE UPDATED: Successfully saved category for resume ID {resume_id}",
                        extra={"resume_id": resume_id, "category": category}
                    )
                else:
                    logger.error(f"[CATEGORY] DATABASE UPDATE FAILED: Resume ID {resume_id} - record not found")
            else:
                logger.warning(
                    f"[CATEGORY] SAVING NULL: No category found for resume ID {resume_id}, saving as NULL",
                    extra={"resume_id": resume_id, "file_name": filename, "mastercategory": mastercategory}
                )
                await self.resume_repo.update(resume_id, {"category": None})
            
            return category
            
        except Exception as e:
            logger.error(
                f"[CATEGORY] ERROR: Failed to extract and save category for resume ID {resume_id}: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "resume_id": resume_id,
                    "mastercategory": mastercategory,
                    "file_name": filename,
                },
                exc_info=True
            )
            # Save None on error
            try:
                await self.resume_repo.update(resume_id, {"category": None})
                logger.info(f"[CATEGORY] Saved NULL category for resume ID {resume_id} after extraction failure")
            except Exception as db_error:
                logger.error(f"[CATEGORY] Failed to update database with NULL category: {db_error}")
            
            return None

