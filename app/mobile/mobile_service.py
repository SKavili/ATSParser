"""Service for extracting and saving mobile phone number to database."""
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.mobile.mobile_extractor import MobileExtractor
from app.repositories.resume_repo import ResumeRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)


class MobileService:
    """Service for extracting mobile phone number from resume and saving to database."""
    
    def __init__(self, session: AsyncSession):
        self.mobile_extractor = MobileExtractor()
        self.resume_repo = ResumeRepository(session)
    
    async def extract_and_save_mobile(
        self,
        resume_text: str,
        resume_id: int,
        filename: str = "resume"
    ) -> Optional[str]:
        """
        Extract mobile phone number from resume text and update the database record.
        
        Args:
            resume_text: The text content of the resume
            resume_id: The ID of the resume record in the database
            filename: Name of the resume file (for logging)
        
        Returns:
            The extracted mobile phone number or None if not found
        """
        try:
            logger.info(
                f"🔍 STARTING MOBILE EXTRACTION for resume ID {resume_id}",
                extra={
                    "resume_id": resume_id, 
                    "file_name": filename,
                }
            )
            
            mobile = await self.mobile_extractor.extract_mobile(resume_text, filename)
            
            logger.info(
                f"📊 MOBILE EXTRACTION RESULT for resume ID {resume_id}: {mobile}",
                extra={"resume_id": resume_id, "mobile": mobile, "file_name": filename}
            )
            
            if mobile:
                logger.info(
                    f"💾 UPDATING DATABASE: Resume ID {resume_id} with mobile: '{mobile}'",
                    extra={"resume_id": resume_id, "mobile": mobile, "file_name": filename}
                )
                
                updated_resume = await self.resume_repo.update(resume_id, {"mobile": mobile})
                if updated_resume:
                    logger.info(
                        f"✅ DATABASE UPDATED: Successfully saved mobile for resume ID {resume_id}",
                        extra={"resume_id": resume_id, "mobile": mobile}
                    )
                else:
                    logger.error(f"❌ DATABASE UPDATE FAILED: Resume ID {resume_id} - record not found")
            else:
                logger.warning(
                    f"💾 SAVING NULL: No mobile found for resume ID {resume_id}, saving as NULL",
                    extra={"resume_id": resume_id, "file_name": filename}
                )
                await self.resume_repo.update(resume_id, {"mobile": None})
            
            return mobile
            
        except Exception as e:
            logger.error(
                f"ERROR: Failed to extract and save mobile for resume ID {resume_id}: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "resume_id": resume_id,
                    "file_name": filename,
                },
                exc_info=True
            )
            try:
                await self.resume_repo.update(resume_id, {"mobile": None})
                logger.info(f"Saved NULL mobile for resume ID {resume_id} after extraction failure")
            except Exception as db_error:
                logger.error(f"Failed to update database with NULL mobile: {db_error}")
            
            return None

