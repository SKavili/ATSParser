"""Service for extracting and saving designation to database."""
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.designation.designation_extractor import DesignationExtractor
from app.repositories.resume_repo import ResumeRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)


class DesignationService:
    """Service for extracting designation from resume and saving to database."""
    
    def __init__(self, session: AsyncSession):
        self.designation_extractor = DesignationExtractor()
        self.resume_repo = ResumeRepository(session)
    
    async def extract_and_save_designation(
        self,
        resume_text: str,
        resume_id: int,
        filename: str = "resume"
    ) -> Optional[str]:
        """
        Extract designation from resume text and update the database record.
        
        Args:
            resume_text: The text content of the resume
            resume_id: The ID of the resume record in the database
            filename: Name of the resume file (for logging)
        
        Returns:
            The extracted designation string or None if not found
        """
        try:
            # Extract designation using the dedicated extractor
            logger.info(
                f"🔍 STARTING DESIGNATION EXTRACTION for resume ID {resume_id}",
                extra={
                    "resume_id": resume_id, 
                    "file_name": filename,  # Use file_name instead of filename (reserved in LogRecord)
                    "resume_text_length": len(resume_text),
                    "resume_text_preview": resume_text[:200]
                }
            )
            print(f"\n{'='*60}")
            print(f"🔍 STARTING DESIGNATION EXTRACTION")
            print(f"   Resume ID: {resume_id}")
            print(f"   File: {filename}")
            print(f"   Resume text length: {len(resume_text)} characters")
            print(f"   Text preview: {resume_text[:150]}...")
            print(f"{'='*60}")
            
            designation = await self.designation_extractor.extract_designation(resume_text, filename)
            
            logger.info(
                f"📊 DESIGNATION EXTRACTION RESULT for resume ID {resume_id}: {designation}",
                extra={"resume_id": resume_id, "designation": designation, "file_name": filename}
            )
            
            # Update the database record with the extracted designation
            if designation:
                logger.info(
                    f"💾 UPDATING DATABASE: Resume ID {resume_id} with designation: '{designation}'",
                    extra={"resume_id": resume_id, "designation": designation, "file_name": filename}
                )
                print(f"\n💾 UPDATING DATABASE: Resume ID {resume_id}")
                print(f"   Designation: '{designation}'")
                
                updated_resume = await self.resume_repo.update(resume_id, {"designation": designation})
                if updated_resume:
                    logger.info(
                        f"✅ DATABASE UPDATED: Successfully saved designation '{designation}' for resume ID {resume_id}",
                        extra={"resume_id": resume_id, "designation": designation}
                    )
                    print(f"✅ DATABASE UPDATED: Designation '{designation}' saved to resume ID {resume_id}\n")
                else:
                    logger.error(f"❌ DATABASE UPDATE FAILED: Resume ID {resume_id} - record not found")
                    print(f"❌ DATABASE UPDATE FAILED: Resume ID {resume_id} not found\n")
            else:
                logger.warning(
                    f"💾 SAVING NULL: No designation found for resume ID {resume_id}, saving as NULL",
                    extra={"resume_id": resume_id, "file_name": filename}  # Use file_name instead of filename
                )
                print(f"\n💾 SAVING NULL: No designation found, saving NULL to database for resume ID {resume_id}\n")
                await self.resume_repo.update(resume_id, {"designation": None})
            
            return designation
            
        except Exception as e:
            logger.error(
                f"ERROR: Failed to extract and save designation for resume ID {resume_id}: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "resume_id": resume_id,
                    "file_name": filename,  # Use file_name instead of filename (reserved in LogRecord)
                    "resume_text_length": len(resume_text) if resume_text else 0
                },
                exc_info=True
            )
            # Try to save NULL to database even if extraction failed
            try:
                await self.resume_repo.update(resume_id, {"designation": None})
                logger.info(f"Saved NULL designation for resume ID {resume_id} after extraction failure")
            except Exception as db_error:
                logger.error(f"Failed to update database with NULL designation: {db_error}")
            
            # Don't raise - allow the resume upload to complete even if designation extraction fails
            # Just log the error and return None
            return None

