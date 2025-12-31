"""Service for extracting and saving job role to database."""
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.role.role_extractor import RoleExtractor
from app.repositories.resume_repo import ResumeRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)


class RoleService:
    """Service for extracting job role from resume and saving to database."""
    
    def __init__(self, session: AsyncSession):
        self.role_extractor = RoleExtractor()
        self.resume_repo = ResumeRepository(session)
    
    async def extract_and_save_role(
        self,
        resume_text: str,
        resume_id: int,
        filename: str = "resume"
    ) -> Optional[str]:
        """
        Extract job role from resume text and update the database record.
        
        Args:
            resume_text: The text content of the resume
            resume_id: The ID of the resume record in the database
            filename: Name of the resume file (for logging)
        
        Returns:
            The extracted job role string or None if not found
        """
        try:
            # Extract role using the dedicated extractor
            logger.info(
                f"🔍 STARTING ROLE EXTRACTION for resume ID {resume_id}",
                extra={
                    "resume_id": resume_id, 
                    "file_name": filename,  # Use file_name instead of filename (reserved in LogRecord)
                    "resume_text_length": len(resume_text),
                    "resume_text_preview": resume_text[:200]
                }
            )
            print(f"\n{'='*60}")
            print(f"🔍 STARTING ROLE EXTRACTION")
            print(f"   Resume ID: {resume_id}")
            print(f"   File: {filename}")
            print(f"   Resume text length: {len(resume_text)} characters")
            print(f"   Text preview: {resume_text[:150]}...")
            print(f"{'='*60}")
            
            role = await self.role_extractor.extract_role(resume_text, filename)
            
            logger.info(
                f"📊 ROLE EXTRACTION RESULT for resume ID {resume_id}: {role}",
                extra={"resume_id": resume_id, "role": role, "file_name": filename}
            )
            
            # Update the database record with the extracted role
            if role:
                logger.info(
                    f"💾 UPDATING DATABASE: Resume ID {resume_id} with jobrole: '{role}'",
                    extra={"resume_id": resume_id, "role": role, "file_name": filename}
                )
                print(f"\n💾 UPDATING DATABASE: Resume ID {resume_id}")
                print(f"   Job Role: '{role}'")
                
                updated_resume = await self.resume_repo.update(resume_id, {"jobrole": role})
                if updated_resume:
                    logger.info(
                        f"✅ DATABASE UPDATED: Successfully saved jobrole '{role}' for resume ID {resume_id}",
                        extra={"resume_id": resume_id, "role": role}
                    )
                    print(f"✅ DATABASE UPDATED: Job Role '{role}' saved to resume ID {resume_id}\n")
                else:
                    logger.error(f"❌ DATABASE UPDATE FAILED: Resume ID {resume_id} - record not found")
                    print(f"❌ DATABASE UPDATE FAILED: Resume ID {resume_id} not found\n")
            else:
                logger.warning(
                    f"💾 SAVING NULL: No role found for resume ID {resume_id}, saving as NULL",
                    extra={"resume_id": resume_id, "file_name": filename}  # Use file_name instead of filename
                )
                print(f"\n💾 SAVING NULL: No role found, saving NULL to database for resume ID {resume_id}\n")
                await self.resume_repo.update(resume_id, {"jobrole": None})
            
            return role
            
        except Exception as e:
            logger.error(
                f"ERROR: Failed to extract and save role for resume ID {resume_id}: {e}",
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
                await self.resume_repo.update(resume_id, {"jobrole": None})
                logger.info(f"Saved NULL jobrole for resume ID {resume_id} after extraction failure")
            except Exception as db_error:
                logger.error(f"Failed to update database with NULL jobrole: {db_error}")
            
            # Don't raise - allow the resume upload to complete even if role extraction fails
            # Just log the error and return None
            return None

