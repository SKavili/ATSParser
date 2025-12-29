"""Service for extracting and saving email to database."""
from pathlib import Path
from typing import Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession

from app.email.email_extractor import EmailExtractor
from app.repositories.resume_repo import ResumeRepository
from app.services.resume_parser import ResumeParser
from app.utils.logging import get_logger

logger = get_logger(__name__)


class EmailService:
    """Service for extracting email from resume and saving to database."""
    
    def __init__(self, session: AsyncSession):
        self.email_extractor = EmailExtractor()
        self.resume_repo = ResumeRepository(session)
    
    async def extract_and_save_email(
        self,
        resume_text: str,
        resume_id: int,
        filename: str = "resume"
    ) -> Optional[str]:
        """
        Extract email from resume text and update the database record.
        
        Args:
            resume_text: The text content of the resume
            resume_id: The ID of the resume record in the database
            filename: Name of the resume file (for logging)
        
        Returns:
            The extracted email string or None if not found
        """
        try:
            logger.info(
                f"🔍 STARTING EMAIL EXTRACTION for resume ID {resume_id}",
                extra={
                    "resume_id": resume_id, 
                    "file_name": filename,
                }
            )
            
            email = await self.email_extractor.extract_email(resume_text, filename)
            
            logger.info(
                f"📊 EMAIL EXTRACTION RESULT for resume ID {resume_id}: {email}",
                extra={"resume_id": resume_id, "email": email, "file_name": filename}
            )
            
            if email:
                logger.info(
                    f"💾 UPDATING DATABASE: Resume ID {resume_id} with email: '{email}'",
                    extra={"resume_id": resume_id, "email": email, "file_name": filename}
                )
                
                updated_resume = await self.resume_repo.update(resume_id, {"email": email})
                if updated_resume:
                    logger.info(
                        f"✅ DATABASE UPDATED: Successfully saved email for resume ID {resume_id}",
                        extra={"resume_id": resume_id, "email": email}
                    )
                else:
                    logger.error(f"❌ DATABASE UPDATE FAILED: Resume ID {resume_id} - record not found")
            else:
                logger.warning(
                    f"💾 SAVING NULL: No email found for resume ID {resume_id}, saving as NULL",
                    extra={"resume_id": resume_id, "file_name": filename}
                )
                await self.resume_repo.update(resume_id, {"email": None})
            
            return email
            
        except Exception as e:
            logger.error(
                f"ERROR: Failed to extract and save email for resume ID {resume_id}: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "resume_id": resume_id,
                    "file_name": filename,
                },
                exc_info=True
            )
            try:
                await self.resume_repo.update(resume_id, {"email": None})
                logger.info(f"Saved NULL email for resume ID {resume_id} after extraction failure")
            except Exception as db_error:
                logger.error(f"Failed to update database with NULL email: {db_error}")
            
            return None
    
    async def reprocess_from_file(
        self,
        resume_id: int,
        filename: str,
        resume_parser: ResumeParser,
        resume_files_dirs: list = None
    ) -> Dict:
        """
        Reprocess email extraction from resume file.
        Finds the resume file and re-extracts email.
        
        Args:
            resume_id: The ID of the resume record
            filename: Name of the resume file
            resume_parser: ResumeParser instance for text extraction
            resume_files_dirs: List of directories to search for resume files
        
        Returns:
            Dictionary with status and extracted email
        """
        result = {
            "resume_id": resume_id,
            "filename": filename,
            "email_extracted": None,
            "status": "error",
            "error": None
        }
        
        if resume_files_dirs is None:
            resume_files_dirs = [
                Path("Resumes"),
                Path("resumes"),
                Path("uploads"),
                Path("data/resumes"),
                Path("storage/resumes"),
                Path("files/resumes"),
                Path("."),
            ]
        
        try:
            # Find resume file
            file_path = None
            for base_dir in resume_files_dirs:
                path = base_dir / filename
                if path.exists() and path.is_file():
                    file_path = path
                    break
            
            if not file_path or not file_path.exists():
                result["error"] = f"Resume file not found: {filename}"
                return result
            
            # Read and extract text
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            resume_text = await resume_parser.extract_text(file_content, filename)
            if not resume_text or len(resume_text.strip()) < 50:
                result["error"] = "Insufficient text extracted from resume"
                return result
            
            # Extract email
            email = await self.extract_and_save_email(resume_text, resume_id, filename)
            result["email_extracted"] = email
            result["status"] = "success"
            return result
            
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            logger.error(f"Error reprocessing email for resume ID {resume_id}: {e}", exc_info=True)
            return result

