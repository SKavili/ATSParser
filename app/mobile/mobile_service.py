"""Service for extracting and saving mobile phone number to database."""
from pathlib import Path
from typing import Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession

from app.mobile.mobile_extractor import MobileExtractor
from app.repositories.resume_repo import ResumeRepository
from app.services.resume_parser import ResumeParser
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
    
    async def reprocess_from_file(
        self,
        resume_id: int,
        filename: str,
        resume_parser: ResumeParser,
        resume_files_dirs: list = None
    ) -> Dict:
        """
        Reprocess mobile extraction from resume file.
        Finds the resume file and re-extracts mobile.
        
        Args:
            resume_id: The ID of the resume record
            filename: Name of the resume file
            resume_parser: ResumeParser instance for text extraction
            resume_files_dirs: List of directories to search for resume files
        
        Returns:
            Dictionary with status and extracted mobile
        """
        result = {
            "resume_id": resume_id,
            "filename": filename,
            "mobile_extracted": None,
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
            
            # Extract mobile
            mobile = await self.extract_and_save_mobile(resume_text, resume_id, filename)
            result["mobile_extracted"] = mobile
            result["status"] = "success"
            return result
            
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            logger.error(f"Error reprocessing mobile for resume ID {resume_id}: {e}", exc_info=True)
            return result

