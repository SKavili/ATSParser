"""Service for extracting and saving mobile phone number to database."""
from pathlib import Path
from typing import Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession

from app.mobile.mobile_extractor import MobileExtractor
from app.repositories.resume_repo import ResumeRepository
from app.services.resume_parser import ResumeParser
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Configuration: Path where resume files are stored (for file-based reprocessing)
RESUME_FILES_DIRS = [
    Path("Resumes"),  # Capital R (common location)
    Path("resumes"),  # Lowercase
    Path("uploads"),
    Path("data/resumes"),
    Path("storage/resumes"),
    Path("files/resumes"),
    Path("."),  # Current directory
]


class MobileService:
    """Service for extracting mobile phone number from resume and saving to database."""
    
    def __init__(self, session: AsyncSession):
        self.mobile_extractor = MobileExtractor()
        self.resume_repo = ResumeRepository(session)
        self.resume_parser = ResumeParser()
    
    async def extract_and_save_mobile(
        self,
        resume_text: str,
        resume_id: int,
        filename: str = "resume",
        retry_on_null: bool = True
    ) -> Optional[str]:
        """
        Extract mobile phone number from resume text and update the database record.
        Automatically retries with improved extraction if first attempt returns None.
        
        Args:
            resume_text: The text content of the resume
            resume_id: The ID of the resume record in the database
            filename: Name of the resume file (for logging)
            retry_on_null: If True, automatically retry extraction if first attempt returns None
        
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
            
            # First extraction attempt
            mobile = await self.mobile_extractor.extract_mobile(resume_text, filename)
            
            # If first attempt returned None and retry is enabled, try again with more aggressive extraction
            if not mobile and retry_on_null and resume_text:
                logger.info(
                    f"🔄 First mobile extraction returned None, retrying with full text scan for resume ID {resume_id}",
                    extra={"resume_id": resume_id, "file_name": filename}
                )
                # On retry, scan the FULL text more aggressively (not just header/footer)
                # The extractor already scans full text, but retry ensures we catch edge cases
                mobile = await self.mobile_extractor.extract_mobile(resume_text, filename)
                
                if mobile:
                    logger.info(
                        f"✅ RETRY SUCCESS: Found mobile on retry for resume ID {resume_id}: {mobile}",
                        extra={"resume_id": resume_id, "mobile": mobile, "file_name": filename}
                    )
            
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
    
    async def reprocess_mobile_from_file(
        self,
        resume_id: int,
        filename: str
    ) -> Optional[str]:
        """
        Reprocess mobile extraction from resume file on disk.
        This method finds the file, extracts text, and then extracts mobile.
        Useful for reprocessing resumes with NULL mobile values.
        
        Args:
            resume_id: The ID of the resume record in the database
            filename: Name of the resume file to find on disk
        
        Returns:
            The extracted mobile phone number or None if not found
        """
        try:
            logger.info(
                f"🔄 REPROCESSING MOBILE from file for resume ID {resume_id}: {filename}",
                extra={"resume_id": resume_id, "file_name": filename}
            )
            
            # Try to find the resume file in multiple possible locations
            file_path = None
            possible_paths = []
            
            # Add all configured directories
            for base_dir in RESUME_FILES_DIRS:
                possible_paths.append(base_dir / filename)
            
            # Also try as absolute path or in current directory
            possible_paths.append(Path(filename))
            
            # Try each path
            for path in possible_paths:
                if path.exists() and path.is_file():
                    file_path = path
                    logger.info(f"Found resume file at: {file_path}")
                    break
            
            if not file_path or not file_path.exists():
                logger.warning(f"Resume file not found for ID {resume_id}: {filename}")
                return None
            
            # Read file content
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            logger.info(f"Processing resume ID {resume_id}: {filename} ({len(file_content)} bytes)")
            
            # Extract text from file
            try:
                resume_text = await self.resume_parser.extract_text(file_content, filename)
                
                if not resume_text or len(resume_text.strip()) < 50:
                    logger.warning(f"Insufficient text for resume ID {resume_id}")
                    return None
                
                logger.info(f"Extracted {len(resume_text)} characters from resume ID {resume_id}")
                
            except Exception as e:
                logger.error(f"Text extraction failed for resume ID {resume_id}: {e}")
                return None
            
            # Extract mobile using the existing method (which includes retry logic)
            mobile = await self.extract_and_save_mobile(
                resume_text=resume_text,
                resume_id=resume_id,
                filename=filename,
                retry_on_null=True
            )
            
            if mobile:
                logger.info(f"✅ REPROCESS SUCCESS: Extracted mobile for resume ID {resume_id}: {mobile}")
            else:
                logger.warning(f"⚠️ REPROCESS: No mobile found for resume ID {resume_id}")
            
            return mobile
            
        except Exception as e:
            logger.error(
                f"ERROR: Failed to reprocess mobile from file for resume ID {resume_id}: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "resume_id": resume_id,
                    "file_name": filename,
                },
                exc_info=True
            )
            return None
    
    async def batch_reprocess_null_mobiles(
        self,
        limit: Optional[int] = None
    ) -> dict:
        """
        Batch reprocess all resumes with NULL mobile values.
        Finds resumes with NULL mobile, reads files from disk, and re-extracts mobile.
        
        Args:
            limit: Optional limit on number of resumes to process
        
        Returns:
            dict with summary statistics:
            {
                "total_found": int,
                "processed": int,
                "mobiles_extracted": int,
                "errors": int,
                "results": list of dicts with resume_id, filename, mobile, status
            }
        """
        result = {
            "total_found": 0,
            "processed": 0,
            "mobiles_extracted": 0,
            "errors": 0,
            "results": []
        }
        
        try:
            # Get all resumes with NULL email or mobile
            null_resumes = await self.resume_repo.get_resumes_with_null_email_or_mobile(limit=limit)
            
            # Filter to only those with NULL mobile
            null_mobile_resumes = [r for r in null_resumes if r.mobile is None]
            result["total_found"] = len(null_mobile_resumes)
            
            if not null_mobile_resumes:
                logger.info("No resumes found with NULL mobile")
                return result
            
            logger.info(f"Found {len(null_mobile_resumes)} resumes with NULL mobile, starting batch reprocessing...")
            
            for resume in null_mobile_resumes:
                try:
                    mobile = await self.reprocess_mobile_from_file(
                        resume_id=resume.id,
                        filename=resume.filename
                    )
                    
                    result["processed"] += 1
                    if mobile:
                        result["mobiles_extracted"] += 1
                        result["results"].append({
                            "resume_id": resume.id,
                            "filename": resume.filename,
                            "mobile": mobile,
                            "status": "success"
                        })
                    else:
                        result["results"].append({
                            "resume_id": resume.id,
                            "filename": resume.filename,
                            "mobile": None,
                            "status": "not_found"
                        })
                except Exception as e:
                    result["errors"] += 1
                    logger.error(f"Error reprocessing mobile for resume ID {resume.id}: {e}")
                    result["results"].append({
                        "resume_id": resume.id,
                        "filename": resume.filename,
                        "mobile": None,
                        "status": "error",
                        "error": str(e)
                    })
            
            logger.info(
                f"Batch reprocessing complete: {result['mobiles_extracted']} mobiles extracted from {result['processed']} resumes"
            )
            
        except Exception as e:
            logger.error(f"Error in batch reprocess: {e}", exc_info=True)
            result["errors"] += 1
        
        return result

