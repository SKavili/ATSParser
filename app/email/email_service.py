"""Service for extracting and saving email to database."""
from pathlib import Path
from typing import Optional, Dict, TYPE_CHECKING
from sqlalchemy.ext.asyncio import AsyncSession

from app.email.email_extractor import EmailExtractor
from app.repositories.resume_repo import ResumeRepository
from app.services.resume_parser import ResumeParser
from app.utils.logging import get_logger

if TYPE_CHECKING:
    from app.mobile.mobile_service import MobileService

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


class EmailService:
    """Service for extracting email from resume and saving to database."""
    
    def __init__(self, session: AsyncSession):
        self.email_extractor = EmailExtractor()
        self.resume_repo = ResumeRepository(session)
        self.resume_parser = ResumeParser()
    
    async def extract_and_save_email(
        self,
        resume_text: str,
        resume_id: int,
        filename: str = "resume",
        retry_on_null: bool = True
    ) -> Optional[str]:
        """
        Extract email from resume text and update the database record.
        Automatically retries with improved extraction if first attempt returns None.
        
        Args:
            resume_text: The text content of the resume
            resume_id: The ID of the resume record in the database
            filename: Name of the resume file (for logging)
            retry_on_null: If True, automatically retry extraction if first attempt returns None
        
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
            
            # Extract ALL emails from resume FIRST (this is the primary method now)
            # This is the CRITICAL method - it finds ALL emails, not just one
            all_emails_str = await self.email_extractor.extract_all_emails(resume_text, filename)
            
            # Log what we found
            if all_emails_str and all_emails_str.strip():
                email_count = len(all_emails_str.split(','))
                logger.info(
                    f"✅ extract_all_emails found {email_count} email(s) for resume ID {resume_id}",
                    extra={
                        "resume_id": resume_id,
                        "file_name": filename,
                        "email_count": email_count,
                        "all_emails": all_emails_str
                    }
                )
            
            # If all_emails_str is empty, try again with retry logic
            if (not all_emails_str or not all_emails_str.strip()) and retry_on_null and resume_text:
                logger.warning(
                    f"🔄 extract_all_emails returned empty for resume ID {resume_id}, retrying...",
                    extra={"resume_id": resume_id, "file_name": filename, "text_length": len(resume_text)}
                )
                # Re-extract all emails on retry
                all_emails_str = await self.email_extractor.extract_all_emails(resume_text, filename)
                if all_emails_str and all_emails_str.strip():
                    email_count = len(all_emails_str.split(','))
                    logger.info(
                        f"✅ RETRY SUCCESS: extract_all_emails found {email_count} email(s) for resume ID {resume_id}",
                        extra={
                            "resume_id": resume_id,
                            "file_name": filename,
                            "email_count": email_count,
                            "all_emails": all_emails_str
                        }
                    )
            
            # Also get the best/primary email for backward compatibility and logging ONLY
            # This is NOT used for storage - we ALWAYS use all_emails_str if available
            email = None
            if not all_emails_str or not all_emails_str.strip():
                # Only call extract_email if extract_all_emails found nothing
                email = await self.email_extractor.extract_email(resume_text, filename)
                # If extract_email found something but extract_all_emails didn't, 
                # use the single email found as a fallback
                if email and (not all_emails_str or not all_emails_str.strip()):
                    logger.warning(
                        f"⚠️ extract_email found '{email}' but extract_all_emails found nothing for {filename}",
                        extra={"resume_id": resume_id, "file_name": filename, "found_email": email}
                    )
                    # Use the single email found as a fallback
                    all_emails_str = email
            
            # CRITICAL: ALWAYS use all_emails_str if available - this contains ALL emails found
            # This is the PRIMARY source - we do NOT use the single email from extract_email()
            # if all_emails_str has multiple emails
            if all_emails_str and all_emails_str.strip():
                email_to_store = all_emails_str.strip()
                # Truncate if too long for VARCHAR(255) - but log a warning
                if len(email_to_store) > 255:
                    logger.warning(
                        f"⚠️ Email string too long ({len(email_to_store)} chars), truncating to 255 for resume ID {resume_id}",
                        extra={"resume_id": resume_id, "original_length": len(email_to_store), "truncated": email_to_store[:255]}
                    )
                    email_to_store = email_to_store[:255]
            elif email:
                # Fallback: Only use single email if all_emails_str is empty
                email_to_store = email
                logger.warning(
                    f"⚠️ Using single email fallback for resume ID {resume_id} (extract_all_emails found nothing)",
                    extra={"resume_id": resume_id, "email": email}
                )
            else:
                email_to_store = None
            
            logger.info(
                f"📊 EMAIL EXTRACTION RESULT for resume ID {resume_id}",
                extra={
                    "resume_id": resume_id, 
                    "primary_email": email,
                    "all_emails": all_emails_str,
                    "email_to_store": email_to_store,
                    "file_name": filename
                }
            )
            
            if email_to_store:
                logger.info(
                    f"💾 UPDATING DATABASE: Resume ID {resume_id} with all emails: '{email_to_store}'",
                    extra={"resume_id": resume_id, "email": email_to_store, "file_name": filename}
                )
                
                updated_resume = await self.resume_repo.update(resume_id, {"email": email_to_store})
                if updated_resume:
                    logger.info(
                        f"✅ DATABASE UPDATED: Successfully saved all emails for resume ID {resume_id}",
                        extra={"resume_id": resume_id, "email": email_to_store}
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
    
    async def reprocess_email_from_file(
        self,
        resume_id: int,
        filename: str
    ) -> Optional[str]:
        """
        Reprocess email extraction from resume file on disk.
        This method finds the file, extracts text, and then extracts email.
        Useful for reprocessing resumes with NULL email values.
        
        Args:
            resume_id: The ID of the resume record in the database
            filename: Name of the resume file to find on disk
        
        Returns:
            The extracted email string or None if not found
        """
        try:
            logger.info(
                f"🔄 REPROCESSING EMAIL from file for resume ID {resume_id}: {filename}",
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
            
            # Extract email using the existing method (which includes retry logic)
            email = await self.extract_and_save_email(
                resume_text=resume_text,
                resume_id=resume_id,
                filename=filename,
                retry_on_null=True
            )
            
            if email:
                logger.info(f"✅ REPROCESS SUCCESS: Extracted email for resume ID {resume_id}: {email}")
            else:
                logger.warning(f"⚠️ REPROCESS: No email found for resume ID {resume_id}")
            
            return email
            
        except Exception as e:
            logger.error(
                f"ERROR: Failed to reprocess email from file for resume ID {resume_id}: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "resume_id": resume_id,
                    "file_name": filename,
                },
                exc_info=True
            )
            return None
    
    async def batch_reprocess_null_emails(
        self,
        limit: Optional[int] = None
    ) -> dict:
        """
        Batch reprocess all resumes with NULL email values.
        Finds resumes with NULL email, reads files from disk, and re-extracts email.
        
        Args:
            limit: Optional limit on number of resumes to process
        
        Returns:
            dict with summary statistics:
            {
                "total_found": int,
                "processed": int,
                "emails_extracted": int,
                "errors": int,
                "results": list of dicts with resume_id, filename, email, status
            }
        """
        result = {
            "total_found": 0,
            "processed": 0,
            "emails_extracted": 0,
            "errors": 0,
            "results": []
        }
        
        try:
            # Get all resumes with NULL email or mobile
            null_resumes = await self.resume_repo.get_resumes_with_null_email_or_mobile(limit=limit)
            
            # Filter to only those with NULL email
            null_email_resumes = [r for r in null_resumes if r.email is None]
            result["total_found"] = len(null_email_resumes)
            
            if not null_email_resumes:
                logger.info("No resumes found with NULL email")
                return result
            
            logger.info(f"Found {len(null_email_resumes)} resumes with NULL email, starting batch reprocessing...")
            
            for resume in null_email_resumes:
                try:
                    email = await self.reprocess_email_from_file(
                        resume_id=resume.id,
                        filename=resume.filename
                    )
                    
                    result["processed"] += 1
                    if email:
                        result["emails_extracted"] += 1
                        result["results"].append({
                            "resume_id": resume.id,
                            "filename": resume.filename,
                            "email": email,
                            "status": "success"
                        })
                    else:
                        result["results"].append({
                            "resume_id": resume.id,
                            "filename": resume.filename,
                            "email": None,
                            "status": "not_found"
                        })
                except Exception as e:
                    result["errors"] += 1
                    logger.error(f"Error reprocessing email for resume ID {resume.id}: {e}")
                    result["results"].append({
                        "resume_id": resume.id,
                        "filename": resume.filename,
                        "email": None,
                        "status": "error",
                        "error": str(e)
                    })
            
            logger.info(
                f"Batch reprocessing complete: {result['emails_extracted']} emails extracted from {result['processed']} resumes"
            )
            
        except Exception as e:
            logger.error(f"Error in batch reprocess: {e}", exc_info=True)
            result["errors"] += 1
        
        return result
    
    async def batch_reprocess_null_email_and_mobile(
        self,
        mobile_service: 'MobileService',  # type: ignore
        limit: Optional[int] = None
    ) -> dict:
        """
        Batch reprocess all resumes with NULL email OR mobile values.
        This is a convenience method that calls both email and mobile batch reprocess.
        
        Args:
            mobile_service: Instance of MobileService to use for mobile reprocessing
            limit: Optional limit on number of resumes to process
        
        Returns:
            dict with combined summary statistics from both email and mobile reprocessing
        """
        result = {
            "email_total_found": 0,
            "email_processed": 0,
            "email_extracted": 0,
            "email_errors": 0,
            "mobile_total_found": 0,
            "mobile_processed": 0,
            "mobile_extracted": 0,
            "mobile_errors": 0,
            "total_processed": 0,
            "total_extracted": 0
        }
        
        try:
            # Reprocess NULL emails
            email_result = await self.batch_reprocess_null_emails(limit=limit)
            result["email_total_found"] = email_result["total_found"]
            result["email_processed"] = email_result["processed"]
            result["email_extracted"] = email_result["emails_extracted"]
            result["email_errors"] = email_result["errors"]
            
            # Reprocess NULL mobiles
            mobile_result = await mobile_service.batch_reprocess_null_mobiles(limit=limit)
            result["mobile_total_found"] = mobile_result["total_found"]
            result["mobile_processed"] = mobile_result["processed"]
            result["mobile_extracted"] = mobile_result["mobiles_extracted"]
            result["mobile_errors"] = mobile_result["errors"]
            
            # Combined totals
            result["total_processed"] = result["email_processed"] + result["mobile_processed"]
            result["total_extracted"] = result["email_extracted"] + result["mobile_extracted"]
            
            logger.info(
                f"Combined batch reprocessing complete: "
                f"{result['email_extracted']} emails and {result['mobile_extracted']} mobiles extracted "
                f"from {result['total_processed']} resumes"
            )
            
        except Exception as e:
            logger.error(f"Error in combined batch reprocess: {e}", exc_info=True)
        
        return result

