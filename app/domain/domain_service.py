"""Service for extracting and saving domain to database."""
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.domain_extractor import DomainExtractor
from app.repositories.resume_repo import ResumeRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)


class DomainService:
    """Service for extracting domain from resume and saving to database."""
    
    def __init__(self, session: AsyncSession):
        self.domain_extractor = DomainExtractor()
        self.resume_repo = ResumeRepository(session)
    
    async def extract_and_save_domain(
        self,
        resume_text: str,
        resume_id: int,
        filename: str = "resume"
    ) -> Optional[str]:
        """
        Extract domain from resume text and update the database record.
        
        Args:
            resume_text: The text content of the resume
            resume_id: The ID of the resume record in the database
            filename: Name of the resume file (for logging)
        
        Returns:
            The extracted domain string or None if not found
        """
        try:
            logger.info(
                f"🔍 STARTING DOMAIN EXTRACTION for resume ID {resume_id}",
                extra={
                    "resume_id": resume_id, 
                    "file_name": filename,
                }
            )
            
            domain = await self.domain_extractor.extract_domain(resume_text, filename)
            
            logger.info(
                f"📊 DOMAIN EXTRACTION RESULT for resume ID {resume_id}: {domain}",
                extra={
                    "resume_id": resume_id, 
                    "domain": domain, 
                    "file_name": filename,
                    "resume_text_length": len(resume_text) if resume_text else 0
                }
            )
            
            # Enhanced logging for null domain cases
            if not domain:
                logger.warning(
                    f"⚠️ Domain extraction returned NULL for resume ID {resume_id}",
                    extra={
                        "resume_id": resume_id,
                        "file_name": filename,
                        "text_length": len(resume_text.strip()) if resume_text else 0,
                        "text_preview": resume_text[:500] if resume_text else "",
                        "possible_reasons": [
                            "No domain information in resume",
                            "Text extraction may have failed",
                            "LLM could not identify domain",
                            "Resume may need OCR retry"
                        ]
                    }
                )
                
                # If resume_text seems insufficient, log for potential OCR retry
                if resume_text and len(resume_text.strip()) < 200:
                    logger.warning(
                        f"⚠️ Domain extraction returned NULL with short text ({len(resume_text.strip())} chars). "
                        f"Consider OCR retry for resume ID {resume_id}",
                        extra={
                            "resume_id": resume_id,
                            "file_name": filename,
                            "text_length": len(resume_text.strip())
                        }
                    )
            
            if domain:
                logger.info(
                    f"💾 UPDATING DATABASE: Resume ID {resume_id} with domain: '{domain}'",
                    extra={"resume_id": resume_id, "domain": domain, "file_name": filename}
                )
                
                updated_resume = await self.resume_repo.update(resume_id, {"domain": domain})
                if updated_resume:
                    logger.info(
                        f"✅ DATABASE UPDATED: Successfully saved domain for resume ID {resume_id}",
                        extra={"resume_id": resume_id, "domain": domain}
                    )
                else:
                    logger.error(f"❌ DATABASE UPDATE FAILED: Resume ID {resume_id} - record not found")
            else:
                logger.warning(
                    f"💾 SAVING NULL: No domain found for resume ID {resume_id}, saving as NULL",
                    extra={"resume_id": resume_id, "file_name": filename}
                )
                await self.resume_repo.update(resume_id, {"domain": None})
            
            return domain
            
        except Exception as e:
            logger.error(
                f"ERROR: Failed to extract and save domain for resume ID {resume_id}: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "resume_id": resume_id,
                    "file_name": filename,
                },
                exc_info=True
            )
            try:
                await self.resume_repo.update(resume_id, {"domain": None})
                logger.info(f"Saved NULL domain for resume ID {resume_id} after extraction failure")
            except Exception as db_error:
                logger.error(f"Failed to update database with NULL domain: {db_error}")
            
            return None

