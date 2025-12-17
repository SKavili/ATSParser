"""Controller for resume-related operations."""
import gc
from typing import Optional
from fastapi import UploadFile, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
 
from app.services.resume_parser import ResumeParser
from app.services.embedding_service import EmbeddingService
from app.services.vector_db_service import VectorDBService
from app.designation import DesignationService
from app.skills import SkillsService
from app.email import EmailService
from app.mobile import MobileService
from app.experience import ExperienceService
from app.name import NameService
from app.domain import DomainService
from app.education import EducationService
from app.repositories.resume_repo import ResumeRepository
from app.models.resume_models import ResumeUpload, ResumeUploadResponse
from app.utils.cleaning import sanitize_filename
from app.utils.logging import get_logger
from app.config import settings
from app.constants.resume_status import (
    STATUS_PENDING, STATUS_PROCESSING, STATUS_COMPLETED,
    get_failure_status, FAILURE_FILE_TOO_LARGE, FAILURE_INVALID_FILE_TYPE,
    FAILURE_EMPTY_FILE, FAILURE_INSUFFICIENT_TEXT, FAILURE_EXTRACTION_ERROR,
    FAILURE_DESIGNATION_EXTRACTION_FAILED, FAILURE_DATABASE_ERROR, FAILURE_UNKNOWN_ERROR
)
 
logger = get_logger(__name__)
 
 
class ResumeController:
    """Controller for handling resume upload and processing."""
 
    def __init__(
        self,
        resume_parser: ResumeParser,
        embedding_service: EmbeddingService,
        vector_db: VectorDBService,
        resume_repo: ResumeRepository,
        session: AsyncSession
    ):
        self.resume_parser = resume_parser
        self.embedding_service = embedding_service
        self.vector_db = vector_db
        self.resume_repo = resume_repo
        self.designation_service = DesignationService(session)
        self.skills_service = SkillsService(session)
        self.email_service = EmailService(session)
        self.mobile_service = MobileService(session)
        self.experience_service = ExperienceService(session)
        self.name_service = NameService(session)
        self.domain_service = DomainService(session)
        self.education_service = EducationService(session)
   
    def _parse_extract_modules(self, extract_modules: Optional[str]) -> set:
        """
        Parse extract_modules parameter and return set of modules to extract.
       
        Args:
            extract_modules: String like "all" or "designation,skills,name" or "1,2,3"
       
        Returns:
            Set of module names to extract
        """
        if not extract_modules or extract_modules.lower().strip() == "all":
            # Extract all modules
            return {"designation", "name", "email", "mobile", "experience", "domain", "education", "skills"}
       
        # Map numbers to module names
        module_map = {
            "1": "designation",
            "2": "name",
            "3": "email",
            "4": "mobile",
            "5": "experience",
            "6": "domain",
            "7": "education",
            "8": "skills",
        }
       
        # Parse comma-separated list
        modules = set()
        parts = [p.strip().lower() for p in extract_modules.split(",")]
       
        for part in parts:
            if not part:
                continue
            # Check if it's a number
            if part in module_map:
                modules.add(module_map[part])
            # Check if it's a direct module name
            elif part in {"designation", "name", "email", "mobile", "experience", "domain", "education", "skills"}:
                modules.add(part)
            else:
                logger.warning(f"Unknown module option: {part}, ignoring")
       
        return modules
   
    async def upload_resume(
        self,
        file: UploadFile,
        metadata: Optional[ResumeUpload] = None,
        extract_modules: Optional[str] = "all"
    ) -> ResumeUploadResponse:
        """Handle resume upload, parsing, and embedding."""
        try:
            # Sanitize filename early
            safe_filename = sanitize_filename(file.filename or "resume.pdf")
           
            # Validate file type
            allowed_extensions = {'.pdf', '.docx', '.doc', '.txt'}
            file_ext = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
            if file_ext not in allowed_extensions:
                # Create record with failed status for invalid file type
                try:
                    db_record = {
                        "candidatename": None,
                        "jobrole": None,
                        "designation": None,
                        "experience": None,
                        "domain": None,
                        "mobile": None,
                        "email": None,
                        "education": None,
                        "filename": safe_filename,
                        "skillset": "",
                        "status": get_failure_status(FAILURE_INVALID_FILE_TYPE),
                    }
                    resume_metadata = await self.resume_repo.create(db_record)
                    logger.warning(
                        f"Invalid file type rejected, created record with failed status: {resume_metadata.id}",
                        extra={"resume_id": resume_metadata.id, "file_name": safe_filename, "file_ext": file_ext}
                    )
                except Exception as e:
                    logger.error(f"Failed to create record for invalid file: {e}", extra={"error": str(e)})
               
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
                )
           
            # Read file content with size limit check
            file_content = await file.read()
            if not file_content:
                # Create record with failed status for empty file
                try:
                    db_record = {
                        "candidatename": None,
                        "jobrole": None,
                        "designation": None,
                        "experience": None,
                        "domain": None,
                        "mobile": None,
                        "email": None,
                        "education": None,
                        "filename": safe_filename,
                        "skillset": "",
                        "status": get_failure_status(FAILURE_EMPTY_FILE),
                    }
                    resume_metadata = await self.resume_repo.create(db_record)
                    logger.warning(
                        f"Empty file rejected, created record with failed status: {resume_metadata.id}",
                        extra={"resume_id": resume_metadata.id, "file_name": safe_filename}
                    )
                except Exception as e:
                    logger.error(f"Failed to create record for empty file: {e}", extra={"error": str(e)})
               
                raise HTTPException(status_code=400, detail="Empty file")
           
            # Check file size limit (memory optimization)
            file_size_mb = len(file_content) / (1024 * 1024)
            if file_size_mb > settings.max_file_size_mb:
                # Create record with failed status for file too large
                try:
                    db_record = {
                        "candidatename": None,
                        "jobrole": None,
                        "designation": None,
                        "experience": None,
                        "domain": None,
                        "mobile": None,
                        "email": None,
                        "education": None,
                        "filename": safe_filename,
                        "skillset": "",
                        "status": get_failure_status(FAILURE_FILE_TOO_LARGE),
                    }
                    resume_metadata = await self.resume_repo.create(db_record)
                    logger.warning(
                        f"File too large rejected, created record with failed status: {resume_metadata.id}",
                        extra={"resume_id": resume_metadata.id, "file_name": safe_filename, "file_size_mb": file_size_mb}
                    )
                except Exception as e:
                    logger.error(f"Failed to create record for large file: {e}", extra={"error": str(e)})
               
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large: {file_size_mb:.2f}MB. Maximum allowed: {settings.max_file_size_mb}MB"
                )
           
            # Check if file with same filename already exists
            existing_resume = await self.resume_repo.get_by_filename(safe_filename)
            if existing_resume:
                logger.info(
                    f"File already processed, skipping: {safe_filename}",
                    extra={"resume_id": existing_resume.id, "file_name": safe_filename}
                )
                # Return existing resume data
                return ResumeUploadResponse(
                    id=existing_resume.id,
                    candidateName=existing_resume.candidatename or "",
                    jobrole=existing_resume.jobrole or "",
                    designation=existing_resume.designation,
                    experience=existing_resume.experience or "",
                    domain=existing_resume.domain or "",
                    mobile=existing_resume.mobile or "",
                    email=existing_resume.email or "",
                    education=existing_resume.education or "",
                    filename=existing_resume.filename,
                    skillset=existing_resume.skillset or "",
                    status=existing_resume.status or STATUS_PENDING,
                    created_at=existing_resume.created_at.isoformat() if existing_resume.created_at else "",
                )
           
            # Prepare database record first (before text extraction)
            # This ensures resume_metadata exists for error handling
            candidate_name = None
            job_role = None
            if metadata:
                candidate_name = metadata.candidate_name
                job_role = metadata.job_role
           
            db_record = {
                "candidatename": candidate_name,
                "jobrole": job_role,
                "designation": None,  # Will be extracted and updated separately
                "experience": None,
                "domain": None,
                "mobile": None,
                "email": None,
                "education": None,
                "filename": safe_filename,
                "skillset": "",
                "status": STATUS_PROCESSING,  # Set to processing when starting
            }
           
            # Save to database first (before text extraction for proper error handling)
            resume_metadata = await self.resume_repo.create(db_record)
           
            # Extract text from file
            try:
                resume_text = await self.resume_parser.extract_text(file_content, safe_filename)
            except Exception as e:
                # Update status to failed for extraction error
                await self.resume_repo.update(
                    resume_metadata.id,
                    {"status": get_failure_status(FAILURE_EXTRACTION_ERROR)}
                )
                logger.error(
                    f"Text extraction failed for resume {resume_metadata.id}: {e}",
                    extra={"resume_id": resume_metadata.id, "file_name": safe_filename, "error": str(e)}
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to extract text from file: {str(e)}"
                )
           
            # Clear file_content from memory early (memory optimization)
            if settings.enable_memory_cleanup:
                del file_content
                gc.collect()
           
            if not resume_text or len(resume_text.strip()) < 50:
                # Update status to failed for insufficient text
                await self.resume_repo.update(
                    resume_metadata.id,
                    {"status": get_failure_status(FAILURE_INSUFFICIENT_TEXT)}
                )
                raise HTTPException(status_code=400, detail="Could not extract sufficient text from resume")
           
            # Limit text length to prevent excessive memory usage
            if len(resume_text) > settings.max_resume_text_length:
                logger.warning(
                    f"Resume text truncated from {len(resume_text)} to {settings.max_resume_text_length} characters",
                    extra={"original_length": len(resume_text), "truncated_length": settings.max_resume_text_length}
                )
                resume_text = resume_text[:settings.max_resume_text_length]
           
            # Parse extract_modules parameter
            # Accepts: "all" or comma-separated list like "designation,skills,name"
            # Valid options: designation, name, email, mobile, experience, domain, education, skills
            modules_to_extract = self._parse_extract_modules(extract_modules)
           
            # Extract and save selected profile fields using dedicated services (SEQUENTIAL)
            # These run one by one - if any fails, the resume upload still succeeds
            logger.info(
                f"[START] PROFILE EXTRACTION PROCESS for resume ID {resume_metadata.id}",
                extra={
                    "resume_id": resume_metadata.id,
                    "file_name": safe_filename,
                    "modules_to_extract": list(modules_to_extract)
                }
            )
            print(f"\n[START] PROFILE EXTRACTION PROCESS")
            print(f"   Resume ID: {resume_metadata.id}")
            print(f"   Filename: {safe_filename}")
            print(f"   Modules to extract: {', '.join(sorted(modules_to_extract)) if modules_to_extract else 'None'}")
            print(f"   [SESSION ISOLATION] Each extraction uses a fresh, isolated LLM context")
           
            # Sequential extraction with session isolation
            # Each extraction creates a fresh HTTP client and includes system messages
            # to ensure no context bleeding between extractions
            # Extract designation (1)
            if "designation" in modules_to_extract:
                try:
                    await self.designation_service.extract_and_save_designation(
                        resume_text=resume_text,
                        resume_id=resume_metadata.id,
                        filename=safe_filename
                    )
                except Exception as e:
                    logger.error(f"[ERROR] DESIGNATION EXTRACTION FAILED: {e}", extra={"resume_id": resume_metadata.id, "error": str(e)})
           
            # Extract name (2)
            if "name" in modules_to_extract:
                try:
                    await self.name_service.extract_and_save_name(
                        resume_text=resume_text,
                        resume_id=resume_metadata.id,
                        filename=safe_filename
                    )
                except Exception as e:
                    logger.error(f"[ERROR] NAME EXTRACTION FAILED: {e}", extra={"resume_id": resume_metadata.id, "error": str(e)})
           
            # Extract email (3)
            if "email" in modules_to_extract:
                try:
                    await self.email_service.extract_and_save_email(
                        resume_text=resume_text,
                        resume_id=resume_metadata.id,
                        filename=safe_filename
                    )
                except Exception as e:
                    logger.error(f"[ERROR] EMAIL EXTRACTION FAILED: {e}", extra={"resume_id": resume_metadata.id, "error": str(e)})
           
            # Extract mobile (4)
            if "mobile" in modules_to_extract:
                try:
                    await self.mobile_service.extract_and_save_mobile(
                        resume_text=resume_text,
                        resume_id=resume_metadata.id,
                        filename=safe_filename
                    )
                except Exception as e:
                    logger.error(f"[ERROR] MOBILE EXTRACTION FAILED: {e}", extra={"resume_id": resume_metadata.id, "error": str(e)})
           
            # Extract experience (5)
            if "experience" in modules_to_extract:
                try:
                    await self.experience_service.extract_and_save_experience(
                        resume_text=resume_text,
                        resume_id=resume_metadata.id,
                        filename=safe_filename
                    )
                except Exception as e:
                    logger.error(f"[ERROR] EXPERIENCE EXTRACTION FAILED: {e}", extra={"resume_id": resume_metadata.id, "error": str(e)})
           
            # Extract domain (6)
            if "domain" in modules_to_extract:
                try:
                    await self.domain_service.extract_and_save_domain(
                        resume_text=resume_text,
                        resume_id=resume_metadata.id,
                        filename=safe_filename
                    )
                except Exception as e:
                    logger.error(f"[ERROR] DOMAIN EXTRACTION FAILED: {e}", extra={"resume_id": resume_metadata.id, "error": str(e)})
           
            # Extract education (7)
            if "education" in modules_to_extract:
                try:
                    await self.education_service.extract_and_save_education(
                        resume_text=resume_text,
                        resume_id=resume_metadata.id,
                        filename=safe_filename
                    )
                except Exception as e:
                    logger.error(f"[ERROR] EDUCATION EXTRACTION FAILED: {e}", extra={"resume_id": resume_metadata.id, "error": str(e)})
           
            # Extract skills (8)
            if "skills" in modules_to_extract:
                try:
                    await self.skills_service.extract_and_save_skills(
                        resume_text=resume_text,
                        resume_id=resume_metadata.id,
                        filename=safe_filename
                    )
                except Exception as e:
                    logger.error(f"[ERROR] SKILLS EXTRACTION FAILED: {e}", extra={"resume_id": resume_metadata.id, "error": str(e)})
           
            # All extractions completed - session contexts are automatically cleared
            # Each extraction used a fresh HTTP client with isolated context
            logger.info(
                f"[SUCCESS] PROFILE EXTRACTION PROCESS COMPLETED for resume ID {resume_metadata.id}",
                extra={"resume_id": resume_metadata.id, "file_name": safe_filename}
            )
            print(f"\n[SUCCESS] PROFILE EXTRACTION PROCESS COMPLETED")
            print(f"   Resume ID: {resume_metadata.id}")
            print(f"   [SESSION CLEARED] All extraction contexts have been isolated and cleared")
           
            # Final refresh to get all updated fields from database
            await self.resume_repo.session.refresh(resume_metadata)
           
            # Log all extracted fields for verification
            logger.info(
                f"[SUCCESS] PROFILE EXTRACTION COMPLETED for resume ID {resume_metadata.id}",
                extra={
                    "resume_id": resume_metadata.id,
                    "candidatename": resume_metadata.candidatename,
                    "designation": resume_metadata.designation,
                    "email": resume_metadata.email,
                    "mobile": resume_metadata.mobile,
                    "experience": resume_metadata.experience,
                    "domain": resume_metadata.domain,
                    "education": resume_metadata.education[:100] if resume_metadata.education else None,
                    "skillset": resume_metadata.skillset[:100] if resume_metadata.skillset else None,
                }
            )
            print(f"\n[SUCCESS] PROFILE EXTRACTION COMPLETED")
            print(f"   Resume ID: {resume_metadata.id}")
            print(f"   Name: {resume_metadata.candidatename}")
            print(f"   Designation: {resume_metadata.designation}")
            print(f"   Email: {resume_metadata.email}")
            print(f"   Mobile: {resume_metadata.mobile}")
            print(f"   Experience: {resume_metadata.experience}")
            print(f"   Domain: {resume_metadata.domain}")
            print(f"   Education: {resume_metadata.education[:50] if resume_metadata.education else None}...")
            print(f"   Skills: {resume_metadata.skillset[:50] if resume_metadata.skillset else None}...")
            print()
           
            # ============================================================
            # EMBEDDINGS DISABLED - Commented out to avoid FAISS errors
            # ============================================================
            # # Generate embeddings for resume text
            # chunk_embeddings = await self.embedding_service.generate_chunk_embeddings(
            #     resume_text,
            #     metadata={
            #         "resume_id": resume_metadata.id,
            #         "filename": safe_filename,
            #         "candidate_name": candidate_name or "",
            #     }
            # )
            #
            # # Store embeddings in vector DB
            # vectors_to_store = []
            # for chunk_data in chunk_embeddings:
            #     vector_id = f"resume_{resume_metadata.id}_chunk_{chunk_data['chunk_index']}"
            #     vectors_to_store.append({
            #         "id": vector_id,
            #         "embedding": chunk_data["embedding"],
            #         "metadata": {
            #             **chunk_data["metadata"],
            #             "type": "resume",  # Mark as resume vector
            #             "chunk_index": chunk_data["chunk_index"],
            #             "text_preview": chunk_data["text"][:200],
            #         }
            #     })
            #
            # if vectors_to_store:
            #     await self.vector_db.upsert_vectors(vectors_to_store)
            #     logger.info(
            #         f"Stored {len(vectors_to_store)} embeddings for resume {resume_metadata.id}",
            #         extra={"resume_id": resume_metadata.id, "vector_count": len(vectors_to_store)}
            #     )
            #     # Clear embeddings from memory after storing
            #     if settings.enable_memory_cleanup:
            #         del vectors_to_store
            #         del chunk_embeddings
            #         gc.collect()
           
            logger.info(f"Embeddings disabled - skipping vector generation for resume {resume_metadata.id}")
           
            # Update status to completed on success
            await self.resume_repo.update(
                resume_metadata.id,
                {"status": STATUS_COMPLETED}
            )
           
            # Final refresh to ensure all extracted fields are loaded from database
            await self.resume_repo.session.refresh(resume_metadata)
           
            # Verify all fields are updated (log for debugging)
            logger.info(
                f"[RESULT] FINAL DATABASE STATE for resume ID {resume_metadata.id}",
                extra={
                    "resume_id": resume_metadata.id,
                    "candidatename": resume_metadata.candidatename,
                    "designation": resume_metadata.designation,
                    "email": resume_metadata.email,
                    "mobile": resume_metadata.mobile,
                    "experience": resume_metadata.experience,
                    "domain": resume_metadata.domain,
                    "education": resume_metadata.education[:100] if resume_metadata.education else None,
                    "skillset": resume_metadata.skillset[:100] if resume_metadata.skillset else None,
                    "status": resume_metadata.status,
                }
            )
           
            # Build response with all extracted fields
            return ResumeUploadResponse(
                id=resume_metadata.id,
                candidateName=resume_metadata.candidatename or "",
                jobrole=resume_metadata.jobrole or "",
                designation=resume_metadata.designation or "",  # Extracted designation
                experience=resume_metadata.experience or "",  # Extracted experience
                domain=resume_metadata.domain or "",  # Extracted domain
                mobile=resume_metadata.mobile or "",  # Extracted mobile
                email=resume_metadata.email or "",  # Extracted email
                education=resume_metadata.education or "",  # Extracted education
                filename=resume_metadata.filename,
                skillset=resume_metadata.skillset or "",  # Extracted skills
                status=resume_metadata.status or STATUS_PENDING,
                created_at=resume_metadata.created_at.isoformat() if resume_metadata.created_at else "",
            )
       
        except HTTPException:
            raise
        except Exception as e:
            # Update status to failed if we have a resume_metadata record
            if 'resume_metadata' in locals() and resume_metadata and resume_metadata.id:
                try:
                    await self.resume_repo.update(
                        resume_metadata.id,
                        {"status": get_failure_status(FAILURE_UNKNOWN_ERROR)}
                    )
                except Exception as update_error:
                    logger.error(
                        f"Failed to update status after error: {update_error}",
                        extra={"resume_id": resume_metadata.id, "error": str(update_error)}
                    )
           
            logger.error(
                f"Error uploading resume: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "file_name": getattr(file, "filename", None) if file else None,
                    "file_size": len(file_content) if 'file_content' in locals() else None,
                },
                exc_info=True
            )
            raise HTTPException(status_code=500, detail=f"Failed to process resume: {str(e)}")