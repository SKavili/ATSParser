"""Controller for resume-related operations."""
import gc
from pathlib import Path
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
from app.mastercategory import MasterCategoryService
from app.category import CategoryService
from app.role import RoleService
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
        self.mastercategory_service = MasterCategoryService(session)
        self.category_service = CategoryService(session)
        self.role_service = RoleService(session)
    
    def _parse_extract_modules(self, extract_modules: Optional[str]) -> set:
        """
        Parse extract_modules parameter and return set of modules to extract.
        
        Args:
            extract_modules: String like "all", "0", or "designation,skills,name" or "1,2,3"
        
        Returns:
            Set of module names to extract
        """
        if not extract_modules or extract_modules.lower().strip() in ("all", "0"):
            # Extract all modules (supports "all", "0", or empty/None)
            return {"designation", "name", "role", "email", "mobile", "experience", "domain", "education", "skills"}
        
        # Map numbers to module names
        module_map = {
            "1": "designation",
            "2": "name",
            "3": "role",
            "4": "email",
            "5": "mobile",
            "6": "experience",
            "7": "domain",
            "8": "education",
            "9": "skills",
        }
        
        # Parse comma-separated list
        modules = set()
        parts = [p.strip().lower() for p in extract_modules.split(",")]
        
        # If "0" or "all" appears anywhere in the list, extract all modules
        if "0" in parts or "all" in parts:
            return {"designation", "name", "role", "email", "mobile", "experience", "domain", "education", "skills"}
        
        for part in parts:
            if not part:
                continue
            # Check if it's a number
            if part in module_map:
                modules.add(module_map[part])
            # Check if it's a direct module name
            elif part in {"designation", "name", "role", "email", "mobile", "experience", "domain", "education", "skills"}:
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
            allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.jpg', '.jpeg', '.png', '.html', '.htm'}
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
            
            # Prepare metadata values
            candidate_name = None
            job_role = None
            if metadata:
                candidate_name = metadata.candidate_name
                job_role = metadata.job_role
            
            # Check if file with same filename already exists
            # If exists: Update existing record with extracted values for requested modules
            # If not: Create new record
            existing_resume = await self.resume_repo.get_by_filename(safe_filename)
            
            if existing_resume:
                # Use existing record - will update specific columns based on modules requested
                resume_metadata = existing_resume
                logger.info(
                    f"File with same filename exists, will update existing record with extracted values: {safe_filename}",
                    extra={"resume_id": resume_metadata.id, "file_name": safe_filename}
                )
                
                # Update candidate_name and job_role if provided in metadata
                update_data = {}
                if candidate_name:
                    update_data["candidatename"] = candidate_name
                if job_role:
                    update_data["jobrole"] = job_role
                
                # Set status to PROCESSING when starting extraction
                # Note: mastercategory will be updated in Step 2 below
                update_data["status"] = STATUS_PROCESSING
                
                # Update the existing record with new metadata and status
                if update_data:
                    await self.resume_repo.update(resume_metadata.id, update_data)
                    await self.resume_repo.session.refresh(resume_metadata)
            else:
                # Create new record if filename doesn't exist
                logger.info(
                    f"Creating new record for filename: {safe_filename}",
                    extra={"file_name": safe_filename}
                )
                db_record = {
                    "mastercategory": None,  # Will be extracted in Step 2
                    "category": None,  # Will be extracted later
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
                    "resume_text": None,  # Will be stored after text extraction
                }
                
                # STEP 1: Create new record in database (before text extraction for proper error handling)
                resume_metadata = await self.resume_repo.create(db_record)
                # Store resume_id early to avoid MissingGreenlet errors in exception handlers
                resume_id = resume_metadata.id
                logger.info(
                    f"[STEP 1] Database record created for resume ID {resume_id}",
                    extra={"resume_id": resume_id, "file_name": safe_filename}
                )
            
            # Extract text from file
            try:
                resume_text = await self.resume_parser.extract_text(file_content, safe_filename)
                
                # ✅ ADD LOG HERE
                logger.info(
                    f"📄 EXTRACTED RESUME TEXT for resume ID {resume_id}",
                    extra={
                        "resume_id": resume_id,
                        "file_name": safe_filename,
                        "text_length": len(resume_text),
                        "text_preview": resume_text[:500],  # First 500 chars
                        "full_text": resume_text  # Full text (be careful with large resumes)
                    }
                )
                
                # Store resume text in database
                await self.resume_repo.update(
                    resume_id,
                    {"resume_text": resume_text}
                )
                logger.info(
                    f"✅ Stored resume text in database for resume ID {resume_id}",
                    extra={"resume_id": resume_id, "text_length": len(resume_text)}
                )
            except Exception as e:
                # Update status to failed for extraction error
                # Use stored resume_id to avoid MissingGreenlet errors
                resume_id_safe = getattr(resume_metadata, 'id', None) if 'resume_metadata' in locals() else resume_id if 'resume_id' in locals() else None
                if resume_id_safe:
                    await self.resume_repo.update(
                        resume_id_safe,
                        {"status": get_failure_status(FAILURE_EXTRACTION_ERROR)}
                    )
                logger.error(
                    f"Text extraction failed for resume {resume_id_safe}: {e}",
                    extra={"resume_id": resume_id_safe, "file_name": safe_filename, "error": str(e)}
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
                # Use stored resume_id to avoid MissingGreenlet errors
                await self.resume_repo.update(
                    resume_id,
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
            
            # ============================================================================
            # STEP 2: MASTER CATEGORY CLASSIFICATION (IT vs NON_IT)
            # ============================================================================
            logger.info(
                f"[STEP 2] STARTING MASTER CATEGORY CLASSIFICATION for resume ID {resume_id}",
                extra={"resume_id": resume_id, "file_name": safe_filename}
            )
            try:
                mastercategory = await self.mastercategory_service.extract_and_save_mastercategory(
                    resume_text=resume_text,
                    resume_id=resume_id,
                    filename=safe_filename
                )
                logger.info(
                    f"[STEP 2] Master category classification completed: {mastercategory}",
                    extra={"resume_id": resume_id, "mastercategory": mastercategory}
                )
            except Exception as e:
                # Use stored resume_id to avoid MissingGreenlet errors in exception handler
                logger.error(
                    f"[STEP 2] MASTER CATEGORY CLASSIFICATION FAILED: {e}",
                    extra={"resume_id": resume_id, "error": str(e)}
                )
                # Continue processing even if mastercategory extraction fails
            
            # ============================================================================
            # STEP 3: CATEGORY CLASSIFICATION (based on mastercategory)
            # ============================================================================
            # Refresh to get latest mastercategory from database
            await self.resume_repo.session.refresh(resume_metadata)
            mastercategory_from_db = resume_metadata.mastercategory
            
            if mastercategory_from_db:
                logger.info(
                    f"[STEP 3] STARTING CATEGORY CLASSIFICATION for resume ID {resume_id}",
                    extra={
                        "resume_id": resume_id,
                        "mastercategory": mastercategory_from_db,
                        "file_name": safe_filename
                    }
                )
                try:
                    category = await self.category_service.extract_and_save_category(
                        resume_text=resume_text,
                        resume_id=resume_id,
                        mastercategory=mastercategory_from_db,
                        filename=safe_filename
                    )
                    logger.info(
                        f"[STEP 3] Category classification completed: {category}",
                        extra={
                            "resume_id": resume_id,
                            "mastercategory": mastercategory_from_db,
                            "category": category
                        }
                    )
                except Exception as e:
                    # Use stored resume_id to avoid MissingGreenlet errors in exception handler
                    logger.error(
                        f"[STEP 3] CATEGORY CLASSIFICATION FAILED: {e}",
                        extra={
                            "resume_id": resume_id,
                            "mastercategory": mastercategory_from_db,
                            "error": str(e)
                        }
                    )
                    # Continue processing even if category extraction fails
            else:
                logger.warning(
                    f"[STEP 3] SKIPPING CATEGORY CLASSIFICATION: mastercategory is None for resume ID {resume_id}",
                    extra={"resume_id": resume_id, "file_name": safe_filename}
                )
            
            # Parse extract_modules parameter
            # Accepts: "all" or comma-separated list like "designation,skills,name,role"
            # Valid options: designation, name, role, email, mobile, experience, domain, education, skills
            modules_to_extract = self._parse_extract_modules(extract_modules)
            
            # Extract and save selected profile fields using dedicated services (SEQUENTIAL)
            # These run one by one - if any fails, the resume upload still succeeds
            logger.info(
                f"🚀 STARTING PROFILE EXTRACTION PROCESS for resume ID {resume_id}",
                extra={
                    "resume_id": resume_id, 
                    "file_name": safe_filename,
                    "modules_to_extract": list(modules_to_extract)
                }
            )
            print(f"\n🚀 STARTING PROFILE EXTRACTION PROCESS")
            print(f"   Resume ID: {resume_id}")
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
                        resume_id=resume_id,
                        filename=safe_filename
                    )
                except Exception as e:
                    logger.error(f"❌ DESIGNATION EXTRACTION FAILED: {e}", extra={"resume_id": resume_id, "error": str(e)})
            
            # Extract name (2)
            if "name" in modules_to_extract:
                try:
                    await self.name_service.extract_and_save_name(
                        resume_text=resume_text,
                        resume_id=resume_id,
                        filename=safe_filename
                    )
                except Exception as e:
                    logger.error(f"❌ NAME EXTRACTION FAILED: {e}", extra={"resume_id": resume_id, "error": str(e)})
            
            # Extract role (3)
            if "role" in modules_to_extract:
                try:
                    await self.role_service.extract_and_save_role(
                        resume_text=resume_text,
                        resume_id=resume_id,
                        filename=safe_filename
                    )
                except Exception as e:
                    logger.error(f"❌ ROLE EXTRACTION FAILED: {e}", extra={"resume_id": resume_id, "error": str(e)})
            
            # Extract email (4)
            if "email" in modules_to_extract:
                try:
                    await self.email_service.extract_and_save_email(
                        resume_text=resume_text,
                        resume_id=resume_id,
                        filename=safe_filename
                    )
                except Exception as e:
                    logger.error(f"❌ EMAIL EXTRACTION FAILED: {e}", extra={"resume_id": resume_id, "error": str(e)})
            
            # Extract mobile (5)
            if "mobile" in modules_to_extract:
                try:
                    await self.mobile_service.extract_and_save_mobile(
                        resume_text=resume_text,
                        resume_id=resume_id,
                        filename=safe_filename
                    )
                except Exception as e:
                    logger.error(f"❌ MOBILE EXTRACTION FAILED: {e}", extra={"resume_id": resume_id, "error": str(e)})
            
            # Extract experience (6)
            if "experience" in modules_to_extract:
                try:
                    await self.experience_service.extract_and_save_experience(
                        resume_text=resume_text,
                        resume_id=resume_id,
                        filename=safe_filename
                    )
                except Exception as e:
                    logger.error(f"❌ EXPERIENCE EXTRACTION FAILED: {e}", extra={"resume_id": resume_id, "error": str(e)})
            
            # Extract domain (7)
            if "domain" in modules_to_extract:
                try:
                    await self.domain_service.extract_and_save_domain(
                        resume_text=resume_text,
                        resume_id=resume_id,
                        filename=safe_filename
                    )
                except Exception as e:
                    logger.error(f"❌ DOMAIN EXTRACTION FAILED: {e}", extra={"resume_id": resume_id, "error": str(e)})
            
            # Extract education (8)
            if "education" in modules_to_extract:
                try:
                    await self.education_service.extract_and_save_education(
                        resume_text=resume_text,
                        resume_id=resume_id,
                        filename=safe_filename
                    )
                except Exception as e:
                    logger.error(f"❌ EDUCATION EXTRACTION FAILED: {e}", extra={"resume_id": resume_id, "error": str(e)})
            
            # Extract skills (9)
            # Refresh resume metadata to ensure we have latest mastercategory and category
            await self.resume_repo.session.refresh(resume_metadata)
            if "skills" in modules_to_extract:
                try:
                    await self.skills_service.extract_and_save_skills(
                        resume_text=resume_text,
                        resume_id=resume_id,
                        filename=safe_filename
                    )
                except Exception as e:
                    logger.error(f"❌ SKILLS EXTRACTION FAILED: {e}", extra={"resume_id": resume_id, "error": str(e)})
            
            # All extractions completed - session contexts are automatically cleared
            # Each extraction used a fresh HTTP client with isolated context
            logger.info(
                f"✅ PROFILE EXTRACTION PROCESS COMPLETED for resume ID {resume_id}",
                extra={"resume_id": resume_id, "file_name": safe_filename}
            )
            print(f"\n✅ PROFILE EXTRACTION PROCESS COMPLETED")
            print(f"   Resume ID: {resume_id}")
            print(f"   [SESSION CLEARED] All extraction contexts have been isolated and cleared")
            
            # Final refresh to get all updated fields from database
            await self.resume_repo.session.refresh(resume_metadata)
            
            # Log all extracted fields for verification
            logger.info(
                f"✅ PROFILE EXTRACTION COMPLETED for resume ID {resume_id}",
                extra={
                    "resume_id": resume_id,
                    "candidatename": resume_metadata.candidatename,
                    "jobrole": resume_metadata.jobrole,
                    "designation": resume_metadata.designation,
                    "email": resume_metadata.email,
                    "mobile": resume_metadata.mobile,
                    "experience": resume_metadata.experience,
                    "domain": resume_metadata.domain,
                    "education": resume_metadata.education[:100] if resume_metadata.education else None,
                    "skillset": resume_metadata.skillset[:100] if resume_metadata.skillset else None,
                }
            )
            print(f"\n✅ PROFILE EXTRACTION COMPLETED")
            print(f"   Resume ID: {resume_id}")
            print(f"   Name: {resume_metadata.candidatename}")
            print(f"   Job Role: {resume_metadata.jobrole}")
            print(f"   Designation: {resume_metadata.designation}")
            print(f"   Email: {resume_metadata.email}")
            print(f"   Mobile: {resume_metadata.mobile}")
            print(f"   Experience: {resume_metadata.experience}")
            print(f"   Domain: {resume_metadata.domain}")
            print(f"   Master Category: {resume_metadata.mastercategory}")
            print(f"   Category: {resume_metadata.category}")
            print(f"   Education: {resume_metadata.education[:50] if resume_metadata.education else None}...")
            print(f"   Skills: {resume_metadata.skillset[:50] if resume_metadata.skillset else None}...")
            print()
            
            # ============================================================================
            # EMBEDDINGS & PINECONE STORAGE - TEMPORARILY DISABLED FOR DEVELOPMENT
            # ============================================================================
            # TODO: Re-enable when ready to use embeddings and vector search
            # This section generates embeddings and stores them in Pinecone for semantic search.
            # Currently disabled to focus on resume parsing functionality.
            # 
            # To re-enable:
            # 1. Uncomment the code below
            # 2. Ensure EMBEDDING_DIMENSION matches your embedding model (1024 for mxbai-embed-large, 768 for nomic-embed-text)
            # 3. Ensure Pinecone index dimension matches EMBEDDING_DIMENSION
            # ============================================================================
            
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
            
            # Update status to completed on success
            await self.resume_repo.update(
                resume_id,
                {"status": STATUS_COMPLETED}
            )
            
            # Final refresh to ensure all extracted fields are loaded from database
            await self.resume_repo.session.refresh(resume_metadata)
            
            # Verify all fields are updated (log for debugging)
            logger.info(
                f"📊 FINAL DATABASE STATE for resume ID {resume_id}",
                extra={
                    "resume_id": resume_id,
                    "mastercategory": resume_metadata.mastercategory,
                    "category": resume_metadata.category,
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
                id=resume_id,
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
            # Use safe attribute access to avoid MissingGreenlet errors
            resume_id_safe = None
            if 'resume_id' in locals():
                resume_id_safe = resume_id
            elif 'resume_metadata' in locals() and resume_metadata:
                try:
                    # Try to get ID without triggering lazy load
                    resume_id_safe = getattr(resume_metadata, 'id', None)
                except Exception:
                    # If that fails, try to access it directly (might work if object is still attached)
                    try:
                        resume_id_safe = resume_metadata.id
                    except Exception:
                        resume_id_safe = None
            
            if resume_id_safe:
                try:
                    await self.resume_repo.update(
                        resume_id_safe,
                        {"status": get_failure_status(FAILURE_UNKNOWN_ERROR)}
                    )
                except Exception as update_error:
                    logger.error(
                        f"Failed to update status after error: {update_error}",
                        extra={"resume_id": resume_id_safe, "error": str(update_error)}
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
    
    async def retry_failed_resume_with_ocr(
        self,
        resume_id: int,
        extract_modules: Optional[str] = "all"
    ) -> ResumeUploadResponse:
        """
        Retry processing a resume that failed with insufficient_text status.
        Finds the file from disk, retries extraction with OCR, and re-runs extraction modules.
        
        Args:
            resume_id: The ID of the failed resume to retry
            extract_modules: Modules to extract (default: "all")
        
        Returns:
            ResumeUploadResponse with updated data
        """
        # Configuration: Path where resume files are stored
        RESUME_FILES_DIRS = [
            Path("Resumes"),  # Capital R (common location)
            Path("resumes"),  # Lowercase
            Path("uploads"),
            Path("data/resumes"),
            Path("storage/resumes"),
            Path("files/resumes"),
            Path("."),  # Current directory
        ]
        
        try:
            # Get resume record from database
            resume_metadata = await self.resume_repo.get_by_id(resume_id)
            if not resume_metadata:
                raise HTTPException(status_code=404, detail=f"Resume with ID {resume_id} not found")
            
            # Check if status is failed:insufficient_text
            if not resume_metadata.status or not resume_metadata.status.startswith("failed:insufficient_text"):
                logger.warning(
                    f"Resume {resume_id} does not have failed:insufficient_text status. Current status: {resume_metadata.status}",
                    extra={"resume_id": resume_id, "current_status": resume_metadata.status}
                )
                # Still allow retry, but log warning
            
            filename = resume_metadata.filename
            if not filename:
                raise HTTPException(status_code=400, detail=f"Resume {resume_id} has no filename")
            
            logger.info(
                f"🔄 RETRYING RESUME with OCR: ID {resume_id}, filename: {filename}",
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
                raise HTTPException(
                    status_code=404,
                    detail=f"Resume file not found for ID {resume_id}: {filename}. Searched in: {[str(p) for p in possible_paths]}"
                )
            
            # Read file content
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            if not file_content:
                raise HTTPException(status_code=400, detail="File is empty")
            
            logger.info(f"Processing resume ID {resume_id}: {filename} ({len(file_content)} bytes)")
            
            # Update status to processing
            await self.resume_repo.update(
                resume_id,
                {"status": STATUS_PROCESSING}
            )
            
            # Store file_content for potential OCR retry later
            self._last_file_content = file_content
            
            # Extract text from file with OCR fallback
            resume_text = None
            try:
                resume_text = await self.resume_parser.extract_text(file_content, filename)
            except Exception as e:
                logger.warning(
                    f"Initial text extraction failed for resume {resume_id}, trying OCR fallback: {e}",
                    extra={"resume_id": resume_id, "file_name": filename, "error": str(e)}
                )
                resume_text = None
            
            # If extraction failed or returned insufficient text, try OCR fallback
            if not resume_text or len(resume_text.strip()) < 50:
                logger.info(
                    f"🔄 Text extraction returned insufficient text ({len(resume_text.strip()) if resume_text else 0} chars), "
                    f"trying OCR fallback for {filename}",
                    extra={
                        "resume_id": resume_id,
                        "file_name": filename,
                        "initial_text_length": len(resume_text.strip()) if resume_text else 0
                    }
                )
                try:
                    # Try OCR fallback using extract_text_with_fallback
                    ocr_text = await self.resume_parser.extract_text_with_fallback(
                        file_content, 
                        filename, 
                        original_text=resume_text
                    )
                    if ocr_text and len(ocr_text.strip()) > len(resume_text.strip() if resume_text else ""):
                        logger.info(
                            f"✅ OCR fallback succeeded: extracted {len(ocr_text.strip())} chars "
                            f"(vs {len(resume_text.strip()) if resume_text else 0} from initial extraction)",
                            extra={
                                "resume_id": resume_id,
                                "file_name": filename,
                                "ocr_text_length": len(ocr_text.strip()),
                                "initial_text_length": len(resume_text.strip()) if resume_text else 0
                            }
                        )
                        resume_text = ocr_text
                    elif ocr_text and len(ocr_text.strip()) >= 50:
                        logger.info(
                            f"✅ OCR fallback provided sufficient text: {len(ocr_text.strip())} chars",
                            extra={
                                "resume_id": resume_id,
                                "file_name": filename,
                                "ocr_text_length": len(ocr_text.strip())
                            }
                        )
                        resume_text = ocr_text
                except Exception as ocr_error:
                    logger.warning(
                        f"OCR fallback also failed for {filename}: {ocr_error}",
                        extra={
                            "resume_id": resume_id,
                            "file_name": filename,
                            "error": str(ocr_error)
                        }
                    )
            
            # If still no sufficient text after OCR fallback, update status and raise error
            if not resume_text or len(resume_text.strip()) < 50:
                await self.resume_repo.update(
                    resume_id,
                    {"status": get_failure_status(FAILURE_INSUFFICIENT_TEXT)}
                )
                logger.error(
                    f"❌ All text extraction methods failed for resume {resume_id}. "
                    f"Initial: {len(resume_text.strip()) if resume_text else 0} chars",
                    extra={
                        "resume_id": resume_id,
                        "file_name": filename,
                        "final_text_length": len(resume_text.strip()) if resume_text else 0
                    }
                )
                raise HTTPException(
                    status_code=400, 
                    detail="Could not extract sufficient text from resume even with OCR fallback"
                )
            
            # Limit text length to prevent excessive memory usage
            if len(resume_text) > settings.max_resume_text_length:
                logger.warning(
                    f"Resume text truncated from {len(resume_text)} to {settings.max_resume_text_length} characters",
                    extra={"original_length": len(resume_text), "truncated_length": settings.max_resume_text_length}
                )
                resume_text = resume_text[:settings.max_resume_text_length]
            
            # Store resume text in database
            await self.resume_repo.update(
                resume_id,
                {"resume_text": resume_text}
            )
            logger.info(
                f"✅ Stored resume text in database for resume ID {resume_id} (retry)",
                extra={"resume_id": resume_id, "text_length": len(resume_text)}
            )
            
            # Parse extract_modules parameter
            modules_to_extract = self._parse_extract_modules(extract_modules)
            
            # Extract and save selected profile fields using dedicated services (SEQUENTIAL)
            logger.info(
                f"[START] PROFILE EXTRACTION PROCESS (RETRY) for resume ID {resume_id}",
                extra={
                    "resume_id": resume_id,
                    "file_name": filename,
                    "modules_to_extract": list(modules_to_extract)
                }
            )
            
            # Sequential extraction with session isolation
            if "designation" in modules_to_extract:
                try:
                    await self.designation_service.extract_and_save_designation(
                        resume_text=resume_text,
                        resume_id=resume_id,
                        filename=filename
                    )
                except Exception as e:
                    logger.error(f"[ERROR] DESIGNATION EXTRACTION FAILED: {e}", extra={"resume_id": resume_id, "error": str(e)})
            
            if "name" in modules_to_extract:
                try:
                    await self.name_service.extract_and_save_name(
                        resume_text=resume_text,
                        resume_id=resume_id,
                        filename=filename
                    )
                except Exception as e:
                    logger.error(f"[ERROR] NAME EXTRACTION FAILED: {e}", extra={"resume_id": resume_id, "error": str(e)})
            
            if "email" in modules_to_extract:
                try:
                    await self.email_service.extract_and_save_email(
                        resume_text=resume_text,
                        resume_id=resume_id,
                        filename=filename
                    )
                except Exception as e:
                    logger.error(f"[ERROR] EMAIL EXTRACTION FAILED: {e}", extra={"resume_id": resume_id, "error": str(e)})
            
            if "mobile" in modules_to_extract:
                try:
                    await self.mobile_service.extract_and_save_mobile(
                        resume_text=resume_text,
                        resume_id=resume_id,
                        filename=filename
                    )
                except Exception as e:
                    logger.error(f"[ERROR] MOBILE EXTRACTION FAILED: {e}", extra={"resume_id": resume_id, "error": str(e)})
            
            if "experience" in modules_to_extract:
                try:
                    await self.experience_service.extract_and_save_experience(
                        resume_text=resume_text,
                        resume_id=resume_id,
                        filename=filename
                    )
                except Exception as e:
                    logger.error(f"[ERROR] EXPERIENCE EXTRACTION FAILED: {e}", extra={"resume_id": resume_id, "error": str(e)})
            
            if "domain" in modules_to_extract:
                try:
                    await self.domain_service.extract_and_save_domain(
                        resume_text=resume_text,
                        resume_id=resume_id,
                        filename=filename
                    )
                except Exception as e:
                    logger.error(f"[ERROR] DOMAIN EXTRACTION FAILED: {e}", extra={"resume_id": resume_id, "error": str(e)})
            
            if "education" in modules_to_extract:
                try:
                    await self.education_service.extract_and_save_education(
                        resume_text=resume_text,
                        resume_id=resume_id,
                        filename=filename
                    )
                except Exception as e:
                    logger.error(f"[ERROR] EDUCATION EXTRACTION FAILED: {e}", extra={"resume_id": resume_id, "error": str(e)})
            
            if "skills" in modules_to_extract:
                try:
                    await self.skills_service.extract_and_save_skills(
                        resume_text=resume_text,
                        resume_id=resume_id,
                        filename=filename
                    )
                except Exception as e:
                    logger.error(f"[ERROR] SKILLS EXTRACTION FAILED: {e}", extra={"resume_id": resume_id, "error": str(e)})
            
            # Clear file_content from memory
            if settings.enable_memory_cleanup:
                if hasattr(self, '_last_file_content'):
                    del self._last_file_content
                del file_content
                gc.collect()
            
            logger.info(
                f"[SUCCESS] PROFILE EXTRACTION PROCESS COMPLETED (RETRY) for resume ID {resume_id}",
                extra={"resume_id": resume_id, "file_name": filename}
            )
            
            # Update status to completed on success
            try:
                await self.resume_repo.update(
                    resume_id,
                    {"status": STATUS_COMPLETED}
                )
                logger.info(
                    f"✅ Status updated to COMPLETED for resume ID {resume_id}",
                    extra={"resume_id": resume_id}
                )
            except Exception as status_error:
                logger.error(
                    f"❌ Failed to update status to COMPLETED: {status_error}",
                    extra={"resume_id": resume_id, "error": str(status_error)}
                )
            
            # Final refresh to get all updated fields from database
            try:
                await self.resume_repo.session.refresh(resume_metadata)
            except Exception as refresh_error:
                logger.warning(
                    f"Failed to refresh resume metadata: {refresh_error}",
                    extra={"resume_id": resume_id}
                )
            
            # Build response with all extracted fields
            return ResumeUploadResponse(
                id=resume_id,
                candidateName=resume_metadata.candidatename or "",
                jobrole=resume_metadata.jobrole or "",
                designation=resume_metadata.designation or "",
                experience=resume_metadata.experience or "",
                domain=resume_metadata.domain or "",
                mobile=resume_metadata.mobile or "",
                email=resume_metadata.email or "",
                education=resume_metadata.education or "",
                filename=resume_metadata.filename,
                skillset=resume_metadata.skillset or "",
                status=resume_metadata.status or STATUS_PENDING,
                created_at=resume_metadata.created_at.isoformat() if resume_metadata.created_at else "",
            )
        
        except HTTPException:
            raise
        except Exception as e:
            # Update status to failed if we have a resume record ID
            try:
                await self.resume_repo.update(
                    resume_id,
                    {"status": get_failure_status(FAILURE_EXTRACTION_ERROR)}
                )
            except Exception as update_error:
                logger.error(
                    f"Failed to update status after retry error: {update_error}",
                    extra={"resume_id": resume_id, "error": str(update_error)}
                )
            
            logger.error(
                f"Error retrying resume with OCR: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "resume_id": resume_id,
                },
                exc_info=True
            )
            raise HTTPException(status_code=500, detail=f"Failed to retry resume with OCR: {str(e)}")
