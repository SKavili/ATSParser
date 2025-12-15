"""Controller for resume-related operations."""
import gc
from typing import Optional
from fastapi import UploadFile, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.resume_parser import ResumeParser
from app.services.embedding_service import EmbeddingService
from app.services.vector_db_service import VectorDBService
from app.services.designation_service import DesignationService
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
    
    async def upload_resume(
        self,
        file: UploadFile,
        metadata: Optional[ResumeUpload] = None
    ) -> ResumeUploadResponse:
        """Handle resume upload, parsing, and embedding."""
        try:
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
            
            # Sanitize filename (already done above, but ensure it's set)
            if 'safe_filename' not in locals():
                safe_filename = sanitize_filename(file.filename or "resume.pdf")
            
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
            
            # Merge metadata if provided
            candidate_name = None
            job_role = None
            if metadata:
                candidate_name = metadata.candidate_name
                job_role = metadata.job_role
            
            # Prepare database record (without designation initially, status = processing)
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
            
            # Save to database first
            resume_metadata = await self.resume_repo.create(db_record)
            
            # Extract and save designation using dedicated service
            # This runs independently - if it fails, the resume upload still succeeds
            logger.info(
                f"🚀 STARTING DESIGNATION EXTRACTION PROCESS for resume ID {resume_metadata.id}",
                extra={"resume_id": resume_metadata.id, "file_name": safe_filename}
            )
            print(f"\n🚀 STARTING DESIGNATION EXTRACTION PROCESS")
            print(f"   Resume ID: {resume_metadata.id}")
            print(f"   Filename: {safe_filename}")
            
            try:
                designation = await self.designation_service.extract_and_save_designation(
                    resume_text=resume_text,
                    resume_id=resume_metadata.id,
                    filename=safe_filename
                )
                # Refresh to get updated designation
                await self.resume_repo.session.refresh(resume_metadata)
                logger.info(
                    f"✅ DESIGNATION EXTRACTION COMPLETED for resume ID {resume_metadata.id}",
                    extra={
                        "resume_id": resume_metadata.id, 
                        "designation": designation,
                        "designation_in_db": resume_metadata.designation
                    }
                )
                print(f"\n✅ DESIGNATION EXTRACTION COMPLETED")
                print(f"   Resume ID: {resume_metadata.id}")
                print(f"   Extracted: {designation}")
                print(f"   In Database: {resume_metadata.designation}\n")
            except Exception as e:
                # Log error but don't fail the upload
                logger.error(
                    f"❌ DESIGNATION EXTRACTION FAILED for resume ID {resume_metadata.id}, but upload succeeded: {e}",
                    extra={"error": str(e), "resume_id": resume_metadata.id, "file_name": safe_filename},
                    exc_info=True
                )
                print(f"\n❌ DESIGNATION EXTRACTION FAILED (upload still succeeded)")
                print(f"   Resume ID: {resume_metadata.id}")
                print(f"   Error: {e}\n")
            
            # Generate embeddings for resume text
            chunk_embeddings = await self.embedding_service.generate_chunk_embeddings(
                resume_text,
                metadata={
                    "resume_id": resume_metadata.id,
                    "filename": safe_filename,
                    "candidate_name": candidate_name or "",
                }
            )
            
            # Store embeddings in vector DB
            vectors_to_store = []
            for chunk_data in chunk_embeddings:
                vector_id = f"resume_{resume_metadata.id}_chunk_{chunk_data['chunk_index']}"
                vectors_to_store.append({
                    "id": vector_id,
                    "embedding": chunk_data["embedding"],
                    "metadata": {
                        **chunk_data["metadata"],
                        "type": "resume",  # Mark as resume vector
                        "chunk_index": chunk_data["chunk_index"],
                        "text_preview": chunk_data["text"][:200],
                    }
                })
            
            if vectors_to_store:
                await self.vector_db.upsert_vectors(vectors_to_store)
                logger.info(
                    f"Stored {len(vectors_to_store)} embeddings for resume {resume_metadata.id}",
                    extra={"resume_id": resume_metadata.id, "vector_count": len(vectors_to_store)}
                )
                # Clear embeddings from memory after storing
                if settings.enable_memory_cleanup:
                    del vectors_to_store
                    del chunk_embeddings
                    gc.collect()
            
            # Update status to completed on success
            await self.resume_repo.update(
                resume_metadata.id,
                {"status": STATUS_COMPLETED}
            )
            await self.resume_repo.session.refresh(resume_metadata)
            
            # Build response
            return ResumeUploadResponse(
                id=resume_metadata.id,
                candidateName=resume_metadata.candidatename or "",
                jobrole=resume_metadata.jobrole or "",
                designation=resume_metadata.designation,  # Extracted designation
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
