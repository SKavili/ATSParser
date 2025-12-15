# """Controller for resume-related operations."""
# from typing import Optional
# from fastapi import UploadFile, HTTPException
# from sqlalchemy.ext.asyncio import AsyncSession

# from app.services.resume_parser import ResumeParser
# from app.services.embedding_service import EmbeddingService
# from app.services.vector_db_service import VectorDBService
# from app.repositories.resume_repo import ResumeRepository
# from app.models.resume_models import ResumeUpload, ResumeUploadResponse
# from app.utils.cleaning import sanitize_filename
# from app.utils.logging import get_logger
# from app.config import settings
# import uuid

# logger = get_logger(__name__)


# class ResumeController:
#     """Controller for handling resume upload and processing."""
    
#     def __init__(
#         self,
#         resume_parser: ResumeParser,
#         embedding_service: EmbeddingService,
#         vector_db: VectorDBService,
#         resume_repo: ResumeRepository
#     ):
#         self.resume_parser = resume_parser
#         self.embedding_service = embedding_service
#         self.vector_db = vector_db
#         self.resume_repo = resume_repo
    
#     async def upload_resume(
#         self,
#         file: UploadFile,
#         metadata: Optional[ResumeUpload] = None
#     ) -> ResumeUploadResponse:
#         """Handle resume upload, parsing, and embedding."""
#         try:
#             # Validate file type
#             allowed_extensions = {'.pdf', '.docx', '.doc', '.txt'}
#             file_ext = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
#             if file_ext not in allowed_extensions:
#                 raise HTTPException(
#                     status_code=400,
#                     detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
#                 )
            
#             # Read file content
#             file_content = await file.read()
#             if not file_content:
#                 raise HTTPException(status_code=400, detail="Empty file")
            
#             # Sanitize filename
#             safe_filename = sanitize_filename(file.filename or "resume.pdf")
            
#             # Extract text from file
#             resume_text = await self.resume_parser.extract_text(file_content, safe_filename)
            
#             if not resume_text or len(resume_text.strip()) < 50:
#                 raise HTTPException(status_code=400, detail="Could not extract sufficient text from resume")
            
#             # Parse resume using LLM
#             parsed_data = await self.resume_parser.parse_resume(resume_text, safe_filename)
            
#             # Merge metadata if provided
#             if metadata:
#                 if metadata.candidate_name:
#                     parsed_data["candidateName"] = metadata.candidate_name
#                 if metadata.job_role:
#                     parsed_data["jobrole"] = metadata.job_role
            
#             # Prepare database record
#             db_record = {
#                 "candidatename": parsed_data.get("candidateName"),
#                 "jobrole": parsed_data.get("jobrole"),
#                 "experience": parsed_data.get("experience"),
#                 "domain": parsed_data.get("domain"),
#                 "mobile": parsed_data.get("mobile"),
#                 "email": parsed_data.get("email"),
#                 "education": parsed_data.get("education"),
#                 "filename": safe_filename,
#                 "skillset": ", ".join(parsed_data.get("skillset", [])) if isinstance(parsed_data.get("skillset"), list) else parsed_data.get("skillset", ""),
#             }
            
#             # Save to database
#             resume_metadata = await self.resume_repo.create(db_record)
            
#             # Generate embeddings for resume text
#             chunk_embeddings = await self.embedding_service.generate_chunk_embeddings(
#                 resume_text,
#                 metadata={
#                     "resume_id": resume_metadata.id,
#                     "filename": safe_filename,
#                     "candidate_name": parsed_data.get("candidateName"),
#                 }
#             )
            
#             # Store embeddings in vector DB
#             vectors_to_store = []
#             for chunk_data in chunk_embeddings:
#                 vector_id = f"resume_{resume_metadata.id}_chunk_{chunk_data['chunk_index']}"
#                 vectors_to_store.append({
#                     "id": vector_id,
#                     "embedding": chunk_data["embedding"],
#                     "metadata": {
#                         **chunk_data["metadata"],
#                         "type": "resume",  # Mark as resume vector
#                         "chunk_index": chunk_data["chunk_index"],
#                         "text_preview": chunk_data["text"][:200],
#                     }
#                 })
            
#             if vectors_to_store:
#                 await self.vector_db.upsert_vectors(vectors_to_store)
#                 logger.info(
#                     f"Stored {len(vectors_to_store)} embeddings for resume {resume_metadata.id}",
#                     extra={"resume_id": resume_metadata.id, "vector_count": len(vectors_to_store)}
#                 )
            
#             # Build response
#             return ResumeUploadResponse(
#                 id=resume_metadata.id,
#                 candidateName=resume_metadata.candidatename or "",
#                 jobrole=resume_metadata.jobrole or "",
#                 experience=resume_metadata.experience or "",
#                 domain=resume_metadata.domain or "",
#                 mobile=resume_metadata.mobile or "",
#                 email=resume_metadata.email or "",
#                 education=resume_metadata.education or "",
#                 filename=resume_metadata.filename,
#                 skillset=resume_metadata.skillset or "",
#                 created_at=resume_metadata.created_at.isoformat() if resume_metadata.created_at else "",
#             )
        
#         except HTTPException:
#             raise
#         except Exception as e:
#             logger.error(
#                 f"Error uploading resume: {e}",
#                 extra={
#                     "error": str(e),
#                     "error_type": type(e).__name__,
#                     "file_name": file.filename,
#                     "file_size": len(file_content) if 'file_content' in locals() else None,
#                 },
#                 exc_info=True
#             )
#             raise HTTPException(status_code=500, detail=f"Failed to process resume: {str(e)}")





"""Controller for resume-related operations."""
from typing import Optional
from fastapi import UploadFile, HTTPException

from app.services.resume_parser import ResumeParser
from app.services.embedding_service import EmbeddingService
from app.services.vector_db_service import VectorDBService
from app.repositories.resume_repo import ResumeRepository
from app.models.resume_models import ResumeUpload, ResumeUploadResponse
from app.utils.cleaning import sanitize_filename
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ResumeController:
    """Controller for handling resume upload and processing."""

    def __init__(
        self,
        resume_parser: ResumeParser,
        embedding_service: EmbeddingService,
        vector_db: VectorDBService,
        resume_repo: ResumeRepository
    ):
        self.resume_parser = resume_parser
        self.embedding_service = embedding_service
        self.vector_db = vector_db
        self.resume_repo = resume_repo

    async def upload_resume(
        self,
        file: UploadFile,
        metadata: Optional[ResumeUpload] = None
    ) -> ResumeUploadResponse:
        """Handle resume upload, parsing, and embedding."""
        try:
            # Validate file type
            allowed_extensions = {'.pdf', '.docx', '.doc', '.txt'}
            file_ext = (
                '.' + file.filename.split('.')[-1].lower()
                if '.' in file.filename else ''
            )

            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
                )

            # Read file content
            file_content = await file.read()
            if not file_content:
                raise HTTPException(status_code=400, detail="Empty file")

            # Sanitize filename
            safe_filename = sanitize_filename(file.filename or "resume.pdf")

            # Extract text from file
            resume_text = await self.resume_parser.extract_text(
                file_content,
                safe_filename
            )

            if not resume_text or len(resume_text.strip()) < 50:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract sufficient text from resume"
                )

            # Parse resume using LLM
            parsed_data = await self.resume_parser.parse_resume(
                resume_text,
                safe_filename
            )

            # Override with user metadata if provided
            if metadata:
                if metadata.candidate_name:
                    parsed_data["candidateName"] = metadata.candidate_name
                if metadata.job_role:
                    parsed_data["jobrole"] = metadata.job_role

            # Prepare DB record
            db_record = {
                "candidatename": parsed_data.get("candidateName"),
                "jobrole": parsed_data.get("jobrole"),
                "experience": parsed_data.get("experience"),
                "domain": parsed_data.get("domain"),
                "mobile": parsed_data.get("mobile"),
                "email": parsed_data.get("email"),
                "education": parsed_data.get("education"),
                "filename": safe_filename,
                "skillset": (
                    ", ".join(parsed_data.get("skillset", []))
                    if isinstance(parsed_data.get("skillset"), list)
                    else parsed_data.get("skillset", "")
                ),
            }

            # Save in database
            resume_metadata = await self.resume_repo.create(db_record)

            # Generate embeddings
            chunk_embeddings = await self.embedding_service.generate_chunk_embeddings(
                resume_text,
                metadata={
                    "resume_id": resume_metadata.id,
                    "filename": safe_filename,
                    "candidate_name": parsed_data.get("candidateName"),
                }
            )

            # Prepare vectors for vector DB
            vectors_to_store = []
            for chunk_data in chunk_embeddings:
                vector_id = (
                    f"resume_{resume_metadata.id}_chunk_{chunk_data['chunk_index']}"
                )

                vectors_to_store.append({
                    "id": vector_id,
                    "embedding": chunk_data["embedding"],
                    "metadata": {
                        **chunk_data["metadata"],
                        "type": "resume",
                        "chunk_index": chunk_data["chunk_index"],
                        "text_preview": chunk_data["text"][:200],
                    }
                })

            # Store vectors
            if vectors_to_store:
                await self.vector_db.upsert_vectors(vectors_to_store)
                logger.info(
                    f"Stored {len(vectors_to_store)} embeddings for resume {resume_metadata.id}",
                    extra={
                        "resume_id": resume_metadata.id,
                        "vector_count": len(vectors_to_store),
                    }
                )

            # Build API Response
            return ResumeUploadResponse(
                id=resume_metadata.id,
                candidateName=resume_metadata.candidatename or "",
                jobrole=resume_metadata.jobrole or "",
                experience=resume_metadata.experience or "",
                domain=resume_metadata.domain or "",
                mobile=resume_metadata.mobile or "",
                email=resume_metadata.email or "",
                education=resume_metadata.education or "",
                filename=resume_metadata.filename,
                skillset=resume_metadata.skillset or "",
                created_at=(
                    resume_metadata.created_at.isoformat()
                    if resume_metadata.created_at else ""
                ),
            )

        except HTTPException:
            raise

        except Exception as e:
            logger.error(
                f"Error uploading resume: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "file_name": getattr(file, "filename", None),
                    "file_size": len(file_content) if 'file_content' in locals() else None,
                },
                exc_info=True
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process resume: {str(e)}"
            )
