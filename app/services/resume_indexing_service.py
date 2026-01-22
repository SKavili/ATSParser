"""Service for indexing resumes to Pinecone with embeddings."""
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.embedding_service import EmbeddingService
from app.services.pinecone_automation import PineconeAutomation
from app.repositories.resume_repo import ResumeRepository
from app.database.models import ResumeMetadata
from app.utils.logging import get_logger
from app.utils.safe_logger import safe_extra
from app.utils.cleaning import normalize_skill_list

logger = get_logger(__name__)


class ResumeIndexingService:
    """Service for indexing resumes to Pinecone with embeddings."""
    
    def __init__(self, session: AsyncSession):
        """
        Initialize ResumeIndexingService.
        
        Args:
            session: Database session
        """
        self.session = session
        self.embedding_service = EmbeddingService()
        self.pinecone_automation = PineconeAutomation()
        self.resume_repo = ResumeRepository(session)
    
    async def initialize_pinecone(self) -> None:
        """Initialize Pinecone client and indexes."""
        try:
            await self.pinecone_automation.initialize_pinecone()
            await self.pinecone_automation.create_indexes()
            logger.info("Pinecone initialized successfully for resume indexing")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}", extra={"error": str(e)})
            raise
    
    async def index_resumes(
        self,
        limit: Optional[int] = None,
        resume_ids: Optional[List[int]] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Index pending resumes to Pinecone.
        
        Args:
            limit: Optional limit on number of resumes to process
            resume_ids: Optional list of specific resume IDs to process
            force: If True, re-index resumes even if already indexed
        
        Returns:
            Dictionary with indexing results:
            {
                "indexed_count": int,
                "failed_count": int,
                "processed_ids": List[int],
                "failed_ids": List[int],
                "skipped_ids": List[int]
            }
        """
        try:
            # Initialize Pinecone if not already done
            if not self.pinecone_automation.pc:
                await self.initialize_pinecone()
            
            # Get pending resumes
            pending_resumes = await self.resume_repo.get_pending_pinecone_resumes(
                limit=limit,
                resume_ids=resume_ids,
                force=force
            )
            
            if not pending_resumes:
                logger.info("No pending resumes to index")
                return {
                    "indexed_count": 0,
                    "failed_count": 0,
                    "processed_ids": [],
                    "failed_ids": [],
                    "skipped_ids": [],
                    "message": "No pending resumes to index"
                }
            
            logger.info(
                f"Starting Pinecone indexing for {len(pending_resumes)} resumes",
                extra={"resume_count": len(pending_resumes)}
            )
            
            indexed_count = 0
            failed_count = 0
            processed_ids = []
            failed_ids = []
            skipped_ids = []
            
            # Process each resume
            for resume in pending_resumes:
                try:
                    # Validate required fields
                    if not resume.resume_text:
                        logger.warning(
                            f"Skipping resume {resume.id}: missing resume_text",
                            extra={"resume_id": resume.id}
                        )
                        skipped_ids.append(resume.id)
                        continue
                    
                    if not resume.mastercategory:
                        logger.warning(
                            f"Skipping resume {resume.id}: missing mastercategory",
                            extra={"resume_id": resume.id}
                        )
                        skipped_ids.append(resume.id)
                        continue
                    
                    # Index the resume
                    success = await self._index_single_resume(resume)
                    
                    if success:
                        indexed_count += 1
                        processed_ids.append(resume.id)
                        logger.info(
                            f"Successfully indexed resume {resume.id} to Pinecone",
                            extra={"resume_id": resume.id}
                        )
                    else:
                        failed_count += 1
                        failed_ids.append(resume.id)
                        logger.error(
                            f"Failed to index resume {resume.id} to Pinecone",
                            extra={"resume_id": resume.id}
                        )
                
                except Exception as e:
                    failed_count += 1
                    failed_ids.append(resume.id)
                    logger.error(
                        f"Error indexing resume {resume.id}: {e}",
                        extra={"resume_id": resume.id, "error": str(e)},
                        exc_info=True
                    )
                    # Continue with next resume even if this one failed
            
            result = {
                "indexed_count": indexed_count,
                "failed_count": failed_count,
                "processed_ids": processed_ids,
                "failed_ids": failed_ids,
                "skipped_ids": skipped_ids,
                "message": f"Indexed {indexed_count} resumes into Pinecone. Failed: {failed_count}. Skipped: {len(skipped_ids)}"
            }
            
            # Use safe_extra to prevent LogRecord conflicts (message is reserved)
            logger.info(
                f"Pinecone indexing completed: {indexed_count} indexed, {failed_count} failed, {len(skipped_ids)} skipped",
                extra=safe_extra({
                    "indexed_count": indexed_count,
                    "failed_count": failed_count,
                    "processed_ids": processed_ids,
                    "failed_ids": failed_ids,
                    "skipped_ids": skipped_ids
                })
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error in index_resumes: {e}", extra={"error": str(e)}, exc_info=True)
            raise
    
    async def _index_single_resume(self, resume: ResumeMetadata) -> bool:
        """
        Index a single resume to Pinecone.
        
        Args:
            resume: ResumeMetadata object to index
        
        Returns:
            True if indexing was successful, False otherwise
        """
        try:
            # Generate chunked embeddings for resume text
            logger.info(
                f"Generating embeddings for resume {resume.id}",
                extra={"resume_id": resume.id, "text_length": len(resume.resume_text)}
            )
            
            # Parse skillset string to array for filtering
            # Normalize skills to canonical forms (e.g., "react.js" → "react", "angularjs" → "angular")
            skills_array = []
            if resume.skillset:
                raw_skills = [s.strip() for s in resume.skillset.split(",") if s.strip()]
                skills_array = normalize_skill_list(raw_skills)
            
            # Extract experience_years from experience string
            experience_years = None
            if resume.experience:
                import re
                match = re.search(r'(\d+(?:\.\d+)?)', resume.experience)
                if match:
                    experience_years = int(float(match.group(1)))
            
            # Prepare base metadata with all resume fields
            # Normalize designation and jobrole to lowercase for case-insensitive filtering
            normalized_designation = (resume.designation or "").lower().strip()
            normalized_jobrole = (resume.jobrole or "").lower().strip()
            
            base_metadata = {
                "resume_id": resume.id,
                "candidate_id": f"C{resume.id}",  # Generate candidate_id
                "filename": resume.filename or "unknown",
                "candidate_name": resume.candidatename or "",
                "name": resume.candidatename or "",  # Alias for compatibility
                "jobrole": normalized_jobrole,  # Lowercase for case-insensitive filtering
                "designation": normalized_designation,  # Lowercase for case-insensitive filtering
                "experience": resume.experience or "",
                "experience_years": experience_years,  # Numeric for filtering
                "domain": resume.domain or "",
                "mobile": resume.mobile or "",
                "email": resume.email or "",
                "education": resume.education or "",
                "skillset": resume.skillset or "",  # Keep original string
                "skills": skills_array,  # Array for Pinecone filtering
            }
            
            chunk_embeddings = await self.embedding_service.generate_chunk_embeddings(
                resume.resume_text,
                metadata=base_metadata
            )
            
            if not chunk_embeddings:
                logger.warning(
                    f"No embeddings generated for resume {resume.id}",
                    extra={"resume_id": resume.id}
                )
                return False
            
            # Format vectors for Pinecone
            vectors_to_store = []
            for chunk_data in chunk_embeddings:
                vector_id = f"resume_{resume.id}_chunk_{chunk_data['chunk_index']}"
                
                # Get full chunk text (not just preview)
                chunk_text = chunk_data["text"]
                
                # Include full resume_text in metadata (truncate if too large to avoid Pinecone limits)
                # Pinecone metadata limit is ~40KB, so we'll limit resume_text to 30KB to be safe
                full_resume_text = resume.resume_text or ""
                if len(full_resume_text) > 30000:
                    full_resume_text = full_resume_text[:30000] + "...[truncated]"
                
                vectors_to_store.append({
                    "id": vector_id,
                    "embedding": chunk_data["embedding"],
                    "metadata": {
                        **chunk_data["metadata"],  # Includes all base_metadata fields
                        "type": "resume",  # Mark as resume vector
                        "chunk_index": chunk_data["chunk_index"],
                        "chunk_text": chunk_text,  # Full chunk text (not just preview)
                        "resume_text": full_resume_text,  # Full resume text (truncated if too large)
                    }
                })
            
            # Store embeddings in Pinecone using PineconeAutomation
            # This handles routing to correct index (IT/Non-IT) and namespace (category)
            await self.pinecone_automation.insert_vectors(
                vectors=vectors_to_store,
                resume_text=resume.resume_text,
                mastercategory=resume.mastercategory,
                filename=resume.filename or "unknown",
                category=resume.category  # Use category from database if available
            )
            
            # Update pinecone_status to 1 (indexed) only after successful storage
            success = await self.resume_repo.update_pinecone_status(resume.id, 1)
            
            if success:
                logger.info(
                    f"Successfully indexed and updated status for resume {resume.id}",
                    extra={
                        "resume_id": resume.id,
                        "vector_count": len(vectors_to_store),
                        "mastercategory": resume.mastercategory,
                        "category": resume.category
                    }
                )
                return True
            else:
                logger.warning(
                    f"Indexed resume {resume.id} to Pinecone but failed to update status",
                    extra={"resume_id": resume.id}
                )
                # Still return True since Pinecone storage succeeded
                return True
        
        except Exception as e:
            logger.error(
                f"Error indexing resume {resume.id}: {e}",
                extra={"resume_id": resume.id, "error": str(e)},
                exc_info=True
            )
            # Don't update status on error - leave it as 0 so it can be retried
            return False

