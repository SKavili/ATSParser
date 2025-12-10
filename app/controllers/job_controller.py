"""Controller for job-related operations."""
from typing import Optional, Dict, List
from fastapi import HTTPException

from app.services.job_parser import JobParser
from app.services.embedding_service import EmbeddingService
from app.services.vector_db_service import VectorDBService
from app.services.job_cache import job_cache
from app.models.job_models import JobCreate, JobCreateResponse, MatchRequest, MatchResult, MatchResponse
from app.repositories.resume_repo import ResumeRepository
from app.utils.logging import get_logger
from app.config import settings
import uuid

logger = get_logger(__name__)


class JobController:
    """Controller for handling job creation and matching."""
    
    def __init__(
        self,
        job_parser: JobParser,
        embedding_service: EmbeddingService,
        vector_db: VectorDBService,
        resume_repo: ResumeRepository
    ):
        self.job_parser = job_parser
        self.embedding_service = embedding_service
        self.vector_db = vector_db
        self.resume_repo = resume_repo
    
    async def create_job(self, job_data: JobCreate) -> JobCreateResponse:
        """Create a job posting and generate embeddings."""
        try:
            # Parse job description using LLM
            parsed_job = await self.job_parser.parse_job(
                title=job_data.title,
                description=job_data.description,
                job_id=job_data.job_id
            )
            
            # Use provided job_id or generated one
            job_id = parsed_job.get("job_id") or job_data.job_id or f"job_{uuid.uuid4().hex[:12]}"
            
            # Generate embedding for job summary
            summary_text = parsed_job.get("summary_for_embedding", f"{job_data.title}. {job_data.description}")
            embedding = await self.embedding_service.generate_embedding(summary_text)
            
            # Store job embedding in vector DB
            vector_id = f"job_{job_id}"
            job_metadata = {
                "type": "job",
                "job_id": job_id,
                "title": parsed_job.get("title"),
                "location": parsed_job.get("location"),
                "summary": summary_text,
                "full_embedding": True,  # Mark as full job embedding (not chunked)
            }
            await self.vector_db.upsert_vectors([{
                "id": vector_id,
                "embedding": embedding,
                "metadata": job_metadata
            }])
            
            # Cache job embedding for quick retrieval
            job_cache.store_job(job_id, embedding, job_metadata)
            
            logger.info(
                f"Created job: {job_id}",
                extra={"job_id": job_id, "title": parsed_job.get("title")}
            )
            
            return JobCreateResponse(
                job_id=job_id,
                title=parsed_job.get("title", job_data.title),
                embedding_id=vector_id,
                message="Job created successfully"
            )
        
        except Exception as e:
            logger.error(f"Error creating job: {e}", extra={"error": str(e)})
            raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")
    
    async def match_job(self, match_request: MatchRequest) -> MatchResponse:
        """Match resumes to a job description."""
        try:
            # Determine job embedding
            job_embedding = None
            job_id = match_request.job_id
            
            if job_id:
                # Retrieve existing job embedding from cache or regenerate
                logger.info(f"Matching against job_id: {job_id}")
                
                # Try to get from cache first
                cached_job = job_cache.get_job(job_id)
                if cached_job:
                    job_embedding = cached_job["embedding"]
                    logger.info(f"Retrieved job embedding from cache: {job_id}")
                else:
                    # Cache miss - try to get from vector DB metadata and regenerate
                    # This is a fallback, ideally jobs should be in cache
                    logger.warning(f"Job {job_id} not in cache, attempting to retrieve from vector DB")
                    
                    # Try to query vector DB for job metadata
                    # Use a dummy query to filter by metadata
                    dummy_vector = [0.0] * settings.embedding_dimension
                    job_results = await self.vector_db.query_vectors(
                        query_vector=dummy_vector,
                        top_k=100,  # Get enough results to find our job
                        filter_dict=None  # FAISS doesn't support filters well, we'll filter in code
                    )
                    
                    # Find our job in results
                    job_metadata = None
                    for result in job_results:
                        meta = result.get("metadata", {})
                        if meta.get("job_id") == job_id and meta.get("type") == "job":
                            job_metadata = meta
                            break
                    
                    if job_metadata and job_metadata.get("summary"):
                        # Regenerate embedding from stored summary
                        job_embedding = await self.embedding_service.generate_embedding(
                            job_metadata["summary"]
                        )
                        # Store in cache for future use
                        job_cache.store_job(job_id, job_embedding, job_metadata)
                    else:
                        raise HTTPException(
                            status_code=404,
                            detail=f"Job with ID {job_id} not found. Please create the job first via /create-job"
                        )
            
            elif match_request.job_description:
                # Generate embedding for provided job description
                job_embedding = await self.embedding_service.generate_embedding(match_request.job_description)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Either job_id or job_description must be provided"
                )
            
            # Determine top_k
            top_k = match_request.top_k or settings.top_k_results
            
            # Query similar resumes from vector DB
            # Get more results to filter by threshold
            query_results = await self.vector_db.query_vectors(
                query_vector=job_embedding,
                top_k=top_k * 3,  # Get more results to filter
                filter_dict=None  # We'll filter resume vectors by checking metadata
            )
            
            # Apply similarity threshold and filter for resume vectors only
            threshold = settings.similarity_threshold
            filtered_results = [
                r for r in query_results
                if (r["score"] >= threshold 
                    and r["metadata"].get("resume_id") 
                    and r["metadata"].get("type") != "job")  # Exclude job vectors
            ]
            
            # Limit to top_k
            filtered_results = filtered_results[:top_k]
            
            # Fetch resume metadata from database
            matches = []
            for result in filtered_results:
                resume_id = result["metadata"].get("resume_id")
                if not resume_id:
                    continue
                
                resume_metadata = await self.resume_repo.get_by_id(resume_id)
                if not resume_metadata:
                    continue
                
                # Get candidate summary from metadata or generate from resume data
                candidate_summary = result["metadata"].get("summary") or \
                    f"{resume_metadata.candidatename or 'Candidate'} with {resume_metadata.experience or 'N/A'} experience in {resume_metadata.domain or 'various domains'}."
                
                matches.append(MatchResult(
                    resume_id=resume_metadata.id,
                    candidate_name=resume_metadata.candidatename or "Unknown",
                    similarity_score=result["score"],
                    candidate_summary=candidate_summary,
                    filename=resume_metadata.filename
                ))
            
            logger.info(
                f"Found {len(matches)} matches",
                extra={"match_count": len(matches), "job_id": job_id}
            )
            
            return MatchResponse(
                matches=matches,
                total_results=len(matches),
                job_id=job_id
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error matching job: {e}", extra={"error": str(e)})
            raise HTTPException(status_code=500, detail=f"Failed to match job: {str(e)}")

