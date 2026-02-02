"""API route definitions."""
from fastapi import APIRouter, Depends, UploadFile, File, Form, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List

from app.controllers.resume_controller import ResumeController
from app.controllers.job_controller import JobController
from app.services.resume_parser import ResumeParser
from app.services.job_parser import JobParser
from app.services.embedding_service import EmbeddingService
from app.services.vector_db_service import get_vector_db_service, VectorDBService
from app.repositories.resume_repo import ResumeRepository
from app.repositories.prompt_repo import PromptRepository
from app.database.connection import get_db_session
from app.models.resume_models import ResumeUpload, ResumeUploadResponse
from app.models.job_models import JobCreate, JobCreateResponse, MatchRequest, MatchResponse
from app.models.ai_search_models import AISearchRequest, AISearchResponse
from app.services.resume_indexing_service import ResumeIndexingService
from app.skills.skills_service import SkillsService
from app.ai_search.ai_search_controller import AISearchController
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


# Dependency factories
async def get_resume_controller(
    session: AsyncSession = Depends(get_db_session),
    vector_db: VectorDBService = Depends(get_vector_db_service)
) -> ResumeController:
    """Create ResumeController with dependencies."""
    resume_parser = ResumeParser()
    embedding_service = EmbeddingService()
    resume_repo = ResumeRepository(session)
    return ResumeController(resume_parser, embedding_service, vector_db, resume_repo, session)


async def get_job_controller(
    session: AsyncSession = Depends(get_db_session),
    vector_db: VectorDBService = Depends(get_vector_db_service)
) -> JobController:
    """Create JobController with dependencies."""
    job_parser = JobParser()
    embedding_service = EmbeddingService()
    resume_repo = ResumeRepository(session)
    return JobController(job_parser, embedding_service, vector_db, resume_repo)


async def get_ai_search_controller(
    session: AsyncSession = Depends(get_db_session)
) -> AISearchController:
    """Create AISearchController with dependencies."""
    from app.services.pinecone_automation import PineconeAutomation
    
    embedding_service = EmbeddingService()
    pinecone_automation = PineconeAutomation()
    resume_repo = ResumeRepository(session)
    return AISearchController(session, embedding_service, pinecone_automation, resume_repo)


@router.post("/upload-resume", response_model=ResumeUploadResponse, status_code=200)
async def upload_resume(
    file: UploadFile = File(...),
    candidate_name: Optional[str] = Form(None),
    job_role: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    extract_modules: Optional[str] = Form("all"),
    controller: ResumeController = Depends(get_resume_controller)
):
    """
    Upload and parse a resume file.
    
    Accepts multipart/form-data with:
    - file: Resume file (PDF, DOCX, or TXT)
    - candidate_name: Optional candidate name
    - job_role: Optional job role
    - source: Optional source identifier
    - extract_modules: Modules to extract (default: "all")
        Options:
        - "all" - Extract all modules (default)
        - "1" or "designation" - Extract designation only
        - "2" or "name" - Extract name only
        - "3" or "email" - Extract email only
        - "4" or "mobile" - Extract mobile only
        - "5" or "experience" - Extract experience only
        - "6" or "domain" - Extract domain only
        - "7" or "education" - Extract education only
        - "8" or "skills" - Extract skills only
        - Comma-separated: "1,2,3" or "designation,name,email" - Extract multiple modules
    
    Examples:
    - extract_modules="all" - Extract all 8 modules
    - extract_modules="1" - Extract only designation
    - extract_modules="designation" - Extract only designation
    - extract_modules="1,2,3" - Extract designation, name, and email
    - extract_modules="designation,skills" - Extract designation and skills
    """
    metadata = ResumeUpload(
        candidateName=candidate_name,
        jobRole=job_role,
        source=source
    )
    return await controller.upload_resume(file, metadata, extract_modules)


@router.post("/create-job", response_model=JobCreateResponse, status_code=200)
async def create_job(
    job_data: JobCreate,
    controller: JobController = Depends(get_job_controller)
):
    """
    Create a job posting and generate embeddings.
    
    Request body:
    - title: Job title
    - description: Job description
    - required_skills: List of required skills
    - location: Job location (optional)
    - job_id: Custom job ID (optional)
    """
    return await controller.create_job(job_data)


@router.post("/match", response_model=MatchResponse, status_code=200)
async def match_job(
    match_request: MatchRequest,
    controller: JobController = Depends(get_job_controller)
):
    """
    Match resumes to a job description.
    
    Request body:
    - job_id: Job ID to match against (optional if job_description provided)
    - job_description: Job description text (optional if job_id provided)
    - top_k: Number of results to return (optional, defaults to configured value)
    """
    return await controller.match_job(match_request)


@router.post("/retry-failed-resume/{resume_id}", response_model=ResumeUploadResponse, status_code=200)
async def retry_failed_resume(
    resume_id: int,
    extract_modules: Optional[str] = Form("all"),
    controller: ResumeController = Depends(get_resume_controller)
):
    """
    Retry processing a resume that failed with insufficient_text status.
    Finds the file from disk, retries extraction with OCR, and re-runs extraction modules.
    
    Args:
        resume_id: The ID of the failed resume to retry
        extract_modules: Modules to extract (default: "all")
            Options:
            - "all" - Extract all modules (default)
            - "1" or "designation" - Extract designation only
            - "2" or "name" - Extract name only
            - "3" or "email" - Extract email only
            - "4" or "mobile" - Extract mobile only
            - "5" or "experience" - Extract experience only
            - "6" or "domain" - Extract domain only
            - "7" or "education" - Extract education only
            - "8" or "skills" - Extract skills only
            - Comma-separated: "1,2,3" or "designation,name,email" - Extract multiple modules
    """
    return await controller.retry_failed_resume_with_ocr(resume_id, extract_modules)


@router.post("/index-pinecone")
async def index_pinecone(
    limit: Optional[int] = Query(None, description="Maximum number of resumes to process"),
    resume_ids: Optional[List[int]] = Query(None, description="Specific resume IDs to process"),
    force: bool = Query(False, description="Force re-indexing even if already indexed"),
    session: AsyncSession = Depends(get_db_session)
):
    """
    Index resumes to Pinecone with embeddings.
    
    This endpoint:
    - Queries resumes where pinecone_status = 0 (not indexed)
    - Generates chunked embeddings for each resume
    - Stores embeddings in Pinecone (routed to IT/Non-IT index based on mastercategory)
    - Updates pinecone_status = 1 only after successful storage
    - Normalizes skills to canonical forms (e.g., "react.js" → "react", "angularjs" → "angular")
    
    Query Parameters:
    - limit: Optional limit on number of resumes to process (default: all pending)
    - resume_ids: Optional list of specific resume IDs to process
    - force: If True, re-index resumes even if pinecone_status = 1 (default: False)
    
    Returns:
    {
        "indexed_count": 5,
        "failed_count": 0,
        "processed_ids": [1, 2, 3, 4, 5],
        "failed_ids": [],
        "skipped_ids": [],
        "message": "Indexed 5 resumes into Pinecone. Failed: 0. Skipped: 0"
    }
    """
    try:
        indexing_service = ResumeIndexingService(session)
        result = await indexing_service.index_resumes(
            limit=limit,
            resume_ids=resume_ids,
            force=force
        )
        return result
    except Exception as e:
        logger.error(f"Error in index-pinecone endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to index resumes: {str(e)}")


@router.post("/reindex-resumes")
async def reindex_resumes(
    limit: Optional[int] = Query(None, description="Maximum number of resumes to re-index"),
    resume_ids: Optional[List[int]] = Query(None, description="Specific resume IDs to re-index"),
    session: AsyncSession = Depends(get_db_session)
):
    """
    Re-index resumes to Pinecone with skill normalization.
    
    This endpoint re-indexes resumes with the new skill normalization logic:
    - Normalizes skills to canonical forms (e.g., "react.js" → "react", "angularjs" → "angular")
    - Re-generates chunked embeddings for each resume
    - Updates Pinecone vectors with normalized skills
    - This is required after implementing skill normalization to update existing resumes
    
    Query Parameters:
    - limit: Optional limit on number of resumes to re-index (default: all resumes)
    - resume_ids: Optional list of specific resume IDs to re-index
    
    Examples:
    - Re-index all resumes: POST /reindex-resumes
    - Re-index first 100 resumes: POST /reindex-resumes?limit=100
    - Re-index specific resumes: POST /reindex-resumes?resume_ids=1&resume_ids=2&resume_ids=3
    
    Returns:
    {
        "indexed_count": 5,
        "failed_count": 0,
        "processed_ids": [1, 2, 3, 4, 5],
        "failed_ids": [],
        "skipped_ids": [],
        "message": "Re-indexed 5 resumes with skill normalization. Failed: 0. Skipped: 0"
    }
    """
    try:
        indexing_service = ResumeIndexingService(session)
        # Force re-indexing to update with normalized skills
        result = await indexing_service.index_resumes(
            limit=limit,
            resume_ids=resume_ids,
            force=True  # Always force re-indexing for this endpoint
        )
        
        # Update message to indicate re-indexing
        if result.get("message"):
            result["message"] = result["message"].replace("Indexed", "Re-indexed")
            result["message"] += " with skill normalization"
        
        logger.info(
            f"Re-indexing completed: {result.get('indexed_count', 0)} resumes processed",
            extra={
                "indexed_count": result.get("indexed_count", 0),
                "failed_count": result.get("failed_count", 0),
                "processed_ids": result.get("processed_ids", [])
            }
        )
        
        return result
    except Exception as e:
        logger.error(f"Error in reindex-resumes endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to re-index resumes: {str(e)}")


@router.post("/ai-search", response_model=AISearchResponse, status_code=200)
async def ai_search(
    request: AISearchRequest,
    controller: AISearchController = Depends(get_ai_search_controller)
):
    """
    AI-powered candidate search using natural language queries.
    
    This endpoint:
    - Parses natural language queries to extract skills, experience, location, etc.
    - Performs semantic search using Pinecone
    - Enforces mandatory requirements strictly
    - Ranks candidates by relevance
    - Returns results categorized into fit tiers
    
    Request body:
    - query: Natural language search query (e.g., "python developer with 5 years experience")
    - mastercategory: Mastercategory (IT/NON_IT) - optional, if provided searches only that category
    - category: Category namespace - optional, if provided searches only that namespace
    - user_id: Optional user ID for tracking
    - top_k: Number of results to return (default: 20)
    
    Note: If both mastercategory and category are provided, searches only that specific namespace.
          If not provided, performs broad search across relevant namespaces using smart filtering.
    
    Response:
    - query: Original query
    - mastercategory: Mastercategory used for search (None if broad search)
    - category: Category namespace used for search (None if broad search)
    - total_results: Number of results found
    - results: List of candidates with fit tiers
    """
    try:
        return await controller.search(
            query=request.query,
            mastercategory=request.mastercategory,
            category=request.category,
            user_id=request.user_id,
            top_k=request.top_k or 20
        )
    except RuntimeError as e:
        # Query parsing failed
        logger.error(f"AI search query parsing failed: {e}", extra={"error": str(e)})
        raise HTTPException(
            status_code=400,
            detail=f"Query parsing failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"AI search failed: {e}", extra={"error": str(e)}, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/health")
async def health_check(session: AsyncSession = Depends(get_db_session)):
    """
    Health check endpoint with prompt validation.
    
    Validates that required 'other' prompts exist in the database.
    """
    health_status = {
        "status": "healthy",
        "service": "ATS Backend",
        "checks": {
            "database": "ok",
            "prompts": "unknown"
        }
    }
    
    try:
        # Validate required prompts
        prompt_repo = PromptRepository(session)
        skills_service = SkillsService(session)
        
        is_valid, missing_prompts = await skills_service.validate_required_prompts()
        
        if is_valid:
            health_status["checks"]["prompts"] = "ok"
            health_status["prompt_validation"] = {
                "status": "valid",
                "message": "All required 'other' prompts exist in database"
            }
        else:
            health_status["status"] = "degraded"
            health_status["checks"]["prompts"] = "missing"
            health_status["prompt_validation"] = {
                "status": "invalid",
                "message": f"Missing required prompts: {', '.join(missing_prompts)}",
                "missing_prompts": missing_prompts,
                "action_required": "Please add the missing prompts to the prompts table"
            }
            logger.warning(
                f"Health check: Missing required prompts: {missing_prompts}",
                extra={"missing_prompts": missing_prompts}
            )
    
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["checks"]["prompts"] = "error"
        health_status["prompt_validation"] = {
            "status": "error",
            "message": f"Failed to validate prompts: {str(e)}"
        }
        logger.error(f"Health check: Failed to validate prompts: {e}", exc_info=True)
    
    return health_status

