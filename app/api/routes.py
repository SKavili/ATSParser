"""API route definitions."""
from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.controllers.resume_controller import ResumeController
from app.controllers.job_controller import JobController
from app.services.resume_parser import ResumeParser
from app.services.job_parser import JobParser
from app.services.embedding_service import EmbeddingService
from app.services.vector_db_service import get_vector_db_service, VectorDBService
from app.repositories.resume_repo import ResumeRepository
from app.database.connection import get_db_session
from app.models.resume_models import ResumeUpload, ResumeUploadResponse
from app.models.job_models import JobCreate, JobCreateResponse, MatchRequest, MatchResponse
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


@router.post("/upload-resume", response_model=ResumeUploadResponse, status_code=200)
async def upload_resume(
    file: UploadFile = File(...),
    candidate_name: Optional[str] = Form(None),
    job_role: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    controller: ResumeController = Depends(get_resume_controller)
):
    """
    Upload and parse a resume file.
    
    Accepts multipart/form-data with:
    - file: Resume file (PDF, DOCX, or TXT)
    - candidate_name: Optional candidate name
    - job_role: Optional job role
    - source: Optional source identifier
    """
    metadata = ResumeUpload(
        candidateName=candidate_name,
        jobRole=job_role,
        source=source
    )
    return await controller.upload_resume(file, metadata)


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


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ATS Backend"}

