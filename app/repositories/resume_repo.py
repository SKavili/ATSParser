"""Repository for resume metadata operations."""
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.database.models import ResumeMetadata
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ResumeRepository:
    """Repository for resume metadata CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, resume_data: dict) -> ResumeMetadata:
        """Create a new resume record."""
        try:
            resume = ResumeMetadata(**resume_data)
            self.session.add(resume)
            await self.session.commit()
            await self.session.refresh(resume)
            logger.info(
                f"Created resume record: id={resume.id}",
                extra={"resume_id": resume.id, "file_name": resume.filename}
            )
            return resume
        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"Failed to create resume: {e}", extra={"error": str(e)})
            raise
    
    async def get_by_id(self, resume_id: int) -> Optional[ResumeMetadata]:
        """Get resume by ID."""
        result = await self.session.execute(
            select(ResumeMetadata).where(ResumeMetadata.id == resume_id)
        )
        return result.scalar_one_or_none()
    
    async def get_all(self, limit: Optional[int] = None, offset: int = 0) -> List[ResumeMetadata]:
        """Get all resumes with optional pagination."""
        query = select(ResumeMetadata).offset(offset)
        if limit:
            query = query.limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def update(self, resume_id: int, update_data: dict) -> Optional[ResumeMetadata]:
        """Update resume record."""
        resume = await self.get_by_id(resume_id)
        if not resume:
            return None
        
        for key, value in update_data.items():
            if hasattr(resume, key):
                setattr(resume, key, value)
        
        await self.session.commit()
        await self.session.refresh(resume)
        logger.info(f"Updated resume record: id={resume_id}")
        return resume

