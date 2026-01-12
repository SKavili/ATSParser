"""Repository for resume metadata operations using SQLAlchemy Core for async background tasks."""
import asyncio
from typing import Optional, List, Dict, Any, Union
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, Table
from sqlalchemy.exc import IntegrityError, OperationalError

from app.database.models import ResumeMetadata
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ResumeRepository:
    """
    Repository for resume metadata CRUD operations.
    
    Uses SQLAlchemy Core for UPDATE operations to avoid ORM state conflicts
    in async background tasks. This ensures:
    - No object expiration issues after commit
    - No lazy loading across greenlets
    - Predictable behavior in concurrent async contexts
    - Alembic compatibility (uses same table metadata)
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        # Get Core Table object from ORM model for Alembic compatibility
        # This ensures we use the exact same table definition as migrations
        self.table: Table = ResumeMetadata.__table__
    
    async def create(self, resume_data: dict) -> ResumeMetadata:
        """Create a new resume record."""
        try:
            resume = ResumeMetadata(**resume_data)
            self.session.add(resume)
            await self.session.flush()  # Flush to ensure data is in database before commit
            await self.session.commit()
            await self.session.refresh(resume)
            # Convert filename to string to avoid any LogRecord conflicts
            filename_str = str(resume.filename) if resume.filename else ""
            logger.info(
                f"Created resume record: id={resume.id}",
                extra={"resume_id": resume.id, "file_name": filename_str}
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
    
    async def get_by_filename(self, filename: str) -> Optional[ResumeMetadata]:
        """Get resume by filename (for duplicate detection)."""
        result = await self.session.execute(
            select(ResumeMetadata).where(ResumeMetadata.filename == filename)
        )
        return result.scalar_one_or_none()
    
    async def update(self, resume_id: int, update_data: dict, return_orm: bool = False) -> Union[Optional[Dict[str, Any]], Optional[ResumeMetadata]]:
        """
        Update resume record using SQLAlchemy Core UPDATE statement.
        
        Uses pure Core (not ORM) to avoid state conflicts in async background tasks:
        - No ORM object state tracking
        - No lazy loading issues
        - No object expiration after commit
        - Safe across greenlets/async contexts
        - Alembic compatible (uses same table metadata)
        
        This method is safe for concurrent updates because:
        1. Uses Core UPDATE statement (no ORM state tracking)
        2. Updates to different columns don't conflict
        3. Each update commits independently as a separate transaction
        4. Validates NOT NULL constraints before updating
        5. Returns plain dict by default (no ORM object state issues)
        
        Args:
            resume_id: The ID of the resume to update
            update_data: Dictionary of column names (matching model attributes) to values
            return_orm: If True, returns ORM object (use with caution in background tasks).
                      If False, returns dict (recommended for async tasks).
        
        Returns:
            Dict with updated data or ResumeMetadata ORM object (if return_orm=True), or None if not found
        
        Raises:
            ValueError: If NOT NULL constraint would be violated
            IntegrityError: If database constraint is violated
        """
        # Define NOT NULL columns and their validation rules
        NOT_NULL_COLUMNS = {
            'filename': {
                'nullable': False,
                'validator': lambda v: v is not None and isinstance(v, str) and len(v.strip()) > 0,
                'error_msg': 'filename cannot be None or empty string (NOT NULL constraint)'
            }
        }
        
        # Define valid updatable columns (excluding read-only: id, created_at, updated_at)
        valid_columns = {
            'mastercategory', 'category',
            'candidatename', 'jobrole', 'designation', 'experience', 'domain',
            'mobile', 'email', 'education', 'filename', 'skillset', 'status', 'resume_text', 'pinecone_status'
        }
        
        # Filter out invalid keys and read-only columns
        filtered_data = {
            key: value 
            for key, value in update_data.items() 
            if key in valid_columns
        }
        
        if not filtered_data:
            logger.warning(f"No valid columns to update for resume ID {resume_id}")
            if return_orm:
                return await self.get_by_id(resume_id)
            else:
                result = await self.get_by_id_dict(resume_id)
                return result
        
        # Validate NOT NULL constraints BEFORE attempting update
        # This prevents async SQLAlchemy from generating invalid UPDATE statements
        for column_name, constraint_info in NOT_NULL_COLUMNS.items():
            if column_name in filtered_data:
                value = filtered_data[column_name]
                if not constraint_info['validator'](value):
                    error_msg = f"{constraint_info['error_msg']} for resume ID {resume_id}"
                    logger.error(
                        error_msg,
                        extra={
                            "resume_id": resume_id,
                            "column": column_name,
                            "value": value,
                            "value_type": type(value).__name__
                        }
                    )
                    raise ValueError(error_msg)
        
        # Additional defensive validation: ensure None values are only set for nullable columns
        nullable_columns = {
            'candidatename', 'jobrole', 'designation', 'experience', 'domain',
            'mobile', 'email', 'education', 'skillset', 'status', 'resume_text', 'pinecone_status'
        }
        
        for key, value in filtered_data.items():
            if value is None and key not in nullable_columns:
                error_msg = f"Cannot set {key} to None (NOT NULL constraint) for resume ID {resume_id}"
                logger.error(
                    error_msg,
                    extra={"resume_id": resume_id, "column": key, "value": None}
                )
                raise ValueError(error_msg)
        
        # Simple deadlock retry strategy for MySQL error 1213
        max_retries = 3
        base_delay = 0.1  # seconds
        attempt = 0

        while True:
            try:
                # Use SQLAlchemy Core UPDATE with table object (not ORM model)
                # This ensures Alembic compatibility and avoids ORM state issues
                stmt = (
                    update(self.table)
                    .where(self.table.c.id == resume_id)
                    .values(**filtered_data)
                )

                result = await self.session.execute(stmt)
                await self.session.flush()  # Flush to ensure data is in database before commit
                await self.session.commit()

                # Check if any rows were updated
                if result.rowcount == 0:
                    logger.warning(
                        f"No rows updated for resume ID {resume_id} - record may not exist",
                        extra={"resume_id": resume_id},
                    )
                    return None

                logger.info(
                    f"Updated resume record: id={resume_id}, fields: {list(filtered_data.keys())}",
                    extra={"resume_id": resume_id, "updated_fields": list(filtered_data.keys())},
                )

                # Return appropriate format based on return_orm flag
                if return_orm:
                    # Return ORM object (use with caution in background tasks)
                    return await self.get_by_id(resume_id)
                else:
                    # Return plain dict (recommended for async background tasks)
                    # No ORM state issues, safe across greenlets
                    return await self.get_by_id_dict(resume_id)

            except OperationalError as e:
                # MySQL deadlock error code 1213 -> retry transaction
                orig = getattr(e, "orig", None)
                err_code = getattr(orig, "args", [None])[0] if orig and hasattr(orig, "args") else None

                if err_code == 1213 and attempt < max_retries:
                    await self.session.rollback()
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"MySQL deadlock detected on resume ID {resume_id}, retrying "
                        f"(attempt {attempt + 1}/{max_retries}) after {delay:.2f}s",
                        extra={
                            "resume_id": resume_id,
                            "error": str(e),
                            "update_data": filtered_data,
                            "error_type": "OperationalError",
                            "mysql_error_code": err_code,
                        },
                    )
                    attempt += 1
                    await asyncio.sleep(delay)
                    continue

                await self.session.rollback()
                logger.error(
                    f"Database operational error updating resume ID {resume_id}: {e}",
                    extra={
                        "resume_id": resume_id,
                        "error": str(e),
                        "update_data": filtered_data,
                        "error_type": "OperationalError",
                        "mysql_error_code": err_code,
                    },
                    exc_info=True,
                )
                raise

            except IntegrityError as e:
                await self.session.rollback()
                logger.error(
                    f"Database integrity error updating resume ID {resume_id}: {e}",
                    extra={
                        "resume_id": resume_id,
                        "error": str(e),
                        "update_data": filtered_data,
                        "error_type": "IntegrityError",
                    },
                    exc_info=True,
                )
                raise

            except Exception as e:
                await self.session.rollback()
                logger.error(
                    f"Failed to update resume ID {resume_id}: {e}",
                    extra={"resume_id": resume_id, "error": str(e), "update_data": filtered_data},
                    exc_info=True,
                )
                raise
    
    async def get_by_id_dict(self, resume_id: int) -> Optional[Dict[str, Any]]:
        """
        Get resume by ID using SQLAlchemy Core, returning plain dict.
        
        Safe for async background tasks - no ORM object state issues.
        
        Args:
            resume_id: The ID of the resume to fetch
        
        Returns:
            Dict with resume data or None if not found
        """
        stmt = select(self.table).where(self.table.c.id == resume_id)
        result = await self.session.execute(stmt)
        row = result.fetchone()
        
        if row is None:
            return None
        
        # Convert Row to dict - safe across greenlets, no ORM state
        return dict(row._mapping)
    
    async def get_resumes_with_null_email_or_mobile(self, limit: Optional[int] = None) -> List[ResumeMetadata]:
        """
        Get resumes that have NULL email or NULL mobile.
        Useful for reprocessing resumes that failed to extract contact info.
        
        Args:
            limit: Optional limit on number of results
        
        Returns:
            List of ResumeMetadata records with NULL email or mobile
        """
        query = select(ResumeMetadata).where(
            (ResumeMetadata.email.is_(None)) | (ResumeMetadata.mobile.is_(None))
        )
        if limit:
            query = query.limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_resumes_with_failed_status(self, failure_reason: str = "insufficient_text", limit: Optional[int] = None) -> List[ResumeMetadata]:
        """
        Get resumes that have failed status with specific failure reason.
        Useful for reprocessing resumes that failed due to insufficient text.
        
        Args:
            failure_reason: The failure reason to filter by (default: "insufficient_text")
            limit: Optional limit on number of results
        
        Returns:
            List of ResumeMetadata records with the specified failure status
        """
        failure_status = f"failed:{failure_reason}"
        query = select(ResumeMetadata).where(ResumeMetadata.status == failure_status)
        if limit:
            query = query.limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_pending_pinecone_resumes(
        self, 
        limit: Optional[int] = None,
        resume_ids: Optional[List[int]] = None,
        force: bool = False
    ) -> List[ResumeMetadata]:
        """
        Get resumes that need to be indexed in Pinecone.
        
        Args:
            limit: Optional limit on number of results
            resume_ids: Optional list of specific resume IDs to process
            force: If True, return resumes even if pinecone_status = 1 (for re-indexing)
        
        Returns:
            List of ResumeMetadata records that need Pinecone indexing
        """
        query = select(ResumeMetadata)
        
        # Filter by pinecone_status
        if force:
            # Force re-index: get all resumes (or specific IDs)
            if resume_ids:
                query = query.where(ResumeMetadata.id.in_(resume_ids))
        else:
            # Normal: only get resumes where pinecone_status = 0 (not indexed)
            if resume_ids:
                query = query.where(
                    (ResumeMetadata.pinecone_status == 0) | (ResumeMetadata.pinecone_status.is_(None)),
                    ResumeMetadata.id.in_(resume_ids)
                )
            else:
                query = query.where(
                    (ResumeMetadata.pinecone_status == 0) | (ResumeMetadata.pinecone_status.is_(None))
                )
        
        # Only get resumes with resume_text and mastercategory (required for indexing)
        query = query.where(
            ResumeMetadata.resume_text.isnot(None),
            ResumeMetadata.mastercategory.isnot(None)
        )
        
        if limit:
            query = query.limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def update_pinecone_status(self, resume_id: int, status: int) -> bool:
        """
        Update pinecone_status for a resume.
        
        Args:
            resume_id: The ID of the resume to update
            status: 0 = not indexed, 1 = indexed
        
        Returns:
            True if update was successful, False otherwise
        """
        try:
            await self.update(resume_id, {"pinecone_status": status})
            logger.info(
                f"Updated pinecone_status for resume {resume_id} to {status}",
                extra={"resume_id": resume_id, "pinecone_status": status}
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to update pinecone_status for resume {resume_id}: {e}",
                extra={"resume_id": resume_id, "status": status, "error": str(e)}
            )
            return False

