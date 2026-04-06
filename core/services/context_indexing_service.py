"""Bulk indexing service for context-based candidate embeddings into Pinecone."""
import asyncio
from typing import Any, Dict, List, Optional

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.utils.logging import get_logger
from core.services.context_indexer import CONTEXT_INDEX_NAME, ContextIndexer

logger = get_logger(__name__)


class ContextIndexingService:
    """
    Index MySQL resume_metadata rows into Pinecone `all-ats-context`.

    This service is intentionally separate from existing ATS indexing flows.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.indexer = ContextIndexer()

    @staticmethod
    def _resume_to_row(resume: Any) -> Dict[str, Any]:
        """Map ORM ResumeMetadata object to context index input row."""
        def pick(key: str) -> Any:
            if isinstance(resume, dict):
                return resume.get(key)
            return getattr(resume, key, None)

        return {
            "id": pick("id"),
            "candidatename": pick("candidatename"),
            "designation": pick("designation"),
            "category": pick("category"),
            "mastercategory": pick("mastercategory"),
            "jobrole": pick("jobrole"),
            "domain": pick("domain"),
            "experience": pick("experience"),
            "skillset": pick("skillset"),
            "education": pick("education"),
            "location": pick("location"),
            "email": pick("email"),
            "mobile": pick("mobile"),
            "filename": pick("filename"),
            "resume_text": pick("resume_text"),
            "created_at": pick("created_at"),
            "updated_at": pick("updated_at"),
        }

    async def index_resumes(
        self,
        limit: Optional[int] = None,
        resume_ids: Optional[List[int]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Bulk index resumes into the context Pinecone index.

        Args:
            limit: Max rows to process. ``0`` or ``None`` means no cap (all eligible).
            resume_ids: Optional specific resume IDs.
            force: Mirrors existing style. True means process all eligible rows.

        Returns:
            Summary payload similar to existing indexing endpoints.
        """
        resumes = await self._get_pending_context_resumes(
            limit=limit,
            resume_ids=resume_ids,
            force=force,
        )

        if not resumes:
            return {
                "indexed_count": 0,
                "failed_count": 0,
                "processed_ids": [],
                "failed_ids": [],
                "skipped_ids": [],
                "message": f"No resumes to index into `{CONTEXT_INDEX_NAME}`",
            }

        await asyncio.to_thread(self.indexer.ensure_index)

        indexed_count = 0
        failed_count = 0
        processed_ids: List[int] = []
        failed_ids: List[int] = []
        skipped_ids: List[int] = []

        for resume in resumes:
            resume_id = resume.get("id") if isinstance(resume, dict) else getattr(resume, "id", None)
            if resume_id is None:
                continue

            resume_text = resume.get("resume_text") if isinstance(resume, dict) else getattr(resume, "resume_text", None)
            if not resume_text:
                skipped_ids.append(resume_id)
                continue

            try:
                row = self._resume_to_row(resume)
                count = await asyncio.to_thread(self.indexer.upsert_candidates, [row])
                if count > 0:
                    await self._update_context_status(resume_id, 1)
                    indexed_count += 1
                    processed_ids.append(resume_id)
                else:
                    await self._update_context_status(resume_id, 0)
                    failed_count += 1
                    failed_ids.append(resume_id)
            except Exception as e:
                await self._update_context_status(resume_id, 0)
                failed_count += 1
                failed_ids.append(resume_id)
                logger.error(
                    f"Failed context indexing for resume {resume_id}: {e}",
                    extra={"resume_id": resume_id, "error": str(e)},
                    exc_info=True,
                )

        return {
            "indexed_count": indexed_count,
            "failed_count": failed_count,
            "processed_ids": processed_ids,
            "failed_ids": failed_ids,
            "skipped_ids": skipped_ids,
            "message": (
                f"Indexed {indexed_count} resumes into Pinecone `{CONTEXT_INDEX_NAME}`. "
                f"Failed: {failed_count}. Skipped: {len(skipped_ids)}"
            ),
        }

    async def _get_pending_context_resumes(
        self,
        limit: Optional[int],
        resume_ids: Optional[List[int]],
        force: bool,
    ) -> List[Any]:
        """
        Fetch eligible resumes for context indexing.

        Rules:
        - Only status='completed'
        - Must have resume_text and mastercategory
        - If force=False, only context_pinecone_status is 0/NULL
        - If force=True, ignore context_pinecone_status and index all eligible rows
        """
        base_sql = """
            SELECT
                id, candidatename, designation, category, mastercategory, jobrole,
                domain, experience, skillset, education, location, email, mobile,
                filename, resume_text, created_at, updated_at, context_pinecone_status
            FROM resume_metadata
            WHERE status = 'completed'
              AND resume_text IS NOT NULL
              AND mastercategory IS NOT NULL
        """
        params: Dict[str, Any] = {}

        if not force:
            base_sql += " AND (context_pinecone_status = 0 OR context_pinecone_status IS NULL)"

        if resume_ids:
            base_sql += " AND id IN :resume_ids"

        base_sql += " ORDER BY id ASC"
        if limit is not None and limit > 0:
            base_sql += " LIMIT :limit"
            params["limit"] = limit

        stmt = text(base_sql)
        if resume_ids:
            stmt = stmt.bindparams(bindparam("resume_ids", expanding=True))
            params["resume_ids"] = resume_ids

        result = await self.session.execute(stmt, params)
        rows = result.mappings().all()
        return rows

    async def _update_context_status(self, resume_id: int, status: int) -> None:
        """Update context_pinecone_status for a resume row."""
        stmt = text(
            """
            UPDATE resume_metadata
            SET context_pinecone_status = :status
            WHERE id = :resume_id
            """
        )
        await self.session.execute(stmt, {"status": status, "resume_id": resume_id})
        await self.session.commit()

