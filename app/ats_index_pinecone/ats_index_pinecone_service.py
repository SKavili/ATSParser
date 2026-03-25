"""Index resumes into Pinecone index `ats` (single index, no namespaces).

This service is intentionally separate from the existing IT/NON-IT + namespace
indexing used by `ai-search` / `ai-search-1`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import ResumeMetadata
from app.repositories.resume_repo import ResumeRepository
from app.services.embedding_service import EmbeddingService
from app.services.vector_db_service import VectorDBService, get_vector_db_service
from app.utils.cleaning import normalize_skill_list
from app.utils.logging import get_logger
from app.utils.safe_logger import safe_extra

logger = get_logger(__name__)


class AtsIndexPineconeService:
    """Index resumes into Pinecone `ats` index only (no namespaces)."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.embedding_service = EmbeddingService()
        self.resume_repo = ResumeRepository(session)
        self.vector_db: Optional[VectorDBService] = None

    async def _get_vector_db(self) -> VectorDBService:
        if self.vector_db is None:
            self.vector_db = await get_vector_db_service()
        return self.vector_db

    async def index_resumes(
        self,
        limit: Optional[int] = None,
        resume_ids: Optional[List[int]] = None,
        force: bool = True,
    ) -> Dict[str, Any]:
        """Index resumes into Pinecone `ats`.

        Args:
            limit: Max number of resumes to process.
            resume_ids: Optional list of specific resume IDs.
            force: When True, indexes all `completed` resumes (ignores old
                `pinecone_status`). When False, indexes only those whose
                `pinecone_status` is still 0/NULL (incremental mode).
        """
        try:
            vector_db = await self._get_vector_db()

            pending_resumes = await self.resume_repo.get_pending_pinecone_resumes(
                limit=limit,
                resume_ids=resume_ids,
                force=force,
            )

            # For initial `ats` build, you typically want `force=True`
            if not pending_resumes:
                return {
                    "indexed_count": 0,
                    "failed_count": 0,
                    "processed_ids": [],
                    "failed_ids": [],
                    "skipped_ids": [],
                    "message": "No resumes to index into `ats`",
                }

            indexed_count = 0
            failed_count = 0
            processed_ids: List[int] = []
            failed_ids: List[int] = []
            skipped_ids: List[int] = []

            for resume in pending_resumes:
                try:
                    if not resume.resume_text:
                        logger.warning(
                            "Skipping resume: missing resume_text",
                            extra={"resume_id": resume.id},
                        )
                        skipped_ids.append(resume.id)
                        continue

                    if not resume.mastercategory:
                        logger.warning(
                            "Skipping resume: missing mastercategory",
                            extra={"resume_id": resume.id},
                        )
                        skipped_ids.append(resume.id)
                        continue

                    success = await self._index_single_resume(vector_db, resume)
                    if success:
                        indexed_count += 1
                        processed_ids.append(resume.id)
                    else:
                        failed_count += 1
                        failed_ids.append(resume.id)
                except Exception as e:
                    failed_count += 1
                    failed_ids.append(resume.id)
                    logger.error(
                        f"Error indexing resume {resume.id} into `ats`: {e}",
                        extra={"resume_id": resume.id, "error": str(e)},
                        exc_info=True,
                    )

            return {
                "indexed_count": indexed_count,
                "failed_count": failed_count,
                "processed_ids": processed_ids,
                "failed_ids": failed_ids,
                "skipped_ids": skipped_ids,
                "message": (
                    f"Indexed {indexed_count} resumes into Pinecone `ats`. "
                    f"Failed: {failed_count}. Skipped: {len(skipped_ids)}"
                ),
            }
        except Exception as e:
            logger.error(
                f"Failed to index resumes into `ats`: {e}",
                extra={"error": str(e)},
                exc_info=True,
            )
            raise

    async def _index_single_resume(self, vector_db: VectorDBService, resume: ResumeMetadata) -> bool:
        try:
            # Parse skillset string to array for filtering.
            # Normalize skills to canonical forms (e.g., "react.js" -> "react").
            skills_array: List[str] = []
            if resume.skillset:
                raw_skills = [s.strip() for s in resume.skillset.split(",") if s.strip()]
                skills_array = normalize_skill_list(raw_skills)

            # Extract experience_years from experience string.
            experience_years = None
            if resume.experience:
                match = re.search(r"(\d+(?:\.\d+)?)", resume.experience)
                if match:
                    experience_years = int(float(match.group(1)))

            # Normalize designation/jobrole/location for case-insensitive filtering.
            normalized_designation = (resume.designation or "").lower().strip()
            normalized_jobrole = (resume.jobrole or "").lower().strip()

            raw_location = (resume.location or "").strip()
            normalized_location = ""
            if raw_location:
                normalized_location = raw_location.lower().split(",", 1)[0].strip()

            base_metadata: Dict[str, Any] = {
                "resume_id": resume.id,
                "candidate_id": f"C{resume.id}",  # Keep stable shape vs other codepaths
                "filename": resume.filename or "unknown",
                "candidate_name": resume.candidatename or "",
                "name": resume.candidatename or "",  # Alias for compatibility
                "jobrole": normalized_jobrole,
                "designation": normalized_designation,
                "experience": resume.experience or "",
                "experience_years": experience_years,
                "domain": resume.domain or "",
                "mobile": resume.mobile or "",
                "email": resume.email or "",
                "education": resume.education or "",
                "location": normalized_location,
                "skillset": resume.skillset or "",
                "skills": skills_array,
                # Include these explicitly since `VectorDBService` does not add them.
                "mastercategory": (resume.mastercategory or "").upper().strip(),
                "category": (resume.category or "").strip() or "uncategorized",
            }

            chunk_embeddings = await self.embedding_service.generate_chunk_embeddings(
                resume.resume_text,
                metadata=base_metadata,
            )

            if not chunk_embeddings:
                logger.warning(
                    "No embeddings generated for resume",
                    extra={"resume_id": resume.id},
                )
                return False

            vectors_to_store: List[Dict[str, Any]] = []
            for chunk_data in chunk_embeddings:
                vector_id = f"resume_{resume.id}_chunk_{chunk_data['chunk_index']}"

                # Truncate to keep Pinecone metadata under limit.
                full_resume_text = resume.resume_text or ""
                if len(full_resume_text) > 30000:
                    full_resume_text = full_resume_text[:30000] + "...[truncated]"

                # Chunk metadata for filtering/ranking.
                metadata = {
                    **chunk_data.get("metadata", {}),
                    "type": "resume",
                    "chunk_index": chunk_data["chunk_index"],
                    "chunk_text": chunk_data["text"],
                    "resume_text": full_resume_text,
                }

                vectors_to_store.append(
                    {
                        "id": vector_id,
                        "embedding": chunk_data["embedding"],
                        "metadata": metadata,
                    }
                )

            # Upsert into a single Pinecone index `ats` (no namespaces).
            await vector_db.upsert_vectors(vectors_to_store)

            logger.info(
                "Successfully indexed resume into `ats`",
                extra=safe_extra(
                    {
                        "resume_id": resume.id,
                        "vector_count": len(vectors_to_store),
                        "mastercategory": resume.mastercategory,
                        "category": resume.category,
                    }
                ),
            )
            return True
        except Exception as e:
            logger.error(
                f"Error indexing resume {resume.id} into `ats`: {e}",
                extra={"resume_id": resume.id, "error": str(e)},
                exc_info=True,
            )
            return False

