"""Pinecone indexer for context-based candidate memory."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pinecone import Pinecone, ServerlessSpec

from app.config import settings
from core.services.context_builder import build_candidate_context
from core.services.context_embedding import generate_embedding

CONTEXT_INDEX_NAME = "ats-context"
CONTEXT_MODEL_DIMENSION = 1536  # text-embedding-3-small


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ContextIndexer:
    """Handles context-index creation and candidate upserts."""

    def __init__(self) -> None:
        api_key = getattr(settings, "pinecone_api_key", None)
        if not api_key or not str(api_key).strip():
            raise ValueError("PINECONE_API_KEY is required for context indexing.")

        self._pc = Pinecone(api_key=str(api_key).strip())
        self._index = None

    def ensure_index(self) -> None:
        """Create `ats-context` index if missing, then bind index handle."""
        existing = [idx.name for idx in self._pc.list_indexes()]
        if CONTEXT_INDEX_NAME not in existing:
            self._pc.create_index(
                name=CONTEXT_INDEX_NAME,
                dimension=CONTEXT_MODEL_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud,
                    region=settings.pinecone_region,
                ),
            )

        self._index = self._pc.Index(CONTEXT_INDEX_NAME)

    def _metadata_from_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        now = _iso_now()
        return {
            "entity": "candidate",
            "candidate_id": row.get("id"),
            "name": str(row.get("candidatename") or ""),
            "role": str(row.get("designation") or ""),
            "category": str(row.get("category") or ""),
            "mastercategory": str(row.get("mastercategory") or ""),
            "jobrole": str(row.get("jobrole") or ""),
            "domain": str(row.get("domain") or ""),
            "experience": str(row.get("experience") or ""),
            "skills": str(row.get("skillset") or ""),
            "location": str(row.get("location") or ""),
            "email": str(row.get("email") or ""),
            "mobile": str(row.get("mobile") or ""),
            "filename": str(row.get("filename") or ""),
            "created_at": str(row.get("created_at") or now),
            "updated_at": str(row.get("updated_at") or now),
        }

    def upsert_candidates(self, rows: List[Dict[str, Any]]) -> int:
        """Build context + embedding and upsert candidates into `ats-context`."""
        if self._index is None:
            self.ensure_index()

        vectors = []
        for row in rows:
            candidate_id = row.get("id")
            if candidate_id is None:
                continue

            context_text = build_candidate_context(row)
            embedding = generate_embedding(context_text)
            metadata = self._metadata_from_row(row)

            vectors.append(
                {
                    "id": f"candidate-{candidate_id}",
                    "values": embedding,
                    "metadata": metadata,
                }
            )

        if vectors:
            self._index.upsert(vectors=vectors)
        return len(vectors)

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Search similar candidates from context memory."""
        if self._index is None:
            self.ensure_index()

        return self._index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=metadata_filter,
        )

