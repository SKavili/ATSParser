"""Persistence for ai_search_1 — same tables as ai_search (ai_search_queries / ai_search_results)."""

from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.database.models import AISearchQuery, AISearchResult
from app.utils.logging import get_logger

logger = get_logger(__name__)


class AISearch2Repository:
    """Repository for ai_search_2 DB operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_query(self, query_text: str, user_id: Optional[int] = None) -> AISearchQuery:
        try:
            query = AISearchQuery(query_text=query_text, user_id=user_id)
            self.session.add(query)
            await self.session.flush()
            await self.session.commit()
            await self.session.refresh(query)
            logger.info(
                f"Created AI search v1 query: id={query.id}",
                extra={"query_id": query.id, "query_text": query_text[:100]},
            )
            return query
        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"Failed to create search query: {e}", extra={"error": str(e)})
            raise

    async def create_result(self, search_query_id: int, results_json: Dict[str, Any]) -> AISearchResult:
        try:
            result = AISearchResult(search_query_id=search_query_id, results_json=results_json)
            self.session.add(result)
            await self.session.flush()
            await self.session.commit()
            await self.session.refresh(result)
            logger.info(
                f"Created AI search v1 result: id={result.id}, query_id={search_query_id}",
                extra={"result_id": result.id, "search_query_id": search_query_id},
            )
            return result
        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"Failed to create search result: {e}", extra={"error": str(e)})
            raise

    async def get_query_by_id(self, query_id: int) -> Optional[AISearchQuery]:
        result = await self.session.execute(select(AISearchQuery).where(AISearchQuery.id == query_id))
        return result.scalar_one_or_none()
