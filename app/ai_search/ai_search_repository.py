"""Repository for AI search queries and results."""
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.database.models import AISearchQuery, AISearchResult
from app.utils.logging import get_logger

logger = get_logger(__name__)


class AISearchRepository:
    """Repository for AI search database operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_query(self, query_text: str, user_id: Optional[int] = None) -> AISearchQuery:
        """Create a new search query record."""
        try:
            query = AISearchQuery(
                query_text=query_text,
                user_id=user_id
            )
            self.session.add(query)
            await self.session.flush()
            await self.session.commit()
            await self.session.refresh(query)
            logger.info(
                f"Created AI search query: id={query.id}",
                extra={"query_id": query.id, "query_text": query_text[:100]}
            )
            return query
        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"Failed to create search query: {e}", extra={"error": str(e)})
            raise
    
    async def create_result(
        self,
        search_query_id: int,
        results_json: Dict[str, Any]
    ) -> AISearchResult:
        """Create a new search result record."""
        try:
            import json
            result = AISearchResult(
                search_query_id=search_query_id,
                results_json=results_json
            )
            self.session.add(result)
            await self.session.flush()
            await self.session.commit()
            await self.session.refresh(result)
            logger.info(
                f"Created AI search result: id={result.id}, query_id={search_query_id}",
                extra={"result_id": result.id, "search_query_id": search_query_id}
            )
            return result
        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"Failed to create search result: {e}", extra={"error": str(e)})
            raise
    
    async def get_query_by_id(self, query_id: int) -> Optional[AISearchQuery]:
        """Get search query by ID."""
        result = await self.session.execute(
            select(AISearchQuery).where(AISearchQuery.id == query_id)
        )
        return result.scalar_one_or_none()
