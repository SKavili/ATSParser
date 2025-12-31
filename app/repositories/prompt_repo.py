"""Repository for prompt operations."""
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.models import Prompt
from app.utils.logging import get_logger

logger = get_logger(__name__)


class PromptRepository:
    """Repository for prompt CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_by_category(
        self, 
        mastercategory: str, 
        category: str
    ) -> Optional[Prompt]:
        """
        Get prompt by mastercategory and category.
        
        Args:
            mastercategory: "IT" or "NON_IT"
            category: The specific category name (e.g., "Full Stack Development (Java)", "Python", "other")
                     Note: "other" is used as fallback when specific category is not found or category is NULL
        
        Returns:
            Prompt object or None if not found
        """
        try:
            result = await self.session.execute(
                select(Prompt).where(
                    Prompt.mastercategory == mastercategory,
                    Prompt.category == category
                )
            )
            prompt = result.scalar_one_or_none()
            
            if prompt:
                logger.debug(
                    f"Found prompt for mastercategory={mastercategory}, category={category}",
                    extra={"prompt_id": prompt.id}
                )
            else:
                logger.warning(
                    f"No prompt found for mastercategory={mastercategory}, category={category}"
                )
            
            return prompt
        except Exception as e:
            logger.error(
                f"Error fetching prompt: {e}",
                extra={
                    "mastercategory": mastercategory,
                    "category": category,
                    "error": str(e)
                }
            )
            return None
    
    async def get_by_mastercategory(
        self, 
        mastercategory: str
    ) -> Optional[Prompt]:
        """
        Get prompt by mastercategory only (fallback when category doesn't match).
        
        Args:
            mastercategory: "IT" or "NON_IT"
        
        Returns:
            Prompt object or None if not found
        """
        try:
            result = await self.session.execute(
                select(Prompt).where(
                    Prompt.mastercategory == mastercategory,
                    Prompt.category.is_(None)
                )
            )
            prompt = result.scalar_one_or_none()
            
            if prompt:
                logger.debug(
                    f"Found generic prompt for mastercategory={mastercategory}",
                    extra={"prompt_id": prompt.id}
                )
            
            return prompt
        except Exception as e:
            logger.error(
                f"Error fetching prompt by mastercategory: {e}",
                extra={
                    "mastercategory": mastercategory,
                    "error": str(e)
                }
            )
            return None

