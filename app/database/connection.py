"""Database connection pool and session management."""
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import text

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

Base = declarative_base()

# Async engine
engine = create_async_engine(
    settings.mysql_url,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

# Async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database connection pool."""
    try:
        async with engine.begin() as conn:
            logger.info("Database connection established")
            # Test connection
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}", extra={"error": str(e)})
        raise


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()
    logger.info("Database connections closed")

