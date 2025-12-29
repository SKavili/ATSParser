"""Database connection pool and session management."""
import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import text

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

Base = declarative_base()

# Configure SQL logging for async SQLAlchemy
def setup_sql_logging():
    """
    Configure SQLAlchemy SQL statement logging for async engines.
    
    SQLAlchemy async engines log to 'sqlalchemy.engine' logger.
    Set SQL_ECHO=true in .env to enable SQL logging.
    """
    if settings.sql_echo:
        # Enable SQLAlchemy engine logging (works for async engines)
        sqlalchemy_logger = logging.getLogger('sqlalchemy.engine')
        log_level = getattr(logging, settings.sql_log_level.upper(), logging.INFO)
        sqlalchemy_logger.setLevel(log_level)
        
        # Optional: Connection pool logging (usually too verbose)
        pool_logger = logging.getLogger('sqlalchemy.pool')
        pool_logger.setLevel(logging.WARNING)  # Only show warnings
        
        # Optional: Connection logging
        connection_logger = logging.getLogger('sqlalchemy.engine.Engine')
        connection_logger.setLevel(log_level)
        
        logger.info(f"✅ SQL logging enabled at {settings.sql_log_level} level")
        logger.info("   SQL statements will be logged to console/logs")
    else:
        # Disable SQL logging
        logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
        logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)

# Initialize SQL logging
setup_sql_logging()

# Async engine with optional echo for SQL logging
# echo=True enables SQL statement logging to Python logging system
# Reduced pool size to prevent "Too many connections" MySQL error
engine = create_async_engine(
    settings.mysql_url,
    echo=settings.sql_echo,  # Enable SQL statement logging (logs to 'sqlalchemy.engine')
    echo_pool=False,  # Pool logging is usually too verbose
    pool_size=5,  # Reduced from 10 to prevent connection limit issues
    max_overflow=5,  # Reduced from 20 to prevent connection limit issues
    pool_pre_ping=True,  # Verify connections before using them
    pool_recycle=3600,  # Recycle connections after 1 hour to prevent stale connections
    pool_timeout=30,  # Timeout after 30 seconds if no connection available
    connect_args={
        "connect_timeout": 10,  # Connection timeout in seconds
    },
)

# Async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting database session.
    Ensures session is properly closed and connection returned to pool immediately after use.
    The async with context manager automatically closes the session, ensuring connections are returned to pool.
    """
    async with async_session_maker() as session:
        try:
            yield session
            # Commit any pending changes before context manager closes session
            await session.commit()
        except Exception:
            # Rollback on error before context manager closes session
            await session.rollback()
            raise
        # Session is automatically closed by async with context manager
        # Connection is returned to pool immediately



async def init_db() -> None:
    """Initialize database connection pool."""
    try:
        # Use a simpler connection test that doesn't hold a transaction
        async with engine.connect() as conn:
            logger.info("Testing database connection...")
            # Test connection with a simple query (no transaction needed for SELECT)
            # Note: Only execute() is async and needs await. fetchone() is synchronous.
            result = await conn.execute(text("SELECT 1"))
            result.fetchone()  # Consume the result (synchronous, no await)
        logger.info("Database connection established successfully")
    except Exception as e:
        error_msg = str(e)
        if "1040" in error_msg or "Too many connections" in error_msg:
            logger.error(
                "MySQL connection limit reached. Please check:\n"
                "1. Close other applications using MySQL\n"
                "2. Kill stale MySQL connections: SHOW PROCESSLIST; KILL <id>;\n"
                "3. Increase MySQL max_connections if needed",
                extra={"error": error_msg}
            )
        else:
            logger.error(f"Failed to connect to database: {e}", extra={"error": error_msg})
        raise


async def close_db() -> None:
    """Close database connections and dispose of engine."""
    try:
        # Close all connections in the pool
        await engine.dispose(close=True)
        logger.info("Database connections closed and engine disposed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}", extra={"error": str(e)})
        raise

