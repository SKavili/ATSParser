"""FastAPI application bootstrap."""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

from app.config import settings
from app.api.routes import router
from app.database.connection import init_db, close_db
from app.services.vector_db_service import get_vector_db_service
from app.utils.logging import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Initialize Sentry if DSN provided
if settings.sentry_dsn:
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        integrations=[
            FastApiIntegration(),
            SqlalchemyIntegration(),
        ],
        traces_sample_rate=0.1,
        environment="production",
    )
    logger.info("Sentry initialized")


# Startup/shutdown lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting ATS Backend application")
    
    try:
        # Initialize database
        await init_db()
        
        # Initialize vector database
        vector_db = await get_vector_db_service()
        logger.info("Vector database initialized")
        
        # Store vector_db in app state for dependency injection
        app.state.vector_db = vector_db
        
        logger.info("Application startup complete")
    
    except Exception as e:
        logger.error(f"Failed to start application: {e}", extra={"error": str(e)})
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down ATS Backend application")
    await close_db()
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="ATS Backend API",
    description="Resume Parsing + Embedding based ATS System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "code": "VALIDATION_ERROR",
            "details": exc.errors()
        }
    )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(
        f"Unhandled exception: {exc}",
            extra={
                "error": str(exc),
                "path": request.url.path,
                "method": request.method
            },
        exc_info=True
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred"
        }
    )


# Include routers
app.include_router(router, prefix="/api/v1", tags=["ATS"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "ATS Backend",
        "version": "1.0.0",
        "status": "running"
    }

