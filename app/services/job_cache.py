"""In-memory cache for job embeddings."""
from typing import Dict, Optional
from app.utils.logging import get_logger

logger = get_logger(__name__)


class JobCache:
    """Simple in-memory cache for job embeddings."""
    
    def __init__(self):
        self._cache: Dict[str, Dict] = {}
    
    def store_job(self, job_id: str, embedding: list, metadata: dict) -> None:
        """Store job embedding in cache."""
        self._cache[job_id] = {
            "embedding": embedding,
            "metadata": metadata
        }
        logger.info(f"Stored job in cache: {job_id}", extra={"job_id": job_id})
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """Retrieve job from cache."""
        return self._cache.get(job_id)
    
    def delete_job(self, job_id: str) -> None:
        """Remove job from cache."""
        if job_id in self._cache:
            del self._cache[job_id]
            logger.info(f"Deleted job from cache: {job_id}", extra={"job_id": job_id})


# Global job cache instance
job_cache = JobCache()

