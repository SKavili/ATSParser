"""In-memory cache for job embeddings with size limits."""
from typing import Dict, Optional
from collections import OrderedDict
from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class JobCache:
    """Simple in-memory cache for job embeddings with LRU eviction."""
    
    def __init__(self, max_size: Optional[int] = None):
        self.max_size = max_size or settings.job_cache_max_size
        # Use OrderedDict for LRU eviction
        self._cache: OrderedDict[str, Dict] = OrderedDict()
    
    def store_job(self, job_id: str, embedding: list, metadata: dict) -> None:
        """Store job embedding in cache with LRU eviction if cache is full."""
        # Remove oldest entry if cache is full
        if len(self._cache) >= self.max_size and job_id not in self._cache:
            # Remove least recently used (first item)
            oldest_id, _ = self._cache.popitem(last=False)
            logger.debug(
                f"Cache full, evicted job: {oldest_id}",
                extra={"evicted_job_id": oldest_id, "cache_size": len(self._cache)}
            )
        
        # Store new entry (moves to end if already exists)
        if job_id in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(job_id)
        
        self._cache[job_id] = {
            "embedding": embedding,
            "metadata": metadata
        }
        logger.info(
            f"Stored job in cache: {job_id}",
            extra={"job_id": job_id, "cache_size": len(self._cache), "max_size": self.max_size}
        )
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """Retrieve job from cache (moves to end for LRU)."""
        if job_id in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(job_id)
            return self._cache[job_id]
        return None
    
    def delete_job(self, job_id: str) -> None:
        """Remove job from cache."""
        if job_id in self._cache:
            del self._cache[job_id]
            logger.info(f"Deleted job from cache: {job_id}", extra={"job_id": job_id})


# Global job cache instance
job_cache = JobCache()

