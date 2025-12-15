"""Service for generating embeddings using OLLAMA."""
import asyncio
import gc
from typing import List, Optional
import httpx
from httpx import Timeout
import numpy as np

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating embeddings using OLLAMA API."""
    
    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.primary_model = "nomic-embed-text"
        self.fallback_model = "mxbai-embed-large"
        self.model = None
        self.embedding_dimension = settings.embedding_dimension
    
    async def _initialize_model(self) -> str:
        """Check which embedding model is available and set it."""
        if self.model:
            return self.model
        
        # Try primary model first
        if await self._check_model_available(self.primary_model):
            self.model = self.primary_model
            logger.info(f"Using embedding model: {self.primary_model}")
            return self.model
        
        # Fallback to secondary model
        if await self._check_model_available(self.fallback_model):
            self.model = self.fallback_model
            logger.warning(f"Primary model unavailable, using fallback: {self.fallback_model}")
            return self.model
        
        raise RuntimeError(f"Neither {self.primary_model} nor {self.fallback_model} is available")
    
    async def _check_model_available(self, model_name: str) -> bool:
        """Check if a model is available via OLLAMA."""
        try:
            async with httpx.AsyncClient(timeout=Timeout(10.0)) as client:
                response = await client.get(f"{self.ollama_host}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
                available = any(m.get("name", "").startswith(model_name) for m in models)
                return available
        except Exception as e:
            logger.warning(f"Failed to check model availability: {e}", extra={"model": model_name})
            return False
    
    async def generate_embedding(self, text: str, retries: int = 3) -> List[float]:
        """Generate embedding for a single text with retry logic."""
        await self._initialize_model()
        
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(timeout=Timeout(600.0)) as client:
                    response = await client.post(
                        f"{self.ollama_host}/api/embeddings",
                        json={
                            "model": self.model,
                            "prompt": text,
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    embedding = result.get("embedding", [])
                    
                    if not embedding:
                        raise ValueError("Empty embedding returned")
                    
                    # Normalize embedding
                    embedding_array = np.array(embedding)
                    norm = np.linalg.norm(embedding_array)
                    if norm > 0:
                        embedding_array = embedding_array / norm
                    
                    return embedding_array.tolist()
                    
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Embedding generation failed, retrying in {wait_time}s: {e}",
                        extra={"attempt": attempt + 1, "error": str(e)}
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to generate embedding after {retries} attempts: {e}")
                    raise RuntimeError(f"Failed to generate embedding: {e}")
        
        raise RuntimeError("Failed to generate embedding")
    
    def chunk_text(self, text: str, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> List[str]:
        """Split text into overlapping chunks."""
        if chunk_size is None:
            chunk_size = settings.chunk_size
        if chunk_overlap is None:
            chunk_overlap = settings.chunk_overlap
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= len(text):
                break
            
            # Move start by chunk_size - overlap for next chunk
            start += chunk_size - chunk_overlap
        
        return chunks
    
    async def generate_chunk_embeddings(self, text: str, metadata: Optional[dict] = None) -> List[dict]:
        """
        Generate embeddings for text chunks in memory-efficient batches.
        Returns list of dicts with 'embedding', 'text', and 'metadata'.
        """
        chunks = self.chunk_text(text)
        batch_size = settings.embedding_batch_size
        logger.info(
            f"Generating embeddings for {len(chunks)} chunks (batch size: {batch_size})",
            extra={"chunk_count": len(chunks), "text_length": len(text), "batch_size": batch_size}
        )
        
        embeddings = []
        
        # Process chunks in batches to reduce memory usage
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            batch_embeddings = []
            for idx, chunk in enumerate(batch_chunks):
                chunk_idx = batch_start + idx
                try:
                    embedding = await self.generate_embedding(chunk)
                    batch_embeddings.append({
                        "embedding": embedding,
                        "text": chunk,
                        "chunk_index": chunk_idx,
                        "metadata": metadata or {},
                    })
                except Exception as e:
                    logger.error(
                        f"Failed to generate embedding for chunk {chunk_idx}: {e}",
                        extra={"chunk_index": chunk_idx, "error": str(e)}
                    )
                    # Continue with other chunks
            
            embeddings.extend(batch_embeddings)
            
            # Memory cleanup after each batch
            if settings.enable_memory_cleanup:
                del batch_chunks
                del batch_embeddings
                gc.collect()
        
        return embeddings


