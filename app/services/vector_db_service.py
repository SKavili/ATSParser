"""Vector database service with Pinecone and FAISS fallback."""
import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import numpy as np

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Try importing Pinecone
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("Pinecone not available, will use FAISS fallback")

# Try importing FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available")


class VectorDBService(ABC):
    """Abstract base class for vector database operations."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector database."""
        pass
    
    @abstractmethod
    async def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> None:
        """Upsert vectors with metadata."""
        pass
    
    @abstractmethod
    async def query_vectors(
        self,
        query_vector: List[float],
        top_k: int,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query similar vectors."""
        pass
    
    @abstractmethod
    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors by IDs."""
        pass


class PineconeVectorDB(VectorDBService):
    """Pinecone implementation of vector database service."""
    
    def __init__(self):
        self.pc = None
        self.index = None
        self.index_name = settings.pinecone_index_name
        self.dimension = settings.embedding_dimension
    
    async def initialize(self) -> None:
        """Initialize Pinecone client and index."""
        if not PINECONE_AVAILABLE:
            raise RuntimeError("Pinecone library not installed")
        
        if not settings.use_pinecone:
            raise RuntimeError("Pinecone API key not configured")
        
        try:
            self.pc = Pinecone(api_key=settings.pinecone_api_key)
            
            # Check if index exists
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=settings.pinecone_cloud,
                        region=settings.pinecone_region
                    )
                )
                # Wait for index to be ready
                import time
                while self.index_name not in [idx.name for idx in self.pc.list_indexes()]:
                    time.sleep(1)
            
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Pinecone initialized: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}", extra={"error": str(e)})
            raise
    
    async def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> None:
        """Upsert vectors to Pinecone."""
        if not self.index:
            raise RuntimeError("Pinecone index not initialized")
        
        try:
            # Format vectors for Pinecone
            pinecone_vectors = []
            for vec_data in vectors:
                vector_id = vec_data.get("id")
                embedding = vec_data.get("embedding")
                metadata = vec_data.get("metadata", {})
                
                if not vector_id or not embedding:
                    continue
                
                pinecone_vectors.append({
                    "id": str(vector_id),
                    "values": embedding,
                    "metadata": metadata
                })
            
            if pinecone_vectors:
                # Pinecone upsert is synchronous, but we're in async context
                self.index.upsert(vectors=pinecone_vectors)
                logger.info(f"Upserted {len(pinecone_vectors)} vectors to Pinecone")
        
        except Exception as e:
            logger.error(f"Failed to upsert vectors to Pinecone: {e}", extra={"error": str(e)})
            raise
    
    async def query_vectors(
        self,
        query_vector: List[float],
        top_k: int,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query similar vectors from Pinecone."""
        if not self.index:
            raise RuntimeError("Pinecone index not initialized")
        
        try:
            # Pinecone query is synchronous, run in thread pool for async compatibility
            import asyncio
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_dict
                )
            )
            
            matches = []
            for match in results.get("matches", []):
                matches.append({
                    "id": match.get("id"),
                    "score": match.get("score", 0.0),
                    "metadata": match.get("metadata", {})
                })
            
            return matches
        
        except Exception as e:
            logger.error(f"Failed to query Pinecone: {e}", extra={"error": str(e)})
            raise
    
    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from Pinecone."""
        if not self.index:
            raise RuntimeError("Pinecone index not initialized")
        
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.index.delete, [str(id) for id in ids])
            logger.info(f"Deleted {len(ids)} vectors from Pinecone")
        except Exception as e:
            logger.error(f"Failed to delete vectors from Pinecone: {e}", extra={"error": str(e)})
            raise


class FAISSVectorDB(VectorDBService):
    """FAISS implementation of vector database service (fallback)."""
    
    def __init__(self):
        self.index = None
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self.dimension = settings.embedding_dimension
        self.faiss_index_path = "faiss_index.pkl"
        self.metadata_path = "faiss_metadata.pkl"
    
    async def initialize(self) -> None:
        """Initialize FAISS index."""
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS library not installed")
        
        try:
            # Try to load existing index
            if os.path.exists(self.faiss_index_path) and os.path.exists(self.metadata_path):
                logger.info("Loading existing FAISS index from disk")
                with open(self.faiss_index_path, "rb") as f:
                    self.index = pickle.load(f)
                with open(self.metadata_path, "rb") as f:
                    data = pickle.load(f)
                    self.metadata_store = data.get("metadata", {})
                    self.id_to_index = data.get("id_to_index", {})
                    self.index_to_id = data.get("index_to_id", {})
            else:
                # Create new index
                logger.info("Creating new FAISS index")
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            
            logger.warning(
                "Using FAISS as fallback vector database. Data stored locally.",
                extra={"index_path": self.faiss_index_path}
            )
        
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}", extra={"error": str(e)})
            raise
    
    def _save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            with open(self.faiss_index_path, "wb") as f:
                pickle.dump(self.index, f)
            with open(self.metadata_path, "wb") as f:
                pickle.dump({
                    "metadata": self.metadata_store,
                    "id_to_index": self.id_to_index,
                    "index_to_id": self.index_to_id,
                }, f)
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}", extra={"error": str(e)})
    
    async def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> None:
        """Upsert vectors to FAISS."""
        if not self.index:
            raise RuntimeError("FAISS index not initialized")
        
        try:
            # FAISS operations are CPU-bound, run in thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            
            def _upsert():
                vectors_to_add = []
                ids_to_add = []
                
                for vec_data in vectors:
                    vector_id = str(vec_data.get("id"))
                    embedding = vec_data.get("embedding")
                    metadata = vec_data.get("metadata", {})
                    
                    if not vector_id or not embedding:
                        continue
                    
                    # Convert to numpy array and normalize
                    embedding_array = np.array(embedding, dtype=np.float32)
                    norm = np.linalg.norm(embedding_array)
                    if norm > 0:
                        embedding_array = embedding_array / norm
                    
                    # Check if ID already exists (update)
                    if vector_id in self.id_to_index:
                        idx = self.id_to_index[vector_id]
                        # FAISS doesn't support update, so we'll replace
                        # For simplicity, we'll just add new entries
                        pass
                    
                    vectors_to_add.append(embedding_array)
                    ids_to_add.append(vector_id)
                    self.metadata_store[vector_id] = metadata
                
                if vectors_to_add:
                    vectors_matrix = np.vstack(vectors_to_add).astype(np.float32)
                    start_idx = self.index.ntotal
                    self.index.add(vectors_matrix)
                    
                    # Update mappings
                    for i, vector_id in enumerate(ids_to_add):
                        idx = start_idx + i
                        self.id_to_index[vector_id] = idx
                        self.index_to_id[idx] = vector_id
                    
                    self._save_index()
                    return len(vectors_to_add)
                return 0
            
            count = await loop.run_in_executor(None, _upsert)
            if count > 0:
                logger.info(f"Upserted {count} vectors to FAISS")
        
        except Exception as e:
            logger.error(f"Failed to upsert vectors to FAISS: {e}", extra={"error": str(e)})
            raise
    
    async def query_vectors(
        self,
        query_vector: List[float],
        top_k: int,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query similar vectors from FAISS."""
        if not self.index:
            raise RuntimeError("FAISS index not initialized")
        
        try:
            # FAISS operations are CPU-bound, run in thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            
            def _query():
                # Normalize query vector
                query_array = np.array(query_vector, dtype=np.float32).reshape(1, -1)
                norm = np.linalg.norm(query_array)
                if norm > 0:
                    query_array = query_array / norm
                
                # Search
                k = min(top_k * 2, self.index.ntotal)  # Get more results to filter
                if k == 0:
                    return []
                
                distances, indices = self.index.search(query_array, k)
                
                matches = []
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx == -1:  # FAISS returns -1 for invalid results
                        continue
                    
                    vector_id = self.index_to_id.get(idx)
                    if not vector_id:
                        continue
                    
                    metadata = self.metadata_store.get(vector_id, {})
                    
                    # Skip deleted vectors
                    if metadata.get("_deleted"):
                        continue
                    
                    # Apply filter if provided
                    if filter_dict:
                        if not all(metadata.get(k) == v for k, v in filter_dict.items()):
                            continue
                    
                    matches.append({
                        "id": vector_id,
                        "score": float(distance),  # Cosine similarity from inner product
                        "metadata": metadata
                    })
                
                # Sort by score descending
                matches.sort(key=lambda x: x["score"], reverse=True)
                return matches[:top_k]
            
            return await loop.run_in_executor(None, _query)
        
        except Exception as e:
            logger.error(f"Failed to query FAISS: {e}", extra={"error": str(e)})
            raise
    
    async def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from FAISS (marked for removal)."""
        # FAISS doesn't support deletion directly
        # We'll mark them in metadata and skip during query
        try:
            for vector_id in ids:
                if vector_id in self.metadata_store:
                    self.metadata_store[vector_id]["_deleted"] = True
            self._save_index()
            logger.warning("FAISS doesn't support direct deletion. Vectors marked as deleted in metadata.")
        except Exception as e:
            logger.error(f"Failed to delete vectors from FAISS: {e}", extra={"error": str(e)})
            raise


async def get_vector_db_service() -> VectorDBService:
    """
    Factory function to get appropriate vector DB service.
    Uses Pinecone if available, otherwise falls back to FAISS.
    """
    if settings.use_pinecone and PINECONE_AVAILABLE:
        try:
            service = PineconeVectorDB()
            await service.initialize()
            return service
        except Exception as e:
            logger.warning(
                f"Failed to initialize Pinecone, falling back to FAISS: {e}",
                extra={"error": str(e)}
            )
    
    # Fallback to FAISS
    if FAISS_AVAILABLE:
        service = FAISSVectorDB()
        await service.initialize()
        return service
    
    raise RuntimeError("No vector database backend available (Pinecone or FAISS)")


