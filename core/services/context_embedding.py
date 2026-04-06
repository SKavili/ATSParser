"""Generate embeddings for context pipeline via Ollama (same stack as resume embeddings)."""
import logging
from typing import Optional

import httpx
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# Must match settings.embedding_dimension (768 for nomic-embed-text)
EMBED_MODEL = "nomic-embed-text"

_ollama_model_cache: Optional[str] = None


def _resolve_model(client: httpx.Client) -> str:
    """Use installed `nomic-embed-text` tag; same dimension as resume pipeline."""
    global _ollama_model_cache
    if _ollama_model_cache:
        return _ollama_model_cache
    try:
        r = client.get(f"{settings.ollama_host}/api/tags", timeout=30.0)
        r.raise_for_status()
        models = r.json().get("models", [])
        names = [m.get("name", "") for m in models]
        for n in names:
            if n.startswith(EMBED_MODEL):
                _ollama_model_cache = n
                logger.info("Context embedding using Ollama model: %s", _ollama_model_cache)
                return _ollama_model_cache
    except Exception as e:
        logger.warning("Could not list Ollama models: %s", e)
    raise RuntimeError(
        f"Ollama embedding model `{EMBED_MODEL}` not found. Run: ollama pull {EMBED_MODEL}"
    )


def generate_embedding(text: str) -> list[float]:
    """Sync embedding via Ollama; L2-normalized vector (same as EmbeddingService)."""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    if not settings.ollama_host:
        raise ValueError("OLLAMA_HOST is not configured")

    with httpx.Client(timeout=120.0) as client:
        model = _resolve_model(client)
        r = client.post(
            f"{settings.ollama_host}/api/embeddings",
            json={"model": model, "prompt": text.strip()},
            timeout=120.0,
        )
        r.raise_for_status()
        data = r.json()
        emb = data.get("embedding")
        if not emb:
            raise RuntimeError("Ollama returned no embedding")
        if len(emb) != settings.embedding_dimension:
            raise RuntimeError(
                f"Embedding length {len(emb)} != EMBEDDING_DIMENSION={settings.embedding_dimension}. "
                "Prefer `nomic-embed-text` for 768-dim context indexing, or align EMBEDDING_DIMENSION "
                "with your Ollama embedding model and recreate the Pinecone index."
            )

    arr = np.array(emb, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()
