"""Embedding generation for context-aware candidate memory."""
from typing import List

import httpx
from httpx import Timeout

from app.config import settings

OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
CONTEXT_EMBEDDING_MODEL = "text-embedding-3-small"


def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding using existing project OpenAI configuration.

    Reuses the same OPENAI_API_KEY source already used in the ATS project
    (settings.openai_api_key). Does not introduce any new OpenAI setup.
    """
    api_key = getattr(settings, "openai_api_key", None)
    if not api_key or not str(api_key).strip():
        raise ValueError(
            "OPENAI_API_KEY is required for context embedding generation."
        )

    payload = {
        "model": CONTEXT_EMBEDDING_MODEL,
        "input": text,
    }

    with httpx.Client(timeout=Timeout(60.0)) as client:
        response = client.post(
            OPENAI_EMBEDDINGS_URL,
            headers={
                "Authorization": f"Bearer {str(api_key).strip()}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    items = data.get("data", [])
    if not items:
        raise RuntimeError("OpenAI embeddings response contains no data")

    embedding = items[0].get("embedding", [])
    if not embedding:
        raise RuntimeError("OpenAI embeddings response contains empty embedding")

    return embedding

