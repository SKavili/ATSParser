"""Create Pinecone index `all-ats-context` only.

Pinecone allows only lowercase letters, digits, and hyphens in index names (no underscores).

This file is standalone: it does not import app.config (avoids pulling the full Settings parse).

Run from project root:

    python create_all_ats_context_index.py

Set in `.env` or the environment: PINECONE_API_KEY, optionally PINECONE_CLOUD, PINECONE_REGION, EMBEDDING_DIMENSION.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Same index name as `CONTEXT_INDEX_NAME` in core/services/context_indexer.py
INDEX_NAME = "all-ats-context"


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")

    api_key = (os.environ.get("PINECONE_API_KEY") or "").strip()
    if not api_key:
        print("ERROR: Set PINECONE_API_KEY in .env or environment.")
        sys.exit(1)

    dim = int(os.environ.get("EMBEDDING_DIMENSION", "768"))
    cloud = (os.environ.get("PINECONE_CLOUD") or "aws").strip()
    region = (os.environ.get("PINECONE_REGION") or "us-east-1").strip()

    print(f"Creating index name={INDEX_NAME!r} dimension={dim} (must match Ollama nomic-embed-text).")

    pc = Pinecone(api_key=api_key)
    existing = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME in existing:
        print(f"Index {INDEX_NAME!r} already exists.")
        return

    pc.create_index(
        name=INDEX_NAME,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    print(f"Created Pinecone index {INDEX_NAME!r} dimension={dim} metric=cosine.")


if __name__ == "__main__":
    main()
