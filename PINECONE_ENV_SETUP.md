# Pinecone Environment Variables Setup

## Required Environment Variables for Pinecone

Add these to your `.env` file in the `ATSParser` directory:

```env
# Pinecone Configuration (REQUIRED for Pinecone automation)
# Get your API key from: https://app.pinecone.io/
PINECONE_API_KEY=your_pinecone_api_key_here

# Cloud provider (default: aws)
PINECONE_CLOUD=aws

# Region (default: us-east-1)
PINECONE_REGION=us-east-1

# Embedding dimension (default: 768 for nomic-embed-text)
# Must match the dimension of your embedding model
# nomic-embed-text: 768, mxbai-embed-large: 1024, OpenAI: 1536
EMBEDDING_DIMENSION=768
```

## Important Notes

1. **PINECONE_API_KEY** (REQUIRED)
   - Get your API key from [Pinecone Console](https://app.pinecone.io/)
   - This is the only mandatory field for Pinecone setup
   - Without this, the system will not be able to connect to Pinecone

2. **PINECONE_CLOUD** (Optional, defaults to "aws")
   - Options: `aws`, `gcp`, `azure`
   - Must match your Pinecone account setup

3. **PINECONE_REGION** (Optional, defaults to "us-east-1")
   - Examples: `us-east-1`, `us-west-2`, `eu-west-1`, etc.
   - Must match your Pinecone account region

4. **EMBEDDING_DIMENSION** (Optional, defaults to 768)
   - Must match the dimension of your embedding model
   - **nomic-embed-text**: 768 (default, used by this system)
   - **mxbai-embed-large**: 1024 (fallback model)
   - **OpenAI text-embedding-ada-002**: 1536
   - **Other models**: Check model documentation

5. **PINECONE_INDEX_NAME** (NOT USED)
   - The system uses hardcoded index names: `it` and `non-it`
   - This variable is kept for backward compatibility but is ignored
   - You don't need to set this

## Index Names

The system automatically creates and uses two indexes:
- **`it`** - For IT resumes (22 namespaces)
- **`non-it`** - For Non-IT resumes (30 namespaces)

These are hardcoded in the code and cannot be changed via environment variables.

## Setup Steps

1. Copy `.env.example` to `.env` (if available)
2. Add your `PINECONE_API_KEY` to `.env`
3. Optionally set `PINECONE_CLOUD` and `PINECONE_REGION` if different from defaults
4. Run: `python initialize_pinecone_indexes.py`

## Verification

After setting up, verify your configuration:

```python
from app.config import settings

print(f"API Key: {'SET' if settings.pinecone_api_key else 'NOT SET'}")
print(f"Cloud: {settings.pinecone_cloud}")
print(f"Region: {settings.pinecone_region}")
print(f"Dimension: {settings.embedding_dimension}")
```

