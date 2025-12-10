# ATS Backend - Resume Parsing + Embedding System

Enterprise-ready backend for an Applicant Tracking System (ATS) using Python, FastAPI, OLLAMA, and vector databases.

## Features

- **Resume Parsing**: Extracts candidate information from PDF, DOCX, and TXT files using LLM (llama3.1)
- **Embedding Generation**: Creates embeddings using OLLAMA (nomic-embed-text or mxbai-embed-large)
- **Vector Search**: Pinecone integration with FAISS fallback for similarity search
- **Job Matching**: Match resumes to job descriptions using semantic similarity
- **Clean Architecture**: Controllers → Services → Repositories → Models
- **Async/Await**: Fully asynchronous I/O operations
- **Structured Logging**: JSON-formatted logs
- **Error Handling**: Comprehensive error handling with proper HTTP status codes

## Technology Stack

- Python 3.10+
- FastAPI
- MySQL (async with aiomysql)
- OLLAMA (LLM + Embeddings)
- Pinecone (with FAISS fallback)
- Alembic (database migrations)

## Prerequisites

1. Python 3.10 or higher
2. MySQL server (5.7+ or 8.0+)
3. OLLAMA installed and running (default: http://localhost:11434)
   - Install OLLAMA: https://ollama.ai
   - Pull required models:
     ```bash
     ollama pull llama3.1
     ollama pull nomic-embed-text
     # Or fallback:
     ollama pull mxbai-embed-large
     ```

## Setup

1. **Clone the repository** (if applicable)

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

   Required variables:
   - `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`
   - `OLLAMA_HOST` (default: http://localhost:11434)
   - Optional: `PINECONE_API_KEY` for Pinecone (falls back to FAISS if not provided)

5. **Create MySQL database**:
   ```sql
   CREATE DATABASE ats_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   ```

6. **Run database migrations**:
   
   Option A: Using Alembic
   ```bash
   alembic upgrade head
   ```
   
   Option B: Using SQL file
   ```bash
   mysql -u root -p ats_db < migrations/001_create_resume_metadata.sql
   ```

7. **Start the server**:
   
   Option A: Using batch file (Windows):
   ```bash
   start_api.bat
   ```
   
   Option B: Using command line:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   
   Or using Python:
   ```bash
   python -m uvicorn app.main:app --reload
   ```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### POST /api/v1/upload-resume

Upload and parse a resume file.

**Request**: multipart/form-data
- `file`: Resume file (PDF, DOCX, or TXT)
- `candidate_name` (optional): Candidate name
- `job_role` (optional): Job role
- `source` (optional): Source identifier

**Response**:
```json
{
  "id": 1,
  "candidateName": "John Doe",
  "jobrole": "Software Engineer",
  "experience": "5 years",
  "domain": "Technology",
  "mobile": "+1234567890",
  "email": "john.doe@example.com",
  "education": "BS in Computer Science",
  "filename": "resume.pdf",
  "skillset": "Python, FastAPI, MySQL",
  "created_at": "2024-01-01T00:00:00"
}
```

### POST /api/v1/create-job

Create a job posting and generate embeddings.

**Request Body**:
```json
{
  "title": "Senior Software Engineer",
  "description": "We are looking for an experienced software engineer...",
  "required_skills": ["Python", "FastAPI", "MySQL"],
  "location": "San Francisco, CA",
  "job_id": "optional-custom-id"
}
```

**Response**:
```json
{
  "job_id": "job_abc123",
  "title": "Senior Software Engineer",
  "embedding_id": "job_job_abc123",
  "message": "Job created successfully"
}
```

### POST /api/v1/match

Match resumes to a job description.

**Request Body**:
```json
{
  "job_id": "job_abc123",
  "top_k": 5
}
```

Or:
```json
{
  "job_description": "We are looking for an experienced software engineer...",
  "top_k": 5
}
```

**Response**:
```json
{
  "matches": [
    {
      "resume_id": 1,
      "candidate_name": "John Doe",
      "similarity_score": 0.85,
      "candidate_summary": "John Doe with 5 years experience in Technology.",
      "filename": "resume.pdf"
    }
  ],
  "total_results": 1,
  "job_id": "job_abc123"
}
```

### GET /api/v1/health

Health check endpoint.

## Sample cURL Requests

### Upload Resume

```bash
curl -X POST "http://localhost:8000/api/v1/upload-resume" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/resume.pdf" \
  -F "candidate_name=John Doe" \
  -F "job_role=Software Engineer"
```

### Create Job

```bash
curl -X POST "http://localhost:8000/api/v1/create-job" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Senior Software Engineer",
    "description": "We are looking for an experienced software engineer with Python and FastAPI experience.",
    "required_skills": ["Python", "FastAPI", "MySQL"],
    "location": "San Francisco, CA"
  }'
```

### Match Job

```bash
curl -X POST "http://localhost:8000/api/v1/match" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "We are looking for an experienced software engineer with Python and FastAPI experience.",
    "top_k": 5
  }'
```

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app bootstrap
│   ├── config.py               # Configuration management
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py           # API route definitions
│   ├── controllers/
│   │   ├── __init__.py
│   │   ├── resume_controller.py
│   │   └── job_controller.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── resume_parser.py    # LLM-based resume parsing
│   │   ├── job_parser.py       # Job description parsing
│   │   ├── embedding_service.py # Embedding generation
│   │   └── vector_db_service.py # Vector DB (Pinecone/FAISS)
│   ├── repositories/
│   │   ├── __init__.py
│   │   └── resume_repo.py      # Database operations
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resume_models.py    # Pydantic models
│   │   └── job_models.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py       # DB connection pool
│   │   └── models.py           # SQLAlchemy models
│   └── utils/
│       ├── __init__.py
│       ├── cleaning.py         # Data cleaning utilities
│       └── logging.py          # Structured logging
├── alembic/                    # Database migrations
├── migrations/                 # SQL migration files
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

Key environment variables:

- **Database**: `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`, `MYSQL_PORT`
- **Chunking**: `CHUNK_SIZE` (default: 1000), `CHUNK_OVERLAP` (default: 200)
- **Matching**: `TOP_K_RESULTS` (default: 5), `SIMILARITY_THRESHOLD` (default: 0.5)
- **Embeddings**: `EMBEDDING_DIMENSION` (default: 1536)
- **Pinecone**: `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_CLOUD`, `PINECONE_REGION`
- **OLLAMA**: `OLLAMA_HOST` (default: http://localhost:11434)
- **Monitoring**: `SENTRY_DSN` (optional), `LOG_LEVEL` (default: INFO)

## Vector Database

The system supports two vector database backends:

1. **Pinecone** (preferred): Requires `PINECONE_API_KEY`. Automatically creates index if not exists.
2. **FAISS** (fallback): Used automatically if Pinecone is not configured. Stores index locally as `faiss_index.pkl`.

## Error Handling

The API returns standardized error responses:

```json
{
  "error": "Error message",
  "code": "ERROR_CODE",
  "details": {}
}
```

Common error codes:
- `VALIDATION_ERROR`: Request validation failed
- `INTERNAL_ERROR`: Unexpected server error

## Logging

Logs are structured JSON format, suitable for log aggregation systems. Logs include:
- Timestamp
- Log level
- Message
- Module/function/line
- Additional context fields

## Development

### Running Tests

```bash
# Add test files and run
pytest
```

### Code Quality

```bash
# Format code
black app/

# Lint
ruff check app/
```

## Production Deployment

1. Set `LOG_LEVEL=WARNING` for production
2. Configure proper CORS origins in `app/main.py`
3. Use a production ASGI server (e.g., Gunicorn with Uvicorn workers)
4. Set up reverse proxy (Nginx)
5. Configure SSL/TLS certificates
6. Set up monitoring and alerting (Sentry)
7. Use connection pooling for database
8. Configure rate limiting
9. Set up backups for MySQL and vector database

## License

[Specify your license here]

## Support

[Add support contact information]

