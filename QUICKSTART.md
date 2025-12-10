# Quick Start Guide

## Prerequisites

1. **Python 3.10+** installed
2. **MySQL** server running
3. **OLLAMA** installed and running

## Setup Steps

### 1. Install OLLAMA Models

```bash
# Install and start OLLAMA (if not already installed)
# Visit https://ollama.ai for installation instructions

# Pull required models
ollama pull llama3.1
ollama pull nomic-embed-text
```

### 2. Setup Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your MySQL credentials
# At minimum, set:
# - MYSQL_HOST
# - MYSQL_USER
# - MYSQL_PASSWORD
# - MYSQL_DATABASE
```

### 4. Create Database

```sql
-- Connect to MySQL and run:
CREATE DATABASE ats_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 5. Run Migrations

Option A: Using Alembic
```bash
alembic upgrade head
```

Option B: Using SQL file
```bash
mysql -u root -p ats_db < migrations/001_create_resume_metadata.sql
```

### 6. Start the Server

**Option A: Using batch file (Windows)**
```bash
start_api.bat
```

**Option B: Using command line**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Test the API

```bash
# Health check
curl http://localhost:8000/api/v1/health

# View API documentation
# Open browser to: http://localhost:8000/docs
```

## Common Issues

### OLLAMA Connection Error

If you get errors connecting to OLLAMA:
- Ensure OLLAMA is running: `ollama serve`
- Check OLLAMA_HOST in .env matches your OLLAMA instance
- Verify models are pulled: `ollama list`

### MySQL Connection Error

- Verify MySQL server is running
- Check credentials in .env file
- Ensure database exists: `CREATE DATABASE ats_db;`

### Pinecone Not Available

The system will automatically fallback to FAISS if Pinecone API key is not set. This is normal and expected if you're not using Pinecone.

### FAISS Index Errors

If FAISS index files are corrupted:
- Delete `faiss_index.pkl` and `faiss_metadata.pkl`
- The system will recreate them on next startup

## Next Steps

1. Upload a test resume using the `/upload-resume` endpoint
2. Create a job posting using `/create-job`
3. Match resumes to jobs using `/match`

See `README.md` for detailed API documentation and examples.

