# ATS Parser - Complete Project Explanation

## 📋 Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Complete Flow: Resume Upload to Database Storage](#complete-flow)
4. [File Responsibilities](#file-responsibilities)
5. [Key Components Deep Dive](#key-components)

---

## 🎯 Overview

**ATS Parser** is an **Applicant Tracking System (ATS) Backend** that automatically parses resume files (PDF, DOCX, DOC, TXT) and extracts structured candidate information using **Large Language Models (LLMs)** via **OLLAMA**. The system uses a modular architecture to extract 8 different types of information from resumes and stores them in a MySQL database.

### Key Features:
- **Multi-format Resume Parsing**: Supports PDF, DOCX, DOC, and TXT files
- **LLM-based Extraction**: Uses OLLAMA (llama3.1) for intelligent information extraction
- **8 Extraction Modules**: Designation, Name, Email, Mobile, Experience, Domain, Education, Skills
- **Database Storage**: MySQL database with async operations
- **Status Tracking**: Tracks processing status (pending, processing, completed, failed)
- **Error Handling**: Comprehensive error handling with detailed failure reasons
- **Memory Optimization**: Built-in memory management for large files

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│  (app/main.py) - Entry point, middleware, error handlers    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    API Routes Layer                          │
│  (app/api/routes.py) - HTTP endpoints, request validation   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Controller Layer                            │
│  (app/controllers/) - Business logic orchestration          │
│  - resume_controller.py: Handles resume upload flow         │
│  - job_controller.py: Handles job matching                 │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Services   │  │ Extractors   │  │ Repository   │
│   Layer      │  │   Layer      │  │   Layer      │
└──────────────┘  └──────────────┘  └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Database Layer                              │
│  (app/database/) - MySQL connection, models, migrations     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Complete Flow: Resume Upload to Database Storage

### Step-by-Step Process:

#### **1. Client Request** 
```
POST /api/v1/upload-resume
Content-Type: multipart/form-data
- file: resume.pdf
- candidate_name: "John Doe" (optional)
- job_role: "Software Engineer" (optional)
- extract_modules: "all" (optional, default: "all")
```

#### **2. API Route Handler** (`app/api/routes.py`)
- Receives the HTTP request
- Validates request format
- Creates dependency injection chain:
  - `ResumeController` ← `ResumeParser` + `EmbeddingService` + `VectorDBService` + `ResumeRepository`
- Calls `controller.upload_resume()`

#### **3. Controller: Initial Validation** (`app/controllers/resume_controller.py`)
- **File Validation**:
  - Checks file extension (`.pdf`, `.docx`, `.doc`, `.txt`)
  - Validates file is not empty
  - Checks file size (max 10MB by default)
  - Sanitizes filename
  
- **Duplicate Check**:
  - Queries database for existing resume with same filename
  - If found, returns existing record (avoids reprocessing)

#### **4. Database Record Creation** (`app/repositories/resume_repo.py`)
- Creates initial database record with:
  - `status = "processing"`
  - `filename = sanitized_filename`
  - `candidatename`, `jobrole` (from metadata if provided)
  - All other fields = `NULL`
- Returns `resume_metadata` object with `id`

#### **5. Text Extraction** (`app/services/resume_parser.py`)
- Reads file content into memory
- Determines file type from extension
- Calls appropriate extraction method:
  
  **For PDF files**:
  - Uses `PyPDF2` to extract text from each page
  - Normalizes whitespace
  
  **For DOCX files**:
  - Primary: Uses `python-docx` to extract:
    - Paragraphs
    - Tables
    - Headers and footers
  - Fallback: Uses Apache Tika for comprehensive extraction
  
  **For DOC files** (older Word format):
  - Method 1: Apache Tika (PRIMARY - requires Java)
  - Method 2: LibreOffice headless conversion
  - Method 3: antiword command-line tool
  - Method 4: python-docx fallback
  - Method 5: olefile (basic binary extraction)
  
  **For TXT files**:
  - Direct UTF-8 decoding

- Returns extracted text as string
- **Memory Cleanup**: Deletes file_content from memory if enabled

#### **6. Text Validation**
- Checks if extracted text is sufficient (minimum 50 characters)
- Truncates text if exceeds max length (50,000 chars by default)
- If insufficient, updates status to `failed:insufficient_text` and raises error

#### **7. Module Selection** (`_parse_extract_modules()`)
- Parses `extract_modules` parameter:
  - `"all"` → Extract all 8 modules
  - `"1,2,3"` or `"designation,name,email"` → Extract specific modules
- Returns set of modules to extract

#### **8. Sequential Information Extraction** (8 Modules)

Each module follows the same pattern:
1. **Extractor** calls OLLAMA LLM with specialized prompt
2. **LLM** returns JSON with extracted data
3. **Service** updates database record with extracted value

**Module 1: Designation** (`app/designation/`)
- Extractor: `designation_extractor.py`
- Service: `designation_service.py`
- Updates: `resume_metadata.designation`
- Prompt: Extracts current/most recent job title

**Module 2: Name** (`app/name/`)
- Extractor: `name_extractor.py`
- Service: `name_service.py`
- Updates: `resume_metadata.candidatename`
- Prompt: Extracts candidate's full name

**Module 3: Email** (`app/email/`)
- Extractor: `email_extractor.py`
- Service: `email_service.py`
- Updates: `resume_metadata.email`
- Prompt: Extracts email address

**Module 4: Mobile** (`app/mobile/`)
- Extractor: `mobile_extractor.py`
- Service: `mobile_service.py`
- Updates: `resume_metadata.mobile`
- Prompt: Extracts phone number

**Module 5: Experience** (`app/experience/`)
- Extractor: `experience_extractor.py`
- Service: `experience_service.py`
- Updates: `resume_metadata.experience`
- Prompt: Extracts years of experience

**Module 6: Domain** (`app/domain/`)
- Extractor: `domain_extractor.py`
- Service: `domain_service.py`
- Updates: `resume_metadata.domain`
- Prompt: Extracts industry domain

**Module 7: Education** (`app/education/`)
- Extractor: `education_extractor.py`
- Service: `education_service.py`
- Updates: `resume_metadata.education`
- Prompt: Extracts educational qualifications

**Module 8: Skills** (`app/skills/`)
- Extractor: `skills_extractor.py`
- Service: `skills_service.py`
- Updates: `resume_metadata.skillset`
- Prompt: Extracts technical skills (comma-separated)

**Important**: Each extraction is **isolated** - uses fresh HTTP client and LLM context to prevent context bleeding.

#### **9. OLLAMA LLM Interaction** (Each Extractor)
- Creates HTTP client with timeout
- Checks OLLAMA connection (`/api/tags`)
- Sends POST request to `/api/generate` or `/api/chat`:
  ```json
  {
    "model": "llama3.1",
    "prompt": "<specialized prompt> + resume_text",
    "stream": false,
    "options": {
      "temperature": 0.1,
      "top_p": 0.9
    }
  }
  ```
- Parses JSON response from LLM
- Handles errors gracefully (returns `None` if extraction fails)

#### **10. Database Updates** (After Each Extraction)
- Each service calls `resume_repo.update(resume_id, {field: value})`
- Repository:
  - Fetches latest record from database
  - Updates specified fields
  - Commits transaction
  - Refreshes object from database

#### **11. Status Update**
- After all extractions complete:
  - Updates `status = "completed"`
  - Logs all extracted fields
  - Refreshes `resume_metadata` object from database

#### **12. Response Building** (`app/models/resume_models.py`)
- Creates `ResumeUploadResponse` object with:
  - All extracted fields
  - Status
  - Timestamps
- Returns JSON response to client

#### **13. Error Handling**
If any step fails:
- Updates status to `failed:<reason>`:
  - `failed:file_too_large`
  - `failed:invalid_file_type`
  - `failed:empty_file`
  - `failed:insufficient_text`
  - `failed:extraction_error`
  - `failed:database_error`
  - `failed:unknown_error`
- Logs error with context
- Returns appropriate HTTP status code

---

## 📁 File Responsibilities

### **Core Application Files**

#### `app/main.py`
**Responsibility**: FastAPI application bootstrap and lifecycle management
- Creates FastAPI app instance
- Configures CORS middleware
- Sets up exception handlers (validation errors, global errors)
- Manages application lifespan:
  - **Startup**: Initializes database connection pool, vector database
  - **Shutdown**: Closes database connections
- Includes API router
- Initializes Sentry (error monitoring) if configured
- Root endpoint for health checks

#### `app/config.py`
**Responsibility**: Configuration management using Pydantic Settings
- Loads environment variables from `.env` file
- Validates required configuration (MySQL credentials)
- Provides settings for:
  - Database connection (MySQL host, user, password, database, port)
  - Chunking (chunk size, overlap)
  - Embeddings (dimension, batch size)
  - File limits (max file size, max text length)
  - OLLAMA (host, API key)
  - Pinecone (API key, index name, region)
  - Monitoring (Sentry DSN, log level)
- Generates MySQL connection URL with URL encoding

---

### **API Layer**

#### `app/api/routes.py`
**Responsibility**: HTTP endpoint definitions and request routing
- Defines API routes:
  - `POST /api/v1/upload-resume`: Upload and parse resume
  - `POST /api/v1/create-job`: Create job posting
  - `POST /api/v1/match`: Match resumes to jobs
  - `GET /api/v1/health`: Health check
- Creates dependency injection factories:
  - `get_resume_controller()`: Creates ResumeController with dependencies
  - `get_job_controller()`: Creates JobController with dependencies
- Validates request models using Pydantic
- Returns response models

---

### **Controller Layer**

#### `app/controllers/resume_controller.py`
**Responsibility**: Orchestrates resume upload and processing workflow
- **Main Method**: `upload_resume()`
  - Validates file (type, size, empty check)
  - Checks for duplicate files
  - Creates initial database record
  - Extracts text from file
  - Parses `extract_modules` parameter
  - Calls extraction services sequentially
  - Updates status to "completed"
  - Builds and returns response
- **Helper Method**: `_parse_extract_modules()`
  - Parses module selection string ("all", "1,2,3", "designation,skills")
  - Maps numbers to module names
  - Returns set of modules to extract
- **Error Handling**: Updates status to failed with reason on errors
- **Memory Management**: Cleans up file content from memory

#### `app/controllers/job_controller.py`
**Responsibility**: Handles job posting and matching operations
- `create_job()`: Creates job posting, generates embeddings
- `match_job()`: Matches resumes to job descriptions using vector similarity

---

### **Service Layer**

#### `app/services/resume_parser.py`
**Responsibility**: Extracts text from resume files
- **Main Method**: `extract_text(file_content, filename)`
  - Determines file type from extension
  - Routes to appropriate extraction method
- **PDF Extraction**: `_extract_pdf_text()`
  - Uses PyPDF2 to read pages
  - Normalizes whitespace
- **DOCX Extraction**: `_extract_docx_text()`
  - Primary: python-docx (paragraphs, tables, headers, footers)
  - Fallback: Apache Tika
- **DOC Extraction**: `_extract_doc_text()`
  - Multiple fallback methods (Tika, LibreOffice, antiword, python-docx, olefile)
- **TXT Extraction**: Direct UTF-8 decoding
- **Error Handling**: Raises ValueError with helpful messages

#### `app/services/fileconverter.py`
**Responsibility**: File format conversion utilities
- `doc_to_docx_with_libreoffice()`: Converts .doc to .docx using LibreOffice
- `doc_to_docx_with_pandoc()`: Converts .docx using pandoc (note: pandoc doesn't support .doc)
- `convert_doc_to_docx_in_memory()`: In-memory conversion
- `doc_to_text_with_pandoc()`: Converts .doc to plain text

#### `app/services/embedding_service.py`
**Responsibility**: Generates embeddings for text chunks
- Chunks resume text into smaller pieces
- Calls OLLAMA embedding API
- Returns embeddings for vector database storage
- **Note**: Currently disabled in resume upload flow

#### `app/services/vector_db_service.py`
**Responsibility**: Manages vector database operations
- Supports Pinecone (cloud) and FAISS (local) backends
- `upsert_vectors()`: Stores embeddings
- `query_vectors()`: Searches for similar vectors
- **Note**: Currently not used in resume upload flow

#### `app/services/job_parser.py`
**Responsibility**: Parses job descriptions
- Extracts structured information from job postings
- Used in job matching functionality

#### `app/services/job_cache.py`
**Responsibility**: Caches job embeddings
- Reduces redundant embedding generation
- LRU cache implementation

---

### **Extraction Modules** (8 Modules)

Each module has the same structure:
- **`*_extractor.py`**: Calls OLLAMA LLM with specialized prompt
- **`*_service.py`**: Orchestrates extraction and database update

#### `app/designation/`
- **`designation_extractor.py`**: Extracts job title using LLM
- **`designation_service.py`**: Updates `resume_metadata.designation`

#### `app/name/`
- **`name_extractor.py`**: Extracts candidate name
- **`name_service.py`**: Updates `resume_metadata.candidatename`

#### `app/email/`
- **`email_extractor.py`**: Extracts email address
- **`email_service.py`**: Updates `resume_metadata.email`

#### `app/mobile/`
- **`mobile_extractor.py`**: Extracts phone number
- **`mobile_service.py`**: Updates `resume_metadata.mobile`

#### `app/experience/`
- **`experience_extractor.py`**: Extracts years of experience
- **`experience_service.py`**: Updates `resume_metadata.experience`

#### `app/domain/`
- **`domain_extractor.py`**: Extracts industry domain
- **`domain_service.py`**: Updates `resume_metadata.domain`

#### `app/education/`
- **`education_extractor.py`**: Extracts educational qualifications
- **`education_service.py`**: Updates `resume_metadata.education`

#### `app/skills/`
- **`skills_extractor.py`**: Extracts technical skills
- **`skills_service.py`**: Updates `resume_metadata.skillset` (comma-separated)

**Common Pattern**:
1. Extractor creates HTTP client
2. Sends prompt + resume text to OLLAMA
3. Parses JSON response
4. Service updates database record
5. Returns extracted value or `None`

---

### **Repository Layer**

#### `app/repositories/resume_repo.py`
**Responsibility**: Database CRUD operations for resume metadata
- **`create(resume_data)`**: Creates new resume record
- **`get_by_id(resume_id)`**: Fetches resume by ID
- **`get_all(limit, offset)`**: Lists all resumes with pagination
- **`get_by_filename(filename)`**: Finds resume by filename (duplicate check)
- **`update(resume_id, update_data)`**: Updates specific fields
  - Thread-safe: Reads latest state, updates, commits
- **`get_resumes_with_null_email_or_mobile()`**: Utility for reprocessing

---

### **Database Layer**

#### `app/database/connection.py`
**Responsibility**: Database connection pool and session management
- Creates async SQLAlchemy engine with connection pooling
- Creates async session factory
- **`get_db_session()`**: Dependency for FastAPI dependency injection
- **`init_db()`**: Tests database connection on startup
- **`close_db()`**: Closes connections on shutdown

#### `app/database/models.py`
**Responsibility**: SQLAlchemy ORM models
- **`ResumeMetadata`**: Database table model
  - Fields: `id`, `candidatename`, `jobrole`, `designation`, `experience`, `domain`, `mobile`, `email`, `education`, `filename`, `skillset`, `status`, `created_at`, `updated_at`
  - Table name: `resume_metadata`

---

### **Models Layer** (Pydantic)

#### `app/models/resume_models.py`
**Responsibility**: Request/response validation models
- **`ResumeUpload`**: Request model for upload metadata
- **`ResumeUploadResponse`**: Response model with all extracted fields

#### `app/models/job_models.py`
**Responsibility**: Job-related models
- **`JobCreate`**: Request model for job creation
- **`JobCreateResponse`**: Response model
- **`MatchRequest`**: Request model for job matching
- **`MatchResponse`**: Response model with matches

---

### **Utilities**

#### `app/utils/logging.py`
**Responsibility**: Structured logging setup
- Configures JSON-formatted logging
- Sets log level from environment
- Provides `get_logger()` function

#### `app/utils/safe_logger.py`
**Responsibility**: Prevents LogRecord attribute conflicts
- `safe_extra()`: Sanitizes extra fields for logging
- Prevents conflicts with reserved LogRecord attributes (e.g., `filename`)

#### `app/utils/cleaning.py`
**Responsibility**: Data cleaning utilities
- `normalize_text()`: Removes extra whitespace
- `sanitize_filename()`: Cleans filename for safe storage

---

### **Constants**

#### `app/constants/resume_status.py`
**Responsibility**: Status constants and helpers
- Status values: `pending`, `processing`, `completed`, `failed`
- Failure reasons: `file_too_large`, `invalid_file_type`, etc.
- `get_failure_status()`: Combines status with reason
- `parse_failure_status()`: Parses status string

---

### **Database Migrations**

#### `alembic/`
**Responsibility**: Database schema version control
- **`env.py`**: Alembic configuration
- **`versions/001_initial_resume_metadata.py`**: Creates initial table
- **`versions/002_add_designation_field.py`**: Adds designation column
- **`versions/003_add_status_field.py`**: Adds status column

#### `alembic.ini`
**Responsibility**: Alembic configuration file

---

### **Configuration Files**

#### `requirements.txt`
**Responsibility**: Python package dependencies
- FastAPI, Uvicorn
- SQLAlchemy, aiomysql
- OLLAMA client
- File parsing: PyPDF2, python-docx, tika, olefile
- Vector DB: pinecone-client, faiss-cpu
- Monitoring: sentry-sdk

#### `.env` (not in repo, example provided)
**Responsibility**: Environment variables
- Database credentials
- OLLAMA host
- Pinecone API key (optional)
- File size limits
- Log level

---

### **Scripts**

#### `1start_api.sh` / `1start_api.bat`
**Responsibility**: Starts the FastAPI server
- Activates virtual environment
- Sets environment variables
- Runs `uvicorn app.main:app`

#### `2upload_resume_test.sh` / `2upload_resume_test.bat`
**Responsibility**: Test script for resume upload
- Sends sample resume file to API
- Tests the upload endpoint

---

### **Legacy Files**

#### `ResumeParser.py`
**Responsibility**: Standalone resume parser (legacy, not used in main flow)
- Original single-file parser for Google Colab
- Can be used independently
- Currently commented out

---

## 🔑 Key Components Deep Dive

### **1. OLLAMA Integration**
- **Purpose**: Provides LLM capabilities for information extraction
- **Model**: llama3.1 (default)
- **API Endpoints**: `/api/generate` or `/api/chat`
- **Configuration**: `OLLAMA_HOST` environment variable
- **Error Handling**: Falls back to HTTP API if Python client unavailable

### **2. Database Schema**
```sql
CREATE TABLE resume_metadata (
    id INT PRIMARY KEY AUTO_INCREMENT,
    candidatename VARCHAR(255),
    jobrole VARCHAR(255),
    designation VARCHAR(255),
    experience VARCHAR(100),
    domain VARCHAR(255),
    mobile VARCHAR(50),
    email VARCHAR(255),
    education TEXT,
    filename VARCHAR(512) NOT NULL,
    skillset TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

### **3. Status Flow**
```
pending → processing → completed
                    ↓
                 failed:<reason>
```

### **4. Memory Optimization**
- File content deleted after text extraction
- Text truncated if exceeds limit
- Garbage collection after large operations
- Configurable via `ENABLE_MEMORY_CLEANUP`

### **5. Error Recovery**
- Each extraction module is independent
- If one module fails, others continue
- Database record always created (even on failure)
- Status updated to reflect failure reason

---

## 📊 Data Flow Summary

```
Client Request
    ↓
API Route (routes.py)
    ↓
Controller (resume_controller.py)
    ↓
[File Validation] → [Duplicate Check] → [Create DB Record]
    ↓
Resume Parser (resume_parser.py)
    ↓
[Text Extraction: PDF/DOCX/DOC/TXT]
    ↓
[Module Selection]
    ↓
[Sequential Extraction - 8 Modules]
    ├─→ Designation Extractor → OLLAMA → DB Update
    ├─→ Name Extractor → OLLAMA → DB Update
    ├─→ Email Extractor → OLLAMA → DB Update
    ├─→ Mobile Extractor → OLLAMA → DB Update
    ├─→ Experience Extractor → OLLAMA → DB Update
    ├─→ Domain Extractor → OLLAMA → DB Update
    ├─→ Education Extractor → OLLAMA → DB Update
    └─→ Skills Extractor → OLLAMA → DB Update
    ↓
[Status Update: completed]
    ↓
[Build Response]
    ↓
Client Response (JSON)
```

---

## 🎯 Key Design Decisions

1. **Sequential Extraction**: Modules run one-by-one to prevent LLM context bleeding
2. **Early DB Record Creation**: Record created before extraction for error tracking
3. **Isolated HTTP Clients**: Each extraction uses fresh client for isolation
4. **Graceful Degradation**: If one module fails, others continue
5. **Status Tracking**: Detailed status with failure reasons for debugging
6. **Memory Management**: Proactive cleanup for large file handling
7. **Duplicate Prevention**: Filename-based duplicate detection

---

## 🔧 Configuration

All configuration is environment-based via `.env` file:
- Database: `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`
- OLLAMA: `OLLAMA_HOST` (default: http://localhost:11434)
- File Limits: `MAX_FILE_SIZE_MB` (default: 10), `MAX_RESUME_TEXT_LENGTH` (default: 50000)
- Memory: `ENABLE_MEMORY_CLEANUP` (default: true)

---

## 📝 Summary

This ATS Parser is a **production-ready, enterprise-grade resume parsing system** that:
- Accepts resume files in multiple formats
- Extracts 8 types of structured information using LLM
- Stores data in MySQL with full status tracking
- Handles errors gracefully with detailed logging
- Optimizes memory usage for large files
- Provides RESTful API for integration

The architecture follows **clean architecture principles** with clear separation of concerns:
- **API Layer**: HTTP handling
- **Controller Layer**: Business logic orchestration
- **Service Layer**: Domain logic
- **Repository Layer**: Data access
- **Database Layer**: Persistence

Each component has a single, well-defined responsibility, making the system maintainable and extensible.

