# Memory Optimization Guide

This document describes the memory optimizations implemented for low-memory systems.

## Overview

The application has been optimized to work efficiently on systems with limited memory by:
- Processing embeddings in smaller batches
- Implementing explicit memory cleanup
- Limiting file sizes and text lengths
- Using LRU cache eviction
- Adding garbage collection hints

## Configuration Settings

Add these environment variables to your `.env` file to control memory usage:

```bash
# Memory Optimization Settings
EMBEDDING_BATCH_SIZE=5          # Number of chunks to process at once (default: 5)
MAX_FILE_SIZE_MB=10              # Maximum file size in MB (default: 10)
MAX_RESUME_TEXT_LENGTH=50000      # Maximum resume text length in characters (default: 50000)
JOB_CACHE_MAX_SIZE=100            # Maximum number of jobs in cache (default: 100)
ENABLE_MEMORY_CLEANUP=true       # Enable automatic memory cleanup (default: true)
```

## Optimizations Implemented

### 1. Batch Processing for Embeddings
- **File**: `app/services/embedding_service.py`
- **Change**: Embeddings are now processed in configurable batches instead of all at once
- **Benefit**: Reduces peak memory usage during embedding generation
- **Config**: `EMBEDDING_BATCH_SIZE` (default: 5 chunks per batch)

### 2. File Size Limits
- **Files**: `app/controllers/resume_controller.py`, `process_all_resumes.py`
- **Change**: Files larger than the configured limit are rejected
- **Benefit**: Prevents processing extremely large files that could exhaust memory
- **Config**: `MAX_FILE_SIZE_MB` (default: 10MB)

### 3. Text Length Limits
- **Files**: `app/controllers/resume_controller.py`, `process_all_resumes.py`
- **Change**: Resume text is truncated to a maximum length
- **Benefit**: Prevents excessive memory usage from very long resumes
- **Config**: `MAX_RESUME_TEXT_LENGTH` (default: 50000 characters)

### 4. LRU Cache with Size Limits
- **File**: `app/services/job_cache.py`
- **Change**: Job cache now uses LRU (Least Recently Used) eviction with configurable size limit
- **Benefit**: Prevents unbounded memory growth from cached job embeddings
- **Config**: `JOB_CACHE_MAX_SIZE` (default: 100 jobs)

### 5. Explicit Memory Cleanup
- **Files**: All processing files
- **Change**: Explicit garbage collection after processing batches and files
- **Benefit**: Frees memory immediately after use instead of waiting for automatic GC
- **Config**: `ENABLE_MEMORY_CLEANUP` (default: true)

### 6. Early Memory Release
- **Files**: `app/controllers/resume_controller.py`, `process_all_resumes.py`
- **Change**: File content is deleted from memory immediately after text extraction
- **Benefit**: Reduces memory footprint during processing
- **Config**: Controlled by `ENABLE_MEMORY_CLEANUP`

## Recommended Settings for Low-Memory Systems

For systems with **4GB RAM or less**, use these settings:

```bash
EMBEDDING_BATCH_SIZE=3
MAX_FILE_SIZE_MB=5
MAX_RESUME_TEXT_LENGTH=30000
JOB_CACHE_MAX_SIZE=50
ENABLE_MEMORY_CLEANUP=true
```

For systems with **8GB RAM**, use these settings:

```bash
EMBEDDING_BATCH_SIZE=5
MAX_FILE_SIZE_MB=10
MAX_RESUME_TEXT_LENGTH=50000
JOB_CACHE_MAX_SIZE=100
ENABLE_MEMORY_CLEANUP=true
```

## Monitoring Memory Usage

To monitor memory usage, check your application logs. The system logs:
- Batch processing information
- File size warnings
- Cache eviction events
- Memory cleanup operations

## Performance Impact

These optimizations may slightly increase processing time due to:
- Batch processing overhead
- Garbage collection pauses
- Additional size checks

However, the trade-off is necessary for systems with limited memory to prevent:
- Out-of-memory errors
- System crashes
- Performance degradation

## Troubleshooting

If you encounter memory issues:

1. **Reduce batch size**: Set `EMBEDDING_BATCH_SIZE=2` or `1`
2. **Reduce file size limit**: Set `MAX_FILE_SIZE_MB=5`
3. **Reduce text length**: Set `MAX_RESUME_TEXT_LENGTH=30000`
4. **Reduce cache size**: Set `JOB_CACHE_MAX_SIZE=25`
5. **Ensure cleanup is enabled**: Set `ENABLE_MEMORY_CLEANUP=true`

## Notes

- The resume parser already limits text sent to LLM to 10,000 characters (hardcoded)
- Vector database operations (FAISS/Pinecone) are not optimized here as they handle their own memory management
- Database connections use connection pooling which is already memory-efficient

