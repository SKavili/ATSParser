"""
Script to process all resumes from the Resumes directory one by one.
Extracts designation for each resume and saves to database.
Memory-optimized for low-memory systems.
"""
import asyncio
import gc
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.services.resume_parser import ResumeParser
from app.designation import DesignationService
from app.repositories.resume_repo import ResumeRepository
from app.database.connection import async_session_maker
from app.utils.cleaning import sanitize_filename
from app.utils.logging import get_logger
from app.config import settings
from app.constants.resume_status import (
    STATUS_PENDING, STATUS_PROCESSING, STATUS_COMPLETED,
    get_failure_status, FAILURE_FILE_TOO_LARGE, FAILURE_EMPTY_FILE,
    FAILURE_INSUFFICIENT_TEXT, FAILURE_EXTRACTION_ERROR,
    FAILURE_DESIGNATION_EXTRACTION_FAILED, FAILURE_DATABASE_ERROR, FAILURE_UNKNOWN_ERROR
)

logger = get_logger(__name__)

RESUMES_DIR = Path(__file__).parent / "app" / "Resumes"
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc'}  # Only PDF, DOCX, and DOC files


async def process_single_resume(
    file_path: Path,
    resume_parser: ResumeParser,
    designation_service: Optional[DesignationService],
    resume_repo: Optional[ResumeRepository],
    file_number: int,
    total_files: int
) -> dict:
    """
    Process a single resume file.
    
    Returns:
        dict with processing result: {
            "filename": str,
            "success": bool,
            "resume_id": int | None,
            "designation": str | None,
            "error": str | None
        }
    """
    filename = file_path.name
    safe_filename = sanitize_filename(filename)
    
    print(f"\n{'='*80}")
    print(f"📄 PROCESSING FILE {file_number}/{total_files}: {filename}")
    print(f"{'='*80}")
    
    result = {
        "filename": filename,
        "success": False,
        "resume_id": None,
        "designation": None,
        "status": None,
        "error": None
    }
    
    # Get database session early to handle failures
    # Session will be automatically closed when exiting the context manager
    async with async_session_maker() as session:
        resume_repo = ResumeRepository(session)
        resume_metadata = None
        
        try:
            # Read file content with size check
            print(f"\n[1/4] Reading file: {filename}")
            file_content = file_path.read_bytes()
            if not file_content:
                # Create record with failed status
                db_record = {
                    "candidatename": None,
                    "jobrole": None,
                    "designation": None,
                    "experience": None,
                    "domain": None,
                    "mobile": None,
                    "email": None,
                    "education": None,
                    "filename": safe_filename,
                    "skillset": "",
                    "status": get_failure_status(FAILURE_EMPTY_FILE),
                }
                resume_metadata = await resume_repo.create(db_record)
                result["resume_id"] = resume_metadata.id
                raise ValueError("Empty file")
            
            # Check file size limit
            file_size_mb = len(file_content) / (1024 * 1024)
            if file_size_mb > settings.max_file_size_mb:
                # Create record with failed status
                db_record = {
                    "candidatename": None,
                    "jobrole": None,
                    "designation": None,
                    "experience": None,
                    "domain": None,
                    "mobile": None,
                    "email": None,
                    "education": None,
                    "filename": safe_filename,
                    "skillset": "",
                    "status": get_failure_status(FAILURE_FILE_TOO_LARGE),
                }
                resume_metadata = await resume_repo.create(db_record)
                result["resume_id"] = resume_metadata.id
                raise ValueError(f"File too large: {file_size_mb:.2f}MB. Maximum: {settings.max_file_size_mb}MB")
            
            print(f"      ✅ File read successfully ({len(file_content)} bytes, {file_size_mb:.2f}MB)")
            
            # Create database record with processing status
            print(f"\n[2/4] Creating database record...")
            db_record = {
                "candidatename": None,
                "jobrole": None,
                "designation": None,  # Will be extracted and updated
                "experience": None,
                "domain": None,
                "mobile": None,
                "email": None,
                "education": None,
                "filename": safe_filename,
                "skillset": "",
                "status": STATUS_PROCESSING,  # Set to processing
            }
            resume_metadata = await resume_repo.create(db_record)
            result["resume_id"] = resume_metadata.id
            print(f"      ✅ Database record created (ID: {resume_metadata.id})")
            
            # Extract text from file
            print(f"\n[3/4] Extracting text from file...")
            try:
                resume_text = await resume_parser.extract_text(file_content, safe_filename)
            except Exception as e:
                # Update status to failed for extraction error
                await resume_repo.update(
                    resume_metadata.id,
                    {"status": get_failure_status(FAILURE_EXTRACTION_ERROR)}
                )
                raise ValueError(f"Text extraction failed: {str(e)}")
            
            # Clear file_content from memory early
            if settings.enable_memory_cleanup:
                del file_content
                gc.collect()
            
            if not resume_text or len(resume_text.strip()) < 50:
                # Update status to failed for insufficient text
                await resume_repo.update(
                    resume_metadata.id,
                    {"status": get_failure_status(FAILURE_INSUFFICIENT_TEXT)}
                )
                raise ValueError("Could not extract sufficient text from resume")
            
            # Limit text length
            if len(resume_text) > settings.max_resume_text_length:
                logger.warning(
                    f"Resume text truncated from {len(resume_text)} to {settings.max_resume_text_length}",
                    extra={"file_name": filename, "original_length": len(resume_text)}
                )
                resume_text = resume_text[:settings.max_resume_text_length]
            
            print(f"      ✅ Text extracted successfully ({len(resume_text)} characters)")
            print(f"      Preview: {resume_text[:150]}...")
            
            # Extract and save designation
            print(f"\n[4/4] Extracting designation using OLLAMA...")
            designation_service = DesignationService(session)
            try:
                designation = await designation_service.extract_and_save_designation(
                    resume_text=resume_text,
                    resume_id=resume_metadata.id,
                    filename=safe_filename
                )
            except Exception as e:
                # Update status to failed for designation extraction error
                await resume_repo.update(
                    resume_metadata.id,
                    {"status": get_failure_status(FAILURE_DESIGNATION_EXTRACTION_FAILED)}
                )
                logger.error(
                    f"Designation extraction failed for resume {resume_metadata.id}: {e}",
                    extra={"resume_id": resume_metadata.id, "file_name": filename, "error": str(e)}
                )
                # Don't fail the whole process, just log it
                designation = None
            
            # Update status to completed on success
            await resume_repo.update(
                resume_metadata.id,
                {"status": STATUS_COMPLETED}
            )
            
            # Refresh to get updated designation
            await session.refresh(resume_metadata)
            result["designation"] = resume_metadata.designation
            result["success"] = True
            
            if designation:
                print(f"      ✅ Designation extracted and saved: '{designation}'")
            else:
                print(f"      ⚠️  No designation found (saved as NULL)")
            
            # Memory cleanup after processing each file
            if settings.enable_memory_cleanup:
                del resume_text
                gc.collect()
            
            result["status"] = resume_metadata.status
            print(f"\n✅ SUCCESS: {filename} processed successfully")
            print(f"   Resume ID: {result['resume_id']}")
            print(f"   Designation: {result['designation']}")
            print(f"   Status: {result['status']}")
            
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
            
            # Update status to failed if we have a record
            if resume_metadata and resume_metadata.id:
                try:
                    # Determine failure reason from error message
                    error_str = str(e).lower()
                    if "too large" in error_str or "file size" in error_str:
                        failure_reason = FAILURE_FILE_TOO_LARGE
                    elif "empty" in error_str:
                        failure_reason = FAILURE_EMPTY_FILE
                    elif "insufficient" in error_str or "extract" in error_str:
                        failure_reason = FAILURE_INSUFFICIENT_TEXT
                    elif "extraction" in error_str:
                        failure_reason = FAILURE_EXTRACTION_ERROR
                    else:
                        failure_reason = FAILURE_UNKNOWN_ERROR
                    
                    await resume_repo.update(
                        resume_metadata.id,
                        {"status": get_failure_status(failure_reason)}
                    )
                    await session.refresh(resume_metadata)
                    result["status"] = resume_metadata.status
                except Exception as update_error:
                    logger.error(
                        f"Failed to update status after error: {update_error}",
                        extra={"resume_id": resume_metadata.id if resume_metadata else None, "error": str(update_error)}
                    )
            
            print(f"\n❌ ERROR processing {filename}: {e}")
            if resume_metadata and resume_metadata.id:
                print(f"   Resume ID: {result['resume_id']}")
                print(f"   Status: {result.get('status', 'unknown')}")
            logger.error(
                f"Error processing resume {filename}: {e}",
                extra={"file_name": filename, "error": str(e), "error_type": type(e).__name__},
                exc_info=True
            )
        # Session is automatically closed by async with context manager
        # Connection is returned to pool immediately when exiting the context
    
    return result


async def process_all_resumes():
    """Process all resumes in the Resumes directory one by one."""
    
    print("="*80)
    print("BULK RESUME PROCESSING - Designation Extraction")
    print("="*80)
    print(f"\nResumes Directory: {RESUMES_DIR}")
    print(f"Supported file types: PDF, DOCX, DOC")
    print(f"Files will be processed ONE BY ONE")
    
    # Check if directory exists
    if not RESUMES_DIR.exists():
        print(f"\n❌ ERROR: Resumes directory not found at: {RESUMES_DIR}")
        print("   Please create the directory and add resume files.")
        return
    
    # Find all resume files
    resume_files: List[Path] = []
    for ext in ALLOWED_EXTENSIONS:
        resume_files.extend(RESUMES_DIR.glob(f"*{ext}"))
        resume_files.extend(RESUMES_DIR.glob(f"*{ext.upper()}"))
    
    # Remove duplicates and sort
    resume_files = sorted(set(resume_files))
    
    if not resume_files:
        print(f"\n⚠️  No resume files found in: {RESUMES_DIR}")
        print(f"   Supported formats: {', '.join(ALLOWED_EXTENSIONS)}")
        return
    
    print(f"\n📁 Found {len(resume_files)} resume file(s) (PDF, DOCX, DOC only):")
    for i, file_path in enumerate(resume_files, 1):
        print(f"   {i}. {file_path.name}")
    
    print(f"\n{'='*80}")
    print("STARTING PROCESSING...")
    print(f"Processing files ONE BY ONE: PDF, DOCX, DOC")
    print(f"{'='*80}\n")
    
    # Initialize services
    resume_parser = ResumeParser()
    
    # Process each file one by one
    results = []
    total_files = len(resume_files)
    
    for file_number, file_path in enumerate(resume_files, 1):
        try:
            result = await process_single_resume(
                file_path=file_path,
                resume_parser=resume_parser,
                designation_service=None,  # Will be created in process_single_resume
                resume_repo=None,  # Will be created in process_single_resume
                file_number=file_number,
                total_files=total_files
            )
            results.append(result)
            
            # Memory cleanup between files
            if settings.enable_memory_cleanup:
                gc.collect()
            
            # Wait 30 seconds before processing next file (for system cooling)
            # This prevents CPU/GPU overheating and ensures accurate LLM extractions
            if file_number < total_files:  # Don't wait after the last file
                print(f"\n⏳ Waiting 30 seconds before processing next file (for system cooling)...")
                await asyncio.sleep(30)
        except Exception as e:
            print(f"\n❌ FATAL ERROR processing {file_path.name}: {e}")
            results.append({
                "filename": file_path.name,
                "success": False,
                "resume_id": None,
                "designation": None,
                "error": str(e)
            })
            # Cleanup on error too
            if settings.enable_memory_cleanup:
                gc.collect()
            
            # Wait 30 seconds before processing next file (for system cooling)
            # This prevents CPU/GPU overheating and ensures accurate LLM extractions
            if file_number < total_files:  # Don't wait after the last file
                print(f"\n⏳ Waiting 30 seconds before processing next file (for system cooling)...")
                await asyncio.sleep(30)
    
    # Print summary
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}\n")
    
    success_count = sum(1 for r in results if r["success"])
    failed_count = len(results) - success_count
    
    print(f"Total files processed: {len(results)}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"\n{'='*80}")
    print("DETAILED RESULTS")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results, 1):
        status_icon = "✅" if result["success"] else "❌"
        print(f"{i}. {status_icon} {result['filename']}")
        if result["success"]:
            print(f"   Resume ID: {result['resume_id']}")
            print(f"   Designation: {result['designation'] or 'NULL'}")
            print(f"   Status: {result.get('status', 'unknown')}")
        else:
            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   Error: {result['error']}")
        print()
    
    print(f"{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        asyncio.run(process_all_resumes())
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

