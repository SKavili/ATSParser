"""
Script to process all resumes from the Resumes directory one by one.
Extracts designation for each resume and saves to database.
"""
import asyncio
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.services.resume_parser import ResumeParser
from app.services.designation_service import DesignationService
from app.repositories.resume_repo import ResumeRepository
from app.database.connection import async_session_maker
from app.utils.cleaning import sanitize_filename
from app.utils.logging import get_logger

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
        "error": None
    }
    
    try:
        # Read file content
        print(f"\n[1/4] Reading file: {filename}")
        file_content = file_path.read_bytes()
        if not file_content:
            raise ValueError("Empty file")
        print(f"      ✅ File read successfully ({len(file_content)} bytes)")
        
        # Extract text from file
        print(f"\n[2/4] Extracting text from file...")
        resume_text = await resume_parser.extract_text(file_content, safe_filename)
        
        if not resume_text or len(resume_text.strip()) < 50:
            raise ValueError("Could not extract sufficient text from resume")
        
        print(f"      ✅ Text extracted successfully ({len(resume_text)} characters)")
        print(f"      Preview: {resume_text[:150]}...")
        
        # Create database record
        print(f"\n[3/4] Creating database record...")
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
        }
        
        # Get database session
        async with async_session_maker() as session:
            resume_repo = ResumeRepository(session)
            resume_metadata = await resume_repo.create(db_record)
            result["resume_id"] = resume_metadata.id
            print(f"      ✅ Database record created (ID: {resume_metadata.id})")
            
            # Extract and save designation
            print(f"\n[4/4] Extracting designation using OLLAMA...")
            designation_service = DesignationService(session)
            designation = await designation_service.extract_and_save_designation(
                resume_text=resume_text,
                resume_id=resume_metadata.id,
                filename=safe_filename
            )
            
            # Refresh to get updated designation
            await session.refresh(resume_metadata)
            result["designation"] = resume_metadata.designation
            result["success"] = True
            
            if designation:
                print(f"      ✅ Designation extracted and saved: '{designation}'")
            else:
                print(f"      ⚠️  No designation found (saved as NULL)")
        
        print(f"\n✅ SUCCESS: {filename} processed successfully")
        print(f"   Resume ID: {result['resume_id']}")
        print(f"   Designation: {result['designation']}")
        
    except Exception as e:
        result["error"] = str(e)
        result["success"] = False
        print(f"\n❌ ERROR processing {filename}: {e}")
        logger.error(
            f"Error processing resume {filename}: {e}",
            extra={"file_name": filename, "error": str(e), "error_type": type(e).__name__},  # Use file_name instead of filename (reserved in LogRecord)
            exc_info=True
        )
    
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
        except Exception as e:
            print(f"\n❌ FATAL ERROR processing {file_path.name}: {e}")
            results.append({
                "filename": file_path.name,
                "success": False,
                "resume_id": None,
                "designation": None,
                "error": str(e)
            })
    
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
        status = "✅" if result["success"] else "❌"
        print(f"{i}. {status} {result['filename']}")
        if result["success"]:
            print(f"   Resume ID: {result['resume_id']}")
            print(f"   Designation: {result['designation'] or 'NULL'}")
        else:
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

