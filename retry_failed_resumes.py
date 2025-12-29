"""
Script to automatically retry all resumes with failed:insufficient_text status.
This script finds all failed resumes and retries them with OCR.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.database.connection import async_session_maker
from app.repositories.resume_repo import ResumeRepository
from app.services.resume_parser import ResumeParser
from app.services.embedding_service import EmbeddingService
from app.services.vector_db_service import get_vector_db_service
from app.controllers.resume_controller import ResumeController
from app.constants.resume_status import FAILURE_INSUFFICIENT_TEXT
from app.utils.logging import get_logger

logger = get_logger(__name__)


async def retry_all_failed_resumes(extract_modules: str = "all", limit: int = None):
    """
    Retry all resumes with failed:insufficient_text status.
    
    Args:
        extract_modules: Modules to extract (default: "all")
        limit: Optional limit on number of resumes to retry
    """
    async with async_session_maker() as session:
        resume_repo = ResumeRepository(session)
        vector_db = await get_vector_db_service()
        
        # Get all failed resumes
        failed_resumes = await resume_repo.get_resumes_with_failed_status(
            failure_reason=FAILURE_INSUFFICIENT_TEXT,
            limit=limit
        )
        
        if not failed_resumes:
            print("✅ No resumes with failed:insufficient_text status found.")
            return
        
        print(f"\n📋 Found {len(failed_resumes)} resume(s) with failed:insufficient_text status")
        print("=" * 80)
        
        # Create controller
        resume_parser = ResumeParser()
        embedding_service = EmbeddingService()
        controller = ResumeController(
            resume_parser=resume_parser,
            embedding_service=embedding_service,
            vector_db=vector_db,
            resume_repo=resume_repo,
            session=session
        )
        
        success_count = 0
        failed_count = 0
        
        for idx, resume in enumerate(failed_resumes, 1):
            print(f"\n[{idx}/{len(failed_resumes)}] Retrying resume ID {resume.id}: {resume.filename}")
            print("-" * 80)
            
            try:
                result = await controller.retry_failed_resume_with_ocr(
                    resume_id=resume.id,
                    extract_modules=extract_modules
                )
                
                if result.status == "completed":
                    print(f"✅ SUCCESS: Resume ID {resume.id} processed successfully")
                    print(f"   Status: {result.status}")
                    success_count += 1
                else:
                    print(f"⚠️  WARNING: Resume ID {resume.id} processed but status is: {result.status}")
                    failed_count += 1
                    
            except Exception as e:
                print(f"❌ ERROR: Failed to retry resume ID {resume.id}: {e}")
                failed_count += 1
                logger.error(
                    f"Failed to retry resume {resume.id}: {e}",
                    extra={"resume_id": resume.id, "error": str(e)},
                    exc_info=True
                )
        
        print("\n" + "=" * 80)
        print("📊 RETRY SUMMARY")
        print("=" * 80)
        print(f"Total resumes: {len(failed_resumes)}")
        print(f"Successfully retried: {success_count}")
        print(f"Failed: {failed_count}")
        print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Retry all resumes with failed:insufficient_text status")
    parser.add_argument(
        "--extract-modules",
        type=str,
        default="all",
        help="Modules to extract (default: 'all'). Options: 'all', '1,2,3', 'designation,skills', etc."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of resumes to retry (default: None = all)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔄 RETRYING FAILED RESUMES WITH OCR")
    print("=" * 80)
    print(f"Extract modules: {args.extract_modules}")
    if args.limit:
        print(f"Limit: {args.limit}")
    print("=" * 80)
    
    asyncio.run(retry_all_failed_resumes(
        extract_modules=args.extract_modules,
        limit=args.limit
    ))


