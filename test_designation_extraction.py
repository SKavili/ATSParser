"""Test script to debug designation extraction."""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.designation import DesignationExtractor
from app.utils.logging import get_logger

logger = get_logger(__name__)


async def test_designation_extraction():
    """Test designation extraction with sample resume text."""
    
    # Sample resume text for testing
    sample_resume = """
    JOHN DOE
    Senior Software Engineer
    
    Email: john.doe@example.com
    Phone: +1-555-123-4567
    
    PROFESSIONAL EXPERIENCE
    
    Senior Software Engineer | Tech Corp | 2020 - Present
    - Led development of microservices architecture
    - Managed team of 5 developers
    
    Software Engineer | Startup Inc | 2018 - 2020
    - Developed REST APIs using Python and FastAPI
    - Implemented CI/CD pipelines
    
    EDUCATION
    Bachelor of Science in Computer Science
    University of Technology, 2018
    """
    
    extractor = DesignationExtractor()
    
    print("=" * 60)
    print("Testing Designation Extraction")
    print("=" * 60)
    print(f"\nOLLAMA Host: {extractor.ollama_host}")
    print(f"Model: {extractor.model}")
    print("\nSample Resume Text:")
    print(sample_resume[:200] + "...")
    print("\n" + "=" * 60)
    
    try:
        print("\n[1] Checking OLLAMA connection...")
        is_connected, available_model = await extractor._check_ollama_connection()
        if not is_connected:
            print(f"❌ OLLAMA is not accessible at {extractor.ollama_host}")
            print("   Please ensure OLLAMA is running: ollama serve")
            return
        print("OK: OLLAMA is connected")
        print(f"   Available model: {available_model}")
        
        print("\n[2] Extracting designation...")
        designation = await extractor.extract_designation(sample_resume, "test_resume.txt")
        
        print("\n" + "=" * 60)
        if designation:
            print(f"SUCCESS: Designation extracted: '{designation}'")
        else:
            print("FAILED: No designation found (returned None)")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_designation_extraction())

