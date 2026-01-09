"""Standalone script to initialize Pinecone indexes for ATS resume embedding system.

This script creates the required Pinecone indexes:
- IT
- Non-IT

Usage:
    python initialize_pinecone_indexes.py
"""
import asyncio
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.pinecone_automation import PineconeAutomation
from app.utils.logging import get_logger

logger = get_logger(__name__)


async def main():
    """Main function to initialize Pinecone indexes."""
    try:
        print("=" * 60)
        print("Pinecone Index Initialization Script")
        print("=" * 60)
        logger.info("=" * 60)
        logger.info("Pinecone Index Initialization Script")
        logger.info("=" * 60)
        
        # Initialize Pinecone automation service
        automation = PineconeAutomation()
        
        # Initialize Pinecone client
        print("\nStep 1: Initializing Pinecone client...")
        logger.info("Step 1: Initializing Pinecone client...")
        await automation.initialize_pinecone()
        print("✓ Pinecone client initialized")
        logger.info("✓ Pinecone client initialized")
        
        # Create indexes and namespaces
        print("\nStep 2: Creating Pinecone indexes and namespaces...")
        logger.info("Step 2: Creating Pinecone indexes and namespaces...")
        await automation.create_indexes()
        print("✓ Indexes and namespaces created/verified successfully")
        logger.info("✓ Indexes and namespaces created/verified successfully")
        
        # Show namespace information
        it_categories = automation._get_all_it_categories()
        non_it_categories = automation._get_all_non_it_categories()
        
        print("\n" + "=" * 60)
        print("SUCCESS: Pinecone indexes and namespaces are ready!")
        print(f"  - Index '{automation._determine_index_name('IT')}' (for IT resumes)")
        print(f"    → {len(it_categories)} namespaces pre-created and visible in Pinecone")
        print(f"  - Index '{automation._determine_index_name('NON_IT')}' (for Non-IT resumes)")
        print(f"    → {len(non_it_categories)} namespaces pre-created and visible in Pinecone")
        print("\nNote: All namespaces are pre-created with placeholder vectors.")
        print("      When you process resumes, data will be stored in the namespace")
        print("      that matches the 'category' column from the database.")
        print("=" * 60)
        logger.info("=" * 60)
        logger.info("SUCCESS: Pinecone indexes and namespaces are ready!")
        logger.info(f"  - Index '{automation._determine_index_name('IT')}' (for IT resumes)")
        logger.info(f"    → {len(it_categories)} namespaces pre-created")
        logger.info(f"  - Index '{automation._determine_index_name('NON_IT')}' (for Non-IT resumes)")
        logger.info(f"    → {len(non_it_categories)} namespaces pre-created")
        logger.info("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.error(f"FAILED: Index initialization failed: {e}", extra={"error": str(e)})
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

