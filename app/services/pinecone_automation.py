"""Pinecone automation service for managing indexes and namespaces for ATS resume embedding system."""
import asyncio
import re
import time
from typing import Optional, List, Dict, Any
from pinecone import Pinecone, ServerlessSpec

from app.config import settings
from app.utils.logging import get_logger
from app.category.category_extractor import CategoryExtractor, IT_CATEGORY_PROMPT, NON_IT_CATEGORY_PROMPT

logger = get_logger(__name__)

# Pinecone API key - will use from settings (.env file) or fallback to hardcoded value
# Priority: settings.pinecone_api_key > hardcoded value
PINECONE_API_KEY_FALLBACK = "pcsk_6FByML_NApKAxacNuHFJ4QLaQretqWVT8R1Tk8yqXDHXYTjg1TJGwedDxwqtCCBo7prWxY"

# Index names (must be lowercase for Pinecone)
IT_INDEX_NAME = "it"
NON_IT_INDEX_NAME = "non-it"

# Default namespace for invalid/empty categories
UNCATEGORIZED_NAMESPACE = "uncategorized"


class PineconeAutomation:
    """
    Pinecone automation service for creating and managing indexes and namespaces.
    
    Features:
    - Creates two indexes: "IT" and "Non-IT"
    - Dynamically creates namespaces based on resume categories
    - Routes resumes to correct index based on mastercategory
    - Handles error cases with uncategorized namespace
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Pinecone automation service.
        
        Args:
            api_key: Pinecone API key (optional, will use from settings if not provided)
        """
        # Use provided key, or from settings, or fallback
        if api_key:
            self.api_key = api_key
        elif settings.pinecone_api_key:
            self.api_key = settings.pinecone_api_key
        else:
            self.api_key = PINECONE_API_KEY_FALLBACK
        self.pc: Optional[Pinecone] = None
        self.it_index = None
        self.non_it_index = None
        self.dimension = settings.embedding_dimension
        self.category_extractor = CategoryExtractor()
    
    def _extract_categories_from_prompt(self, prompt_text: str) -> List[str]:
        """
        Extract category names from category_extractor.py prompt text.
        
        Args:
            prompt_text: The IT_CATEGORY_PROMPT or NON_IT_CATEGORY_PROMPT string
            
        Returns:
            List of category names
        """
        categories = []
        lines = prompt_text.split('\n')
        in_category_section = False
        
        for line in lines:
            line = line.strip()
            
            # Start capturing after "SAMPLE IT CATEGORIES:" or "SAMPLE NON-IT CATEGORIES:"
            if "SAMPLE" in line and "CATEGORIES:" in line:
                in_category_section = True
                continue
            
            # Stop capturing when we hit the next section
            if in_category_section and line and not line[0].isdigit() and "ASSESSMENT" in line:
                break
            
            # Extract category from numbered lines (e.g., "1. Full Stack Development (Java)")
            if in_category_section and line and line[0].isdigit():
                # Remove number prefix (e.g., "1. " or "10. ")
                category = re.sub(r'^\d+\.\s*', '', line)
                if category:
                    categories.append(category)
        
        return categories
    
    def _get_all_it_categories(self) -> List[str]:
        """Get all IT categories from category_extractor.py."""
        return self._extract_categories_from_prompt(IT_CATEGORY_PROMPT)
    
    def _get_all_non_it_categories(self) -> List[str]:
        """Get all Non-IT categories from category_extractor.py."""
        return self._extract_categories_from_prompt(NON_IT_CATEGORY_PROMPT)
    
    def _normalize_namespace(self, category: str) -> str:
        """
        Normalize category string to valid Pinecone namespace format.
        
        Namespace normalization rules (as per requirements):
        - Convert to lowercase
        - Replace spaces, slashes, dots, parentheses with underscores
        - Remove all characters except [a-z0-9_]
        - Collapse multiple underscores into one
        
        Example:
        "Full Stack Development (Java)" → "full_stack_development_java"
        
        Args:
            category: Category string from category_extractor.py output
            
        Returns:
            Normalized namespace string
        """
        if not category or not category.strip():
            return UNCATEGORIZED_NAMESPACE
        
        # Convert to lowercase
        normalized = category.lower().strip()
        
        # Replace spaces, slashes, dots, parentheses, and all other special chars with underscores
        # This handles: spaces, slashes (/), dots (.), parentheses (()), and any other non-alphanumeric chars
        normalized = re.sub(r'[^a-z0-9_]+', '_', normalized)
        
        # Collapse multiple consecutive underscores into one
        normalized = re.sub(r'_+', '_', normalized)
        
        # Remove leading/trailing underscores
        normalized = normalized.strip('_')
        
        # Ensure it's not empty after normalization
        if not normalized:
            return UNCATEGORIZED_NAMESPACE
        
        # Final validation: must contain only [a-z0-9_]
        if not re.match(r'^[a-z0-9_]+$', normalized):
            return UNCATEGORIZED_NAMESPACE
        
        return normalized
    
    def _determine_index_name(self, mastercategory: str) -> str:
        """
        Determine which index to use based on mastercategory.
        
        Args:
            mastercategory: "IT" or "NON_IT" or "Non-IT"
            
        Returns:
            Index name: "it" or "non-it" (lowercase as required by Pinecone)
        """
        if not mastercategory:
            logger.warning("Empty mastercategory, defaulting to Non-IT index")
            return NON_IT_INDEX_NAME
        
        mastercategory_upper = mastercategory.upper().strip()
        
        if mastercategory_upper == "IT":
            return IT_INDEX_NAME
        else:
            # Default to Non-IT for NON_IT, Non-IT, or any other value
            return NON_IT_INDEX_NAME
    
    async def initialize_pinecone(self) -> None:
        """
        Initialize Pinecone client.
        
        Raises:
            RuntimeError: If Pinecone initialization fails
        """
        try:
            self.pc = Pinecone(api_key=self.api_key)
            logger.info("Pinecone client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {e}", extra={"error": str(e)})
            raise RuntimeError(f"Pinecone initialization failed: {e}")
    
    async def create_indexes(self) -> None:
        """
        Create Pinecone indexes (IT and Non-IT) if they don't exist.
        
        This method is idempotent - it will not recreate existing indexes.
        
        Raises:
            RuntimeError: If index creation fails
        """
        if not self.pc:
            await self.initialize_pinecone()
        
        try:
            # Get list of existing indexes
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            logger.info(f"Existing Pinecone indexes: {existing_indexes}")
            
            # Create IT index if it doesn't exist
            if IT_INDEX_NAME not in existing_indexes:
                logger.info(f"Creating Pinecone index: {IT_INDEX_NAME}")
                self.pc.create_index(
                    name=IT_INDEX_NAME,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=settings.pinecone_cloud,
                        region=settings.pinecone_region
                    )
                )
                logger.info(f"Index '{IT_INDEX_NAME}' creation initiated")
            else:
                logger.info(f"Index '{IT_INDEX_NAME}' already exists, skipping creation")
            
            # Create Non-IT index if it doesn't exist
            if NON_IT_INDEX_NAME not in existing_indexes:
                logger.info(f"Creating Pinecone index: {NON_IT_INDEX_NAME}")
                self.pc.create_index(
                    name=NON_IT_INDEX_NAME,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=settings.pinecone_cloud,
                        region=settings.pinecone_region
                    )
                )
                logger.info(f"Index '{NON_IT_INDEX_NAME}' creation initiated")
            else:
                logger.info(f"Index '{NON_IT_INDEX_NAME}' already exists, skipping creation")
            
            # Wait for indexes to be ready
            max_wait_time = 300  # 5 minutes
            wait_interval = 2  # 2 seconds
            elapsed_time = 0
            
            while elapsed_time < max_wait_time:
                current_indexes = [idx.name for idx in self.pc.list_indexes()]
                
                it_ready = IT_INDEX_NAME in current_indexes
                non_it_ready = NON_IT_INDEX_NAME in current_indexes
                
                if it_ready and non_it_ready:
                    logger.info("Both indexes are ready")
                    break
                
                if elapsed_time % 10 == 0:  # Log every 10 seconds
                    logger.info(
                        f"Waiting for indexes to be ready... "
                        f"IT: {it_ready}, Non-IT: {non_it_ready}"
                    )
                
                await asyncio.sleep(wait_interval)
                elapsed_time += wait_interval
            
            if IT_INDEX_NAME not in [idx.name for idx in self.pc.list_indexes()]:
                raise RuntimeError(f"Index '{IT_INDEX_NAME}' was not created within timeout period")
            
            if NON_IT_INDEX_NAME not in [idx.name for idx in self.pc.list_indexes()]:
                raise RuntimeError(f"Index '{NON_IT_INDEX_NAME}' was not created within timeout period")
            
            # Initialize index connections
            self.it_index = self.pc.Index(IT_INDEX_NAME)
            self.non_it_index = self.pc.Index(NON_IT_INDEX_NAME)
            
            logger.info(f"Successfully initialized indexes: {IT_INDEX_NAME}, {NON_IT_INDEX_NAME}")
            
            # Pre-create all namespaces so they're visible in Pinecone dashboard
            await self._create_all_namespaces()
            
        except Exception as e:
            logger.error(f"Failed to create Pinecone indexes: {e}", extra={"error": str(e)})
            raise RuntimeError(f"Index creation failed: {e}")
    
    async def _create_all_namespaces(self) -> None:
        """
        Pre-create all namespaces by inserting placeholder vectors.
        
        This makes all namespaces visible in the Pinecone dashboard immediately,
        even before actual resume data is inserted.
        """
        try:
            if not self.it_index:
                self.it_index = self.pc.Index(IT_INDEX_NAME)
            if not self.non_it_index:
                self.non_it_index = self.pc.Index(NON_IT_INDEX_NAME)
            
            # Create a minimal valid embedding vector (all zeros with one small value)
            # Pinecone requires at least one non-zero value
            placeholder_vector = [0.0001] * self.dimension
            
            created_count = 0
            loop = asyncio.get_event_loop()
            
            # Create IT namespaces
            it_categories = self._get_all_it_categories()
            print(f"🔨 Creating {len(it_categories)} IT namespaces...")
            logger.info(f"Creating {len(it_categories)} IT namespaces")
            
            for category in it_categories:
                namespace = self._normalize_namespace(category)
                placeholder_id = f"_namespace_init_{namespace}"
                
                try:
                    def _upsert_it(ns=namespace, pid=placeholder_id, cat=category):
                        return self.it_index.upsert(
                            vectors=[{
                                "id": pid,
                                "values": placeholder_vector,
                                "metadata": {
                                    "type": "namespace_placeholder",
                                    "category": cat,
                                    "namespace": ns,
                                    "mastercategory": "IT"
                                }
                            }],
                            namespace=ns
                        )
                    await loop.run_in_executor(None, _upsert_it)
                    created_count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to create IT namespace '{namespace}': {e}",
                        extra={"namespace": namespace, "category": category, "error": str(e)}
                    )
            
            # Create Non-IT namespaces
            non_it_categories = self._get_all_non_it_categories()
            print(f"🔨 Creating {len(non_it_categories)} Non-IT namespaces...")
            logger.info(f"Creating {len(non_it_categories)} Non-IT namespaces")
            
            for category in non_it_categories:
                namespace = self._normalize_namespace(category)
                placeholder_id = f"_namespace_init_{namespace}"
                
                try:
                    def _upsert_non_it(ns=namespace, pid=placeholder_id, cat=category):
                        return self.non_it_index.upsert(
                            vectors=[{
                                "id": pid,
                                "values": placeholder_vector,
                                "metadata": {
                                    "type": "namespace_placeholder",
                                    "category": cat,
                                    "namespace": ns,
                                    "mastercategory": "NON_IT"
                                }
                            }],
                            namespace=ns
                        )
                    await loop.run_in_executor(None, _upsert_non_it)
                    created_count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to create Non-IT namespace '{namespace}': {e}",
                        extra={"namespace": namespace, "category": category, "error": str(e)}
                    )
            
            # Create uncategorized namespace in both indexes
            for index_name, target_index in [(IT_INDEX_NAME, self.it_index), (NON_IT_INDEX_NAME, self.non_it_index)]:
                placeholder_id = f"_namespace_init_{UNCATEGORIZED_NAMESPACE}"
                try:
                    mc = "IT" if index_name == IT_INDEX_NAME else "NON_IT"
                    def _upsert_uncategorized(idx=target_index, pid=placeholder_id, mastercat=mc):
                        return idx.upsert(
                            vectors=[{
                                "id": pid,
                                "values": placeholder_vector,
                                "metadata": {
                                    "type": "namespace_placeholder",
                                    "category": "Uncategorized",
                                    "namespace": UNCATEGORIZED_NAMESPACE,
                                    "mastercategory": mastercat
                                }
                            }],
                            namespace=UNCATEGORIZED_NAMESPACE
                        )
                    await loop.run_in_executor(None, _upsert_uncategorized)
                    created_count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to create 'uncategorized' namespace in {index_name}: {e}",
                        extra={"namespace": UNCATEGORIZED_NAMESPACE, "index": index_name, "error": str(e)}
                    )
            
            total_namespaces = len(it_categories) + len(non_it_categories) + 2  # +2 for uncategorized in both indexes
            print(f"✅ Created {created_count}/{total_namespaces} namespaces")
            logger.info(
                f"✅ Created {created_count} namespaces (IT: {len(it_categories)}, Non-IT: {len(non_it_categories)}, Uncategorized: 2)",
                extra={"created_count": created_count, "it_count": len(it_categories), "non_it_count": len(non_it_categories)}
            )
            
        except Exception as e:
            logger.error(
                f"Failed to create namespaces: {e}",
                extra={"error": str(e)}
            )
            # Don't raise - namespace creation failure shouldn't block index creation
            print(f"⚠️ Warning: Some namespaces may not have been created: {e}")
    
    async def delete_placeholder_vectors(self) -> None:
        """
        Delete all placeholder vectors from both indexes.
        
        This removes the placeholder vectors that were created during namespace setup.
        Namespaces will remain and will be populated when actual resume data is inserted.
        """
        try:
            if not self.pc:
                await self.initialize_pinecone()
            
            if not self.it_index:
                self.it_index = self.pc.Index(IT_INDEX_NAME)
            if not self.non_it_index:
                self.non_it_index = self.pc.Index(NON_IT_INDEX_NAME)
            
            # Get all categories to find placeholder IDs
            it_categories = self._get_all_it_categories()
            non_it_categories = self._get_all_non_it_categories()
            
            deleted_count = 0
            loop = asyncio.get_event_loop()
            
            # Delete IT placeholder vectors
            for category in it_categories:
                namespace = self._normalize_namespace(category)
                placeholder_id = f"_placeholder_{namespace}"
                try:
                    await loop.run_in_executor(
                        None,
                        lambda ns=namespace, pid=placeholder_id: self.it_index.delete(
                            ids=[pid],
                            namespace=ns
                        )
                    )
                    deleted_count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to delete IT placeholder '{placeholder_id}': {e}",
                        extra={"namespace": namespace, "placeholder_id": placeholder_id, "error": str(e)}
                    )
            
            # Delete Non-IT placeholder vectors
            for category in non_it_categories:
                namespace = self._normalize_namespace(category)
                placeholder_id = f"_placeholder_{namespace}"
                try:
                    await loop.run_in_executor(
                        None,
                        lambda ns=namespace, pid=placeholder_id: self.non_it_index.delete(
                            ids=[pid],
                            namespace=ns
                        )
                    )
                    deleted_count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to delete Non-IT placeholder '{placeholder_id}': {e}",
                        extra={"namespace": namespace, "placeholder_id": placeholder_id, "error": str(e)}
                    )
            
            # Delete uncategorized placeholder from both indexes
            placeholder_id = f"_placeholder_{UNCATEGORIZED_NAMESPACE}"
            for index_name, target_index in [(IT_INDEX_NAME, self.it_index), (NON_IT_INDEX_NAME, self.non_it_index)]:
                try:
                    await loop.run_in_executor(
                        None,
                        lambda idx=target_index, pid=placeholder_id: idx.delete(
                            ids=[pid],
                            namespace=UNCATEGORIZED_NAMESPACE
                        )
                    )
                    deleted_count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to delete '{placeholder_id}' from {index_name}: {e}",
                        extra={"namespace": UNCATEGORIZED_NAMESPACE, "index": index_name, "error": str(e)}
                    )
            
            logger.info(
                f"✅ Deleted {deleted_count} placeholder vectors from Pinecone indexes",
                extra={"deleted_count": deleted_count}
            )
            
        except Exception as e:
            logger.error(
                f"Failed to delete placeholder vectors: {e}",
                extra={"error": str(e)}
            )
            raise
    
    async def get_category_from_extractor(
        self,
        resume_text: str,
        mastercategory: str,
        filename: str = "resume"
    ) -> Optional[str]:
        """
        Get category from category_extractor.py.
        
        CRITICAL: category_extractor.py is the SINGLE SOURCE OF TRUTH for categories.
        It contains predefined IT and Non-IT category mappings.
        DO NOT manually define categories in this code.
        
        Args:
            resume_text: Resume text content
            mastercategory: "IT" or "NON_IT"
            filename: Resume filename for logging
            
        Returns:
            Category string from category_extractor.py or None if extraction fails
        """
        try:
            category = await self.category_extractor.extract_category(
                resume_text=resume_text,
                mastercategory=mastercategory,
                filename=filename
            )
            
            if category:
                logger.info(
                    f"Category extracted: {category}",
                    extra={
                        "category": category,
                        "mastercategory": mastercategory,
                        "file_name": filename,
                    }
                )
            else:
                logger.warning(
                    f"Category extraction returned None",
                    extra={
                        "mastercategory": mastercategory,
                        "file_name": filename,
                    }
                )
            
            return category
            
        except Exception as e:
            logger.error(
                f"Category extraction failed: {e}",
                extra={
                    "mastercategory": mastercategory,
                    "file_name": filename,
                    "error": str(e),
                }
            )
            return None
    
    async def insert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        resume_text: str,
        mastercategory: str,
        filename: str = "resume",
        category: Optional[str] = None
    ) -> None:
        """
        Insert vectors into the correct Pinecone index and namespace.
        
        Args:
            vectors: List of vector dictionaries with 'id', 'embedding', and 'metadata'
            resume_text: Resume text (used for fallback category extraction if category not provided)
            mastercategory: "IT" or "NON_IT"
            filename: Resume filename for logging
            category: Category from database (category column). If provided, this will be used.
                     If not provided, will extract from category_extractor.py
            
        Raises:
            RuntimeError: If insertion fails
        """
        if not vectors:
            logger.warning("No vectors provided for insertion")
            return
        
        try:
            # Determine target index
            index_name = self._determine_index_name(mastercategory)
            target_index = self.it_index if index_name == IT_INDEX_NAME else self.non_it_index
            
            if not target_index:
                # Initialize index connection if not already done
                if index_name == IT_INDEX_NAME:
                    self.it_index = self.pc.Index(IT_INDEX_NAME)
                    target_index = self.it_index
                else:
                    self.non_it_index = self.pc.Index(NON_IT_INDEX_NAME)
                    target_index = self.non_it_index
            
            # Use category from database if provided, otherwise extract from category_extractor.py
            if not category:
                # Fallback: Get category from category_extractor.py
                # CRITICAL: category_extractor.py is the SINGLE SOURCE OF TRUTH for categories
                category = await self.get_category_from_extractor(
                    resume_text=resume_text,
                    mastercategory=mastercategory,
                    filename=filename
                )
                logger.info(
                    f"Category extracted from resume text (not from database)",
                    extra={"category": category, "file_name": filename}
                )
            else:
                logger.info(
                    f"Using category from database",
                    extra={"category": category, "file_name": filename}
                )
            
            # Normalize category to namespace
            # Namespace is derived from the category (either from database or category_extractor.py)
            namespace = self._normalize_namespace(category) if category else UNCATEGORIZED_NAMESPACE
            
            # Verify namespace exists in pre-created list (log warning if not, but still use it)
            if category:
                # Get all valid categories for this mastercategory
                if mastercategory.upper() == "IT":
                    valid_categories = self._get_all_it_categories()
                else:
                    valid_categories = self._get_all_non_it_categories()
                
                # Check if the category matches any pre-created category
                category_matches = any(
                    self._normalize_namespace(valid_cat) == namespace 
                    for valid_cat in valid_categories
                )
                
                if not category_matches:
                    # Category from DB doesn't match pre-created namespaces
                    # Still use it, but log a warning - namespace will be created automatically by Pinecone
                    logger.warning(
                        f"Category '{category}' from database does not match any pre-created namespace, but will be used anyway",
                        extra={
                            "category": category,
                            "namespace": namespace,
                            "mastercategory": mastercategory,
                            "file_name": filename
                        }
                    )
                    # Note: Pinecone will create the namespace automatically when we insert vectors
            
            print(f"🔍 [PINECONE DEBUG] Inserting {len(vectors)} vectors into index '{index_name}', namespace '{namespace}'")
            logger.info(
                f"Inserting vectors into index '{index_name}', namespace '{namespace}'",
                extra={
                    "index_name": index_name,
                    "namespace": namespace,
                    "category": category,
                    "mastercategory": mastercategory,
                    "vector_count": len(vectors),
                    "file_name": filename,
                }
            )
            
            # Format vectors for Pinecone
            pinecone_vectors = []
            for vec_data in vectors:
                vector_id = vec_data.get("id")
                embedding = vec_data.get("embedding")
                metadata = vec_data.get("metadata", {})
                
                if not vector_id or not embedding:
                    logger.warning(f"Skipping vector with missing id or embedding")
                    continue
                
                # Add category and mastercategory to metadata
                metadata["category"] = category or "uncategorized"
                metadata["mastercategory"] = mastercategory
                metadata["namespace"] = namespace
                
                pinecone_vectors.append({
                    "id": str(vector_id),
                    "values": embedding,
                    "metadata": metadata
                })
            
            if not pinecone_vectors:
                logger.warning("No valid vectors to insert after filtering")
                return
            
            # Insert vectors into namespace
            # Pinecone upsert is synchronous, but we're in async context
            def _upsert_vectors(idx, vecs, ns):
                return idx.upsert(vectors=vecs, namespace=ns)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                _upsert_vectors,
                target_index,
                pinecone_vectors,
                namespace
            )
            
            print(f"✅ [PINECONE DEBUG] Successfully inserted {len(pinecone_vectors)} vectors into index '{index_name}', namespace '{namespace}'")
            logger.info(
                f"Successfully inserted {len(pinecone_vectors)} vectors into "
                f"index '{index_name}', namespace '{namespace}'",
                extra={
                    "index_name": index_name,
                    "namespace": namespace,
                    "vector_count": len(pinecone_vectors),
                    "file_name": filename,
                }
            )
            
        except Exception as e:
            error_msg = f"Failed to insert vectors: {e}"
            logger.error(
                error_msg,
                extra={
                    "mastercategory": mastercategory,
                    "file_name": filename,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "vector_count": len(vectors) if vectors else 0,
                    "index_name": index_name if 'index_name' in locals() else "unknown",
                    "namespace": namespace if 'namespace' in locals() else "unknown"
                }
            )
            # Print to console for immediate visibility
            print(f"❌ PINECONE INSERTION ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Vector insertion failed: {e}")
    
    async def get_all_namespaces(self, mastercategory: str) -> List[str]:
        """
        Get all namespaces from the specified index.
        
        Args:
            mastercategory: "IT" or "NON_IT" to determine index
            
        Returns:
            List of namespace names
        """
        try:
            # Determine target index
            index_name = self._determine_index_name(mastercategory)
            target_index = self.it_index if index_name == IT_INDEX_NAME else self.non_it_index
            
            if not target_index:
                # Initialize index connection if not already done
                if index_name == IT_INDEX_NAME:
                    self.it_index = self.pc.Index(IT_INDEX_NAME)
                    target_index = self.it_index
                else:
                    self.non_it_index = self.pc.Index(NON_IT_INDEX_NAME)
                    target_index = self.non_it_index
            
            # Get index stats which includes namespace information
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(None, lambda: target_index.describe_index_stats())
            
            # Extract namespaces from stats
            namespaces = []
            if stats and "namespaces" in stats:
                namespaces = list(stats["namespaces"].keys())
            
            # Filter out placeholder namespaces (they don't have real data)
            namespaces = [ns for ns in namespaces if not ns.startswith("_namespace_init_")]
            
            # Also check if default namespace has data
            # Default namespace shows up as empty string "" or might be in total_vector_count
            if stats and "total_vector_count" in stats:
                total_count = stats.get("total_vector_count", 0)
                # If there are vectors but no namespaces listed, they might be in default namespace
                # But we'll rely on the namespaces dict which should include default if it has data
            
            logger.info(
                f"Found {len(namespaces)} namespaces in index '{index_name}'",
                extra={"index_name": index_name, "namespace_count": len(namespaces), "namespaces": namespaces[:10]}  # Log first 10
            )
            
            return namespaces
            
        except Exception as e:
            logger.error(
                f"Failed to get namespaces: {e}",
                extra={"mastercategory": mastercategory, "error": str(e)}
            )
            # Return empty list on error - search will still work with default namespace
            return []
    
    async def query_vectors(
        self,
        query_vector: List[float],
        mastercategory: str,
        namespace: Optional[str] = None,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query vectors from the correct Pinecone index and namespace.
        
        Args:
            query_vector: Query embedding vector
            mastercategory: "IT" or "NON_IT" to determine index
            namespace: Optional namespace to query (if None, queries default namespace)
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of matching vectors with metadata
        """
        try:
            # Determine target index
            index_name = self._determine_index_name(mastercategory)
            target_index = self.it_index if index_name == IT_INDEX_NAME else self.non_it_index
            
            if not target_index:
                # Initialize index connection if not already done
                if index_name == IT_INDEX_NAME:
                    self.it_index = self.pc.Index(IT_INDEX_NAME)
                    target_index = self.it_index
                else:
                    self.non_it_index = self.pc.Index(NON_IT_INDEX_NAME)
                    target_index = self.non_it_index
            
            # Query specific namespace (or default namespace if None)
            query_namespace = namespace if namespace else ""
            
            # Query Pinecone
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: target_index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=query_namespace if query_namespace else None,
                    filter=filter_dict
                )
            )
            
            matches = []
            for match in results.get("matches", []):
                matches.append({
                    "id": match.get("id"),
                    "score": match.get("score", 0.0),
                    "metadata": match.get("metadata", {})
                })
            
            logger.info(
                f"Query returned {len(matches)} results from index '{index_name}', namespace '{query_namespace or 'default'}'",
                extra={
                    "index_name": index_name,
                    "namespace": query_namespace,
                    "result_count": len(matches)
                }
            )
            
            return matches
            
        except Exception as e:
            logger.error(
                f"Failed to query vectors: {e}",
                extra={
                    "mastercategory": mastercategory,
                    "namespace": namespace,
                    "error": str(e)
                }
            )
            raise RuntimeError(f"Vector query failed: {e}")
    
    async def delete_vectors(
        self,
        vector_ids: List[str],
        mastercategory: str,
        namespace: Optional[str] = None
    ) -> None:
        """
        Delete vectors from the correct Pinecone index and namespace.
        
        Args:
            vector_ids: List of vector IDs to delete
            mastercategory: "IT" or "NON_IT" to determine index
            namespace: Optional namespace (if None, deletes from default namespace)
        """
        try:
            # Determine target index
            index_name = self._determine_index_name(mastercategory)
            target_index = self.it_index if index_name == IT_INDEX_NAME else self.non_it_index
            
            if not target_index:
                # Initialize index connection if not already done
                if index_name == IT_INDEX_NAME:
                    self.it_index = self.pc.Index(IT_INDEX_NAME)
                    target_index = self.it_index
                else:
                    self.non_it_index = self.pc.Index(NON_IT_INDEX_NAME)
                    target_index = self.non_it_index
            
            # Use namespace if provided
            delete_namespace = namespace if namespace else ""
            
            # Delete vectors
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: target_index.delete(
                    ids=[str(vid) for vid in vector_ids],
                    namespace=delete_namespace if delete_namespace else None
                )
            )
            
            logger.info(
                f"Deleted {len(vector_ids)} vectors from index '{index_name}', namespace '{delete_namespace or 'default'}'",
                extra={
                    "index_name": index_name,
                    "namespace": delete_namespace,
                    "vector_count": len(vector_ids)
                }
            )
            
        except Exception as e:
            logger.error(
                f"Failed to delete vectors: {e}",
                extra={
                    "mastercategory": mastercategory,
                    "namespace": namespace,
                    "error": str(e),
                    "vector_count": len(vector_ids)
                }
            )
            raise RuntimeError(f"Vector deletion failed: {e}")

