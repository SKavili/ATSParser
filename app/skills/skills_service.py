"""Service for extracting and saving skills to database."""
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.skills.skills_extractor import SkillsExtractor
from app.repositories.resume_repo import ResumeRepository
from app.repositories.prompt_repo import PromptRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SkillsService:
    """Service for extracting skills from resume and saving to database."""
    
    def __init__(self, session: AsyncSession):
        self.skills_extractor = SkillsExtractor()
        self.resume_repo = ResumeRepository(session)
        self.prompt_repo = PromptRepository(session)
    
    async def validate_required_prompts(self) -> tuple[bool, list[str]]:
        """
        Validate that required 'other' prompts exist in the database.
        
        Returns:
            tuple: (is_valid, missing_prompts)
                - is_valid: True if all required prompts exist
                - missing_prompts: List of missing prompt descriptions
        """
        missing_prompts = []
        
        # Check IT + "other" prompt
        it_other = await self.prompt_repo.get_by_category("IT", "other")
        if not it_other or not it_other.prompt:
            missing_prompts.append("IT + 'other'")
        
        # Check NON_IT + "other" prompt (use normalized format for database lookup)
        non_it_other = await self.prompt_repo.get_by_category("non IT", "other")
        if not non_it_other or not non_it_other.prompt:
            missing_prompts.append("NON_IT + 'other'")
        
        is_valid = len(missing_prompts) == 0
        
        if not is_valid:
            logger.error(
                f"❌ VALIDATION FAILED: Missing required prompts in database: {', '.join(missing_prompts)}",
                extra={"missing_prompts": missing_prompts}
            )
        else:
            logger.info("✅ VALIDATION PASSED: All required 'other' prompts exist in database")
        
        return is_valid, missing_prompts
    
    def _normalize_category(self, category: Optional[str]) -> Optional[str]:
        """
        Normalize category name for consistent matching.
        
        Args:
            category: Category name to normalize
        
        Returns:
            Normalized category name (lowercase, stripped) or None
        """
        if not category:
            return None
        return category.strip().lower()
    
    def _normalize_mastercategory_for_prompt(self, mastercategory: str) -> str:
        """
        Normalize mastercategory to match database format in prompts table.
        
        Converts mastercategory from resume_metadata format to prompts table format:
        - "NON_IT" → "non IT" (for database lookup)
        - "IT" → "IT" (stays same)
        
        This handles the format mismatch where:
        - resume_metadata stores: "NON_IT" (with underscore, uppercase)
        - prompts table stores: "non IT" (with space, lowercase)
        
        Args:
            mastercategory: Mastercategory from resume_metadata (e.g., "NON_IT" or "IT")
        
        Returns:
            Normalized mastercategory for prompt lookup (e.g., "non IT" or "IT")
        """
        if not mastercategory:
            return mastercategory
        
        mastercategory = mastercategory.strip()
        
        # Convert NON_IT to "non IT" format (as stored in prompts table)
        if mastercategory.upper() == "NON_IT":
            return "non IT"
        elif mastercategory.upper() == "IT":
            return "IT"
        
        # Return as-is if unknown format (shouldn't happen after validation)
        return mastercategory
    
    async def extract_and_save_skills(
        self,
        resume_text: str,
        resume_id: int,
        filename: str = "resume"
    ) -> Optional[str]:
        """
        Extract skills from resume text and update the database record.
        Uses prompt from prompts table based on mastercategory and category.
        
        Prompt lookup priority:
        1. If category exists: Try mastercategory + category (specific)
           → If not found: Try mastercategory + "other"
        2. If category is NULL: Use mastercategory + "other" directly
        
        If mastercategory is NULL, skills extraction is skipped.
        If no prompt found in database, extraction is skipped.
        Gateway classification is NOT used - always uses prompts from database.
        
        Args:
            resume_text: The text content of the resume
            resume_id: The ID of the resume record in the database
            filename: Name of the resume file (for logging)
        
        Returns:
            Comma-separated string of skills or None if not found/extraction skipped
        """
        try:
            logger.info(
                f"🔍 STARTING SKILLS EXTRACTION for resume ID {resume_id}",
                extra={
                    "resume_id": resume_id, 
                    "file_name": filename,
                    "resume_text_length": len(resume_text),
                }
            )
            
            # Get resume metadata to fetch mastercategory and category
            resume_metadata = await self.resume_repo.get_by_id(resume_id)
            if not resume_metadata:
                logger.error(f"Resume ID {resume_id} not found in database")
                return None
            
            mastercategory = resume_metadata.mastercategory
            category = resume_metadata.category
            
            # Normalize mastercategory for consistency
            if mastercategory:
                mastercategory = mastercategory.strip()
            
            # STEP 1: Check if mastercategory exists - required for skills extraction
            if not mastercategory:
                error_msg = (
                    f"❌ SKILLS EXTRACTION SKIPPED: Mastercategory is NULL for resume ID {resume_id}. "
                    f"Mastercategory must be extracted before skills extraction can proceed."
                )
                logger.warning(
                    error_msg,
                    extra={
                        "resume_id": resume_id,
                        "file_name": filename,
                        "mastercategory": mastercategory,
                        "category": category,
                        "error_type": "missing_mastercategory"
                    }
                )
                await self.resume_repo.update(resume_id, {"skillset": None})
                return None
            
            # Validate mastercategory value
            if mastercategory.upper() not in ["IT", "NON_IT"]:
                error_msg = (
                    f"❌ SKILLS EXTRACTION SKIPPED: Invalid mastercategory '{mastercategory}' for resume ID {resume_id}. "
                    f"Expected 'IT' or 'NON_IT'."
                )
                logger.error(
                    error_msg,
                    extra={
                        "resume_id": resume_id,
                        "file_name": filename,
                        "mastercategory": mastercategory,
                        "category": category,
                        "error_type": "invalid_mastercategory"
                    }
                )
                await self.resume_repo.update(resume_id, {"skillset": None})
                return None
            
            custom_prompt = None
            normalized_category = self._normalize_category(category) if category else None
            
            # Normalize mastercategory for prompt lookup (handles "NON_IT" → "non IT" conversion)
            normalized_mastercategory = self._normalize_mastercategory_for_prompt(mastercategory)
            
            # STEP 2: Determine which category to use and fetch prompt from database
            if category:
                # Normalize category for lookup (try both original and normalized)
                category_to_try = category.strip()  # Use original first for exact match
                
                # STEP 2A: category exists - try specific category prompt first
                logger.info(
                    f"Fetching prompt from database for mastercategory={mastercategory} (normalized: {normalized_mastercategory}), category={category_to_try}",
                    extra={
                        "resume_id": resume_id,
                        "mastercategory": mastercategory,
                        "normalized_mastercategory": normalized_mastercategory,
                        "category": category_to_try,
                        "normalized_category": normalized_category
                    }
                )
                prompt_record = await self.prompt_repo.get_by_category(normalized_mastercategory, category_to_try)
                
                if prompt_record and prompt_record.prompt:
                    custom_prompt = prompt_record.prompt
                    logger.info(
                        f"✅ Found specific prompt in database for category '{category}'",
                        extra={
                            "resume_id": resume_id,
                            "prompt_id": prompt_record.id,
                            "prompt_length": len(custom_prompt)
                        }
                    )
                else:
                    # STEP 2B: Specific category not found - fallback to "other" category
                    logger.warning(
                        f"⚠️ No specific prompt found for category '{category}', trying 'other' prompt for mastercategory '{mastercategory}' (normalized: {normalized_mastercategory})",
                        extra={
                            "resume_id": resume_id,
                            "category": category,
                            "normalized_category": normalized_category,
                            "mastercategory": mastercategory,
                            "normalized_mastercategory": normalized_mastercategory,
                            "fallback_reason": "specific_category_not_found"
                        }
                    )
                    prompt_record = await self.prompt_repo.get_by_category(normalized_mastercategory, "other")
                    if prompt_record and prompt_record.prompt:
                        custom_prompt = prompt_record.prompt
                        logger.info(
                            f"✅ Found 'other' prompt for mastercategory '{mastercategory}' (fallback from category '{category}')",
                            extra={
                                "resume_id": resume_id,
                                "prompt_id": prompt_record.id,
                                "prompt_length": len(custom_prompt),
                                "original_category": category
                            }
                        )
            else:
                # STEP 2C: category is NULL - use "other" category directly
                logger.info(
                    f"Category is NULL, using 'other' prompt for mastercategory '{mastercategory}' (normalized: {normalized_mastercategory})",
                    extra={
                        "resume_id": resume_id,
                        "mastercategory": mastercategory,
                        "normalized_mastercategory": normalized_mastercategory,
                        "category": category,
                        "fallback_reason": "category_is_null"
                    }
                )
                prompt_record = await self.prompt_repo.get_by_category(normalized_mastercategory, "other")
                if prompt_record and prompt_record.prompt:
                    custom_prompt = prompt_record.prompt
                    logger.info(
                        f"✅ Found 'other' prompt for mastercategory '{mastercategory}'",
                        extra={
                            "resume_id": resume_id,
                            "prompt_id": prompt_record.id,
                            "prompt_length": len(custom_prompt)
                        }
                    )
            
            # STEP 3: Check if prompt was found - required for extraction
            if not custom_prompt:
                error_msg = (
                    f"❌ SKILLS EXTRACTION FAILED: No prompt found in database for "
                    f"mastercategory='{mastercategory}', category='{category or 'other'}'. "
                    f"Please ensure the required prompt exists in the prompts table. "
                    f"For mastercategory='{mastercategory}', at minimum the 'other' category prompt must exist."
                )
                logger.error(
                    error_msg,
                    extra={
                        "resume_id": resume_id,
                        "mastercategory": mastercategory,
                        "category": category,
                        "file_name": filename,
                        "error_type": "prompt_not_found",
                        "required_prompt": f"{mastercategory} + 'other'"
                    }
                )
                await self.resume_repo.update(resume_id, {"skillset": None})
                return None
            
            # STEP 4: Extract skills using prompt from database (NO GATEWAY)
            skills = await self.skills_extractor.extract_skills(
                resume_text, 
                filename,
                custom_prompt=custom_prompt  # Always use DB prompt, never gateway
            )
            
            # Convert list to comma-separated string for database storage
            skillset = ", ".join(skills) if skills else None
            
            logger.info(
                f"📊 SKILLS EXTRACTION RESULT for resume ID {resume_id}: {len(skills)} skills",
                extra={"resume_id": resume_id, "skills_count": len(skills), "file_name": filename}
            )
            
            # Update the database record
            if skillset:
                logger.info(
                    f"💾 UPDATING DATABASE: Resume ID {resume_id} with {len(skills)} skills",
                    extra={"resume_id": resume_id, "skills_count": len(skills), "file_name": filename}
                )
                
                updated_resume = await self.resume_repo.update(resume_id, {"skillset": skillset})
                if updated_resume:
                    logger.info(
                        f"✅ DATABASE UPDATED: Successfully saved skills for resume ID {resume_id}",
                        extra={"resume_id": resume_id, "skills_count": len(skills)}
                    )
                else:
                    logger.error(f"❌ DATABASE UPDATE FAILED: Resume ID {resume_id} - record not found")
            else:
                logger.warning(
                    f"💾 SAVING NULL: No skills found for resume ID {resume_id}, saving as NULL",
                    extra={"resume_id": resume_id, "file_name": filename}
                )
                await self.resume_repo.update(resume_id, {"skillset": None})
            
            return skillset
            
        except Exception as e:
            logger.error(
                f"ERROR: Failed to extract and save skills for resume ID {resume_id}: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "resume_id": resume_id,
                    "file_name": filename,
                    "resume_text_length": len(resume_text) if resume_text else 0
                },
                exc_info=True
            )
            try:
                await self.resume_repo.update(resume_id, {"skillset": None})
                logger.info(f"Saved NULL skillset for resume ID {resume_id} after extraction failure")
            except Exception as db_error:
                logger.error(f"Failed to update database with NULL skillset: {db_error}")
            
            return None

