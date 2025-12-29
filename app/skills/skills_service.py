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
    
    async def extract_and_save_skills(
        self,
        resume_text: str,
        resume_id: int,
        filename: str = "resume"
    ) -> Optional[str]:
        """
        Extract skills from resume text and update the database record.
        Uses prompt from prompts table based on mastercategory and category.
        
        Args:
            resume_text: The text content of the resume
            resume_id: The ID of the resume record in the database
            filename: Name of the resume file (for logging)
        
        Returns:
            Comma-separated string of skills or None if not found
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
            
            custom_prompt = None
            mastercategory = resume_metadata.mastercategory
            category = resume_metadata.category
            
            # Try to fetch prompt from database based on mastercategory and category
            if mastercategory and category:
                logger.info(
                    f"Fetching prompt from database for mastercategory={mastercategory}, category={category}",
                    extra={
                        "resume_id": resume_id,
                        "mastercategory": mastercategory,
                        "category": category
                    }
                )
                prompt_record = await self.prompt_repo.get_by_category(mastercategory, category)
                
                if prompt_record and prompt_record.prompt:
                    custom_prompt = prompt_record.prompt
                    logger.info(
                        f"✅ Found prompt in database for category '{category}'",
                        extra={
                            "resume_id": resume_id,
                            "prompt_id": prompt_record.id,
                            "prompt_length": len(custom_prompt)
                        }
                    )
                else:
                    # Try to get generic prompt for mastercategory (category=NULL)
                    logger.warning(
                        f"No specific prompt found for category '{category}', trying generic prompt for mastercategory '{mastercategory}'",
                        extra={"resume_id": resume_id, "category": category}
                    )
                    prompt_record = await self.prompt_repo.get_by_mastercategory(mastercategory)
                    if prompt_record and prompt_record.prompt:
                        custom_prompt = prompt_record.prompt
                        logger.info(
                            f"✅ Found generic prompt for mastercategory '{mastercategory}'",
                            extra={
                                "resume_id": resume_id,
                                "prompt_id": prompt_record.id,
                                "prompt_length": len(custom_prompt)
                            }
                        )
                    else:
                        logger.warning(
                            f"No prompt found in database, falling back to gateway-based routing",
                            extra={"resume_id": resume_id, "mastercategory": mastercategory, "category": category}
                        )
            else:
                logger.warning(
                    f"Mastercategory or category is None, falling back to gateway-based routing",
                    extra={
                        "resume_id": resume_id,
                        "mastercategory": mastercategory,
                        "category": category
                    }
                )
            
            # Extract skills using custom prompt or gateway routing
            skills = await self.skills_extractor.extract_skills(
                resume_text, 
                filename,
                custom_prompt=custom_prompt
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

