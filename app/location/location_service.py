"""Service for extracting and saving candidate location to database."""
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.location.location_extractor import LocationExtractor
from app.repositories.resume_repo import ResumeRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)


class LocationService:
    """Service for extracting candidate location from resume and saving to database."""

    def __init__(self, session: AsyncSession):
        self.location_extractor = LocationExtractor()
        self.resume_repo = ResumeRepository(session)

    async def extract_and_save_location(
        self,
        resume_text: str,
        resume_id: int,
        filename: str = "resume",
    ) -> Optional[str]:
        """
        Extract candidate location from resume text and update the database record.

        Args:
            resume_text: The text content of the resume.
            resume_id: The ID of the resume record in the database.
            filename: Name of the resume file (for logging).

        Returns:
            The extracted location string or None if not found.
        """
        try:
            logger.info(
                f"🔍 STARTING LOCATION EXTRACTION for resume ID {resume_id}",
                extra={"resume_id": resume_id, "file_name": filename},
            )

            location = await self.location_extractor.extract_location(resume_text, filename)

            logger.info(
                f"📊 LOCATION EXTRACTION RESULT for resume ID {resume_id}: {location}",
                extra={"resume_id": resume_id, "extracted_location": location, "file_name": filename},
            )

            await self.resume_repo.update(resume_id, {"location": location})

            if location:
                logger.info(
                    f"💾 UPDATED resume ID {resume_id} with location: '{location}'",
                    extra={"resume_id": resume_id, "location": location, "file_name": filename},
                )
            else:
                logger.debug(f"Saved NULL location for resume ID {resume_id}")

            return location

        except Exception as e:
            logger.error(
                f"ERROR: Failed to extract and save location for resume ID {resume_id}: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "resume_id": resume_id,
                    "file_name": filename,
                },
                exc_info=True,
            )
            try:
                await self.resume_repo.update(resume_id, {"location": None})
            except Exception as db_error:
                logger.error(f"Failed to update database with NULL location: {db_error}")
            return None
