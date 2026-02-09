"""Location extraction module for extracting candidate location from resumes."""
from app.location.location_extractor import LocationExtractor, LOCATION_PROMPT
from app.location.location_service import LocationService

__all__ = [
    "LocationExtractor",
    "LocationService",
    "LOCATION_PROMPT",
]
