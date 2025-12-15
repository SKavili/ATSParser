"""Constants for resume processing status."""

# Status values
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# Failure reasons (will be combined with STATUS_FAILED)
FAILURE_FILE_TOO_LARGE = "file_too_large"
FAILURE_INVALID_FILE_TYPE = "invalid_file_type"
FAILURE_EMPTY_FILE = "empty_file"
FAILURE_INSUFFICIENT_TEXT = "insufficient_text"
FAILURE_EXTRACTION_ERROR = "extraction_error"
FAILURE_DESIGNATION_EXTRACTION_FAILED = "designation_extraction_failed"
FAILURE_DATABASE_ERROR = "database_error"
FAILURE_UNKNOWN_ERROR = "unknown_error"

# Helper function to create failure status with reason
def get_failure_status(reason: str) -> str:
    """Get failure status string with reason."""
    return f"{STATUS_FAILED}:{reason}"

# Helper function to parse failure status
def parse_failure_status(status: str) -> tuple[str, str | None]:
    """Parse status to get base status and failure reason if any.
    
    Returns:
        tuple: (base_status, failure_reason)
    """
    if status and status.startswith(f"{STATUS_FAILED}:"):
        reason = status[len(f"{STATUS_FAILED}:"):]
        return STATUS_FAILED, reason
    return status, None

# Valid status values
VALID_STATUSES = [
    STATUS_PENDING,
    STATUS_PROCESSING,
    STATUS_COMPLETED,
    STATUS_FAILED,
]

