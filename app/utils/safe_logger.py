"""Safe logging wrapper that prevents LogRecord attribute conflicts."""
import logging
from typing import Any, Dict, Optional

# Reserved LogRecord attributes that cannot be overwritten
RESERVED_LOGRECORD_ATTRS = {
    'name', 'msg', 'args', 'created', 'filename', 'funcName', 
    'levelname', 'levelno', 'lineno', 'module', 'msecs', 'message',
    'pathname', 'process', 'processName', 'relativeCreated', 'thread',
    'threadName', 'exc_info', 'exc_text', 'stack_info', 'taskName'
}


def safe_extra(extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sanitize extra dict to remove reserved LogRecord attributes.
    
    Args:
        extra: Dictionary of extra attributes for logging
        
    Returns:
        Sanitized dictionary with reserved attributes removed or renamed
    """
    if not extra:
        return {}
    
    safe_dict = {}
    for key, value in extra.items():
        if key in RESERVED_LOGRECORD_ATTRS:
            # Rename reserved attributes
            if key == 'filename':
                safe_dict['file_name'] = value
            else:
                # For other reserved attrs, prefix with underscore
                safe_dict[f'_{key}'] = value
        else:
            safe_dict[key] = value
    
    return safe_dict


def safe_log(logger: logging.Logger, level: int, msg: str, *args, extra: Optional[Dict[str, Any]] = None, **kwargs):
    """
    Safe logging wrapper that prevents LogRecord attribute conflicts.
    
    Usage:
        from app.utils.safe_logger import safe_log
        safe_log(logger, logging.INFO, "Message", extra={"filename": "test.txt"})
        # Will automatically convert "filename" to "file_name"
    """
    safe_extra_dict = safe_extra(extra)
    logger.log(level, msg, *args, extra=safe_extra_dict if safe_extra_dict else None, **kwargs)

