"""Utility module for isolating education-relevant text from resume content."""
import re
from typing import List, Set
from app.utils.logging import get_logger

logger = get_logger(__name__)


def isolate_education_text(resume_text: str) -> str:
    """
    Extract education-relevant text from resume by finding education/academic keywords
    and extracting surrounding context (1 line before + keyword line + 4 lines after).
    
    Args:
        resume_text: The full resume text content
    
    Returns:
        Concatenated string containing all education-relevant sections
    """
    if not resume_text or not resume_text.strip():
        logger.warning("Empty resume text provided for education isolation")
        return ""
    
    # Split resume text into lines for processing
    lines = resume_text.split('\n')
    total_lines = len(lines)
    
    if total_lines == 0:
        return ""
    
    # Keywords to search for (case-insensitive)
    education_keywords = ['Education',
    'academic',
    'qualification',
    'qualifications',
    'degree',
    'university',
    'college',
    'institute']
    
    # Track which line indices have been extracted (to avoid duplicates)
    extracted_indices: Set[int] = set()
    
    # List to store extracted text chunks
    extracted_chunks: List[str] = []
    
    # Search through all lines for education/academic keywords
    for line_idx, line in enumerate(lines):
        line_lower = line.lower()
        
        # DEBUG 1: Confirm resume scanning
        logger.debug(
            "EDU_SCAN_LINE",
            extra={
                "line_number": line_idx + 1,
                "line_preview": line.strip()[:100]
            }
        )
        
        # DEBUG 2: Detect keyword presence (case-insensitive visibility check)
        visible_match = any(k.lower() in line_lower for k in education_keywords)
        if visible_match:
            logger.warning(
                "EDU_KEYWORD_VISIBLE",
                extra={
                    "line_number": line_idx + 1,
                    "line_text": line.strip()
                }
            )
        
        # EXISTING LOGIC (DO NOT MODIFY)
        # Check if this line contains any education keyword
        contains_keyword = any(keyword in line_lower for keyword in education_keywords)
        
        if contains_keyword:
            # Calculate the range: 1 line before + current line + 4 lines after
            start_idx = max(0, line_idx - 1)
            end_idx = min(total_lines, line_idx + 5)  # +5 because we want line_idx + 0,1,2,3,4 (5 lines total)
            
            # Check if we've already extracted any of these lines (to avoid duplicates)
            range_indices = set(range(start_idx, end_idx))
            
            # DEBUG 3: Overlap skip
            if range_indices & extracted_indices:
                logger.error(
                    "EDU_EXTRACTION_SKIPPED_OVERLAP",
                    extra={
                        "line_number": line_idx + 1,
                        "line_text": line.strip(),
                        "overlap_indices": list(range_indices & extracted_indices)
                    }
                )
                # Some lines already extracted, skip to avoid duplication
                continue
            
            # DEBUG 4: Extraction triggered
            logger.info(
                "EDU_EXTRACTION_TRIGGERED",
                extra={
                    "line_number": line_idx + 1,
                    "start_line": start_idx + 1,
                    "end_line": end_idx,
                    "extracted_preview": "\n".join(lines[start_idx:end_idx])[:200]
                }
            )
            
            # Extract the text chunk (1 before + current + 4 after = 6 lines total)
            chunk_lines = lines[start_idx:end_idx]
            chunk_text = '\n'.join(chunk_lines)
            
            # Add to extracted chunks
            extracted_chunks.append(chunk_text)
            
            # Mark these indices as extracted
            extracted_indices.update(range_indices)
            
            logger.debug(
                f"Found education keyword at line {line_idx + 1}, extracted lines {start_idx + 1}-{end_idx}",
                extra={
                    "line_index": line_idx,
                    "start_index": start_idx,
                    "end_index": end_idx,
                    "chunk_preview": chunk_text[:100]
                }
            )
    
    # Final summary log
    logger.critical(
        "EDU_EXTRACTION_SUMMARY",
        extra={
            "total_lines_scanned": total_lines,
            "sections_extracted": len(extracted_chunks),
            "original_text_length": len(resume_text)
        }
    )
    
    # Concatenate all extracted chunks with double newline separator
    if extracted_chunks:
        isolated_text = '\n\n'.join(extracted_chunks)
        logger.info(
            f"Isolated education text: {len(extracted_chunks)} section(s), {len(isolated_text)} characters",
            extra={
                "num_sections": len(extracted_chunks),
                "isolated_text_length": len(isolated_text),
                "original_text_length": len(resume_text)
            }
        )
        return isolated_text
    else:
        logger.warning("No education keywords found in resume text")
        return ""

