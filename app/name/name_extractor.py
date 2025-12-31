"""Service for extracting candidate names from resumes using optimized regex fallback and OLLAMA LLM."""
import json
import re
from typing import Dict, Optional
import httpx
from httpx import Timeout

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import OLLAMA Python client
try:
    import ollama
    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    OLLAMA_CLIENT_AVAILABLE = False
    logger.warning("OLLAMA Python client not available, using HTTP API directly")

# Detailed prompt for name extraction with anti-hallucination rules
NAME_PROMPT = """IMPORTANT: This is a FRESH, ISOLATED extraction task.
Ignore any previous context, memory, or conversations.

ROLE:
You are an ATS resume parsing expert specializing in US IT staffing profiles.

CONTEXT:
- Candidate resumes may be unstructured, multi-line, or poorly formatted.
- Names may appear with irregular spacing, line breaks, or formatting artifacts.
- Name refers to the candidate's personal full name (first name and last name).

TASK:
Extract the candidate's full name from the profile text.

SELECTION RULES (IN ORDER):
1. Prefer the name appearing in the resume header or top-most section.
2. Else, prefer the name appearing near contact details (email or phone).
3. Extract the most complete explicit personal name found.

NAME NORMALIZATION RULES:
- Preserve the original spelling and capitalization of the name.
- Normalize unintended whitespace artifacts caused by formatting:
- Collapse multiple spaces within a name into a single space ONLY where linguistically valid.
- Do NOT alter legitimate spaces between first and last name.

CONSTRAINTS:
- Do NOT invent, infer, or guess names.
- Do NOT construct names from email IDs, usernames, or file names.
- Do NOT include titles or honorifics (Mr., Ms., Dr., etc.) unless explicitly part of the name.
- If only a partial name is explicitly present, return only that portion.
- Extract exactly ONE name.

ANTI-HALLUCINATION RULES:
- If no explicit personal name is found, return null.
- Never correct spelling.
- Never expand initials.

OUTPUT FORMAT:
Return ONLY valid JSON.
No additional text. No explanations. No markdown.

JSON SCHEMA:
{
  "name": "string | null"
}

VALID EXAMPLES:
{"name": "Dennis Zabluda"}
{"name": "Dennis Z Abluda"}
{"name": "John Doe"}
{"name": null}"""

# Configuration for name extraction
NAME_EXTRACTION_TEXT_LIMIT = 1000  # Use first 1000 characters
NAME_EXTRACTION_MIN_TEXT = 500  # Minimum characters to use
OLLAMA_TIMEOUT = 90.0  # Timeout for OLLAMA API calls (90 seconds to allow sufficient processing time)
OLLAMA_MAX_TOKENS = 100  # Limit response length for speed (enough for JSON response)


class NameExtractor:
    """Service for extracting candidate names from resume text using regex fallback and OLLAMA LLM."""
    
    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.model = "llama3.1"
    
    def _extract_name_regex_fallback(self, text: str) -> Optional[str]:
        """
        Deterministic regex-based fallback for name extraction.
        Extracts the first non-empty line without numbers, emails, or common resume keywords.
        
        Args:
            text: The resume text (should be first 500-1000 characters)
        
        Returns:
            Extracted name or None if not found
        """
        if not text or len(text.strip()) < 3:
            return None
        
        # Split into lines and process
        lines = text.split('\n')
        
        # Patterns to exclude
        email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
        phone_pattern = re.compile(r'[\d\s\-\(\)\+]{10,}')
        url_pattern = re.compile(r'https?://[^\s]+|www\.[^\s]+')
        date_pattern = re.compile(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[/\-]\d{1,2}[/\-]\d{1,2}')
        
        # Common resume keywords to skip
        skip_keywords = [
            'resume', 'cv', 'curriculum vitae', 'objective', 'summary', 'experience',
            'education', 'skills', 'certifications', 'projects', 'references',
            'phone', 'email', 'address', 'linkedin', 'github', 'portfolio'
        ]
        
        for line in lines:
            original_line = line.strip()
            line = original_line
            
            # Skip empty lines
            if not line or len(line) < 2:
                continue
            
            # Skip lines that start with common resume keywords
            line_lower = line.lower()
            if any(line_lower.startswith(keyword) for keyword in skip_keywords):
                continue
            
            # Try to extract name from lines that may contain name + location + email/phone
            # Pattern: Extract first 1-5 words that look like a name (before location indicators)
            # Look for patterns like: "Name City, State | email" or "Name | email" or "Name Location"
            
            # First, try to extract name from the beginning of the line
            # Match 1-5 capitalized words at the start (name pattern)
            name_at_start_pattern = re.compile(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})')
            match = name_at_start_pattern.match(line)
            
            if match:
                potential_name = match.group(1).strip()
                
                # Check if this looks like a name (not a location or other info)
                # Names typically have 2-4 words, locations often have commas or state codes
                if 2 <= len(potential_name.split()) <= 4:
                    # Check if line continues with location indicators (comma, state codes, |, email, phone)
                    remaining = line[len(potential_name):].strip()
                    
                    # If there's a comma, |, @, or phone pattern after, likely the name part is correct
                    if (',' in remaining or '|' in remaining or 
                        email_pattern.search(remaining) or 
                        phone_pattern.search(remaining) or
                        len(remaining) == 0):  # Or it's just the name on the line
                        
                        # Validate the extracted name
                        if len(potential_name) >= 3 and len(potential_name) <= 50:
                            letter_count = len(re.findall(r'[A-Za-z]', potential_name))
                            if letter_count >= 4:  # At least 4 letters total
                                logger.debug(f"Regex fallback extracted name from line with location/email: {potential_name}")
                                return potential_name
            
            # Fallback: Check if entire line looks like a name (for simpler cases)
            if len(line) <= 60:
                # Skip lines with emails, phone, URLs, dates
                if (email_pattern.search(line) or 
                    phone_pattern.search(line) or 
                    url_pattern.search(line) or 
                    date_pattern.search(line)):
                    continue
                
                # More flexible name pattern: 1-5 words (allows single names, middle initials, etc.)
                name_pattern = re.compile(r'^[A-Za-z][A-Za-z\-\'\.]*(?:\s+[A-Za-z][A-Za-z\-\'\.]*){0,4}$')
                
                if name_pattern.match(line):
                    # Clean up common prefixes/suffixes
                    cleaned = re.sub(r'^(Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|Miss\.)\s+', '', line, flags=re.IGNORECASE)
                    cleaned = cleaned.strip()
                    
                    # More lenient validation: at least 2 characters, mostly letters
                    if cleaned and len(cleaned) >= 2:
                        # Check if it's mostly letters (allow hyphens, apostrophes, periods, spaces)
                        letter_count = len(re.findall(r'[A-Za-z]', cleaned))
                        total_chars = len(cleaned.replace(' ', ''))
                        if total_chars > 0 and letter_count / total_chars >= 0.7:  # At least 70% letters
                            logger.debug(f"Regex fallback extracted name: {cleaned}")
                            return cleaned
        
        return None
    
    async def _check_ollama_connection(self) -> tuple[bool, Optional[str]]:
        """Check if OLLAMA is accessible and running. Returns (is_connected, available_model)."""
        try:
            async with httpx.AsyncClient(timeout=Timeout(5.0)) as client:
                response = await client.get(f"{self.ollama_host}/api/tags")
                if response.status_code == 200:
                    models_data = response.json()
                    models = models_data.get("models", [])
                    for model in models:
                        model_name = model.get("name", "")
                        if "llama3.1" in model_name.lower() or "llama3" in model_name.lower():
                            return True, model_name
                    if models:
                        return True, models[0].get("name", "")
                    return True, None
                return False, None
        except Exception as e:
            logger.warning(f"Failed to check OLLAMA connection: {e}", extra={"error": str(e)})
            return False, None
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON object from LLM response."""
        if not text:
            logger.warning("Empty response from LLM")
            return {"name": None}
        
        cleaned_text = text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        start_idx = cleaned_text.find('{')
        end_idx = cleaned_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned_taext = cleaned_text[start_idx:end_idx + 1]
        
        try:
            parsed = json.loads(cleaned_text)
            if isinstance(parsed, dict) and "name" in parsed:
                logger.debug(f"Successfully extracted JSON: {parsed}")
                return parsed
        except json.JSONDecodeError:
            pass
        
        try:
            start_idx = cleaned_text.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(cleaned_text)):
                    if cleaned_text[i] == '{':
                        brace_count += 1
                    elif cleaned_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                if brace_count == 0:
                    json_str = cleaned_text[start_idx:end_idx]
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict) and "name" in parsed:
                        logger.debug(f"Successfully extracted JSON with balanced braces: {parsed}")
                        return parsed
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON with balanced braces: {e}")
        
        logger.error(
            "ERROR: Failed to parse JSON from LLM response", 
            extra={
                "response_preview": text[:500],
                "response_length": len(text),
                "cleaned_preview": cleaned_text[:500]
            }
        )
        return {"name": None}
    
    async def extract_name(self, resume_text: str, filename: str = "resume") -> Optional[str]:
        """
        Extract candidate name from resume text using optimized regex fallback and OLLAMA LLM.
        Uses only first 500-1000 characters for speed and reliability.
        Never blocks the pipeline - returns None on any error.
        
        IMPORTANT: This function expects the FULL original resume text, not truncated text.
        It will handle truncation internally to use only the first 1000 characters.
        
        Args:
            resume_text: The FULL text content of the resume (not truncated)
            filename: Name of the resume file (for logging)
        
        Returns:
            The extracted candidate name or None if not found
        """
        # Step 1: Ensure we have the full original text (not truncated first 2000 + last 1000)
        # If text appears truncated (has pattern of first+last concatenation), use only the first part
        if not resume_text or len(resume_text.strip()) < 10:
            logger.warning(f"Resume text too short for name extraction: {filename}")
            return None
        
        # CRITICAL: Use ONLY the first 1000 characters from the ORIGINAL text
        # Do NOT use any concatenated first 2000 + last 1000 pattern
        # Names are ALWAYS at the top of resumes, so we only need the very beginning
        text_length = len(resume_text)
        
        # Detect if text appears to be concatenated (first 2000 + last 1000 pattern)
        # If so, use ONLY the first part (where the name actually is)
        # Check: if text is around 3000 chars and has a clear break, it might be concatenated
        if text_length > 2500:
            # Check if this looks like concatenated text (first 2000 + last 1000)
            # Names are in the first part, so we'll use only the first 1000 from the start
            logger.debug(
                f"Text length {text_length} might be concatenated, using only first {NAME_EXTRACTION_TEXT_LIMIT} chars",
                extra={"file_name": filename, "text_length": text_length}
            )
        
        # Always take from the absolute beginning - ignore any truncation that may have happened
        # If text was truncated elsewhere, we still want the FIRST characters only
        if text_length >= NAME_EXTRACTION_TEXT_LIMIT:
            # Take first 1000 characters from the start (regardless of how text was prepared)
            limited_text = resume_text[:NAME_EXTRACTION_TEXT_LIMIT]
        elif text_length >= NAME_EXTRACTION_MIN_TEXT:
            # Use all if between 500-1000
            limited_text = resume_text
        else:
            # Use what we have if less than 500
            limited_text = resume_text
        
        # Log what we're using for debugging
        logger.info(
            f"📝 NAME EXTRACTION: Using {len(limited_text)} characters (from {text_length} total) for {filename}",
            extra={
                "file_name": filename, 
                "text_length": text_length, 
                "limited_length": len(limited_text),
                "text_preview": limited_text[:300],  # First 300 chars for debugging
                "note": "Using ONLY first characters, NOT first 2000 + last 1000"
            }
        )
        
        # Step 2: Try LLM extraction first with detailed prompt (as user requested)
        try:
            name = await self._extract_name_with_llm(limited_text, filename)
            if name:
                logger.info(
                    f"✅ NAME EXTRACTED via LLM from {filename}",
                    extra={
                        "file_name": filename,
                        "extracted_name": name,
                        "method": "llm"
                    }
                )
                return name
            else:
                logger.debug(
                    f"LLM did not find name in {filename}, trying regex fallback",
                    extra={
                        "file_name": filename,
                        "first_lines": "\n".join(limited_text.split('\n')[:5])  # First 5 lines for debugging
                    }
                )
        except Exception as e:
            # Never raise - just log and try regex fallback
            logger.warning(
                f"LLM name extraction failed for {filename}, trying regex fallback: {e}",
                extra={
                    "file_name": filename,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
        
        # Step 3: Try regex-based fallback if LLM fails or times out
        try:
            regex_name = self._extract_name_regex_fallback(limited_text)
            if regex_name:
                logger.info(
                    f"✅ NAME EXTRACTED via regex fallback from {filename}",
                    extra={
                        "file_name": filename,
                        "extracted_name": regex_name,
                        "method": "regex_fallback"
                    }
                )
                return regex_name
        except Exception as e:
            logger.warning(
                f"Regex fallback failed for {filename}: {e}",
                extra={"file_name": filename, "error": str(e)},
                exc_info=True
            )
        
        # Step 4: Return None if nothing found (never block the pipeline)
        logger.info(
            f"⚠️  NO NAME FOUND for {filename}",
            extra={"file_name": filename, "status": "not_found"}
        )
        return None
    
    async def _extract_name_with_llm(self, text: str, filename: str) -> Optional[str]:
        """
        Extract name using LLM with strict timeout and anti-hallucination rules.
        Never blocks - returns None on any error.
        
        Args:
            text: Limited resume text (first 500-1000 characters)
            filename: Name of the resume file (for logging)
        
        Returns:
            Extracted name or None
        """
        try:
            # Quick connection check with short timeout
            is_connected, available_model = await self._check_ollama_connection()
            if not is_connected:
                logger.warning(f"OLLAMA not accessible for {filename}, skipping LLM extraction")
                return None
            
            model_to_use = self.model
            if available_model and "llama3.1" not in available_model.lower():
                model_to_use = available_model
            
            # Detailed prompt with anti-hallucination rules and strict JSON requirement
            prompt = f"{NAME_PROMPT}\n\nInput resume text:\n{text}\n\nOutput (JSON only, no other text, no explanations):"
            
            logger.debug(
                f"📤 CALLING OLLAMA API for name extraction (timeout: {OLLAMA_TIMEOUT}s)",
                extra={
                    "file_name": filename,
                    "model": model_to_use,
                    "ollama_host": self.ollama_host,
                    "text_length": len(text)
                }
            )
            
            result = None
            last_error = None
            
            # Use short timeout to prevent blocking
            async with httpx.AsyncClient(timeout=Timeout(OLLAMA_TIMEOUT)) as client:
                try:
                    # Try /api/generate first
                    response = await client.post(
                        f"{self.ollama_host}/api/generate",
                        json={
                            "model": model_to_use,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.0,  # Deterministic, no creativity
                                "top_p": 0.9,
                                "num_predict": OLLAMA_MAX_TOKENS,  # Limit response length
                            }
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    response_text = result.get("response", "") or result.get("text", "")
                    if not response_text and "message" in result:
                        response_text = result.get("message", {}).get("content", "")
                    result = {"response": response_text}
                    logger.debug("✅ Successfully used /api/generate endpoint for name extraction")
                except httpx.HTTPStatusError as e:
                    if e.response.status_code != 404:
                        raise
                    last_error = e
                    logger.debug("OLLAMA /api/generate returned 404, trying /api/chat endpoint")
                
                if result is None:
                    try:
                        # Try /api/chat with minimal context
                        response = await client.post(
                            f"{self.ollama_host}/api/chat",
                            json={
                                "model": model_to_use,
                                "messages": [
                                    {"role": "user", "content": prompt}
                                ],
                                "stream": False,
                                "options": {
                                    "temperature": 0.0,  # Deterministic, no creativity
                                    "top_p": 0.9,
                                    "num_predict": OLLAMA_MAX_TOKENS,  # Limit response length
                                }
                            }
                        )
                        response.raise_for_status()
                        result = response.json()
                        if "message" in result and "content" in result["message"]:
                            result = {"response": result["message"]["content"]}
                        else:
                            raise ValueError("Unexpected response format from OLLAMA chat API")
                        logger.debug("✅ Successfully used /api/chat endpoint for name extraction")
                    except Exception as e2:
                        last_error = e2
                        logger.debug(f"OLLAMA /api/chat also failed: {e2}")
                
                if result is None:
                    logger.warning(
                        f"All OLLAMA API endpoints failed for {filename}",
                        extra={"last_error": str(last_error) if last_error else "Unknown"}
                    )
                    return None
            
            # Extract JSON from response
            raw_output = ""
            if isinstance(result, dict):
                if "response" in result:
                    raw_output = str(result["response"])
                elif "text" in result:
                    raw_output = str(result["text"])
                elif "content" in result:
                    raw_output = str(result["content"])
                elif "message" in result and isinstance(result.get("message"), dict):
                    raw_output = str(result["message"].get("content", ""))
            else:
                raw_output = str(result)
            
            parsed_data = self._extract_json(raw_output)
            name = parsed_data.get("name")
            
            # Log what we got from LLM for debugging
            logger.debug(
                f"LLM raw response for {filename}",
                extra={
                    "file_name": filename,
                    "raw_output_preview": raw_output[:200],
                    "parsed_name": name
                }
            )
            
            # Anti-hallucination: validate name
            if name:
                name = str(name).strip()
                # Reject if empty, "null", "none", or suspiciously long
                if not name or name.lower() in ["null", "none", ""] or len(name) > 100:
                    logger.debug(f"Rejected name (empty/null/too long): {name}")
                    name = None
                # Reject if contains too many numbers or special characters (except hyphens, apostrophes, spaces, periods)
                elif re.search(r'[0-9]{2,}', name):  # Only reject if 2+ consecutive digits
                    logger.warning(f"Rejected suspicious name with multiple digits: {name}")
                    name = None
                elif re.search(r'[@#$%&*()]', name):  # Reject special characters except allowed ones
                    logger.warning(f"Rejected suspicious name with invalid characters: {name}")
                    name = None
                # Additional check: should have at least 2 letters
                elif len(re.findall(r'[A-Za-z]', name)) < 2:
                    logger.warning(f"Rejected name with too few letters: {name}")
                    name = None
            
            if name:
                logger.debug(f"✅ LLM extracted valid name: {name}")
            else:
                logger.debug(f"⚠️ LLM returned null or invalid name for {filename}")
            
            return name
            
        except httpx.TimeoutException:
            logger.warning(f"OLLAMA timeout for name extraction: {filename}")
            return None
        except httpx.HTTPError as e:
            logger.warning(f"HTTP error calling OLLAMA for name extraction: {e}", extra={"file_name": filename})
            return None
        except Exception as e:
            logger.warning(f"Unexpected error in LLM name extraction: {e}", extra={"file_name": filename})
            return None

