"""Service for extracting years of experience from resumes using OLLAMA LLM."""
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

EXPERIENCE_PROMPT = """
IMPORTANT: This is a FRESH, ISOLATED extraction task. Ignore any previous context or conversations.

ROLE:
You are an ATS resume parsing expert specializing in US IT staffing profiles.

CONTEXT:
Candidate profiles and resumes may be unstructured and inconsistently formatted.
Experience refers to the total years of professional work experience of the candidate.

TASK:
Extract the candidate's total years of experience from the profile text.

SELECTION RULES (in priority order):
1. FIRST: Look for explicitly stated total experience in summary sections, profile sections, or header areas.
   Examples: "18+ years of experience", "over 25+ years of experience", "10 years experience", 
   "Total Work Experience: 18 years", "18 years of experience", "Experience: 15+ years",
   "with 20+ years", "having 12 years", "18 years", "25+ years"

2. SECOND: Look for experience stated in work history sections:
   Examples: "Total Work Experience: X years", "Work Experience: X years", "Years of Experience: X"

3. THIRD: If no explicit total is found, calculate from work history entries:
   - Sum up years from all job positions listed
   - Consider date ranges (e.g., "Jan 2020 - Present" = current year - 2020)
   - Consider only professional work experience, not internships or education

4. Look for patterns like:
   - "X years" (e.g., "5 years", "10 years")
   - "X+ years" (e.g., "5+ years", "10+ years")
   - "over X years" (e.g., "over 25 years")
   - "X-Y years" (e.g., "3-5 years") - use the higher number
   - "X to Y years" (e.g., "5 to 10 years") - use the higher number
   - "more than X years" (e.g., "more than 10 years")
   - "X+ years of experience" (e.g., "18+ years of experience")

CONSTRAINTS:
- Return experience as a string (e.g., "5 years", "10+ years", "18+ years", "25+ years").
- Preserve the format as written if explicitly stated (especially "+" signs).
- If calculated, return in format "X years" or "X+ years" if the original had a "+".
- Always include "years" in the response.

ANTI-HALLUCINATION RULES:
- Extract experience ONLY if it is clearly stated or can be reliably calculated from work history.
- Do not count education or internship years unless explicitly stated as experience.
- If experience is ambiguous or cannot be determined, return null.

OUTPUT FORMAT:
Return only valid JSON. No additional text. No explanations. No markdown formatting.

JSON SCHEMA:
{
  "experience": "string | null"
}

Example valid outputs:
{"experience": "5 years"}
{"experience": "10+ years"}
{"experience": "18+ years"}
{"experience": "25+ years"}
{"experience": null}
"""


class ExperienceExtractor:
    """Service for extracting years of experience from resume text using OLLAMA LLM."""
    
    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.model = "llama3.1"
    
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
    
    def _extract_experience_fallback(self, resume_text: str) -> Optional[str]:
        """
        Fallback regex-based extraction if LLM fails.
        Looks for common experience patterns in the resume text.
        """
        if not resume_text:
            logger.warning("Fallback extraction: resume_text is empty")
            return None
        
        logger.info(f"🔍 FALLBACK EXTRACTION: Starting regex-based experience extraction")
        logger.debug(f"Resume text length: {len(resume_text)} characters")
        
        # Common patterns for experience - capture full "X years" or "X+ years" format
        # Order matters - more specific patterns first
        patterns = [
            # Most specific patterns first - these should match "18+ years of experience"
            r'(\d+\+?\s*years?)\s+of\s+experience',  # "18+ years of experience"
            r'(\d+\+?\s*years?)\s+of\s+professional\s+experience',
            r'(?:total\s+)?(?:work\s+)?experience[:\s]+(\d+\+?\s*years?)',  # "Total Work Experience: 18 years"
            r'over\s+(\d+\+?\s*years?)',  # "over 25+ years"
            r'more\s+than\s+(\d+\+?\s*years?)',
            r'(\d+\+?\s*years?)\s+experience',  # "18+ years experience"
            r'experience[:\s]+(\d+\+?\s*years?)',
            r'(\d+\+?\s*years?)\s+in\s+(?:the\s+)?(?:field|industry|profession)',
            r'(\d+\+?\s*years?)\s+professional',
            r'(\d+\+?\s*years?)\s+work',
            r'(\d+\+?\s*years?)\s+in\s+',
            r'with\s+(\d+\+?\s*years?)',  # "with 18+ years"
            r'having\s+(\d+\+?\s*years?)',
        ]
        
        # Search in first 15000 characters (usually contains summary/profile sections)
        search_text = resume_text[:15000]
        logger.debug(f"Searching in first {len(search_text)} characters")
        logger.debug(f"First 500 chars of search text: {search_text[:500]}")
        
        # Try patterns in order, return first match
        for idx, pattern in enumerate(patterns):
            try:
                matches = re.findall(pattern, search_text, re.IGNORECASE)
                logger.debug(f"Pattern {idx+1}/{len(patterns)} '{pattern}': found {len(matches)} matches")
                
                if matches:
                    # Get the first match and normalize it
                    exp_str = matches[0].strip()
                    logger.debug(f"Raw match: '{exp_str}'")
                    
                    # Ensure it has "years" or "year"
                    if 'year' not in exp_str.lower():
                        # Extract number and add "years"
                        num_match = re.search(r'(\d+\+?)', exp_str)
                        if num_match:
                            exp_str = f"{num_match.group(1)} years"
                        else:
                            exp_str = f"{exp_str} years"
                    
                    # Normalize spacing and ensure proper format
                    exp_str = re.sub(r'\s+', ' ', exp_str).strip()
                    
                    # Validate: should be between 0 and 50 years (reasonable range)
                    num_match = re.search(r'(\d+)', exp_str)
                    if num_match:
                        years_num = int(num_match.group(1))
                        if years_num > 50:
                            logger.debug(f"Skipping match '{exp_str}' - value {years_num} > 50 years")
                            continue  # Skip unrealistic values
                    
                    logger.info(f"✅ Fallback regex extracted experience: '{exp_str}' using pattern {idx+1}")
                    return exp_str
            except Exception as e:
                logger.warning(f"Error processing pattern {idx+1} '{pattern}': {e}")
                continue
        
        # If no pattern matched, try a more aggressive search in the first 2000 chars
        logger.debug("No matches found with standard patterns, trying aggressive search in first 2000 chars")
        aggressive_text = resume_text[:2000].lower()
        
        # Look for any "X+ years" or "X years" near "experience" keyword
        aggressive_pattern = r'(\d+\+?\s*years?)'
        aggressive_matches = re.findall(aggressive_pattern, aggressive_text, re.IGNORECASE)
        
        if aggressive_matches:
            # Find the position of each match and check context
            for match in aggressive_matches[:5]:  # Check first 5 matches
                # Escape special regex characters in the match string
                escaped_match = re.escape(match)
                match_obj = re.search(escaped_match, aggressive_text, re.IGNORECASE)
                if match_obj:
                    start_pos = match_obj.start()
                    context_start = max(0, start_pos - 150)
                    context_end = min(len(aggressive_text), start_pos + len(match) + 150)
                    context = aggressive_text[context_start:context_end]
                    
                    # Must be near "experience" keyword
                    if 'experience' in context:
                        # Skip if it's clearly a skill (has dash or bullet before number)
                        if not re.search(r'[-•]\s*\d+\s*years?', context):
                            exp_str = match.strip()
                            if 'year' not in exp_str.lower():
                                num_match = re.search(r'(\d+\+?)', exp_str)
                                if num_match:
                                    exp_str = f"{num_match.group(1)} years"
                            
                            exp_str = re.sub(r'\s+', ' ', exp_str).strip()
                            
                            # Validate years
                            num_match = re.search(r'(\d+)', exp_str)
                            if num_match:
                                years_num = int(num_match.group(1))
                                if 1 <= years_num <= 50:
                                    logger.info(f"✅ Fallback aggressive search extracted experience: '{exp_str}'")
                                    return exp_str
        
        logger.warning("❌ Fallback extraction: No experience pattern found in resume text")
        return None
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON object from LLM response."""
        if not text:
            logger.warning("Empty response from LLM")
            return {"experience": None}
        
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
            cleaned_text = cleaned_text[start_idx:end_idx + 1]
        
        try:
            parsed = json.loads(cleaned_text)
            if isinstance(parsed, dict) and "experience" in parsed:
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
                    if isinstance(parsed, dict) and "experience" in parsed:
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
        return {"experience": None}
    
    async def extract_experience(self, resume_text: str, filename: str = "resume") -> Optional[str]:
        """
        Extract years of experience from resume text using OLLAMA LLM.
        
        Args:
            resume_text: The text content of the resume
            filename: Name of the resume file (for logging)
        
        Returns:
            The extracted experience string or None if not found
        """
        model_to_use = self.model  # Initialize early for error handling
        try:
            is_connected, available_model = await self._check_ollama_connection()
            if not is_connected:
                # Try fallback before raising error
                logger.warning(f"OLLAMA not connected, trying fallback extraction for {filename}")
                experience = self._extract_experience_fallback(resume_text)
                if experience:
                    logger.info(f"✅ EXPERIENCE EXTRACTED via fallback (OLLAMA not connected) from {filename}: {experience}")
                    return experience
                raise RuntimeError(
                    f"OLLAMA is not accessible at {self.ollama_host}. "
                    "Please ensure OLLAMA is running. Start it with: ollama serve"
                )
            
            model_to_use = self.model
            if available_model and "llama3.1" not in available_model.lower():
                logger.warning(
                    f"llama3.1 not found, using available model: {available_model}",
                    extra={"available_model": available_model}
                )
                model_to_use = available_model
            
            # Increase text limit to capture more content, especially from summary sections
            text_to_send = resume_text[:20000]  # Increased from 10000 to capture more content
            
            prompt = f"""{EXPERIENCE_PROMPT}

Input resume text:
{text_to_send}

Output (JSON only, no other text, no explanations):"""
            
            logger.info(
                f"📤 CALLING OLLAMA API for experience extraction",
                extra={
                    "file_name": filename,
                    "model": model_to_use,
                    "ollama_host": self.ollama_host,
                }
            )
            
            result = None
            last_error = None
            
            async with httpx.AsyncClient(timeout=Timeout(1200.0)) as client:
                try:
                    response = await client.post(
                        f"{self.ollama_host}/api/generate",
                        json={
                            "model": model_to_use,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.1,
                                "top_p": 0.9,
                            }
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    response_text = result.get("response", "") or result.get("text", "")
                    if not response_text and "message" in result:
                        response_text = result.get("message", {}).get("content", "")
                    result = {"response": response_text}
                    logger.info("✅ Successfully used /api/generate endpoint for experience extraction")
                except httpx.HTTPStatusError as e:
                    if e.response.status_code != 404:
                        raise
                    last_error = e
                    logger.warning("OLLAMA /api/generate returned 404, trying /api/chat endpoint")
                
                if result is None:
                    try:
                        # Use /api/chat with fresh conversation (no history)
                        # System message ensures complete session isolation
                        response = await client.post(
                            f"{self.ollama_host}/api/chat",
                            json={
                                "model": model_to_use,
                                "messages": [
                                    {"role": "system", "content": "You are a fresh, isolated extraction agent. This is a new, independent task with no previous context. Ignore any previous conversations."},
                                    {"role": "user", "content": prompt}
                                ],
                                "stream": False,
                                "options": {
                                    "temperature": 0.1,
                                    "top_p": 0.9,
                                    "num_predict": 500,  # Limit response length for isolation
                                }
                            }
                        )
                        response.raise_for_status()
                        result = response.json()
                        if "message" in result and "content" in result["message"]:
                            result = {"response": result["message"]["content"]}
                        else:
                            raise ValueError("Unexpected response format from OLLAMA chat API")
                        logger.info("Successfully used /api/chat endpoint for experience extraction")
                    except Exception as e2:
                        last_error = e2
                        logger.error(f"OLLAMA /api/chat also failed: {e2}", extra={"error": str(e2)})
                
                if result is None:
                    raise RuntimeError(
                        f"All OLLAMA API endpoints failed. "
                        f"OLLAMA is running at {self.ollama_host} but endpoints return errors. "
                        f"Last error: {last_error}"
                    )
            
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
            experience = parsed_data.get("experience")
            
            if experience:
                experience = str(experience).strip()
                if not experience or experience.lower() in ["null", "none", ""]:
                    experience = None
            
            # Fallback to regex extraction if LLM returned null
            if not experience:
                logger.warning(
                    f"LLM returned null experience for {filename}, trying fallback regex extraction",
                    extra={"file_name": filename}
                )
                experience = self._extract_experience_fallback(resume_text)
                if experience:
                    logger.info(
                        f"✅ EXPERIENCE EXTRACTED via fallback from {filename}",
                        extra={
                            "file_name": filename,
                            "experience": experience,
                            "method": "regex_fallback"
                        }
                    )
            
            logger.info(
                f"✅ EXPERIENCE EXTRACTED from {filename}",
                extra={
                    "file_name": filename,
                    "experience": experience,
                    "status": "success" if experience else "not_found",
                    "method": "llm" if parsed_data.get("experience") else "fallback"
                }
            )
            
            return experience
            
        except httpx.HTTPError as e:
            error_details = {
                "error": str(e),
                "error_type": type(e).__name__,
                "ollama_host": self.ollama_host,
                "model": model_to_use,
            }
            logger.warning(
                f"HTTP error calling OLLAMA for experience extraction: {e}. Trying fallback regex extraction.",
                extra=error_details
            )
            # Try fallback extraction even when LLM fails
            experience = self._extract_experience_fallback(resume_text)
            if experience:
                logger.info(
                    f"✅ EXPERIENCE EXTRACTED via fallback after LLM failure from {filename}",
                    extra={
                        "file_name": filename,
                        "experience": experience,
                        "method": "regex_fallback_after_llm_error"
                    }
                )
                return experience
            # If fallback also fails, raise the error
            logger.error(
                f"Both LLM and fallback extraction failed for {filename}",
                extra=error_details,
                exc_info=True
            )
            raise RuntimeError(f"Failed to extract experience with LLM and fallback: {e}")
        except Exception as e:
            logger.warning(
                f"Error extracting experience with LLM: {e}. Trying fallback regex extraction.",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "ollama_host": self.ollama_host,
                    "model": model_to_use,
                }
            )
            # Try fallback extraction even when other errors occur
            experience = self._extract_experience_fallback(resume_text)
            if experience:
                logger.info(
                    f"✅ EXPERIENCE EXTRACTED via fallback after error from {filename}",
                    extra={
                        "file_name": filename,
                        "experience": experience,
                        "method": "regex_fallback_after_error"
                    }
                )
                return experience
            # If fallback also fails, raise the original error
            logger.error(
                f"Both LLM and fallback extraction failed for {filename}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "ollama_host": self.ollama_host,
                    "model": model_to_use,
                },
                exc_info=True
            )
            raise

